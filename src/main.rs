mod feature_ext;

use std::{
    collections::HashMap,
    mem,
    path::PathBuf,
    sync::mpsc::{self, Sender, SyncSender},
    thread::{self, JoinHandle},
};

use anyhow::Result;
use gdal::{
    raster::{GdalDataType, RasterBand},
    spatial_ref::SpatialRef,
    vector::{
        Feature, FieldDefn, FieldValue, Geometry, LayerAccess, OGRFieldType, OGRwkbGeometryType,
    },
    Dataset, DriverManager, GeoTransformEx, LayerOptions, Metadata,
};
use ndarray::Array2;

use crate::feature_ext::FeatureExt;

#[derive(Clone, Copy)]
enum BandValue {
    U16(u16),
    I16(i16),
    F32(f32),
}

struct FieldDefinition {
    name: String,
    field_type: OGRFieldType::Type,
    width: Option<i32>,
    precision: Option<i32>,
}

impl FieldDefinition {
    fn to_field_defn(&self) -> gdal::errors::Result<FieldDefn> {
        let field_defn = FieldDefn::new(&self.name, self.field_type)?;
        if let Some(width) = self.width {
            field_defn.set_width(width);
        }
        if let Some(precision) = self.precision {
            field_defn.set_precision(precision);
        }
        Ok(field_defn)
    }
}

#[derive(Debug)]
enum TypedBlock {
    U16(Array2<u16>),
    I16(Array2<i16>),
    F32(Array2<f32>),
}

fn read_typed_block<'d>(
    band: &RasterBand<'d>,
    x: usize,
    y: usize,
) -> gdal::errors::Result<TypedBlock> {
    match band.band_type() {
        GdalDataType::UInt16 => {
            let buf = band.read_block::<u16>((x, y))?;
            Ok(TypedBlock::U16(buf))
        }
        GdalDataType::Int16 => {
            let buf = band.read_block::<i16>((x, y))?;
            Ok(TypedBlock::I16(buf))
        }
        GdalDataType::Float32 => {
            let buf = band.read_block::<f32>((x, y))?;
            Ok(TypedBlock::F32(buf))
        }
        _ => unimplemented!(),
    }
}

struct ThreadedBlockReader<T> {
    workers: Vec<JoinHandle<()>>,
    request_txs: Vec<Sender<(usize, usize, T)>>,
    current_worker: usize,
}

impl<T: Send + 'static> ThreadedBlockReader<T> {
    fn new<R: BlockReducer<InputState = T>, F: BlockFinalizer<Input = R::Output>>(
        path: PathBuf,
        block_finalizer: F,
    ) -> Self {
        let num_threads = 8;

        let mut workers = Vec::with_capacity(num_threads);
        let mut request_txs = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let (tx, rx) = mpsc::channel();
            let path = path.clone();
            let block_finalizer = block_finalizer.clone();

            let worker = thread::spawn(move || {
                let dataset = Dataset::open(path).unwrap();
                let band_count = dataset.raster_count() as usize;

                for (x, y, data) in rx {
                    let mut block_reducer = R::new(data);

                    for band_index in 0..band_count {
                        let band = dataset.rasterband(band_index as isize + 1).unwrap();
                        let block = read_typed_block(&band, x, y).unwrap();
                        block_reducer.push_block(band_index, block);
                    }

                    println!("{x} {y}");
                    block_finalizer.apply(block_reducer.finalize());
                }
            });
            workers.push(worker);
            request_txs.push(tx);
        }

        Self {
            workers,
            request_txs,
            current_worker: 0,
        }
    }

    fn submit(&mut self, x: usize, y: usize, data: T) {
        self.request_txs[self.current_worker]
            .send((x, y, data))
            .unwrap();
        self.current_worker += 1;
        if self.current_worker == self.request_txs.len() {
            self.current_worker = 0;
        }
    }

    fn join(self) {
        for worker in self.workers {
            let _ = worker.join();
        }
    }
}

trait BlockReducer {
    type InputState;
    type Output: Send + 'static;

    fn new(input_state: Self::InputState) -> Self;
    fn push_block(&mut self, band_index: usize, block: TypedBlock);
    fn finalize(self) -> Self::Output;
}

trait BlockFinalizer: Clone + Send + 'static {
    type Input: Send;

    fn apply(&self, input: Self::Input);
}

struct SamplingBlockReducer {
    band_count: usize,
    points: Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
    samples: Vec<BandValue>,
}

impl BlockReducer for SamplingBlockReducer {
    type InputState = (
        usize,
        Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
    );
    type Output = (
        Vec<BandValue>,
        Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
    );

    fn new((band_count, points): Self::InputState) -> Self {
        let samples = vec![BandValue::I16(0); points.len() * band_count];

        Self {
            band_count,
            points,
            samples,
        }
    }

    fn push_block(&mut self, band_index: usize, block: TypedBlock) {
        match block {
            TypedBlock::U16(buf) => {
                for (idx, &(bx, by, ..)) in self.points.iter().enumerate() {
                    let pix = buf[(by, bx)];
                    self.samples[self.band_count * idx + band_index] = BandValue::U16(pix);
                }
            }
            TypedBlock::I16(buf) => {
                for (idx, &(bx, by, ..)) in self.points.iter().enumerate() {
                    let pix = buf[(by, bx)];
                    self.samples[self.band_count * idx + band_index] = BandValue::I16(pix);
                }
            }
            TypedBlock::F32(buf) => {
                for (idx, &(bx, by, ..)) in self.points.iter().enumerate() {
                    let pix = buf[(by, bx)];
                    self.samples[self.band_count * idx + band_index] = BandValue::F32(pix);
                }
            }
        }
    }

    fn finalize(self) -> Self::Output {
        (self.samples, self.points)
    }
}

#[derive(Clone)]
struct BlockSender(
    SyncSender<(
        Vec<BandValue>,
        Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
    )>,
);

impl BlockFinalizer for BlockSender {
    type Input = (
        Vec<BandValue>,
        Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
    );

    fn apply(&self, input: Self::Input) {
        self.0.send(input).unwrap()
    }
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let image = Dataset::open(&args[1])?;
    let geo_transform = image.geo_transform()?;
    let geo_transform = geo_transform.invert()?;
    let block_size = image.rasterband(1)?.block_size();
    dbg!(block_size);

    let band_count = image.raster_count() as usize;
    for band_idx in 0..band_count {
        let band = image.rasterband(band_idx as isize + 1)?;
        assert!(band.block_size() == block_size);
    }
    let mut tile_points = HashMap::<_, Vec<_>>::new();
    let point_ds = Dataset::open(&args[2])?;
    let mut layer = point_ds.layer(0)?;
    dbg!(layer.feature_count());
    for feature in layer.features() {
        let (orig_x, orig_y, _) = feature.geometry().as_ref().unwrap().get_point(0);
        let (x, y) = geo_transform.apply(orig_x, orig_y);
        let (block_x, block_y) = (
            (x / block_size.0 as f64) as usize,
            (y / block_size.1 as f64) as usize,
        );
        let (x, y) = (x as usize, y as usize);
        tile_points.entry((block_x, block_y)).or_default().push((
            x % block_size.0,
            y % block_size.1,
            orig_x,
            orig_y,
            feature.fields().map(|f| f.1).collect::<Vec<_>>(),
        ));
    }
    let spatial_ref = layer.spatial_ref().map(|sr| sr.to_wkt()).transpose()?;
    let mut output_fields = Vec::new();
    for field in layer.defn().fields() {
        let field_definition = FieldDefinition {
            name: field.name(),
            field_type: field.field_type(),
            width: Some(field.width()),
            precision: Some(field.precision()),
        };
        output_fields.push(field_definition);
    }
    for band_idx in 1..=band_count {
        let band = image.rasterband(band_idx as isize)?;
        let name = match band.description() {
            Ok(name) if !name.is_empty() => name,
            _ => format!("band_{band_idx}"),
        };
        let field_type = match band.band_type() {
            GdalDataType::UInt16 => OGRFieldType::OFTInteger,
            GdalDataType::Int16 => OGRFieldType::OFTInteger,
            GdalDataType::Float32 => OGRFieldType::OFTReal,
            _ => unimplemented!(),
        };
        let field_definition = FieldDefinition {
            name,
            field_type,
            width: None,
            precision: None,
        };
        output_fields.push(field_definition);
    }

    let (tx, rx) = mpsc::sync_channel::<(
        Vec<BandValue>,
        Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
    )>(128);
    let output_thread = thread::spawn(move || {
        let mut output =
            DriverManager::get_driver_by_name("SQLite")?.create_vector_only("output.sqlite")?;
        let output_layer = output.create_layer(LayerOptions {
            name: "output",
            srs: spatial_ref
                .map(|wkt| SpatialRef::from_wkt(&wkt))
                .transpose()?
                .as_ref(),
            ty: OGRwkbGeometryType::wkbPoint,
            options: None,
        })?;
        for field in output_fields {
            let field_defn = field.to_field_defn()?;
            field_defn.add_to_layer(&output_layer)?;
        }
        for (sample_values, points) in rx {
            let tx = output.start_transaction()?;
            let output_layer = tx.layer(0)?;
            for (idx, (_bx, _by, x, y, fields)) in points.into_iter().enumerate() {
                let mut feature = Feature::new(output_layer.defn())?;
                let field_offset = fields.len();
                for (field_idx, field_value) in fields.into_iter().enumerate() {
                    if let Some(field_value) = field_value {
                        feature.set_field_by_index(field_idx, &field_value);
                    }
                }
                for band_idx in 0..band_count {
                    match sample_values[band_count * idx + band_idx] {
                        BandValue::U16(value) => {
                            feature
                                .set_field_integer_by_index(band_idx + field_offset, value as i32);
                        }
                        BandValue::I16(value) => {
                            feature
                                .set_field_integer_by_index(band_idx + field_offset, value as i32);
                        }
                        BandValue::F32(value) => {
                            feature
                                .set_field_double_by_index(band_idx + field_offset, value as f64);
                        }
                    }
                }
                let mut geometry = Geometry::empty(OGRwkbGeometryType::wkbPoint)?;
                geometry.add_point_2d((x, y));
                feature.set_geometry(geometry)?;
                feature.create(&output_layer)?;
            }
            tx.commit()?;
        }
        output.close()?;
        mem::forget(output);
        Ok::<_, gdal::errors::GdalError>(())
    });

    let mut tile_points = tile_points.into_iter().collect::<Vec<_>>();
    tile_points.sort_by_key(|t| (t.0 .1, t.0 .0));

    let block_sender = BlockSender(tx);

    let mut block_reader =
        ThreadedBlockReader::new::<SamplingBlockReducer, _>(PathBuf::from(&args[1]), block_sender);
    for ((block_x, block_y), points) in tile_points {
        block_reader.submit(block_x, block_y, (band_count, points));
    }
    drop(block_reader);

    output_thread.join().unwrap()?;

    Ok(())
}
