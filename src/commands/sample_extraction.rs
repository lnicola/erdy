use std::{
    collections::HashMap,
    num::NonZeroUsize,
    ops::Range,
    path::PathBuf,
    sync::{
        mpsc::{self, SyncSender},
        Arc,
    },
    thread::{self, available_parallelism},
};

use anyhow::Result;
use clap::Parser;
use gdal::{
    raster::GdalDataType,
    spatial_ref::SpatialRef,
    vector::{
        Feature, FieldDefn, FieldValue, Geometry, LayerAccess, LayerOptions, OGRFieldType,
        OGRwkbGeometryType,
    },
    Dataset, DriverManager, GeoTransformEx, Metadata,
};

use crate::{
    gdal_ext::{FeatureExt, TypedBlock},
    threaded_block_reader::{BlockFinalizer, BlockReducer, ThreadedBlockReader},
};

#[derive(Debug, Parser)]
pub struct SampleExtractionArgs {
    /// Input image
    input: PathBuf,

    /// Sampling positions
    #[arg(long, value_parser, num_args = 1..)]
    points: Vec<PathBuf>,

    /// Output datasets
    #[arg(long, value_parser, num_args = 1..)]
    outputs: Vec<PathBuf>,

    /// Output format
    #[arg(short, long)]
    format: String,

    /// Output field names
    #[arg(long, value_parser, num_args = 1..)]
    fields: Option<Vec<String>>,

    /// Number of threads to use
    #[arg(long)]
    num_threads: Option<usize>,

    /// Copy the input feature ids
    #[arg(long, default_value_t)]
    copy_fid: bool,
}

#[derive(Clone, Copy)]
enum BandValue {
    U8(u8),
    U16(u16),
    I16(i16),
    F32(f32),
}

#[derive(Clone)]
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

struct SamplingBlockReducer {
    block_points: BlockPoints,
    samples: Vec<BandValue>,
}

impl BlockReducer for SamplingBlockReducer {
    type InputState = BlockPoints;
    type Output = (Vec<BandValue>, BlockPoints);

    fn new(band_count: usize, block_points: Self::InputState) -> Self {
        let samples = vec![BandValue::I16(0); block_points.points.len() * band_count];

        Self {
            block_points,
            samples,
        }
    }

    fn push_block(&mut self, band_index: usize, band_count: usize, block: TypedBlock) {
        match block {
            TypedBlock::U8(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::U8(pix);
                }
            }
            TypedBlock::U16(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::U16(pix);
                }
            }
            TypedBlock::I16(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::I16(pix);
                }
            }
            TypedBlock::F32(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::F32(pix);
                }
            }
        }
    }

    fn finalize(self) -> Self::Output {
        (self.samples, self.block_points)
    }
}

#[derive(Clone)]
struct BlockSender(Vec<SyncSender<(Arc<(Vec<BandValue>, BlockPoints)>, Range<usize>)>>);

impl BlockFinalizer for BlockSender {
    type Input = (Vec<BandValue>, BlockPoints);

    fn apply(&self, input: Self::Input) {
        let input = Arc::new(input);
        for (idx, (&input_index, &start)) in input
            .1
            .input_indexes
            .iter()
            .zip(&input.1.input_start_positions)
            .enumerate()
        {
            let end = input
                .1
                .input_start_positions
                .get(idx + 1)
                .copied()
                .unwrap_or(input.1.points.len());
            self.0[input_index]
                .send((input.clone(), start..end))
                .unwrap()
        }
    }
}

struct SamplingPoint {
    _fid: Option<u64>,
    bx: usize,
    by: usize,
    orig_x: f64,
    orig_y: f64,
    original_fields: Vec<Option<FieldValue>>,
}

#[derive(Default)]
struct BlockPoints {
    points: Vec<SamplingPoint>,
    input_indexes: Vec<usize>,
    input_start_positions: Vec<usize>,
}

impl SampleExtractionArgs {
    pub fn run(&self) -> Result<()> {
        let image = Dataset::open(&self.input)?;
        let geo_transform = image.geo_transform()?;
        let geo_transform = geo_transform.invert()?;
        let block_size = image.rasterband(1)?.block_size();

        let band_count = image.raster_count();
        for band_idx in 0..band_count {
            let band = image.rasterband(band_idx + 1)?;
            assert_eq!(band.block_size(), block_size);
        }

        if let Some(fields) = self.fields.as_ref() {
            assert_eq!(fields.len(), band_count);
        }

        assert_eq!(self.points.len(), self.outputs.len());

        let mut band_fields = Vec::with_capacity(band_count);
        for band_index in 1..=band_count {
            let band = image.rasterband(band_index)?;
            let name = match self.fields.as_ref() {
                Some(fields) => fields[band_index - 1].clone(),
                None => match band.description() {
                    Ok(name) if !name.is_empty() => name,
                    _ => format!("band_{band_index}"),
                },
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
            band_fields.push(field_definition);
        }

        let mut tile_points = HashMap::<_, BlockPoints>::new();
        let mut layer_names = Vec::with_capacity(self.points.len());
        let mut spatial_refs = Vec::with_capacity(self.points.len());
        let mut output_fields = Vec::with_capacity(self.points.len());
        for (input_index, path) in self.points.iter().enumerate() {
            let ds = Dataset::open(path)?;
            let mut layer = ds.layer(0)?;

            let layer_name = layer.name();
            layer_names.push(layer_name);

            dbg!(layer.feature_count());
            for feature in layer.features() {
                if let Some(geometry) = feature.geometry().as_ref() {
                    let (orig_x, orig_y, _) = geometry.get_point(0);
                    let (x, y) = geo_transform.apply(orig_x, orig_y);
                    let (block_x, block_y) = (
                        (x / block_size.0 as f64) as usize,
                        (y / block_size.1 as f64) as usize,
                    );
                    let (x, y) = (x as usize, y as usize);
                    let sampling_point = SamplingPoint {
                        _fid: feature.fid(),
                        bx: x % block_size.0,
                        by: y % block_size.1,
                        orig_x,
                        orig_y,
                        original_fields: feature.fields().map(|f| f.1).collect::<Vec<_>>(),
                    };
                    let block_points = tile_points.entry((block_x, block_y)).or_default();
                    if block_points.input_indexes.last().copied() != Some(input_index) {
                        block_points.input_indexes.push(input_index);
                        block_points
                            .input_start_positions
                            .push(block_points.points.len());
                    }
                    block_points.points.push(sampling_point);
                }
            }

            let spatial_ref = layer.spatial_ref().map(|sr| sr.to_wkt()).transpose()?;
            spatial_refs.push(spatial_ref);

            let mut layer_fields = Vec::new();
            for field in layer.defn().fields() {
                let field_definition = FieldDefinition {
                    name: field.name(),
                    field_type: field.field_type(),
                    width: Some(field.width()),
                    precision: Some(field.precision()),
                };
                layer_fields.push(field_definition);
            }
            layer_fields.extend_from_slice(&band_fields);

            output_fields.push(layer_fields);
        }

        let mut output_txs = Vec::with_capacity(self.points.len());
        thread::scope(|scope| -> Result<()> {
            let mut output_threads = Vec::with_capacity(self.points.len());
            for (path, layer_name, spatial_ref, layer_fields) in self
                .outputs
                .iter()
                .zip(layer_names)
                .zip(spatial_refs)
                .zip(output_fields)
                .map(|(((a, b), c), d)| (a, b, c, d))
            {
                let (tx, rx) =
                    mpsc::sync_channel::<(Arc<(Vec<BandValue>, BlockPoints)>, Range<usize>)>(128);
                output_txs.push(tx);

                let thread = scope.spawn(move || {
                    let mut output = DriverManager::get_driver_by_name(&self.format)?
                        .create_vector_only(path)?;
                    let output_layer = output.create_layer(LayerOptions {
                        name: &layer_name,
                        srs: spatial_ref
                            .map(|wkt| SpatialRef::from_wkt(&wkt))
                            .transpose()?
                            .as_ref(),
                        ty: OGRwkbGeometryType::wkbPoint,
                        options: None,
                    })?;
                    for field in layer_fields {
                        let field_defn = field.to_field_defn()?;
                        field_defn.add_to_layer(&output_layer)?;
                    }
                    for (data, range) in rx {
                        let (sample_values, block_points) = &*data;
                        let tx = output.start_transaction()?;
                        let output_layer = tx.layer(0)?;
                        for (idx, point) in block_points.points[range.clone()].iter().enumerate() {
                            let sample_idx = idx + range.start;
                            let mut feature = Feature::new(output_layer.defn())?;
                            if self.copy_fid {
                                feature.set_fid(point._fid)?;
                            }
                            let field_offset = point.original_fields.len();
                            for (field_idx, field_value) in point.original_fields.iter().enumerate()
                            {
                                if let Some(field_value) = field_value {
                                    feature.set_field_by_index(field_idx, field_value);
                                }
                            }
                            for band_idx in 0..band_count {
                                match sample_values[band_count * sample_idx + band_idx] {
                                    BandValue::U8(value) => {
                                        feature.set_field_integer_by_index(
                                            band_idx + field_offset,
                                            value as i32,
                                        );
                                    }
                                    BandValue::U16(value) => {
                                        feature.set_field_integer_by_index(
                                            band_idx + field_offset,
                                            value as i32,
                                        );
                                    }
                                    BandValue::I16(value) => {
                                        feature.set_field_integer_by_index(
                                            band_idx + field_offset,
                                            value as i32,
                                        );
                                    }
                                    BandValue::F32(value) => {
                                        feature.set_field_double_by_index(
                                            band_idx + field_offset,
                                            value as f64,
                                        );
                                    }
                                }
                            }
                            let mut geometry = Geometry::empty(OGRwkbGeometryType::wkbPoint)?;
                            geometry.add_point_2d((point.orig_x, point.orig_y));
                            feature.set_geometry(geometry)?;
                            feature.create(&output_layer)?;
                        }
                        tx.commit()?;
                    }
                    output.close()?;
                    Ok::<_, gdal::errors::GdalError>(())
                });
                output_threads.push(thread);
            }

            let mut tile_points = tile_points.into_iter().collect::<Vec<_>>();
            tile_points.sort_by_key(|t| (t.0 .1, t.0 .0));

            let block_sender = BlockSender(output_txs);

            let num_threads = NonZeroUsize::new(self.num_threads.unwrap_or(8))
                .unwrap()
                .min(available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap()));
            println!("Using {num_threads} threads");
            let mut block_reader = ThreadedBlockReader::new::<SamplingBlockReducer, _>(
                PathBuf::from(&self.input),
                block_sender,
                num_threads,
            );
            for ((block_x, block_y), points) in tile_points {
                block_reader.submit(block_x, block_y, points);
            }
            drop(block_reader);

            for thread in output_threads {
                thread.join().unwrap()?;
            }
            Ok(())
        })
    }
}
