use std::{
    collections::HashMap,
    mem,
    num::NonZeroUsize,
    path::PathBuf,
    sync::mpsc::{self, SyncSender},
    thread::{self, available_parallelism},
};

use anyhow::Result;
use clap::Parser;
use gdal::{
    raster::GdalDataType,
    spatial_ref::SpatialRef,
    vector::{
        Feature, FieldDefn, FieldValue, Geometry, LayerAccess, OGRFieldType, OGRwkbGeometryType,
    },
    Dataset, DriverManager, GeoTransformEx, LayerOptions, Metadata,
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
    points: PathBuf,

    /// Output dataset
    output: PathBuf,

    /// Output format
    #[arg(short, long)]
    format: String,

    /// Output field names
    #[arg(long, value_parser, num_args = 1..)]
    fields: Option<Vec<String>>,

    /// Number of threads to use
    #[arg(long)]
    num_threads: Option<usize>,
}

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

struct SamplingBlockReducer {
    points: Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
    samples: Vec<BandValue>,
}

impl BlockReducer for SamplingBlockReducer {
    type InputState = Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>;
    type Output = (
        Vec<BandValue>,
        Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
    );

    fn new(band_count: usize, points: Self::InputState) -> Self {
        let samples = vec![BandValue::I16(0); points.len() * band_count];

        Self { points, samples }
    }

    fn push_block(&mut self, band_index: usize, band_count: usize, block: TypedBlock) {
        match block {
            TypedBlock::U16(buf) => {
                for (idx, &(bx, by, ..)) in self.points.iter().enumerate() {
                    let pix = buf[(by, bx)];
                    self.samples[band_count * idx + band_index] = BandValue::U16(pix);
                }
            }
            TypedBlock::I16(buf) => {
                for (idx, &(bx, by, ..)) in self.points.iter().enumerate() {
                    let pix = buf[(by, bx)];
                    self.samples[band_count * idx + band_index] = BandValue::I16(pix);
                }
            }
            TypedBlock::F32(buf) => {
                for (idx, &(bx, by, ..)) in self.points.iter().enumerate() {
                    let pix = buf[(by, bx)];
                    self.samples[band_count * idx + band_index] = BandValue::F32(pix);
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

impl SampleExtractionArgs {
    pub fn run(&self) -> Result<()> {
        let image = Dataset::open(&self.input)?;
        let geo_transform = image.geo_transform()?;
        let geo_transform = geo_transform.invert()?;
        let block_size = image.rasterband(1)?.block_size();

        let band_count = image.raster_count() as usize;
        for band_idx in 0..band_count {
            let band = image.rasterband(band_idx as isize + 1)?;
            assert_eq!(band.block_size(), block_size);
        }
        let mut tile_points = HashMap::<_, Vec<_>>::new();
        let point_ds = Dataset::open(&self.points)?;
        let mut layer = point_ds.layer(0)?;
        let layer_name = layer.name();

        if let Some(fields) = self.fields.as_ref() {
            assert_eq!(fields.len(), band_count);
        }

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
        for band_index in 1..=band_count {
            let band = image.rasterband(band_index as isize)?;
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
            output_fields.push(field_definition);
        }

        let (tx, rx) = mpsc::sync_channel::<(
            Vec<BandValue>,
            Vec<(usize, usize, f64, f64, Vec<Option<FieldValue>>)>,
        )>(128);
        thread::scope(|scope| -> Result<()> {
            let output_thread = scope.spawn(move || {
                let mut output = DriverManager::get_driver_by_name(&self.format)?
                    .create_vector_only(&self.output)?;
                let output_layer = output.create_layer(LayerOptions {
                    name: &layer_name,
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

            let mut block_reader = ThreadedBlockReader::new::<SamplingBlockReducer, _>(
                PathBuf::from(&self.input),
                block_sender,
                NonZeroUsize::new(self.num_threads.unwrap_or(8))
                    .unwrap()
                    .min(available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap())),
            );
            for ((block_x, block_y), points) in tile_points {
                block_reader.submit(block_x, block_y, points);
            }
            drop(block_reader);

            output_thread.join().unwrap()?;
            Ok(())
        })
    }
}
