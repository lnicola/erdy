use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use gdal::{
    errors,
    raster::GdalDataType,
    vector::{Feature, FieldDefn, Geometry, LayerAccess, LayerOptions},
    Dataset, DriverManager, DriverType,
};
use gdal_sys::{
    OGRFieldType::{OFTInteger, OFTInteger64, OFTReal},
    OGRwkbGeometryType::wkbPoint,
};
use rand::{distr::StandardUniform, rngs::StdRng, Rng, SeedableRng};

use crate::gdal_ext::{RasterBandExt as _, TypedBuffer};

#[derive(Debug, Parser)]
pub struct SampleSelectionArgs {
    /// Input image
    #[arg(long, value_parser)]
    input: PathBuf,

    /// Output datasets
    #[arg(long, value_parser, num_args = 1..)]
    outputs: Vec<PathBuf>,

    /// Label field
    #[arg(long)]
    label_field: String,

    /// Output probabilities
    #[arg(long, num_args = 1..)]
    output_probabilities: Option<Vec<f32>>,

    /// Ignored labels
    #[arg(long)]
    ignored_labels: Vec<usize>,

    /// Max target pixel count
    #[arg(long)]
    max_target: Option<usize>,

    /// Target multiplier
    #[arg(long, default_value_t = 1.0)]
    target_multiplier: f32,

    /// Random seed
    #[arg(long)]
    random_seed: Option<String>,
}

impl SampleSelectionArgs {
    pub fn run(&mut self) -> Result<()> {
        let dataset = Dataset::open(&self.input)?;
        assert!(dataset.raster_count() == 1);

        if let Some(output_probs) = &self.output_probabilities {
            if output_probs.len() != self.outputs.len() {
                anyhow::bail!("--output-probabilities and --outputs must have the same length");
            }
        }

        let band = dataset.rasterband(1)?;
        let raster_size = band.size();
        let block_size = band.block_size();
        let geo_transform = dataset.geo_transform()?;

        let (blocks_x, blocks_y) = (
            raster_size.0.div_ceil(block_size.0),
            raster_size.1.div_ceil(block_size.1),
        );

        let mut label_counts = HashMap::<_, usize>::new();
        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let block = band.read_typed_block(block_x, block_y)?;
                let block_shape = band.actual_block_size(block_x, block_y)?;

                let iterator: &mut dyn Iterator<Item = usize> = match block {
                    TypedBuffer::U8(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::I8(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::U16(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::I16(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::U32(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::I32(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::U64(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::I64(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::F32(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::F64(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                };

                let (mut row, mut col) = (0, 0);
                for val in iterator {
                    if col < block_shape.0 && row < block_shape.1 {
                        *label_counts.entry(val).or_default() += 1;
                    }

                    col += 1;
                    if col == block_size.0 {
                        col = 0;
                        row += 1;
                    }
                }
            }
        }

        for x in &self.ignored_labels {
            label_counts.remove(x);
        }

        let mut counts = label_counts.into_iter().collect::<Vec<_>>();
        counts.sort_by_key(|x| x.0);
        for &(label, count) in &counts {
            println!("Pixel count for class {label}: {count}");
        }
        let target = counts
            .iter()
            .map(|x| x.1)
            .min()
            .unwrap_or(0)
            .min(self.max_target.unwrap_or(usize::MAX));
        let targets = counts
            .iter()
            .map(|x| {
                (
                    x.0,
                    (target as f32 * self.target_multiplier).round_ties_even() as usize,
                )
            })
            .collect::<Vec<_>>();
        for &(label, count) in &targets {
            println!("Pixel target for class {label}: {count}");
        }
        let probabilities = targets
            .iter()
            .copied()
            .zip(counts.iter().copied().map(|x| x.1))
            .map(|((label, target), count)| (label, target as f32 / count as f32))
            .collect::<HashMap<_, _>>();

        let mut rng = if let Some(seed) = &self.random_seed {
            let mut seed_buf = <StdRng as SeedableRng>::Seed::default();
            faster_hex::hex_decode(seed.as_bytes(), &mut seed_buf)?;
            StdRng::from_seed(seed_buf)
        } else {
            StdRng::from_os_rng()
        };

        let field_type = match band.band_type() {
            GdalDataType::Unknown => unimplemented!(),
            GdalDataType::UInt8
            | GdalDataType::Int8
            | GdalDataType::UInt16
            | GdalDataType::Int16
            | GdalDataType::UInt32
            | GdalDataType::Int32 => OFTInteger,
            GdalDataType::UInt64 => OFTInteger64,
            GdalDataType::Int64 => OFTInteger64,
            GdalDataType::Float32 | GdalDataType::Float64 => OFTReal,
        };

        let spatial_ref = dataset.spatial_ref()?;
        let mut outputs = Vec::with_capacity(self.outputs.len());

        for output in &self.outputs {
            let driver =
                DriverManager::get_output_driver_for_dataset_name(output, DriverType::Vector)
                    .expect("can't determine output driver");
            let mut dataset = driver.create_vector_only(output)?;

            let name = output
                .file_stem()
                .expect("output is a file")
                .to_string_lossy();
            let layer = dataset.create_layer(LayerOptions {
                name: name.as_ref(),
                srs: Some(&spatial_ref),
                ty: wkbPoint,
                options: None,
            })?;
            let field_defn = FieldDefn::new(&self.label_field, field_type)?;
            field_defn.add_to_layer(&layer)?;

            outputs.push(dataset);
        }

        let output_probabilities_csums = self
            .output_probabilities
            .take()
            .unwrap_or_else(|| vec![1.0 / outputs.len() as f32; outputs.len()])
            .into_iter()
            .scan(0.0, |sum, i| {
                *sum += i;
                Some(*sum)
            })
            .collect::<Vec<_>>();

        let mut counts = HashMap::<_, usize>::new();
        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let block = band.read_typed_block(block_x, block_y)?;
                let block_shape = band.actual_block_size(block_x, block_y)?;

                let iterator: &mut dyn Iterator<Item = usize> = match block {
                    TypedBuffer::U8(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::I8(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::U16(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::I16(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::U32(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::I32(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::U64(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::I64(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::F32(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                    TypedBuffer::F64(buffer) => &mut buffer.into_iter().map(|x| x as usize),
                };

                let transactions = outputs
                    .iter_mut()
                    .map(|ds| ds.start_transaction())
                    .collect::<errors::Result<Vec<_>>>()?;
                let layers = transactions
                    .iter()
                    .map(|tx| tx.layer(0))
                    .collect::<errors::Result<Vec<_>>>()?;

                let (mut row, mut col) = (0, 0);
                for val in iterator {
                    if col < block_shape.0 && row < block_shape.1 {
                        if let Some(probability) = probabilities.get(&val) {
                            if rng.sample::<f32, _>(StandardUniform) < *probability {
                                *counts.entry(val).or_default() += 1;

                                let abs_col = col + block_x * block_size.0;
                                let abs_row = row + block_y * block_size.1;

                                let abs_col = abs_col as f64 + 0.5;
                                let abs_row = abs_row as f64 + 0.5;

                                let mx = geo_transform[0]
                                    + abs_col * geo_transform[1]
                                    + abs_row * geo_transform[2];
                                let my = geo_transform[3]
                                    + abs_col * geo_transform[4]
                                    + abs_row * geo_transform[5];
                                let mut point = Geometry::empty(wkbPoint)?;
                                point.add_point_2d((mx, my));

                                let t = rng.sample::<f32, _>(StandardUniform);
                                let idx = output_probabilities_csums
                                    .iter()
                                    .copied()
                                    .enumerate()
                                    .find(|x| t < x.1)
                                    .map(|x| x.0)
                                    .expect("probabilities add up to 1.0");

                                let layer = &layers[idx];
                                let mut feature = Feature::new(layer.defn())?;
                                feature.set_geometry(point)?;
                                feature.set_field_integer(0, val as i32)?;
                                feature.create(layer)?;
                            }
                        }
                    }

                    col += 1;
                    if col == block_size.0 {
                        col = 0;
                        row += 1;
                    }
                }

                for tx in transactions {
                    tx.commit()?;
                }
            }
        }

        let mut counts = counts.into_iter().collect::<Vec<_>>();
        counts.sort_by_key(|x| x.0);
        for &(label, count) in &counts {
            println!("Output pixel count for class {label}: {count}");
        }

        Ok(())
    }
}
