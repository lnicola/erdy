use std::path::PathBuf;

use clap::Parser;
use gdal::{
    errors::GdalError,
    raster::{Buffer, RasterCreationOptions},
    Dataset, DriverManager,
};
use num_traits::NumCast;

use crate::gdal_ext::{RasterBandExt as _, TypedBuffer};

#[derive(Debug, Parser)]
pub struct BandSelectArgs {
    /// Input images
    #[arg(long, value_parser, num_args = 1..)]
    inputs: Vec<PathBuf>,

    /// Input labels
    #[arg(long, num_args = 1..)]
    input_labels: Vec<usize>,

    /// Input mask
    mask: PathBuf,

    /// Output dataset
    output: PathBuf,

    /// Output format
    #[arg(short, long)]
    format: String,
}

fn gather_labels<T: NumCast>(block: &[T]) -> Vec<usize> {
    let mut labels = Vec::new();
    for mask_pixel in block {
        if let Some(label) = mask_pixel.to_usize() {
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
    }
    labels
}

impl BandSelectArgs {
    pub fn run(&self) -> anyhow::Result<()> {
        assert_eq!(self.inputs.len(), self.input_labels.len());

        let inputs = self
            .inputs
            .iter()
            .map(Dataset::open)
            .collect::<Result<Vec<Dataset>, _>>()?;
        let mask = Dataset::open(&self.mask)?;

        let raster_size = inputs[0].raster_size();
        let block_size = inputs[0].rasterband(1)?.block_size();

        let mut output = DriverManager::get_driver_by_name(&self.format)?
            .create_with_band_type_with_options::<i16, _>(
                &self.output,
                raster_size.0,
                raster_size.1,
                1,
                &RasterCreationOptions::from_iter(["TILED=YES", "COMPRESS=DEFLATE"]),
            )?;
        output.set_projection(&mask.projection())?;
        output.set_geo_transform(&mask.geo_transform()?)?;
        let mut output_band = output.rasterband(1)?;

        let (blocks_x, blocks_y) = (
            raster_size.0.div_ceil(block_size.0),
            raster_size.1.div_ceil(block_size.1),
        );

        for y in 0..blocks_y {
            for x in 0..blocks_x {
                let mask_block = mask.rasterband(1)?.read_typed_block(x, y)?;
                let block_shape = mask_block.shape();
                let required_labels = match &mask_block {
                    TypedBuffer::U8(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::I8(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::U16(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::I16(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::U32(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::I32(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::U64(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::I64(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::F32(mask_block) => gather_labels(mask_block.data()),
                    TypedBuffer::F64(mask_block) => gather_labels(mask_block.data()),
                };
                let input_blocks = required_labels
                    .iter()
                    .copied()
                    .map(|label| {
                        for (&dataset_label, dataset) in self.input_labels.iter().zip(&inputs) {
                            if dataset_label == label {
                                return Ok::<_, GdalError>(Some((
                                    label,
                                    dataset.rasterband(1)?.read_typed_block(x, y)?,
                                )));
                            }
                        }
                        Ok(None)
                    })
                    .filter_map(|e| match e {
                        Ok(Some(e)) => Some(Ok(e)),
                        Ok(None) => None,
                        Err(e) => Some(Err(e)),
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let mask_iterator: &mut dyn Iterator<Item = usize> = match mask_block {
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
                if input_blocks.iter().all(|(_, block)| block.is_i16()) {
                    let input_blocks = input_blocks
                        .into_iter()
                        .map(|(label, block)| (label, block.try_into_i16().unwrap()))
                        .collect::<Vec<_>>();
                    let mut input_iterators = input_blocks
                        .into_iter()
                        .map(|(label, input)| (label, input.into_iter()))
                        .collect::<Vec<_>>();

                    let mut output_block =
                        Buffer::new(block_shape, vec![0; block_shape.0 * block_shape.1]);
                    for (out_pixel, mask_pixel) in
                        output_block.data_mut().iter_mut().zip(mask_iterator)
                    {
                        let mut found = false;
                        for (label, it) in input_iterators.iter_mut() {
                            if *label == mask_pixel {
                                found = true;
                                *out_pixel = it.next().unwrap();
                            } else {
                                it.next().unwrap();
                            }
                        }
                        if !found {
                            *out_pixel = 0;
                        }
                    }
                    output_band.write_block((x, y), &mut output_block)?;
                } else {
                    unimplemented!("only `UInt16` inputs are currently supported");
                }
            }
        }

        Ok(())
    }
}
