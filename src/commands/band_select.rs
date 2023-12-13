use std::path::PathBuf;

use clap::Parser;
use gdal::{errors::GdalError, raster::RasterCreationOption, Dataset, DriverManager};
use ndarray::Array2;

use crate::gdal_ext::{RasterBandExt, TypedBlock};

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

impl BandSelectArgs {
    pub fn run(&self) -> anyhow::Result<()> {
        assert_eq!(self.inputs.len(), self.input_labels.len());

        let inputs = self
            .inputs
            .iter()
            .map(|f| Dataset::open(f))
            .collect::<Result<Vec<Dataset>, _>>()?;
        let mask = Dataset::open(&self.mask)?;

        let raster_size = inputs[0].raster_size();
        let block_size = inputs[0].rasterband(1)?.block_size();

        let mut output = DriverManager::get_driver_by_name(&self.format)?
            .create_with_band_type_with_options::<i16, _>(
                &self.output,
                raster_size.0 as isize,
                raster_size.1 as isize,
                1,
                &[
                    RasterCreationOption {
                        key: "TILED",
                        value: "YES",
                    },
                    RasterCreationOption {
                        key: "COMPRESS",
                        value: "DEFLATE",
                    },
                ],
            )?;
        output.set_projection(&mask.projection())?;
        output.set_geo_transform(&mask.geo_transform()?)?;
        let mut output_band = output.rasterband(1)?;

        let (blocks_x, blocks_y) = (
            (raster_size.0 + block_size.0 - 1) / block_size.0,
            (raster_size.1 + block_size.1 - 1) / block_size.1,
        );

        for y in 0..blocks_y {
            for x in 0..blocks_x {
                let mask_block = mask.rasterband(1)?.read_typed_block(x, y)?;
                match mask_block {
                    TypedBlock::U8(mask_block) => {
                        let mut required_labels = Vec::new();
                        for mask_pixel in &mask_block {
                            if !required_labels.contains(&(*mask_pixel as usize)) {
                                required_labels.push(*mask_pixel as usize);
                            }
                        }
                        let input_blocks = required_labels
                            .iter()
                            .copied()
                            .map(|label| {
                                for (&dataset_label, dataset) in
                                    self.input_labels.iter().zip(&inputs)
                                {
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

                        if input_blocks.iter().all(|(_, block)| block.is_i16()) {
                            let input_blocks = input_blocks
                                .into_iter()
                                .map(|(label, block)| (label, block.try_into_i16().unwrap()))
                                .collect::<Vec<_>>();
                            let mut input_iterators = input_blocks
                                .into_iter()
                                .map(|(label, input)| (label, input.into_iter()))
                                .collect::<Vec<_>>();
                            let mut output_block = Array2::<i16>::zeros(mask_block.raw_dim());
                            for (out_pixel, mask_pixel) in output_block.iter_mut().zip(mask_block) {
                                let mut found = false;
                                for (label, it) in input_iterators.iter_mut() {
                                    if *label == mask_pixel as usize {
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
                            output_band.write_block((x, y), output_block)?;
                        } else {
                            unimplemented!();
                        }
                    }
                    _ => {
                        unimplemented!();
                    }
                }
            }
        }

        Ok(())
    }
}
