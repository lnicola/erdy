use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use clap::Parser;
use gdal::Dataset;
use rayon::prelude::*;

#[derive(Debug, Parser)]
pub struct ComputeClassCountsArgs {
    /// Input images
    #[arg(long, required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,

    /// Output CSV file
    #[arg(short, long)]
    output: PathBuf,
}

impl ComputeClassCountsArgs {
    pub fn run(&self) -> anyhow::Result<()> {
        let global_counts = self
            .inputs
            .par_iter()
            .map(|input_path| {
                let dataset = Dataset::open(input_path)?;
                let rasterband = dataset.rasterband(1)?;
                let raster_size = dataset.raster_size();
                let block_size = rasterband.block_size();

                let blocks_x = raster_size.0.div_ceil(block_size.0);
                let blocks_y = raster_size.1.div_ceil(block_size.1);

                let mut local_counts: HashMap<u32, u64> = HashMap::new();

                for y in 0..blocks_y {
                    for x in 0..blocks_x {
                        let window_x = (x * block_size.0) as isize;
                        let window_y = (y * block_size.1) as isize;
                        let actual_size = rasterband.actual_block_size(x, y)?;
                        let buffer = rasterband.read_as::<u32>(
                            (window_x, window_y),
                            actual_size,
                            actual_size,
                            None,
                        )?;
                        for &value in buffer.data() {
                            *local_counts.entry(value).or_insert(0) += 1;
                        }
                    }
                }

                Ok::<_, anyhow::Error>(local_counts)
            })
            .try_reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                Ok(a)
            })?;

        let mut sorted_counts = global_counts.into_iter().collect::<Vec<_>>();
        sorted_counts.sort_unstable_by_key(|&(k, _)| k);

        let file = File::create(&self.output)?;
        let mut writer = BufWriter::new(file);

        for (value, count) in sorted_counts {
            writeln!(writer, "{},{}", value, count)?;
        }

        Ok(())
    }
}
