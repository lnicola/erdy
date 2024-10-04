use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};

use anyhow::Result;
use clap::Parser;
use gdal::{vector::LayerAccess, Dataset};
use rayon::iter::{IntoParallelIterator as _, IntoParallelRefIterator as _, ParallelIterator as _};

use crate::{confusion_matrix::ConfusionMatrixBuilder, gdal_ext::DefnExt};

#[derive(Debug, Parser)]
pub struct ComputeConfusionMatrixArgs {
    /// Input paths
    #[arg(long, value_parser, num_args = 1..)]
    inputs: Vec<PathBuf>,

    /// Output path
    #[arg(long, value_parser)]
    output: PathBuf,

    /// Reference column name
    #[arg(long)]
    reference: String,

    /// Prediction column name
    #[arg(long)]
    prediction: String,
}

impl ComputeConfusionMatrixArgs {
    pub fn run(&self) -> Result<()> {
        let builders = self
            .inputs
            .par_iter()
            .map(|p| process_file(p, &self.reference, &self.prediction))
            .collect::<Result<Vec<_>>>()?;

        let confusion_matrix_builder =
            builders
                .into_par_iter()
                .reduce(ConfusionMatrixBuilder::new, |mut a, b| {
                    a.merge(&b);
                    a
                });

        let statistics = confusion_matrix_builder.to_statistics();
        let file = File::create(&self.output)?;
        let buf_writer = BufWriter::new(file);

        statistics.write_to(buf_writer)?;

        Ok(())
    }
}

fn process_file(path: &Path, reference: &str, prediction: &str) -> Result<ConfusionMatrixBuilder> {
    let dataset = Dataset::open(path)?;
    let mut layer = dataset.layer(0)?;
    let defn = layer.defn();
    let reference_idx = defn.get_field_index(reference)?;
    let prediction_idx = defn.get_field_index(prediction)?;

    let mut matrix_builder = ConfusionMatrixBuilder::new();
    for feature in layer.features() {
        let Some(reference) = feature.field_as_integer(reference_idx)? else {
            continue;
        };
        let Some(prediction) = feature.field_as_integer(prediction_idx)? else {
            continue;
        };
        let reference = u16::try_from(reference)?;
        let prediction = u16::try_from(prediction)?;
        matrix_builder.add_sample(reference, prediction);
    }

    Ok(matrix_builder)
}
