use std::{
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use clap::Parser;
use gdal::{vector::LayerAccess, Dataset};
use rayon::iter::{IntoParallelIterator as _, IntoParallelRefIterator as _, ParallelIterator as _};

use crate::gdal_ext::DefnExt;

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

struct Statistics {
    labels: Vec<u16>,
    confusion_matrix: Vec<Vec<u64>>,
}

impl Statistics {
    pub fn write_to<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(&mut writer, "{{")?;
        write!(&mut writer, r#"  "labels": "#)?;
        write_array_compact(&mut writer, &self.labels)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "confusion_matrix": ["#)?;
        for (row, values) in self.confusion_matrix.iter().enumerate() {
            if row > 0 {
                write!(&mut writer, "                       ")?;
            }
            write_array_compact(&mut writer, values)?;
            if row + 1 < self.confusion_matrix.len() {
                writeln!(&mut writer, ",")?;
            }
        }
        writeln!(&mut writer, "\n}}")
    }
}

fn write_array_compact<W: Write, T: Display>(mut writer: W, array: &[T]) -> io::Result<()> {
    write!(&mut writer, "[")?;
    for (idx, value) in array.iter().enumerate() {
        write!(&mut writer, "{value}")?;
        if idx + 1 < array.len() {
            write!(&mut writer, ", ")?;
        }
    }
    write!(&mut writer, "]")
}

struct ConfusionMatrixBuilder {
    map: HashMap<(u16, u16), u64>,
}

impl ConfusionMatrixBuilder {
    pub fn new() -> Self {
        let map = HashMap::new();
        Self { map }
    }

    pub fn add_sample(&mut self, reference: u16, prediction: u16) {
        *self.map.entry((reference, prediction)).or_default() += 1;
    }

    pub fn merge(&mut self, other: &Self) {
        for (&(reference, prediction), &count) in other.map.iter() {
            *self.map.entry((reference, prediction)).or_default() += count;
        }
    }

    pub fn to_statistics(&self) -> Statistics {
        let mut labels = self
            .map
            .keys()
            .flat_map(|&(r, p)| [r, p])
            .collect::<Vec<_>>();
        labels.sort_unstable();
        labels.dedup();
        let label_map = labels
            .iter()
            .enumerate()
            .map(|(idx, label)| (label, idx))
            .collect::<HashMap<_, _>>();

        let mut confusion_matrix = vec![vec![0; labels.len()]; labels.len()];
        for (&(reference, prediction), &count) in self.map.iter() {
            let reference_idx = label_map[&reference];
            let prediction_idx = label_map[&prediction];
            confusion_matrix[reference_idx][prediction_idx] += count;
        }

        Statistics {
            labels,
            confusion_matrix,
        }
    }
}
