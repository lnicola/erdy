use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use anyhow::Result;
use clap::Parser;
use gdal::{vector::LayerAccess, Dataset};

use crate::confusion_matrix::Statistics;

#[derive(Debug, Parser)]
pub struct RemapConfusionMatrixArgs {
    /// Input path
    #[arg(long, value_parser)]
    input: PathBuf,

    /// Output path
    #[arg(long, value_parser)]
    output: PathBuf,

    /// Remapping table
    #[arg(long, value_parser)]
    remapping_table: PathBuf,
}

impl RemapConfusionMatrixArgs {
    pub fn run(&self) -> Result<()> {
        let input = File::open(&self.input)?;
        let buf_reader = BufReader::new(input);
        let statistics: Statistics = serde_json::from_reader(buf_reader)?;

        let mut remapping_set = HashMap::new();
        let dataset = Dataset::open(&self.remapping_table)?;
        let mut layer = dataset.layer(0)?;
        for feature in layer.features() {
            let pre = u16::try_from(feature.field_as_integer(0)?.expect("label is set"))?;
            let post = u16::try_from(feature.field_as_integer(1)?.expect("label is set"))?;
            remapping_set.insert(pre, post);
        }

        let mut remapped_labels = statistics
            .labels
            .iter()
            .copied()
            .map(|l| remapping_set.get(&l).copied().unwrap_or(l))
            .collect::<Vec<_>>();
        remapped_labels.sort_unstable_by_key(|l| l.to_string());
        remapped_labels.dedup();
        let remapped_label_map = remapped_labels
            .iter()
            .enumerate()
            .map(|(idx, label)| (label, idx))
            .collect::<HashMap<_, _>>();
        let remapped_label_map = statistics
            .labels
            .iter()
            .copied()
            .map(|l| {
                (
                    l,
                    remapped_label_map[&remapping_set.get(&l).copied().unwrap_or(l)],
                )
            })
            .collect::<HashMap<_, _>>();
        let mut remapped_confusion_matrix =
            vec![vec![0; remapped_labels.len()]; remapped_labels.len()];

        for (row, values) in statistics.confusion_matrix.iter().enumerate() {
            let row_label = statistics.labels[row];
            let row_idx = remapped_label_map[&row_label];
            for (col, count) in values.iter().enumerate() {
                let col_label = statistics.labels[col];
                let col_idx = remapped_label_map[&col_label];
                remapped_confusion_matrix[row_idx][col_idx] += count;
            }
        }
        let statistics = Statistics {
            labels: remapped_labels,
            confusion_matrix: remapped_confusion_matrix,
        };

        let file = File::create(&self.output)?;
        let buf_writer = BufWriter::new(file);
        statistics.write_to(buf_writer)?;

        Ok(())
    }
}
