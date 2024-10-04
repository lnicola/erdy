use std::{
    collections::HashMap,
    fmt::Display,
    io::{self, Write},
};

use serde::Deserialize;

#[derive(Deserialize)]
pub struct Statistics {
    pub labels: Vec<u16>,
    pub confusion_matrix: Vec<Vec<u64>>,
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
        writeln!(&mut writer, "]\n}}")
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

pub struct ConfusionMatrixBuilder {
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
