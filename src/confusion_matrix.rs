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

struct DerivedStatistics {
    class_precision: Vec<f64>,
    class_recall: Vec<f64>,
    class_f1_score: Vec<f64>,
    overall_accuracy: f64,
    balanced_accuracy: f64,
    weighted_precision: f64,
    weighted_recall: f64,
    weighted_f1_score: f64,
    kappa: f64,
}

impl Statistics {
    pub fn write_to<W: Write>(&self, mut writer: W) -> io::Result<()> {
        let derived_statistics = self.derived_statistics();

        writeln!(&mut writer, "{{")?;
        write!(&mut writer, r#"  "labels": "#)?;
        write_array_compact_int(&mut writer, &self.labels)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "confusion_matrix": ["#)?;
        for (row, values) in self.confusion_matrix.iter().enumerate() {
            if row > 0 {
                write!(&mut writer, "                       ")?;
            }
            write_array_compact_int(&mut writer, values)?;
            if row + 1 < self.confusion_matrix.len() {
                writeln!(&mut writer, ",")?;
            }
        }
        writeln!(&mut writer, "],")?;
        write!(&mut writer, r#"  "class_recall": "#)?;
        write_array_compact_f64(&mut writer, &derived_statistics.class_recall)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "class_precision": "#)?;
        write_array_compact_f64(&mut writer, &derived_statistics.class_precision)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "class_f1_score": "#)?;
        write_array_compact_f64(&mut writer, &derived_statistics.class_f1_score)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "overall_accuracy": "#)?;
        write!(&mut writer, "{}", derived_statistics.overall_accuracy)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "balanced_accuracy": "#)?;
        write!(&mut writer, "{}", derived_statistics.balanced_accuracy)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "weighted_precision": "#)?;
        write!(&mut writer, "{}", derived_statistics.weighted_precision)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "weighted_recall": "#)?;
        write!(&mut writer, "{}", derived_statistics.weighted_recall)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "weighted_f1_score": "#)?;
        write!(&mut writer, "{}", derived_statistics.weighted_f1_score)?;
        writeln!(&mut writer, ",")?;
        write!(&mut writer, r#"  "kappa": "#)?;
        write!(&mut writer, "{}", derived_statistics.kappa)?;
        writeln!(&mut writer, "\n}}")
    }

    fn derived_statistics(&self) -> DerivedStatistics {
        let mut row_sums = vec![0; self.confusion_matrix.len()];
        let mut column_sums = vec![0; self.confusion_matrix[0].len()];
        let mut samples = 0;
        for (row, values) in self.confusion_matrix.iter().enumerate() {
            for (col, count) in values.iter().enumerate() {
                row_sums[row] += count;
                column_sums[col] += count;
            }
        }

        let mut class_recall = Vec::with_capacity(row_sums.len());
        let mut class_precision = Vec::with_capacity(column_sums.len());
        let mut p_o = 0.0;

        for (idx, row_sum) in row_sums.iter().copied().enumerate() {
            let val = self.confusion_matrix[idx][idx] as f64;
            class_recall.push(if row_sum > 0 {
                val / row_sum as f64
            } else {
                0.0
            });

            p_o += val;
            samples += row_sum;
        }
        p_o /= samples as f64;

        for (idx, column_sum) in column_sums.iter().copied().enumerate() {
            class_precision.push(self.confusion_matrix[idx][idx] as f64 / column_sum as f64);
        }

        let class_f1_score = class_recall
            .iter()
            .copied()
            .zip(class_precision.iter().copied())
            .map(|(r, p)| 2.0 * r * p / (r + p))
            .collect::<Vec<_>>();

        let mut p_e = 0.0;
        for (row_sum, column_sum) in row_sums.iter().copied().zip(column_sums.iter().copied()) {
            p_e += row_sum as f64 * column_sum as f64;
        }
        p_e /= (samples as f64) * (samples as f64);

        let overall_accuracy = p_o;
        let balanced_accuracy = class_recall.iter().sum::<f64>() / class_recall.len() as f64;
        let kappa = (p_o - p_e) / (1.0 - p_e);

        let mut weighted_precision = 0.0;
        let mut weighted_recall = 0.0;
        let mut weighted_f1_score = 0.0;

        for i in 0..row_sums.len() {
            let weight = row_sums[i] as f64 / samples as f64;
            weighted_precision += class_precision[i].nan_to_zero() * weight;
            weighted_recall += class_recall[i].nan_to_zero() * weight;
            weighted_f1_score += class_f1_score[i].nan_to_zero() * weight;
        }

        DerivedStatistics {
            class_precision,
            class_recall,
            class_f1_score,
            overall_accuracy,
            balanced_accuracy,
            weighted_precision,
            weighted_recall,
            weighted_f1_score,
            kappa,
        }
    }
}

fn write_array_compact_int<W: Write, T: Display>(mut writer: W, array: &[T]) -> io::Result<()> {
    write!(&mut writer, "[")?;
    for (idx, value) in array.iter().enumerate() {
        write!(&mut writer, "{value}")?;
        if idx + 1 < array.len() {
            write!(&mut writer, ", ")?;
        }
    }
    write!(&mut writer, "]")
}

fn write_array_compact_f64<W: Write>(mut writer: W, array: &[f64]) -> io::Result<()> {
    write!(&mut writer, "[")?;
    for (idx, value) in array.iter().enumerate() {
        if value.is_nan() {
            write!(&mut writer, "null")?;
        } else {
            write!(&mut writer, "{value}")?;
        }

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

trait FloatExt {
    fn nan_to_zero(self) -> f64;
}

impl FloatExt for f64 {
    fn nan_to_zero(self) -> f64 {
        if self.is_nan() {
            0.0
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use approx_eq::assert_approx_eq;

    use super::Statistics;

    #[test]
    fn derived_statistics() {
        let labels = vec![
            3, 5, 61, 71, 81, 911, 1111, 1121, 1131, 1141, 1181, 1192, 1421, 1437, 1921,
        ];
        let confusion_matrix = vec![
            vec![80, 42, 1, 5, 3, 0, 2, 2, 0, 0, 17, 0, 5, 0, 0],
            vec![9, 81, 28, 3, 4, 0, 0, 7, 0, 5, 3, 2, 14, 0, 0],
            vec![0, 13, 138, 0, 0, 2, 0, 0, 0, 0, 2, 2, 4, 0, 0],
            vec![7, 2, 0, 102, 35, 5, 0, 1, 0, 1, 3, 0, 2, 0, 0],
            vec![2, 1, 2, 14, 135, 0, 0, 1, 0, 0, 1, 0, 4, 0, 0],
            vec![0, 0, 0, 3, 0, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![14, 0, 0, 0, 0, 0, 44, 32, 0, 0, 92, 0, 25, 0, 0],
            vec![47, 8, 13, 15, 84, 0, 13, 1447, 0, 898, 788, 55, 2209, 3, 9],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 75, 0, 0, 0, 0, 0, 0],
            vec![9, 27, 31, 2, 10, 0, 0, 85, 0, 280, 137, 3, 79, 0, 0],
            vec![
                325, 25, 0, 8, 10, 0, 398, 590, 0, 543, 7233, 12, 1656, 55, 0,
            ],
            vec![0, 1, 0, 0, 0, 0, 0, 19, 0, 19, 0, 47, 15, 0, 0],
            vec![
                36, 29, 48, 19, 21, 0, 534, 602, 4, 609, 957, 141, 16168, 137, 10,
            ],
            vec![0, 0, 0, 3, 23, 0, 0, 3, 0, 0, 0, 0, 5, 15, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0],
        ];
        let statistics = Statistics {
            labels,
            confusion_matrix,
        };

        let derived_statistics = statistics.derived_statistics();
        let expected_recall = [
            0.509554, 0.519231, 0.857143, 0.64557, 0.84375, 0.981481, 0.2125603, 0.2589014, 1.0,
            0.422323, 0.666329, 0.465347, 0.83707, 0.3061224, 0.0,
        ];

        for (computed, expected) in derived_statistics
            .class_recall
            .iter()
            .copied()
            .zip(expected_recall)
        {
            assert_approx_eq!(computed, expected);
        }

        let expected_precision = [
            0.1512287, 0.353712, 0.528736, 0.586207, 0.415385, 0.957831, 0.0443996, 0.518824,
            0.949367, 0.1188455, 0.783386, 0.1793893, 0.800674, 0.0714286, 0.0,
        ];

        for (computed, expected) in derived_statistics
            .class_precision
            .iter()
            .copied()
            .zip(expected_precision)
        {
            assert_approx_eq!(computed, expected);
        }

        let expected_f_score = [
            0.233236,
            0.420779,
            0.654028,
            0.614458,
            0.556701,
            0.969512,
            0.0734558,
            0.3454285,
            0.974026,
            0.185492,
            0.720131,
            0.258953,
            0.818467,
            0.1158301,
            f64::NAN,
        ];

        for (computed, expected) in derived_statistics
            .class_f1_score
            .iter()
            .copied()
            .zip(expected_f_score)
        {
            if expected.is_nan() {
                assert!(computed.is_nan())
            } else {
                assert_approx_eq!(computed, expected);
            }
        }

        assert_approx_eq!(derived_statistics.overall_accuracy, 0.687645);
        assert_approx_eq!(derived_statistics.balanced_accuracy, 0.568359);
        assert_approx_eq!(derived_statistics.weighted_precision, 0.727932);
        assert_approx_eq!(derived_statistics.weighted_recall, 0.687645);
        assert_approx_eq!(derived_statistics.weighted_f1_score, 0.696799);
        assert_approx_eq!(derived_statistics.kappa, 0.515598);
    }
}
