use std::{
    mem,
    num::NonZeroUsize,
    path::PathBuf,
    sync::{self, Arc},
    thread,
};

use anyhow::Result;
use ball_tree::{BallTree, Point};
use clap::Parser;
use gdal::{
    vector::{Feature, FieldDefn, LayerAccess, LayerOptions},
    Dataset, DriverManager,
};
use gdal_sys::{OGRFieldType, OGRwkbGeometryType::wkbNone};
use num_traits::{real::Real, AsPrimitive, Zero};
use rand::{
    rngs::{SmallRng, StdRng},
    seq::SliceRandom,
    Rng, SeedableRng,
};

use crate::gdal_ext::FeatureExt;

#[derive(Debug, Parser)]
pub struct SampleAugmentationArgs {
    /// Input dataset
    input: PathBuf,

    /// Output dataset
    output: PathBuf,

    /// Class field
    #[arg(long)]
    field: String,

    /// Class label to filter by
    #[arg(long)]
    label: i64,

    /// Samples to generate
    #[arg(long)]
    samples: usize,

    /// Number of nearest neighbors
    #[arg(long, default_value_t = 5)]
    neighbors: usize,

    /// Excluded fields
    #[arg(long, num_args = 1..)]
    exclude: Vec<String>,

    /// Number of threads to use
    #[arg(long)]
    num_threads: Option<usize>,

    /// Normalize features
    #[arg(long, default_value_t = false)]
    normalize: bool,
}

struct SampleTable<T> {
    columns: usize,
    data: Vec<T>,
}

impl<T> SampleTable<T> {
    fn new(columns: usize, data: Vec<T>) -> Self {
        Self { columns, data }
    }

    fn rows(&self) -> usize {
        self.data.len() / self.columns
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn sample(&self, idx: usize) -> &[T] {
        &self.data[idx * self.columns..(idx + 1) * self.columns]
    }
}

#[derive(Clone)]
enum SampleStorage<T> {
    Ref {
        storage: Arc<SampleTable<T>>,
        idx: usize,
    },
    Owned {
        data: Vec<T>,
    },
}

impl<T> SampleStorage<T> {
    fn data(&self) -> &[T] {
        match self {
            SampleStorage::Ref { storage, idx } => storage.sample(*idx),
            SampleStorage::Owned { data } => data.as_slice(),
        }
    }
}

impl<T: PartialEq> PartialEq for SampleStorage<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data().eq(other.data())
    }
}

#[derive(Clone, PartialEq)]
struct Sample<T> {
    storage: SampleStorage<T>,
}

impl<T> Sample<T> {
    fn from_ref(storage: Arc<SampleTable<T>>, idx: usize) -> Self {
        let storage = SampleStorage::Ref { storage, idx };
        Self { storage }
    }

    fn from_owned(data: Vec<T>) -> Self {
        let storage = SampleStorage::Owned { data };
        Self { storage }
    }

    fn data(&self) -> &[T] {
        self.storage.data()
    }
}

impl<T> Sample<T>
where
    T: Real + Zero,
    T: AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    fn lerp(&self, other: &Self, alpha: f64, output: &mut [T]) {
        let alpha: T = alpha.as_();
        self.data()
            .iter()
            .copied()
            .zip(other.data().iter().copied())
            .map(|(x, y)| x + alpha * (y - x))
            .zip(output.iter_mut())
            .for_each(|(x, o)| {
                *o = x;
            });
    }
}

impl<T> Point for Sample<T>
where
    T: Real + Zero,
    T: AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    fn distance(&self, other: &Self) -> f64 {
        let mut d = T::zero();
        for (x, y) in self
            .data()
            .iter()
            .copied()
            .zip(other.data().iter().copied())
        {
            d = d + (x - y) * (x - y);
        }
        d.sqrt().as_()
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {
        let distance: T = self.distance(other).as_();
        if distance.is_zero() {
            return self.clone();
        }

        let mut result = self.data().to_owned();
        let scale = d.as_() / distance;

        for (r, o) in result.iter_mut().zip(other.data().iter().copied()) {
            *r = *r + scale * (o - *r);
        }

        Self::from_owned(result)
    }
}

impl SampleAugmentationArgs {
    pub fn run(&self) -> Result<()> {
        let input_dataset = Dataset::open(&self.input)?;
        let mut layer = input_dataset.layer(0)?;
        let input_layer_name = layer.name();

        let layer_defn = layer.defn();
        let mut included_fields = Vec::new();
        for (idx, field) in layer_defn.fields().enumerate() {
            let name = field.name();
            if !self.exclude.contains(&name) && name != self.field {
                included_fields.push((idx, name, field.field_type()));
            }
        }

        let attribute_filter = format!("{}={}", self.field, self.label);
        layer.set_attribute_filter(&attribute_filter)?;
        let mut data = Vec::new();
        for feature in layer.features() {
            for &(idx, _, _) in &included_fields {
                let value = feature.field_as_double(idx as i32)?.unwrap();
                data.push(value);
            }
        }

        let columns = included_fields.len();

        let mut column_max = vec![f64::NEG_INFINITY; columns];
        let mut column_min = vec![f64::INFINITY; columns];

        if self.normalize {
            for (idx, value) in data.iter().copied().enumerate() {
                let col = idx % columns;
                column_max[col] = column_max[col].max(value);
                column_min[col] = column_min[col].min(value);
            }

            for (idx, value) in data.iter_mut().enumerate() {
                let col = idx % columns;
                let d = column_max[col] - column_min[col];
                if d != 0.0 {
                    *value = (*value - column_min[col]) / d;
                } else {
                    *value = 0.0;
                }
            }
        }

        let sample_table = Arc::new(SampleTable::new(columns, data));

        let samples = (0..sample_table.rows())
            .map(|idx| Sample::from_ref(sample_table.clone(), idx))
            .collect::<Vec<_>>();

        if samples.len() == 1 {
            panic!("can't run sample augmentation on a single input sample");
        }

        let k_nearest_neighbors = self.neighbors.min(samples.len() - 1);
        let values = vec![(); samples.len()];

        let ball_tree = BallTree::new(samples.clone(), values);

        let mut rng = StdRng::from_entropy();

        let t = self.samples / sample_table.rows();
        let mut neighbors = vec![t; sample_table.rows()];
        let mut indices = (0..sample_table.rows()).collect::<Vec<_>>();
        let extra_indices = indices
            .partial_shuffle(&mut rng, self.samples % sample_table.rows())
            .0;
        for idx in extra_indices {
            neighbors[*idx] += 1;
        }

        let mut num_threads = self.num_threads.filter(|&t| t > 0).unwrap_or_else(|| {
            thread::available_parallelism()
                .unwrap_or(NonZeroUsize::MIN)
                .get()
        });
        if num_threads >= sample_table.rows() {
            num_threads = 1;
        }

        let mut thread_items = vec![sample_table.rows() / num_threads; num_threads];
        let remainder = sample_table.rows() % num_threads;
        thread_items
            .iter_mut()
            .take(remainder)
            .for_each(|n| *n += 1);
        let thread_start = thread_items
            .iter()
            .scan(0, |sum, i| {
                let s = *sum;
                *sum += i;
                Some(s)
            })
            .collect::<Vec<_>>();

        thread::scope(|s| {
            let (tx, rx) = sync::mpsc::sync_channel(num_threads * 3);

            for thread_idx in 0..num_threads {
                let tx = tx.clone();
                let mut query = ball_tree.query();
                let neighbors = &neighbors;
                let sample_table = &sample_table;

                let thread_start = &thread_start;
                let thread_items = &thread_items;
                let mut rng = SmallRng::from_rng(&mut rng).expect("can't create SmallRng");
                s.spawn(move || {
                    let thread_range_start = thread_start[thread_idx];
                    let thread_range_end = thread_range_start + thread_items[thread_idx];

                    let thread_data = &neighbors[thread_range_start..thread_range_end];

                    let mut output_data = Vec::new();
                    let mut tmp = vec![0.0; sample_table.columns()];
                    for (count, sample_idx) in
                        thread_data.iter().zip(thread_range_start..thread_range_end)
                    {
                        let sample = Sample::from_ref(sample_table.clone(), sample_idx);
                        for _ in 0..*count {
                            let idx = rng.gen_range(0..k_nearest_neighbors);
                            let neighbor = query.nn(&sample).nth(idx + 1).unwrap();
                            let distance = rng.gen_range(0.0..=1.0);
                            sample.lerp(neighbor.0, distance, tmp.as_mut_slice());

                            for value in tmp.iter().copied() {
                                output_data.push(value);
                            }
                        }

                        if output_data.len() / sample_table.columns() >= 4096 {
                            let tmp_data = mem::take(&mut output_data);
                            let output_table = SampleTable::new(sample_table.columns(), tmp_data);
                            if tx.send(output_table).is_err() {
                                break;
                            }
                        }
                    }

                    if !output_data.is_empty() {
                        let output_table = SampleTable::new(sample_table.columns(), output_data);
                        let _ = tx.send(output_table);
                    }
                });
            }
            drop(tx);

            // TODO: add_to_layer error
            let output = || -> anyhow::Result<()> {
                let driver = DriverManager::get_driver_by_name("SQLite")?;
                let mut output_dataset = driver.create_vector_only(&self.output)?;

                let layer = output_dataset.create_layer(LayerOptions {
                    name: &input_layer_name,
                    srs: None,
                    ty: wkbNone,
                    options: None,
                })?;
                for (_, name, field_type) in &included_fields {
                    let field_defn = FieldDefn::new(name, *field_type)?;
                    field_defn.add_to_layer(&layer)?;
                }

                for batch in rx {
                    let tx = output_dataset.start_transaction()?;
                    let layer = tx.layer(0)?;
                    for idx in 0..batch.rows() {
                        let mut feature = Feature::new(layer.defn())?;
                        for (col_idx, (mut value, &(_, _, field_type))) in batch
                            .sample(idx)
                            .iter()
                            .copied()
                            .zip(&included_fields)
                            .enumerate()
                        {
                            if self.normalize {
                                let d = column_max[col_idx] - column_min[col_idx];
                                value = value * d + column_min[col_idx];

                                if field_type == OGRFieldType::OFTInteger
                                    || field_type == OGRFieldType::OFTInteger64
                                {
                                    value = value.round_ties_even();
                                }
                            }
                            feature.set_field_double_by_index(col_idx, value);
                        }
                        feature.create(&layer)?;
                    }
                    tx.commit()?;
                }

                Ok(())
            };
            output().unwrap();
        });

        Ok(())
    }
}
