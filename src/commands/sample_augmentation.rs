use std::{
    mem,
    num::NonZeroUsize,
    path::PathBuf,
    sync::{self, Arc},
    thread,
    time::Instant,
};

use anyhow::Result;
use ball_tree::{BallTree, Point};
use clap::Parser;
use gdal::{
    vector::{Feature, FieldDefn, Geometry, LayerAccess, LayerOptions},
    Dataset, DriverManager,
};
use gdal_sys::{OGRFieldType, OGRwkbGeometryType::wkbNone};
use num_traits::{real::Real, AsPrimitive, Zero};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};

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

    /// Excluded fields
    #[arg(long, num_args = 1..)]
    exclude: Vec<String>,

    /// Number of threads to use
    #[arg(long)]
    num_threads: Option<usize>,
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
        let now = Instant::now();
        let input_dataset = Dataset::open(&self.input)?;
        let mut layer = input_dataset.layer(0)?;
        let input_layer_name = layer.name();

        let layer_defn = layer.defn();
        let mut included_fields = Vec::new();
        for (idx, field) in layer_defn.fields().enumerate() {
            let name = field.name();
            if !self.exclude.contains(&name) && name != self.field {
                included_fields.push(idx);
            }
        }

        let attribute_filter = format!("{}={}", self.field, self.label);
        layer.set_attribute_filter(&attribute_filter)?;
        let mut data = Vec::new();
        for feature in layer.features() {
            for &idx in &included_fields {
                let value = feature.field_as_double(idx as i32)?.unwrap();
                data.push(value);
            }
        }

        let columns = included_fields.len();
        let sample_table = Arc::new(SampleTable::new(columns, data));

        let samples = (0..sample_table.rows())
            .map(|idx| Sample::from_ref(sample_table.clone(), idx))
            .collect::<Vec<_>>();
        println!("{}", (Instant::now() - now).as_millis());

        println!("{} rows", samples.len());

        let values = vec![(); samples.len()];

        let now = Instant::now();
        let ball_tree = BallTree::new(samples.clone(), values);
        println!("{}", (Instant::now() - now).as_millis());

        let mut rng = rand::thread_rng();

        let t = self.samples / sample_table.rows();
        let mut neighbors = (0..sample_table.rows())
            .map(|_| (t, SmallRng::from_rng(&mut rng).unwrap()))
            .collect::<Vec<_>>();
        let mut indices = (0..sample_table.rows()).collect::<Vec<_>>();
        let extra_indices = indices
            .partial_shuffle(&mut rng, self.samples % sample_table.rows())
            .0;
        for idx in extra_indices {
            neighbors[*idx].0 += 1;
        }

        let mut num_threads = self.num_threads.filter(|&t| t > 0).unwrap_or_else(|| {
            thread::available_parallelism()
                .unwrap_or(NonZeroUsize::MIN)
                .get()
        });
        if num_threads >= sample_table.rows() {
            num_threads = 1;
        }

        let now = Instant::now();
        thread::scope(|s| {
            let (tx, rx) = sync::mpsc::sync_channel(num_threads * 3);
            let thread_items = sample_table.rows() / num_threads;
            for thread_idx in 0..num_threads {
                let tx = tx.clone();
                let mut query = ball_tree.query();
                let neighbors = &neighbors;
                let sample_table = &sample_table;
                s.spawn(move || {
                    let thread_range_start = thread_idx * thread_items;
                    let thread_range_end = sample_table.rows().min((thread_idx + 1) * thread_items);
                    let thread_data = &neighbors[thread_range_start..thread_range_end];

                    let mut output_data = Vec::new();
                    let mut tmp = vec![0.0; sample_table.columns()];
                    for ((count, rng), sample_idx) in
                        thread_data.iter().zip(thread_range_start..thread_range_end)
                    {
                        let sample = Sample::from_ref(sample_table.clone(), sample_idx);
                        let mut rng = rng.clone();
                        for _ in 0..*count {
                            let idx = rng.gen_range(0..3);
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
                        dbg!(output_table.rows());
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
                for idx in included_fields {
                    let field_defn =
                        FieldDefn::new(&format!("field_{idx}"), OGRFieldType::OFTReal)?;
                    field_defn.add_to_layer(&layer)?;
                }

                for batch in rx {
                    let tx = output_dataset.start_transaction()?;
                    let layer = tx.layer(0)?;
                    for idx in 0..batch.rows() {
                        let feature = Feature::new(layer.defn())?;
                        for (col_idx, value) in batch.sample(idx).iter().copied().enumerate() {
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
        println!("{}", (Instant::now() - now).as_millis());

        Ok(())
    }
}
