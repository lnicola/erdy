use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use clap::Parser;
use gdal::vector::LayerAccess;
use gdal::{cpl::CslStringList, Dataset};

fn column_to_u32(column: &ArrayRef) -> anyhow::Result<Vec<u32>> {
    Ok(match column.data_type() {
        DataType::Utf8 => column
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or_default().parse().unwrap_or(0))
            .collect(),
        DataType::LargeUtf8 => column
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or_default().parse().unwrap_or(0))
            .collect(),
        DataType::Int32 => column
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as u32)
            .collect(),
        DataType::Int64 => column
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as u32)
            .collect(),
        DataType::UInt32 => column
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .unwrap()
            .values()
            .to_vec(),
        DataType::UInt64 => column
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as u32)
            .collect(),
        _ => anyhow::bail!(
            "Unsupported data type for key column: {:?}",
            column.data_type()
        ),
    })
}

fn column_to_u64(column: &ArrayRef) -> anyhow::Result<Vec<u64>> {
    Ok(match column.data_type() {
        DataType::Utf8 => column
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or_default().parse().unwrap_or(0))
            .collect(),
        DataType::LargeUtf8 => column
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or_default().parse().unwrap_or(0))
            .collect(),
        DataType::Int32 => column
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as u64)
            .collect(),
        DataType::Int64 => column
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as u64)
            .collect(),
        DataType::UInt32 => column
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as u64)
            .collect(),
        DataType::UInt64 => column
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap()
            .values()
            .to_vec(),
        _ => anyhow::bail!(
            "Unsupported data type for value column: {:?}",
            column.data_type()
        ),
    })
}

struct ArrowStreamIterator {
    reader: ArrowArrayStreamReader,
    current_keys: Vec<u32>,
    current_vals: Vec<u64>,
    row_idx: usize,
}

impl ArrowStreamIterator {
    fn new(reader: ArrowArrayStreamReader) -> Self {
        Self {
            reader,
            current_keys: Vec::new(),
            current_vals: Vec::new(),
            row_idx: 0,
        }
    }
}

impl Iterator for ArrowStreamIterator {
    type Item = (u32, u64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row_idx < self.current_keys.len() {
                let key = self.current_keys[self.row_idx];
                let val = self.current_vals[self.row_idx];
                self.row_idx += 1;
                return Some((key, val));
            }

            match self.reader.next() {
                Some(Ok(batch)) => {
                    self.current_keys = column_to_u32(batch.column(0)).unwrap_or_default();
                    self.current_vals = column_to_u64(batch.column(1)).unwrap_or_default();
                    self.row_idx = 0;
                }
                _ => return None,
            }
        }
    }
}

#[derive(Debug, Parser)]
pub struct MergeClassCountsArgs {
    /// Input CSV files
    #[arg(required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,

    /// Output CSV file
    #[arg(short, long)]
    output: PathBuf,
}

impl MergeClassCountsArgs {
    pub fn run(&self) -> anyhow::Result<()> {
        let datasets = self
            .inputs
            .iter()
            .map(|p| Dataset::open(p))
            .collect::<Result<Vec<_>, _>>()?;

        let mut layers = datasets
            .iter()
            .map(|ds| ds.layer(0))
            .collect::<Result<Vec<_>, _>>()?;

        let mut iterators = Vec::new();
        for layer in &mut layers {
            let mut output_stream = FFI_ArrowArrayStream::empty();
            let output_stream_ptr = &mut output_stream as *mut FFI_ArrowArrayStream;
            let mut options = CslStringList::new();
            options.set_name_value("INCLUDE_FID", "NO")?;

            unsafe { layer.read_arrow_stream(output_stream_ptr.cast(), &options)? }

            let reader = ArrowArrayStreamReader::try_new(output_stream)?;
            iterators.push(ArrowStreamIterator::new(reader));
        }

        let mut currents = iterators.iter_mut().map(|it| it.next()).collect::<Vec<_>>();

        let file = File::create(&self.output)?;
        let mut writer = BufWriter::new(file);

        loop {
            let min_key = match currents.iter().filter_map(|&c| c).map(|(k, _)| k).min() {
                Some(k) => k,
                None => break,
            };

            write!(writer, "{}", min_key)?;

            for (i, c) in currents.iter_mut().enumerate() {
                if let Some((k, v)) = c {
                    if *k == min_key {
                        write!(writer, ",{}", v)?;
                        *c = iterators[i].next();
                    } else {
                        write!(writer, ",0")?;
                    }
                } else {
                    write!(writer, ",0")?;
                }
            }
            writeln!(writer)?;
        }

        Ok(())
    }
}
