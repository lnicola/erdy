use crate::catboost::{EApiPredictionType, Model};
use crate::gdal_ext::LayerExt;

use anyhow::{anyhow, Result};
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{
    Array, ArrayRef, Float16Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, RecordBatch, RecordBatchReader, StructArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use crossbeam_channel::bounded;
use gdal::cpl::CslStringList;
use gdal::vector::{LayerAccess, LayerOptions};
use gdal::{Dataset, DriverManager, DriverType};
use itertools::Itertools;
use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelIterator as _, IntoParallelRefIterator as _,
    ParallelBridge as _, ParallelIterator as _,
};
use std::cmp::Ordering;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info};

#[derive(Debug, Parser)]
pub struct VectorPredictArgs {
    /// Path to the model file
    #[arg(long, required = true)]
    pub model: PathBuf,

    /// List of feature column names
    #[arg(long, required = true, num_args = 1..)]
    pub features: Vec<String>,

    /// Input vector files
    #[arg(long, required = true, num_args = 1..)]
    pub inputs: Vec<PathBuf>,

    /// Output vector files
    #[arg(long, required = true, num_args = 1..)]
    pub outputs: Vec<PathBuf>,

    /// Name of the column with the predicted values
    #[arg(long, default_value = "predicted")]
    pub prediction_column: String,

    /// Name of the column with the reference values
    #[arg(long)]
    pub reference_column: Option<String>,

    /// Layer creation options
    #[arg(long, alias = "lco", value_name = "NAME=VALUE", num_args = 1..)]
    pub layer_creation_options: Vec<String>,
}

fn column_to_f32(column: &ArrayRef) -> Result<Vec<f32>> {
    Ok(match column.data_type() {
        DataType::Int8 => column
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DataType::Int16 => column
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DataType::Int32 => column
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DataType::Int64 => column
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DataType::UInt8 => column
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DataType::UInt16 => column
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DataType::UInt32 => column
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DataType::UInt64 => column
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DataType::Float16 => column
            .as_any()
            .downcast_ref::<Float16Array>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap().to_f32())
            .collect(),
        DataType::Float32 => column
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec(),
        DataType::Float64 => column
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        other => anyhow::bail!("Unsupported data type for feature: {:?}", other),
    })
}

impl VectorPredictArgs {
    pub fn run(&self) -> Result<()> {
        if self.inputs.len() != self.outputs.len() {
            anyhow::bail!("The number of input and output files must be the same.");
        }

        let layer_creation_options = self
            .layer_creation_options
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        let layer_creation_options = &layer_creation_options;

        let mut model = Model::load_from_file(&self.model)?;
        model.set_prediction_type(EApiPredictionType::Probability)?;
        let class_names = model
            .params()
            .map(|params| Arc::new(params.data_processing_options.class_names));
        let dimensions_count = model.get_prediction_dimensions_count();

        let model = Arc::new(model);

        self.inputs
            .par_iter()
            .zip(&self.outputs)
            .try_for_each(|(input_path, output_path)| {
                info!("Processing file: {:?} -> {:?}", input_path, output_path);
                let model = Arc::clone(&model);
                let class_names = class_names.clone();

                let (batch_tx, batch_rx) = bounded::<RecordBatch>(16);
                let (result_tx, result_rx) =
                    bounded::<(Vec<u16>, Option<ArrayRef>, RecordBatch)>(16);

                std::thread::scope(move |s| -> Result<()> {
                    let in_dataset = Dataset::open(input_path)?;
                    let mut in_layer = in_dataset.layer(0)?;

                    let mut output_stream = FFI_ArrowArrayStream::empty();
                    let output_stream_ptr = &mut output_stream as *mut FFI_ArrowArrayStream;

                    let mut options = CslStringList::new();
                    options.set_name_value("INCLUDE_FID", "NO")?;

                    unsafe { in_layer.read_arrow_stream(output_stream_ptr.cast(), &options)? }

                    let arrow_stream_reader = ArrowArrayStreamReader::try_new(output_stream)?;
                    let schema = arrow_stream_reader.schema();

                    let geom_column_index = schema
                        .column_with_name("wkb_geometry")
                        .or_else(|| schema.column_with_name("geom"))
                        .or_else(|| schema.column_with_name("geometry"))
                        .or_else(|| schema.column_with_name("GEOMETRY"))
                        .ok_or_else(|| anyhow::anyhow!("No geometry column found"))?
                        .0;

                    let feature_indices = self
                        .features
                        .iter()
                        .map(|name| {
                            schema
                                .column_with_name(name)
                                .map(|(index, _)| index)
                                .ok_or_else(|| {
                                    anyhow::anyhow!("Feature column '{}' not found", name)
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let reference_index = self
                        .reference_column
                        .as_ref()
                        .map(|name| {
                            schema
                                .column_with_name(name)
                                .map(|(index, _)| index)
                                .ok_or_else(|| {
                                    anyhow::anyhow!("Reference column '{}' not found", name)
                                })
                        })
                        .transpose()?;

                    s.spawn(move || {
                        let res = (|| -> Result<()> {
                            for maybe_batch in arrow_stream_reader {
                                let batch = maybe_batch?;
                                batch_tx.send(batch)?;
                            }
                            Ok(())
                        })();
                        if let Err(e) = res {
                            error!("Reader error for {:?}: {}", input_path, e);
                        }
                        drop(batch_tx);
                    });

                    s.spawn(move || {
                        let res = (|| -> Result<()> {
                            let mut it = result_rx.into_iter();
                            let (predictions, reference, batch) = match it.next() {
                                Some(x) => x,
                                None => return Ok(()),
                            };

                            let in_layer = in_dataset.layer(0)?;

                            let geom_field = batch.schema().field(geom_column_index).clone();

                            let driver = DriverManager::get_output_driver_for_dataset_name(
                                output_path,
                                DriverType::Vector,
                            )
                            .ok_or_else(|| anyhow!("Unable to determine output driver"))?;
                            let mut out_dataset = driver.create_vector_only(output_path)?;
                            let mut out_layer = out_dataset.create_layer(LayerOptions {
                                name: &in_layer.name(),
                                srs: in_layer.spatial_ref().as_ref(),
                                options: (!layer_creation_options.is_empty())
                                    .then_some(layer_creation_options),
                                ..Default::default()
                            })?;

                            let mut fields =
                                vec![Field::new(&self.prediction_column, DataType::UInt16, false)];
                            if let Some(idx) = reference_index {
                                let field = batch.schema().field(idx).clone();
                                fields.push(field);
                            }
                            fields.push(geom_field.clone());

                            let schema = Arc::new(Schema::new(fields));
                            let ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())?;

                            for field in schema.fields() {
                                if field.name() == geom_field.name() {
                                    continue;
                                }
                                let field_schema = FFI_ArrowSchema::try_from(field.as_ref())?;
                                out_layer.create_field_from_arrow_schema(&field_schema, None)?;
                            }

                            let mut write_batch = |predictions: Vec<u16>,
                                                   reference: Option<ArrayRef>,
                                                   batch: RecordBatch|
                             -> Result<()> {
                                let mut columns: Vec<ArrayRef> =
                                    vec![Arc::new(UInt16Array::from(predictions))];
                                if let Some(ref_arr) = reference {
                                    columns.push(ref_arr);
                                }
                                columns.push(Arc::clone(batch.column(geom_column_index)));

                                let out_batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
                                let struct_array = StructArray::from(out_batch);
                                let mut ffi_array = FFI_ArrowArray::new(&struct_array.into_data());

                                out_layer.write_arrow_batch(&ffi_schema, &mut ffi_array, None)?;
                                Ok(())
                            };

                            write_batch(predictions, reference, batch)?;

                            for (predictions, reference, batch) in it {
                                write_batch(predictions, reference, batch)?;
                            }

                            Ok(())
                        })();
                        if let Err(e) = res {
                            error!("Writer error for {:?}: {}", output_path, e);
                        }
                    });

                    let worker_res =
                        batch_rx
                            .into_iter()
                            .par_bridge()
                            .try_for_each(|batch| -> Result<()> {
                                let features_data = feature_indices
                                    .par_iter()
                                    .map(|&idx| {
                                        let column = batch.column(idx);
                                        column_to_f32(column)
                                    })
                                    .collect::<Result<Vec<_>>>()?;

                                let doc_count = batch.num_rows();
                                const CHUNK_SIZE: usize = 4096;
                                let num_chunks = doc_count.div_ceil(CHUNK_SIZE);

                                let predictions = (0..num_chunks)
                                    .into_par_iter()
                                    .map(|i| {
                                        let start = i * CHUNK_SIZE;
                                        let end = (start + CHUNK_SIZE).min(doc_count);
                                        let chunk_features = features_data
                                            .iter()
                                            .map(|col| &col[start..end])
                                            .collect::<Vec<_>>();
                                        let mut chunk_predictions = Vec::new();
                                        model.calc_model_prediction_flat_transposed(
                                            &chunk_features,
                                            &mut chunk_predictions,
                                        )?;

                                        Ok(chunk_predictions
                                            .chunks_exact(dimensions_count)
                                            .map(|chunk| {
                                                let pred_idx = chunk
                                                    .iter()
                                                    .position_max_by(|a, b| {
                                                        a.partial_cmp(b).unwrap_or(Ordering::Equal)
                                                    })
                                                    .unwrap_or(0);

                                                if let Some(labels) = &class_names {
                                                    labels[pred_idx]
                                                } else {
                                                    pred_idx as u16
                                                }
                                            })
                                            .collect::<Vec<_>>())
                                    })
                                    .collect::<Result<Vec<_>>>()?
                                    .into_iter()
                                    .flatten()
                                    .collect();

                                let reference =
                                    reference_index.map(|idx| Arc::clone(batch.column(idx)));
                                result_tx.send((predictions, reference, batch))?;
                                Ok(())
                            });

                    drop(result_tx);
                    worker_res
                })
            })?;

        Ok(())
    }
}
