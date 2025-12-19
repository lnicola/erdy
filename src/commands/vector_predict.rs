use crate::catboost::{EApiPredictionType, Model};
use crate::gdal_ext::DatasetExt;

use anyhow::{anyhow, Result};
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{
    Array, ArrayRef, BinaryArray, Float16Array, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, RecordBatch, RecordBatchReader, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_schema::DataType;
use clap::Parser;
use crossbeam_channel::bounded;
use gdal::cpl::CslStringList;
use gdal::vector::{Feature, FieldDefn, Geometry, LayerAccess, LayerOptions};
use gdal::{Dataset, DriverManager, DriverType};
use gdal_sys::OGRFieldType::OFTInteger;
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

                let (batch_tx, batch_rx) =
                    bounded::<(RecordBatch, usize, Vec<usize>, Option<usize>)>(16);
                let (result_tx, result_rx) =
                    bounded::<(Vec<u16>, Option<Vec<u16>>, RecordBatch, usize)>(16);

                std::thread::scope(|s| -> Result<()> {
                    s.spawn(|| {
                        let res = (|| -> Result<()> {
                            let in_dataset = Dataset::open(input_path)?;
                            let mut in_layer = in_dataset.layer(0)?;

                            let mut output_stream = FFI_ArrowArrayStream::empty();
                            let output_stream_ptr = &mut output_stream as *mut FFI_ArrowArrayStream;
                            let gdal_pointer: *mut gdal::ArrowArrayStream =
                                output_stream_ptr.cast();

                            let mut options = CslStringList::new();
                            options.set_name_value("INCLUDE_FID", "NO")?;

                            unsafe { in_layer.read_arrow_stream(gdal_pointer, &options)? }

                            let arrow_stream_reader =
                                ArrowArrayStreamReader::try_new(output_stream)?;
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

                            for maybe_batch in arrow_stream_reader {
                                let batch = maybe_batch?;
                                batch_tx.send((
                                    batch,
                                    geom_column_index,
                                    feature_indices.clone(),
                                    reference_index,
                                ))?;
                            }
                            Ok(())
                        })();
                        if let Err(e) = res {
                            error!("Reader error for {:?}: {}", input_path, e);
                        }
                        drop(batch_tx);
                    });

                    s.spawn(|| {
                        let res = (|| -> Result<()> {
                            let in_dataset = Dataset::open(input_path)?;
                            let in_layer = in_dataset.layer(0)?;

                            let driver = DriverManager::get_output_driver_for_dataset_name(
                                output_path,
                                DriverType::Vector,
                            )
                            .ok_or_else(|| anyhow!("Unable to determine output driver"))?;
                            let mut out_dataset = driver.create_vector_only(output_path)?;
                            let out_layer = out_dataset.create_layer(LayerOptions {
                                name: &in_layer.name(),
                                srs: in_layer.spatial_ref().as_ref(),
                                options: (!layer_creation_options.is_empty())
                                    .then_some(&layer_creation_options),
                                ..Default::default()
                            })?;

                            let pred_field = FieldDefn::new(&self.prediction_column, OFTInteger)?;
                            pred_field.add_to_layer(&out_layer)?;

                            if let Some(ref_col) = &self.reference_column {
                                let ref_field = FieldDefn::new(ref_col, OFTInteger)?;
                                ref_field.add_to_layer(&out_layer)?;
                            }

                            for (predictions, reference_data_u16, batch, geom_idx) in result_rx {
                                let geom_column = batch
                                    .column(geom_idx)
                                    .as_any()
                                    .downcast_ref::<BinaryArray>()
                                    .unwrap();

                                out_dataset.maybe_run_in_batch(|dataset| {
                                    let out_layer = dataset.layer(0)?;
                                    for (i, &class) in predictions.iter().enumerate() {
                                        let mut feature = Feature::new(out_layer.defn())?;
                                        let geom = Geometry::from_wkb(geom_column.value(i))?;
                                        feature.set_geometry(geom)?;

                                        feature.set_field_integer(0, class as i32)?;

                                        if let Some(ref_data) = &reference_data_u16 {
                                            feature.set_field_integer(1, ref_data[i] as i32)?;
                                        }

                                        feature.create(&out_layer)?;
                                    }
                                    Ok(())
                                })?;
                            }
                            Ok(())
                        })();
                        if let Err(e) = res {
                            error!("Writer error for {:?}: {}", output_path, e);
                        }
                    });

                    let worker_res = batch_rx.into_iter().par_bridge().try_for_each(
                        |(batch, geom_idx, feat_indices, ref_idx)| -> Result<()> {
                            let features_data = feat_indices
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
                            let reference_data = ref_idx
                                .map(|idx| {
                                    let ref_data = column_to_f32(batch.column(idx))?;
                                    Ok::<_, anyhow::Error>(
                                        ref_data.iter().copied().map(|r| r as u16).collect(),
                                    )
                                })
                                .transpose()?;

                            result_tx.send((predictions, reference_data, batch, geom_idx))?;

                            Ok(())
                        },
                    );

                    drop(result_tx);
                    worker_res
                })
            })?;

        Ok(())
    }
}
