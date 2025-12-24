use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossbeam_channel::bounded;
use gdal::DriverType;
use gdal::{raster::RasterCreationOptions, Dataset, DriverManager};
use itertools::Itertools as _;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use tracing::{debug, error, info};

use crate::catboost::{EApiPredictionType, Model};

#[derive(Debug, Parser)]
pub struct RasterPredictArgs {
    /// CatBoost model file (.cbm)
    #[arg(short, long)]
    model: PathBuf,

    /// Input rasters. Each band is treated as a feature.
    /// Multiple rasters will be logically concatenated (bands 1..N of image 1, then 1..M of image 2, etc.)
    /// Supports extended filenames like "image.tif?bands=1,3" to select specific bands.
    #[arg(short, long, num_args = 1..)]
    inputs: Vec<String>,

    /// Output raster
    #[arg(short, long)]
    output: PathBuf,

    /// Block size for processing
    #[arg(long, default_value_t = 256)]
    block_size: usize,

    /// GDAL creation options
    #[arg(long, num_args = 0..)]
    creation_options: Vec<String>,

    /// I/O threads
    #[arg(long, default_value_t = 4)]
    io_threads: usize,
}

struct InputSource {
    path: PathBuf,
    bands: Option<Vec<usize>>,
}

impl InputSource {
    fn parse(input: &str) -> Result<Self> {
        if let Some((path_str, query)) = input.split_once('?') {
            let path = PathBuf::from(path_str);
            let mut bands = None;
            for pair in query.split('&') {
                if let Some((key, value)) = pair.split_once('=') {
                    if key == "bands" {
                        let indices = value
                            .split(',')
                            .map(|s| {
                                s.trim()
                                    .parse::<usize>()
                                    .map_err(|e| anyhow!("Invalid band index '{}': {}", s, e))
                            })
                            .collect::<Result<Vec<_>, _>>();
                        bands = Some(indices?);
                    }
                }
            }
            Ok(Self { path, bands })
        } else {
            Ok(Self {
                path: PathBuf::from(input),
                bands: None,
            })
        }
    }
}

struct BlockData {
    x: usize,
    y: usize,
    win_w: usize,
    win_h: usize,
    features: Vec<Vec<f32>>,
}

struct PartialBlockData {
    bx: usize,
    by: usize,
    ds_idx: usize,
    x: usize,
    y: usize,
    win_w: usize,
    win_h: usize,
    features: Vec<Vec<f32>>,
}

struct BlockResult {
    x: usize,
    y: usize,
    win_w: usize,
    win_h: usize,
    data: Vec<u16>,
}

impl RasterPredictArgs {
    pub fn run(&self) -> Result<()> {
        let mut model = Model::load_from_file(&self.model)?;
        model.set_prediction_type(EApiPredictionType::Probability)?;
        let class_names = model
            .params()
            .map(|params| Arc::new(params.data_processing_options.class_names));
        let dimensions_count = model.get_prediction_dimensions_count();

        let model = Arc::new(model);

        if self.inputs.is_empty() {
            return Err(anyhow!("No input rasters provided"));
        }

        let parsed_inputs = self
            .inputs
            .iter()
            .map(|s| InputSource::parse(s))
            .collect::<Result<Vec<_>, _>>()?;

        let (width, height, projection, geo_transform, total_bands) = {
            let first_ds = Dataset::open(&parsed_inputs[0].path)
                .with_context(|| format!("Failed to open {}", parsed_inputs[0].path.display()))?;
            let (w, h) = first_ds.raster_size();
            let mut total = 0;
            for input in &parsed_inputs {
                let ds = Dataset::open(&input.path)
                    .with_context(|| format!("Failed to open {}", input.path.display()))?;
                if ds.raster_size() != (w, h) {
                    anyhow::bail!("Size mismatch in input raster: {}", input.path.display());
                }

                let count = if let Some(bands) = &input.bands {
                    // Verify bands exist
                    let max_band = ds.raster_count() as usize;
                    for &b in bands {
                        if b == 0 || b > max_band {
                            anyhow::bail!(
                                "Band index {} out of range for {:?} (max {})",
                                b,
                                input.path,
                                max_band
                            );
                        }
                    }
                    bands.len()
                } else {
                    ds.raster_count() as usize
                };
                total += count;
            }
            (
                w,
                h,
                first_ds.projection(),
                first_ds.geo_transform()?,
                total,
            )
        };

        info!(
            "Total input bands: {}. Image size: {}x{}",
            total_bands, width, height
        );

        let block_size = self.block_size;
        let x_blocks = width.div_ceil(block_size);
        let y_blocks = height.div_ceil(block_size);

        let (data_tx, data_rx) = bounded::<BlockData>(2);
        let (result_tx, result_rx) = bounded::<BlockResult>(4);

        std::thread::scope(|s| -> Result<()> {
            s.spawn(|| {
                let res = (|| -> Result<()> {
                    let datasets = parsed_inputs
                        .iter()
                        .map(|input| Dataset::open(&input.path).map(Mutex::new))
                        .collect::<Result<Vec<_>, _>>()?;
                    let dataset_count = datasets.len();

                    let mut offsets = Vec::with_capacity(dataset_count);
                    let mut current_offset = 0;

                    let dataset_bands = parsed_inputs
                        .iter()
                        .zip(&datasets)
                        .map(|(input, ds)| {
                            if let Some(bands) = &input.bands {
                                bands.clone()
                            } else {
                                let count = ds.lock().unwrap().raster_count();
                                (1..=count as usize).collect()
                            }
                        })
                        .collect::<Vec<Vec<_>>>();

                    for bands in &dataset_bands {
                        offsets.push(current_offset);
                        current_offset += bands.len();
                    }

                    let tasks = (0..y_blocks)
                        .flat_map(|by| (0..x_blocks).map(move |bx| (bx, by)))
                        .flat_map(|(bx, by)| (0..dataset_count).map(move |i| (bx, by, i)))
                        .collect::<Vec<_>>();
                    let tasks = Mutex::new(tasks.into_iter());

                    let (partial_tx, partial_rx) = bounded::<PartialBlockData>(4);

                    std::thread::scope(|s| {
                        let mut handles = Vec::new();
                        info!("Starting {} reader threads", self.io_threads);
                        for i in 0..self.io_threads {
                            let partial_tx = partial_tx.clone();
                            let tasks = &tasks;
                            let datasets = &datasets;
                            let dataset_bands = &dataset_bands;
                            handles.push(s.spawn(move || -> Result<()> {
                                loop {
                                    let (bx, by, ds_idx) = match tasks.lock().unwrap().next() {
                                        Some(t) => t,
                                        None => break,
                                    };
                                    debug!(
                                        "Thread {} reading block ({}, {}), ds {}",
                                        i, bx, by, ds_idx
                                    );

                                    let y_off = by * block_size;
                                    let win_h = block_size.min(height - y_off);
                                    let x_off = bx * block_size;
                                    let win_w = block_size.min(width - x_off);

                                    let mut features = Vec::new();
                                    {
                                        let ds = datasets[ds_idx].lock().unwrap();
                                        let bands_to_read = &dataset_bands[ds_idx];

                                        for &b in bands_to_read {
                                            let band = ds.rasterband(b)?;
                                            let data = band.read_as::<f32>(
                                                (x_off as isize, y_off as isize),
                                                (win_w, win_h),
                                                (win_w, win_h),
                                                None,
                                            )?;
                                            features.push(data.data().to_vec());
                                        }
                                    }

                                    partial_tx.send(PartialBlockData {
                                        bx,
                                        by,
                                        ds_idx,
                                        x: x_off,
                                        y: y_off,
                                        win_w,
                                        win_h,
                                        features,
                                    })?;
                                    debug!(
                                        "Thread {} finished block ({}, {}), ds {}",
                                        i, bx, by, ds_idx
                                    );
                                }
                                Ok(())
                            }));
                        }
                        drop(partial_tx);

                        #[allow(clippy::type_complexity)]
                        let mut pending: HashMap<
                            (usize, usize),
                            (usize, Vec<Vec<f32>>),
                        > = HashMap::new();

                        for part in partial_rx {
                            let (count, features) = pending
                                .entry((part.bx, part.by))
                                .or_insert_with(|| (0, vec![Vec::new(); total_bands]));

                            let start_idx = offsets[part.ds_idx];
                            for (i, band_data) in part.features.into_iter().enumerate() {
                                features[start_idx + i] = band_data;
                            }
                            *count += 1;

                            if *count == dataset_count {
                                if let Some((_, features)) = pending.remove(&(part.bx, part.by)) {
                                    data_tx.send(BlockData {
                                        x: part.x,
                                        y: part.y,
                                        win_w: part.win_w,
                                        win_h: part.win_h,
                                        features,
                                    })?;
                                    info!("Sent aggregated block ({}, {})", part.bx, part.by);
                                }
                            }
                        }

                        for h in handles {
                            h.join().unwrap()?;
                        }
                        Ok(())
                    })
                })();
                if let Err(e) = res {
                    error!("Reader error: {}", e);
                }
                drop(data_tx);
            });

            s.spawn(|| {
                let res = (|| -> Result<()> {
                    let driver = DriverManager::get_output_driver_for_dataset_name(
                        &self.output,
                        DriverType::Raster,
                    )
                    .ok_or_else(|| anyhow!("Unable to determine output driver"))?;
                    let mut out_dataset = driver.create_with_band_type_with_options::<u16, _>(
                        &self.output,
                        width,
                        height,
                        1,
                        &RasterCreationOptions::from_iter(
                            self.creation_options.iter().map(|s| s.as_str()),
                        ),
                    )?;

                    out_dataset.set_projection(&projection)?;
                    out_dataset.set_geo_transform(&geo_transform)?;

                    let mut out_band = out_dataset.rasterband(1)?;

                    for result in result_rx {
                        let mut out_buffer =
                            gdal::raster::Buffer::new((result.win_w, result.win_h), result.data);
                        out_band.write(
                            (result.x as isize, result.y as isize),
                            (result.win_w, result.win_h),
                            &mut out_buffer,
                        )?;
                    }
                    Ok(())
                })();
                if let Err(e) = res {
                    error!("Writer error: {}", e);
                }
            });

            data_rx.into_iter().par_bridge().try_for_each(|block| {
                let start = Instant::now();
                info!(
                    "Starting prediction for block ({}, {})",
                    block.x / self.block_size,
                    block.y / self.block_size
                );

                let doc_count = block.win_w * block.win_h;
                const CHUNK_SIZE: usize = 4096;
                let num_chunks = doc_count.div_ceil(CHUNK_SIZE);

                let out_data = (0..num_chunks)
                    .into_par_iter()
                    .map(|i| {
                        let start = i * CHUNK_SIZE;
                        let end = (start + CHUNK_SIZE).min(doc_count);
                        let chunk_features = block
                            .features
                            .iter()
                            .map(|col| &col[start..end])
                            .collect::<Vec<_>>();
                        let mut chunk_predictions = Vec::new();
                        let start = Instant::now();
                        model.calc_model_prediction_flat_transposed(
                            &chunk_features,
                            &mut chunk_predictions,
                        )?;
                        let elapsed = start.elapsed();
                        info!(
                            "Finished prediction for chunk in {} ms",
                            elapsed.as_millis()
                        );

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

                let elapsed = start.elapsed();
                info!(
                    "Finished prediction for block ({}, {}) in {} ms",
                    block.x / self.block_size,
                    block.y / self.block_size,
                    elapsed.as_millis()
                );
                result_tx.send(BlockResult {
                    x: block.x,
                    y: block.y,
                    win_w: block.win_w,
                    win_h: block.win_h,
                    data: out_data,
                })?;
                Ok::<(), anyhow::Error>(())
            })?;
            drop(result_tx);

            Ok(())
        })?;

        info!("Raster prediction completed: {:?}", self.output);
        Ok(())
    }
}
