use std::{
    collections::HashMap,
    mem,
    ops::DerefMut,
    panic,
    path::PathBuf,
    sync::Arc,
    thread::{self, JoinHandle},
};

use anyhow::Result;
use clap::Parser;
use flume::{Receiver, Sender};
use gdal::Dataset;
use parking_lot::Mutex;
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};

use crate::gdal_ext::TypedBuffer;

#[derive(Debug, Parser)]
pub struct TemporalResamplingArgs {
    /// Input images
    #[arg(long, value_parser, num_args = 1..)]
    inputs: Vec<PathBuf>,

    /// Input validity masks
    #[arg(long, value_parser, num_args = 1..)]
    masks: Vec<PathBuf>,

    /// Output dataset
    #[arg(long, value_parser)]
    output: Vec<PathBuf>,

    /// I/O threads
    #[arg(long)]
    io_threads: Option<usize>,

    /// Input dates
    #[arg(long, num_args = 1..)]
    input_dates: Vec<i16>,

    /// Output dates
    #[arg(long, num_args = 1..)]
    output_dates: Vec<i16>,
}

type BlockReadHandler = Box<dyn Fn(usize, usize, HashMap<usize, TypedBuffer>) + Send + Sync>;

struct BlockReadRequest {
    dataset: Arc<Mutex<Dataset>>,
    num_datasets: usize,
    idx: usize,
    x: usize,
    y: usize,
    state: BlockReadState,
    handler: Arc<BlockReadHandler>,
}

#[derive(Clone)]
struct BlockReadState {
    blocks: Arc<Mutex<HashMap<usize, TypedBuffer>>>,
    region_size: (usize, usize),
}

struct ParallelBlockReader {
    datasets: Box<[Arc<Mutex<Dataset>>]>,
    region_size: (usize, usize),
    blocks: (usize, usize),
    workers: Vec<JoinHandle<()>>,
    req_tx: Sender<BlockReadRequest>,
}

impl ParallelBlockReader {
    pub fn new(datasets: Box<[Arc<Mutex<Dataset>>]>, threads: usize) -> gdal::errors::Result<Self> {
        let (req_tx, req_rx) = flume::unbounded();

        let mut workers = Vec::new();
        for _ in 0..threads {
            let req_rx: Receiver<BlockReadRequest> = req_rx.clone();

            workers.push(thread::spawn(move || {
                for request in req_rx {
                    let block = {
                        let region_size = request.state.region_size;
                        let dataset = request.dataset.lock();
                        let band = dataset.rasterband(1).unwrap();
                        let size = band.size();
                        let window = (request.x * region_size.0, request.y * region_size.1);
                        let window_size = (
                            if window.0 + region_size.0 <= size.0 {
                                region_size.0
                            } else {
                                size.0 - window.0
                            },
                            if window.1 + region_size.1 <= size.1 {
                                region_size.1
                            } else {
                                size.1 - window.1
                            },
                        );
                        let buffer = band
                            .read_as::<u16>(
                                (window.0 as isize, window.1 as isize),
                                window_size,
                                window_size,
                                None,
                            )
                            .unwrap();

                        TypedBuffer::U16(buffer)
                        // band.read_typed_block(request.x, request.y).unwrap()
                    };
                    let blocks = {
                        let mut blocks = request.state.blocks.lock();
                        blocks.insert(request.idx, block);
                        if blocks.len() == request.num_datasets {
                            let blocks = mem::take(blocks.deref_mut());
                            Some(blocks)
                        } else {
                            None
                        }
                    };
                    if let Some(blocks) = blocks {
                        let BlockReadRequest { handler, .. } = request;
                        (handler)(request.x, request.y, blocks);
                    }
                }
            }));
        }

        let dataset = datasets[0].lock();
        let band = dataset.rasterband(1)?;
        let raster_size = band.size();
        let block_size = band.block_size();
        // let block_size = (1024, 1024);
        let _geo_transform = dataset.geo_transform()?;
        drop(dataset);

        let region_size = block_size;
        let blocks = (
            raster_size.0.div_ceil(block_size.0),
            raster_size.1.div_ceil(block_size.1),
        );

        Ok(Self {
            datasets,
            region_size,
            blocks,
            workers,
            req_tx,
        })
    }

    pub fn run(
        &self,
        block_x: usize,
        block_y: usize,
        dataset_indices: &[usize],
        handler: BlockReadHandler,
    ) {
        let mut pending = HashMap::<_, usize>::new();

        let handler = Arc::new(handler);
        let state = BlockReadState {
            region_size: self.region_size,
            blocks: Arc::new(Mutex::new(HashMap::new())),
        };
        for &idx in dataset_indices {
            *pending.entry((block_x, block_y)).or_default() += 1;
            let request = BlockReadRequest {
                dataset: self.datasets[idx].clone(),
                num_datasets: dataset_indices.len(),
                idx,
                x: block_x,
                y: block_y,
                state: state.clone(),
                handler: handler.clone(),
            };
            self.req_tx.send(request).unwrap();
        }
    }

    pub fn join(self) {
        drop(self.req_tx);

        let mut errors = Vec::new();
        for worker in self.workers {
            if let Err(e) = worker.join() {
                errors.push(e);
            }
        }

        if !errors.is_empty() {
            panic::resume_unwind(Box::new(errors));
        }
    }
}

impl TemporalResamplingArgs {
    pub fn run(&mut self) -> Result<()> {
        let datasets = self
            .inputs
            .par_iter()
            .map(|p| -> gdal::errors::Result<Arc<Mutex<Dataset>>> {
                Ok(Arc::new(Mutex::new(Dataset::open(p)?)))
            })
            .collect::<gdal::errors::Result<Vec<_>>>()?
            .into_boxed_slice();

        println!("Opened");

        let block_reader = ParallelBlockReader::new(datasets, self.io_threads.unwrap_or(32))?;

        let dataset_indices = (0..self.inputs.len()).collect::<Vec<_>>();
        for y in 0..block_reader.blocks.1 {
            for x in 0..block_reader.blocks.0 {
                block_reader.run(
                    x,
                    y,
                    &dataset_indices,
                    Box::new(move |x, y, _blocks| {
                        println!("block ({x}, {y}) done");
                    }),
                );
            }
        }
        println!("done sending read requests");

        block_reader.join();

        println!("done");

        Ok(())
    }
}
