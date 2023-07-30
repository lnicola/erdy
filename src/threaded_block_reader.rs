use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::mpsc::{self, Sender},
    thread::{self, JoinHandle},
};

use gdal::Dataset;

use crate::gdal_ext::{RasterBandExt, TypedBlock};

pub trait BlockReducer {
    type InputState;
    type Output: Send + 'static;

    fn new(band_count: usize, input_state: Self::InputState) -> Self;
    fn push_block(&mut self, band_index: usize, band_count: usize, block: TypedBlock);
    fn finalize(self) -> Self::Output;
}

pub trait BlockFinalizer: Clone + Send + 'static {
    type Input: Send;

    fn apply(&self, input: Self::Input);
}

pub struct ThreadedBlockReader<T> {
    _workers: Vec<JoinHandle<()>>,
    request_txs: Vec<Sender<(usize, usize, T)>>,
    current_worker: usize,
}

impl<T: Send + 'static> ThreadedBlockReader<T> {
    pub fn new<R: BlockReducer<InputState = T>, F: BlockFinalizer<Input = R::Output>>(
        path: PathBuf,
        block_finalizer: F,
        num_threads: NonZeroUsize,
    ) -> Self {
        let num_threads = num_threads.into();
        let mut workers = Vec::with_capacity(num_threads);
        let mut request_txs = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let (tx, rx) = mpsc::channel();
            let path = path.clone();
            let block_finalizer = block_finalizer.clone();

            let worker = thread::spawn(move || {
                let dataset = Dataset::open(path).unwrap();
                let band_count = dataset.raster_count() as usize;

                for (x, y, data) in rx {
                    let mut block_reducer = R::new(band_count, data);

                    for band_index in 0..band_count {
                        let band = dataset.rasterband(band_index as isize + 1).unwrap();
                        let block = band.read_typed_block(x, y).unwrap();
                        block_reducer.push_block(band_index, band_count, block);
                    }

                    println!("{x} {y}");
                    block_finalizer.apply(block_reducer.finalize());
                }
            });
            workers.push(worker);
            request_txs.push(tx);
        }

        Self {
            _workers: workers,
            request_txs,
            current_worker: 0,
        }
    }

    pub fn submit(&mut self, x: usize, y: usize, data: T) {
        self.request_txs[self.current_worker]
            .send((x, y, data))
            .unwrap();
        self.current_worker += 1;
        if self.current_worker == self.request_txs.len() {
            self.current_worker = 0;
        }
    }
}
