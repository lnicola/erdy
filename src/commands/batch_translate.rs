use std::{
    ffi::CString,
    fs, iter,
    os::unix::prelude::OsStrExt,
    path::PathBuf,
    ptr,
    sync::mpsc::{self, Receiver},
    thread,
};

use anyhow::Result;
use clap::Parser;
use gdal::{Dataset, DriverManager, GdalOpenFlags};
use indicatif::{ProgressBar, ProgressStyle};
use tempfile::Builder;
use walkdir::{DirEntry, WalkDir};

#[derive(Debug, Parser)]
pub struct BatchTranslateArgs {
    /// Input path
    input: PathBuf,

    /// Output extension
    #[arg(long)]
    extension: String,

    #[arg(long)]
    remove: bool,

    /// Translate options
    #[arg(last = true)]
    options: Vec<String>,
}

impl BatchTranslateArgs {
    fn processing_thread(&self, options: Vec<CString>, bar: ProgressBar, rx: Receiver<DirEntry>) {
        let mut options = options
            .iter()
            .map(|s| s.as_ptr() as *mut _)
            .chain(iter::once(ptr::null_mut()))
            .collect::<Vec<_>>();
        let translate_options =
            unsafe { gdal_sys::GDALTranslateOptionsNew(options.as_mut_ptr(), ptr::null_mut()) };
        assert_ne!(translate_options, ptr::null_mut());

        for entry in rx {
            struct ProgressGuard<'a>(&'a ProgressBar);
            impl<'a> Drop for ProgressGuard<'a> {
                fn drop(&mut self) {
                    self.0.inc(1);
                }
            }

            let _guard = ProgressGuard(&bar);

            let path_suffix = entry
                .path()
                .strip_prefix(&self.input)
                .unwrap_or_else(|_| entry.path());
            bar.set_message(path_suffix.display().to_string());

            let path = entry.path().display();
            tracing::info!(%path, "processing");

            let Some(parent) = entry.path().parent() else {
                let path = entry.path().display();
                tracing::error!(%path, "unable to determine parent directory");
                continue;
            };

            let destination = entry.path().with_extension(&self.extension);
            let overwrite_input = destination == entry.path();
            if !self.remove && overwrite_input {
                tracing::error!(%path, "destination extension matches source extension, but `--remove` not passed");
                continue;
            }

            let Ok(temp) = Builder::new()
                .prefix(&format!("temp_{}", entry.file_name().to_string_lossy()))
                .suffix(&format!(".{}", self.extension))
                .keep(true)
                .tempfile_in(parent)
            else {
                let path = entry.path().display();
                tracing::error!(%path, "unable to create temporary file");
                continue;
            };

            match entry.metadata() {
                Ok(metadata) => {
                    // FIXME: use Builder::permissions
                    if let Err(e) = temp.as_file().set_permissions(metadata.permissions()) {
                        let path = temp.path().display();
                        tracing::error!(%e, %path, "unable to set output file permissions");
                    }
                }
                Err(e) => {
                    let path = entry.path().display();
                    tracing::error!(%e, %path, "unable to read input file metadata");
                }
            }

            let Ok(c_output) = CString::new(temp.path().as_os_str().as_bytes()) else {
                let path = temp.path().display();
                tracing::error!(%path, "invalid output name");
                continue;
            };
            let Ok(ds) = Dataset::open(entry.path()) else {
                let path = entry.path().display();
                tracing::error!(%path, "unable to open input dataset");
                continue;
            };

            let mut err = 0;
            let ds_ptr = unsafe {
                gdal_sys::GDALTranslate(
                    c_output.as_ptr(),
                    ds.c_dataset(),
                    translate_options as *const _,
                    &mut err,
                )
            };
            if !ds_ptr.is_null() {
                unsafe { gdal_sys::GDALClose(ds_ptr) };
            } else {
                let path = temp.path().display();
                tracing::error!(%path, "unable to close output dataset");
            }

            if self.remove {
                if !overwrite_input {
                    if let Err(e) = fs::remove_file(entry.path()) {
                        let path = entry.path().display();
                        tracing::error!(%e, %path, "unable to remove input file");
                    }
                }

                {
                    let temp_path = temp.path().display();
                    let destination = destination.display();
                    tracing::debug!(%temp_path, %destination, "renaming file");
                }
                if let Err(e) = fs::rename(temp.path(), &destination) {
                    let temp_path = temp.path().display();
                    let destination = destination.display();
                    tracing::error!(%e, %temp_path, %destination, "unable to rename output file");
                }
            }
        }

        unsafe { gdal_sys::GDALTranslateOptionsFree(translate_options) };
    }
}

impl BatchTranslateArgs {
    pub fn run(&self) -> Result<()> {
        DriverManager::register_all();

        let style = ProgressStyle::with_template("{msg} {wide_bar} {pos}/{len}")?;
        let bar = ProgressBar::new(0).with_style(style);
        let (tx, rx) = mpsc::sync_channel::<DirEntry>(4);

        let allowed_drivers = ["JP2OpenJPEG", "GTiff"];
        let allowed_drivers = allowed_drivers
            .iter()
            .map(|s| CString::new(s.as_bytes()))
            .collect::<Result<Vec<_>, _>>()?;

        let options = self
            .options
            .iter()
            .map(|s| CString::new(s.as_bytes()))
            .collect::<Result<Vec<_>, _>>()?;

        let bar_ = bar.clone();

        thread::scope(|scope| -> Result<()> {
            let thread = scope.spawn(move || self.processing_thread(options, bar_, rx));

            let allowed_drivers = allowed_drivers
                .iter()
                .map(|s| s.as_ptr())
                .chain(iter::once(ptr::null()))
                .collect::<Vec<_>>();

            for entry in WalkDir::new(&self.input) {
                let entry = entry?;
                if !entry.metadata()?.is_dir() {
                    let Ok(c_path) = CString::new(entry.path().as_os_str().as_bytes()) else {
                        let path = entry.path().display();
                        tracing::error!(%path, "invalid input name");
                        continue;
                    };

                    let rv = unsafe {
                        gdal_sys::GDALIdentifyDriverEx(
                            c_path.as_ptr(),
                            GdalOpenFlags::GDAL_OF_RASTER.bits(),
                            allowed_drivers.as_ptr(),
                            ptr::null_mut(),
                        )
                    };
                    if !rv.is_null() {
                        bar.inc_length(1);

                        tx.send(entry)?;
                    }
                }
            }
            drop(tx);

            thread.join().unwrap();
            bar.finish();

            Ok(())
        })
    }
}
