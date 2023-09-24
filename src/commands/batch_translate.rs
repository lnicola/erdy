use std::{
    ffi::CString, iter, os::unix::prelude::OsStrExt, path::PathBuf, ptr, sync::mpsc, thread,
};

use anyhow::Result;
use clap::Parser;
use gdal::{DriverManager, GdalOpenFlags};
use indicatif::{ProgressBar, ProgressStyle};
use walkdir::{DirEntry, WalkDir};

#[derive(Debug, Parser)]
pub struct BatchTranslateArgs {
    /// Input path
    input: PathBuf,

    /// Translate options
    #[arg(last = true)]
    options: Vec<String>,
}

impl BatchTranslateArgs {
    pub fn run(&self) -> Result<()> {
        DriverManager::register_all();

        let style = ProgressStyle::with_template("{msg} {wide_bar} {pos}/{len}")?;
        let bar = ProgressBar::new(0).with_style(style);
        let (tx, rx) = mpsc::sync_channel::<DirEntry>(12800);

        let allowed_drivers = ["JP2OpenJPEG"];
        let allowed_drivers = allowed_drivers
            .iter()
            .map(|s| CString::new(s.as_bytes()))
            .collect::<Result<Vec<_>, _>>()?;

        dbg!(&self.options);
        let options = self
            .options
            .iter()
            .map(|s| CString::new(s.as_bytes()))
            .collect::<Result<Vec<_>, _>>()?;

        let bar_ = bar.clone();

        thread::scope(|scope| -> Result<()> {
            let thread = scope.spawn(move || {
                let allowed_drivers = allowed_drivers
                    .iter()
                    .map(|s| s.as_ptr())
                    .chain(iter::once(ptr::null()))
                    .collect::<Vec<_>>();

                let mut options = options
                    .iter()
                    .map(|s| s.as_ptr() as *mut _)
                    .chain(iter::once(ptr::null_mut()))
                    .collect::<Vec<_>>();
                let translate_options = unsafe {
                    gdal_sys::GDALTranslateOptionsNew(options.as_mut_ptr(), ptr::null_mut())
                };
                assert_ne!(translate_options, ptr::null_mut());

                for entry in rx {
                    let path_suffix = entry.path().strip_prefix(&self.input).unwrap();
                    bar_.set_message(path_suffix.display().to_string());

                    let c_path = CString::new(entry.path().as_os_str().as_bytes()).unwrap();
                    let rv = unsafe {
                        gdal_sys::GDALIdentifyDriverEx(
                            c_path.as_ptr(),
                            GdalOpenFlags::GDAL_OF_RASTER.bits(),
                            allowed_drivers.as_ptr(),
                            ptr::null_mut(),
                        )
                    };
                    if rv != ptr::null_mut() {
                        bar_.inc_length(1);

                        let output = entry.path().with_extension("tif");
                        let c_output = CString::new(output.as_os_str().as_bytes()).unwrap();
                        let ds = gdal::Dataset::open(entry.path()).unwrap();

                        let mut err = 0;
                        let ds_ptr = unsafe {
                            gdal_sys::GDALTranslate(
                                c_output.as_ptr(),
                                ds.c_dataset(),
                                translate_options as *const _,
                                &mut err,
                            )
                        };
                        if ds_ptr != ptr::null_mut() {
                            unsafe { gdal_sys::GDALClose(ds_ptr) };
                        } else {
                            dbg!(entry.path());
                        }
                    }

                    bar_.inc(1);
                }

                unsafe { gdal_sys::GDALTranslateOptionsFree(translate_options) };
            });

            for entry in WalkDir::new(&self.input) {
                let entry = entry?;
                if !entry.metadata()?.is_dir() {
                    tx.send(entry)?;
                }
            }
            drop(tx);

            thread.join().unwrap();
            bar.finish();

            Ok(())
        })
    }
}
