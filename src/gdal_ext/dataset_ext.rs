use gdal::{errors::GdalError, Dataset};
use gdal_sys::OGRErr;

pub trait DatasetExt {
    fn maybe_run_in_batch(
        &mut self,
        func: impl FnMut(&mut Dataset) -> Result<(), GdalError>,
    ) -> Result<(), GdalError>;
}

impl DatasetExt for Dataset {
    fn maybe_run_in_batch(
        &mut self,
        mut func: impl FnMut(&mut Dataset) -> Result<(), GdalError>,
    ) -> Result<(), GdalError> {
        let force = 0;
        let rv = unsafe { gdal_sys::GDALDatasetStartTransaction(self.c_dataset(), force) };
        if rv == OGRErr::OGRERR_UNSUPPORTED_OPERATION {
            func(self)
        } else if rv == OGRErr::OGRERR_NONE {
            let res = func(self);
            if res.is_ok() {
                let rv = unsafe { gdal_sys::GDALDatasetCommitTransaction(self.c_dataset()) };
                if rv != OGRErr::OGRERR_NONE {
                    Err(GdalError::OgrError {
                        err: rv,
                        method_name: "GDALDatasetCommitTransaction",
                    })
                } else {
                    res
                }
            } else {
                let _ = unsafe { gdal_sys::GDALDatasetRollbackTransaction(self.c_dataset()) };
                // ignore rollback error because it's not guaranteed to be
                // supported, and the `func` failure is more important.
                res
            }
        } else {
            // transaction supported but failed
            Err(GdalError::OgrError {
                err: rv,
                method_name: "GDALDatasetStartTransaction",
            })
        }
    }
}
