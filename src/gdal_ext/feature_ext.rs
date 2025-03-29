use gdal::{errors::GdalError, vector::Feature};
use gdal_sys::OGRErr;

pub trait FeatureExt {
    fn set_fid(&mut self, fid: Option<u64>) -> gdal::errors::Result<()>;
}

impl FeatureExt for Feature<'_> {
    fn set_fid(&mut self, fid: Option<u64>) -> gdal::errors::Result<()> {
        let rv = unsafe {
            gdal_sys::OGR_F_SetFID(self.c_feature(), fid.map(|fid| fid as i64).unwrap_or(-1))
        };

        if rv != OGRErr::OGRERR_NONE {
            return Err(GdalError::OgrError {
                err: OGRErr::OGRERR_FAILURE,
                method_name: "OGR_F_SetFID",
            });
        }

        Ok(())
    }
}
