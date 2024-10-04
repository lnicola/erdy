use std::ffi::CString;

use gdal::errors::{GdalError, Result};
use gdal::vector::Defn;

pub trait DefnExt {
    fn get_field_index<S: AsRef<str>>(&self, field_name: S) -> Result<i32>;
}

impl DefnExt for Defn {
    fn get_field_index<S: AsRef<str>>(&self, field_name: S) -> Result<i32> {
        let c_str_field_name = CString::new(field_name.as_ref())?;
        let field_id =
            unsafe { gdal_sys::OGR_FD_GetFieldIndex(self.c_defn(), c_str_field_name.as_ptr()) };
        if field_id == -1 {
            return Err(GdalError::InvalidFieldName {
                field_name: field_name.as_ref().to_string(),
                method_name: "OGR_F_GetFieldIndex",
            });
        }
        Ok(field_id)
    }
}
