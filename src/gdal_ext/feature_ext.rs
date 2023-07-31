use std::ffi::{c_double, c_int, c_longlong};

use gdal::{
    errors::GdalError,
    vector::{Feature, FieldValue},
};
use gdal_sys::OGRErr;

pub trait FeatureExt {
    fn set_fid(&self, fid: Option<u64>) -> gdal::errors::Result<()>;
    fn set_field_double_by_index(&self, idx: usize, value: f64);
    fn set_field_integer_by_index(&self, idx: usize, value: i32);
    fn set_field_integer64_by_index(&self, idx: usize, value: i64);
    fn set_field_by_index(&self, idx: usize, value: &FieldValue);
}

impl FeatureExt for Feature<'_> {
    fn set_fid(&self, fid: Option<u64>) -> gdal::errors::Result<()> {
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

    fn set_field_double_by_index(&self, idx: usize, value: f64) {
        unsafe {
            gdal_sys::OGR_F_SetFieldDouble(self.c_feature(), idx as i32, value as c_double);
        }
    }

    fn set_field_integer_by_index(&self, idx: usize, value: i32) {
        unsafe {
            gdal_sys::OGR_F_SetFieldInteger(self.c_feature(), idx as i32, value as c_int);
        }
    }

    fn set_field_integer64_by_index(&self, idx: usize, value: i64) {
        unsafe {
            gdal_sys::OGR_F_SetFieldInteger64(self.c_feature(), idx as i32, value as c_longlong);
        }
    }

    fn set_field_by_index(&self, idx: usize, value: &FieldValue) {
        match value {
            FieldValue::IntegerValue(value) => self.set_field_integer_by_index(idx, *value),
            // FieldValue::IntegerListValue(value) => self.set_field_integer_list(field_name, value),
            FieldValue::Integer64Value(value) => self.set_field_integer64_by_index(idx, *value),
            // FieldValue::Integer64ListValue(value) => {
            //     self.set_field_integer64_list(field_name, value)
            // }
            // FieldValue::StringValue(ref value) => self.set_field_string(field_name, value.as_str()),
            // FieldValue::StringListValue(ref value) => {
            //     let strs = value.iter().map(String::as_str).collect::<Vec<&str>>();
            //     self.set_field_string_list(field_name, &strs)
            // }
            FieldValue::RealValue(value) => self.set_field_double_by_index(idx, *value),
            // FieldValue::RealListValue(value) => self.set_field_double_list(field_name, value),
            // FieldValue::DateValue(value) => {
            //     let dv = value
            //         .and_hms_opt(0, 0, 0)
            //         .ok_or_else(|| GdalError::DateError("offset to midnight".into()))?;
            //     let dt = DateTime::from_utc(
            //         dv,
            //         FixedOffset::east_opt(0)
            //             .ok_or_else(|| GdalError::DateError("utc offset".into()))?,
            //     );
            //     self.set_field_datetime(field_name, dt)
            // }
            // FieldValue::DateTimeValue(value) => self.set_field_datetime(field_name, *value),
            _ => unimplemented!(),
        }
    }
}
