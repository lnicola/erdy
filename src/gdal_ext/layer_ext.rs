use arrow_array::ffi::FFI_ArrowArray;
use arrow_schema::ffi::FFI_ArrowSchema;
use gdal::cpl::CslStringList;
use gdal::errors::GdalError;
use gdal::vector::{Layer, LayerAccess as _};
use std::ffi::{c_char, CStr};
use std::ptr;

pub trait LayerExt {
    fn create_field_from_arrow_schema(
        &self,
        schema: &FFI_ArrowSchema,
        options: Option<&CslStringList>,
    ) -> Result<(), GdalError>;

    fn write_arrow_batch(
        &mut self,
        schema: &FFI_ArrowSchema,
        array: &mut FFI_ArrowArray,
        options: Option<&CslStringList>,
    ) -> Result<(), GdalError>;
}

impl LayerExt for Layer<'_> {
    fn create_field_from_arrow_schema(
        &self,
        schema: &FFI_ArrowSchema,
        options: Option<&CslStringList>,
    ) -> Result<(), GdalError> {
        let papsz_options = options.map_or(ptr::null_mut(), |o| o.as_ptr());
        let success = unsafe {
            gdal_sys::OGR_L_CreateFieldFromArrowSchema(
                self.c_layer(),
                schema as *const _ as *const _,
                papsz_options as *mut _,
            )
        };

        if success {
            Ok(())
        } else {
            Err(_last_cpl_err())
        }
    }

    fn write_arrow_batch(
        &mut self,
        schema: &FFI_ArrowSchema,
        array: &mut FFI_ArrowArray,
        options: Option<&CslStringList>,
    ) -> Result<(), GdalError> {
        let papsz_options = options.map_or(ptr::null_mut(), |o| o.as_ptr());
        let success = unsafe {
            gdal_sys::OGR_L_WriteArrowBatch(
                self.c_layer(),
                schema as *const _ as *const _,
                array as *mut _ as *mut _,
                papsz_options as *mut _,
            )
        };

        if success {
            Ok(())
        } else {
            Err(_last_cpl_err())
        }
    }
}

fn _last_cpl_err() -> GdalError {
    let last_err_class = unsafe { gdal_sys::CPLGetLastErrorType() };
    let last_err_no = unsafe { gdal_sys::CPLGetLastErrorNo() };
    let last_err_msg = _string(unsafe { gdal_sys::CPLGetLastErrorMsg() });
    unsafe { gdal_sys::CPLErrorReset() };

    GdalError::CplError {
        class: last_err_class,
        number: last_err_no,
        msg: last_err_msg.unwrap_or_default(),
    }
}

fn _string(raw_ptr: *const c_char) -> Option<String> {
    if raw_ptr.is_null() {
        None
    } else {
        let c_str = unsafe { CStr::from_ptr(raw_ptr) };
        Some(c_str.to_string_lossy().into_owned())
    }
}
