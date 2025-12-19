use serde::Deserialize;
use std::ffi::{CStr, CString, NulError};
use std::path::Path;
use thiserror::Error;

pub mod sys;

#[derive(Clone, Debug, Error)]
pub enum CatBoostError {
    #[error("CatBoostError: {0}")]
    CatBoost(String),
    #[error(transparent)]
    Nul(#[from] NulError),
    #[error("all feature vectors must have the same length")]
    DifferentLengthVectors,
    #[error("unknown CatBoost error")]
    Unknown,
}

pub fn check_return_value(ret_val: bool) -> Result<(), CatBoostError> {
    if ret_val {
        Ok(())
    } else {
        Err(get_error())
    }
}

fn get_error() -> CatBoostError {
    let error_str = unsafe {
        let ptr = sys::GetErrorString();
        if ptr.is_null() {
            return CatBoostError::Unknown;
        }
        CStr::from_ptr(ptr)
    };
    let message = error_str.to_string_lossy().into_owned();
    CatBoostError::CatBoost(message)
}

#[allow(dead_code)]
#[repr(u32)]
pub enum EApiPredictionType {
    RawFormulaVal = sys::EApiPredictionType_APT_RAW_FORMULA_VAL,
    Exponent = sys::EApiPredictionType_APT_EXPONENT,
    RmseWithUncertainty = sys::EApiPredictionType_APT_RMSE_WITH_UNCERTAINTY,
    Probability = sys::EApiPredictionType_APT_PROBABILITY,
    Class = sys::EApiPredictionType_APT_CLASS,
    MultiProbability = sys::EApiPredictionType_APT_MULTI_PROBABILITY,
}

pub struct Model {
    handle: *mut sys::ModelCalcerHandle,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn load_from_file(path: &Path) -> Result<Self, CatBoostError> {
        let handle = unsafe { sys::ModelCalcerCreate() };
        if handle.is_null() {
            return Err(get_error());
        }

        let c_path = CString::new(path.to_str().unwrap_or_default())?;
        let success = unsafe { sys::LoadFullModelFromFile(handle, c_path.as_ptr()) };

        if success {
            Ok(Self { handle })
        } else {
            let error = get_error();
            unsafe { sys::ModelCalcerDelete(handle) };
            Err(error)
        }
    }

    pub fn set_prediction_type(
        &mut self,
        prediction_type: EApiPredictionType,
    ) -> Result<(), CatBoostError> {
        check_return_value(unsafe { sys::SetPredictionType(self.handle, prediction_type as u32) })
    }

    pub fn get_prediction_dimensions_count(&self) -> usize {
        unsafe { sys::GetPredictionDimensionsCount(self.handle) }
    }

    pub fn get_model_info_value(&self, key: &str) -> Option<String> {
        let value_ptr =
            unsafe { sys::GetModelInfoValue(self.handle, key.as_ptr() as *const _, key.len()) };

        if value_ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(value_ptr).to_string_lossy().into_owned() })
        }
    }

    pub fn calc_model_prediction_flat_transposed(
        &self,
        float_features: &[&[f32]],
        result: &mut Vec<f64>,
    ) -> Result<(), CatBoostError> {
        if float_features.is_empty() {
            result.clear();
            return Ok(());
        }
        let doc_count = float_features[0].len();
        if doc_count == 0 {
            result.clear();
            return Ok(());
        }

        let feature_count = float_features.len();

        for features in float_features.iter().skip(1) {
            if features.len() != doc_count {
                return Err(CatBoostError::DifferentLengthVectors);
            }
        }

        let float_feature_pointers: Vec<*const f32> =
            float_features.iter().map(|f| f.as_ptr()).collect();

        let required_result_size = doc_count * self.get_prediction_dimensions_count();
        result.resize(required_result_size, 0.0);

        check_return_value(unsafe {
            sys::CalcModelPredictionFlatTransposed(
                self.handle,
                doc_count,
                float_feature_pointers.as_ptr(),
                feature_count,
                result.as_mut_ptr(),
                result.len(),
            )
        })
    }

    pub fn params(&self) -> Option<Params> {
        self.get_model_info_value("params")
            .map(|params| serde_json::from_str(&params))
            .transpose()
            .unwrap()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::ModelCalcerDelete(self.handle) };
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct DataProcessingOptions {
    pub class_names: Vec<u16>,
}

#[derive(Debug, Deserialize)]
pub struct Params {
    pub data_processing_options: DataProcessingOptions,
}
