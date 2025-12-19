#![allow(non_upper_case_globals)]

use std::ffi::{c_char, c_void};

pub type ModelCalcerHandle = c_void;

pub const EApiPredictionType_APT_RAW_FORMULA_VAL: EApiPredictionType = 0;
pub const EApiPredictionType_APT_EXPONENT: EApiPredictionType = 1;
pub const EApiPredictionType_APT_RMSE_WITH_UNCERTAINTY: EApiPredictionType = 2;
pub const EApiPredictionType_APT_PROBABILITY: EApiPredictionType = 3;
pub const EApiPredictionType_APT_CLASS: EApiPredictionType = 4;
pub const EApiPredictionType_APT_MULTI_PROBABILITY: EApiPredictionType = 5;
pub type EApiPredictionType = ::std::os::raw::c_uint;

#[link(name = "catboostmodel")]
extern "C" {
    pub fn GetErrorString() -> *const c_char;
    pub fn ModelCalcerCreate() -> *mut ModelCalcerHandle;
    pub fn ModelCalcerDelete(modelHandle: *mut ModelCalcerHandle);
    pub fn LoadFullModelFromFile(
        modelHandle: *mut ModelCalcerHandle,
        filename: *const c_char,
    ) -> bool;
    pub fn SetPredictionType(modelHandle: *mut ModelCalcerHandle, predictionType: u32) -> bool;
    pub fn GetPredictionDimensionsCount(modelHandle: *mut ModelCalcerHandle) -> usize;
    pub fn CalcModelPredictionFlatTransposed(
        modelHandle: *mut ModelCalcerHandle,
        docCount: usize,
        floatFeatures: *const *const f32,
        floatFeaturesSize: usize,
        result: *mut f64,
        resultSize: usize,
    ) -> bool;
    pub fn GetModelInfoValue(
        modelHandle: *mut ModelCalcerHandle,
        keyPtr: *const c_char,
        keySize: usize,
    ) -> *const c_char;
}
