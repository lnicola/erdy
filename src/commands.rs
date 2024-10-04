mod band_select;
mod batch_translate;
mod build_vrt;
mod compute_confusion_matrix;
mod remap_confusion_matrix;
mod sample_augmentation;
mod sample_extraction;
mod sample_selection;

use clap::Subcommand;

pub use band_select::BandSelectArgs;
pub use batch_translate::BatchTranslateArgs;
pub use build_vrt::BuildVrtArgs;
pub use compute_confusion_matrix::ComputeConfusionMatrixArgs;
pub use remap_confusion_matrix::RemapConfusionMatrixArgs;
pub use sample_extraction::SampleExtractionArgs;
pub use sample_selection::SampleSelectionArgs;

use self::sample_augmentation::SampleAugmentationArgs;

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Copy each pixel from one of the input bands according to a mask.
    BandSelect(BandSelectArgs),
    /// Translate a directory of images in batch mode.
    BatchTranslate(BatchTranslateArgs),
    /// Builds a VRT.
    BuildVrt(BuildVrtArgs),
    /// Compute the confusion matrix for a vector dataset
    ComputeConfusionMatrix(ComputeConfusionMatrixArgs),
    /// Remap the labels in a confusion matrix
    RemapConfusionMatrix(RemapConfusionMatrixArgs),
    /// Sample an image at given positions and output a table of the band values.
    SampleExtraction(SampleExtractionArgs),
    /// Augment a vector dataset.
    SampleAugmentation(SampleAugmentationArgs),
    /// Samples labelled points from a raster.
    SampleSelection(SampleSelectionArgs),
}
