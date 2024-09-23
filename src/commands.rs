mod band_select;
mod batch_translate;
mod build_vrt;
mod sample_augmentation;
mod sample_extraction;
mod sample_selection;

use clap::Subcommand;

pub use band_select::BandSelectArgs;
pub use batch_translate::BatchTranslateArgs;
pub use build_vrt::BuildVrtArgs;
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
    /// Sample an image at given positions and output a table of the band values.
    SampleExtraction(SampleExtractionArgs),
    /// Augment a vector dataset.
    SampleAugmentation(SampleAugmentationArgs),
    /// Samples labelled points from a raster.
    SampleSelection(SampleSelectionArgs),
}
