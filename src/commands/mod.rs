mod band_select;
mod batch_translate;
mod sample_extraction;

use clap::Subcommand;

pub use band_select::BandSelectArgs;
pub use batch_translate::BatchTranslateArgs;
pub use sample_extraction::SampleExtractionArgs;

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Copy each pixel from one of the input bands according to a mask.
    BandSelect(BandSelectArgs),
    /// Translate a directory of images in batch mode.
    BatchTranslate(BatchTranslateArgs),
    /// Sample an image at given positions and output a table of the band values.
    SampleExtraction(SampleExtractionArgs),
}
