mod sample_extraction;

use clap::Subcommand;

pub use sample_extraction::SampleExtractionArgs;

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Sample an image at given positions and output a table of the band values.
    SampleExtraction(SampleExtractionArgs),
}
