mod commands;
mod gdal_ext;
mod threaded_block_reader;

use anyhow::Result;
use clap::Parser;
use commands::Command;
use tracing_subscriber::fmt;

#[derive(Debug, Parser)]
#[command(author, version, about)]
/// A collection of useful GIS tools.
struct Args {
    #[command(subcommand)]
    command: Command,
}

fn main() -> Result<()> {
    fmt::init();

    let args = Args::parse();

    match args.command {
        Command::BandSelect(args) => args.run(),
        Command::BatchTranslate(args) => args.run(),
        Command::BuildVrt(args) => args.run(),
        Command::SampleExtraction(args) => args.run(),
        Command::SampleAugmentation(args) => args.run(),
    }
}
