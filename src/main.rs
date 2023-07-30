mod commands;
mod gdal_ext;
mod threaded_block_reader;

use anyhow::Result;
use clap::Parser;
use commands::Command;

#[derive(Debug, Parser)]
#[command(author, version, about)]
/// A collection of useful GIS tools.
struct Args {
    #[command(subcommand)]
    command: Command,
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::SampleExtraction(args) => args.run(),
    }
}
