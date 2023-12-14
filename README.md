# `erdy`

A collection of useful GIS tools, implemented with a focus on performance.

Based on GDAL, using tiled input images is recommended.

## Available tools

### `sample-extraction`

Takes an image and a set of points and extracts a table of the corresponding pixel values from that image.

## Building

Install GDAL, Rust and run `cargo run --release`.

## Docker image

`docker pull docker.io/lnicola/erdy`
