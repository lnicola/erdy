# `erdy`

A collection of potentially useful GIS tools, often faster than the alternatives.

Based on GDAL, using tiled input images is recommended.

## Available tools

### `band-select`

Samples one of a number of input rasters for each pixel, according to a label image.

### `batch-translate`

Recursively converts every image under a path to a different format.

### `build-vrt`

Creates a GDAL VRT from a number of input rasters.
Supports Sentinel-2 MSIL1C products and resamples the reflectance bands to 10m.

### `sample-augmentation`

Takes a table of features and synthetizes extra ones using SMOTE.

### `sample-extraction`

Takes an image and a set of points and extracts a table of the corresponding pixel values from that image.

## Building

Install GDAL, Rust and run `cargo run --release`.

## Docker image

`docker pull docker.io/lnicola/erdy:latest`
