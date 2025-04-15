# `erdy`

A collection of potentially useful GIS tools, often faster than the alternatives.

Based on GDAL, using tiled input images is recommended.

## Available tools

| Command | Description |
|---------|-------------|
| `band-select` | Samples one of a number of input rasters for each pixel, according to a label image. |
| `batch-translate` | Recursively converts every image under a path to a different format. |
| `build-vrt` | Creates a GDAL VRT from a number of input rasters. Supports Sentinel-2 MSIL1C products and resamples the reflectance bands to 10m. |
| `compute-confusion-matrix` | Computes the confusion matrix for a number of vector datasets. |
| `remap-confusion-matrix` | Remaps the labels in a confusion matrix using a remapping table. |
| `sample-augmentation` | Takes a table of features and synthesizes extra ones using SMOTE. |
| `sample-extraction` | Takes an image and a set of points and extracts a table of the corresponding pixel values from that image. |
| `sample-selection` | Samples labelled points from a raster image, with control over class distribution. |
| `temporal-resampling` | Gap-fills and interpolates a time series of images at given output dates. Doesn't actually exist yet. |

## Building

Install GDAL, Rust and run `cargo run --release`.

## Container image

Available at [`ghcr.io/lnicola/erdy:latest`](https://ghcr.io/lnicola/erdy).
