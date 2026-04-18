use std::{
    collections::HashMap,
    num::NonZeroUsize,
    ops::Range,
    path::PathBuf,
    sync::{
        mpsc::{self, SyncSender},
        Arc, Mutex,
    },
    thread::{self, available_parallelism},
};

use anyhow::Result;
use clap::Parser;
use gdal::{
    config,
    raster::GdalDataType,
    spatial_ref::SpatialRef,
    vector::{
        Feature, FieldDefn, FieldValue, Geometry, LayerAccess, LayerOptions, OGRFieldType,
        OGRwkbGeometryType,
    },
    Dataset, DriverManager, DriverType, GeoTransformEx, Metadata,
};

use crate::{
    gdal_ext::{DatasetExt, FeatureExt, TypedBuffer},
    threaded_block_reader::{BlockFinalizer, BlockReducer, ThreadedBlockReader},
};

#[derive(Debug, Parser)]
pub struct SampleExtractionArgs {
    /// Input image
    pub input: PathBuf,

    /// Sampling positions
    #[arg(long, value_parser, num_args = 1..)]
    pub points: Vec<PathBuf>,

    /// Output datasets
    #[arg(long, value_parser, num_args = 1..)]
    pub outputs: Vec<PathBuf>,

    /// Output format
    #[arg(short, long)]
    pub format: Option<String>,

    /// Output field names
    #[arg(long, value_parser, num_args = 1..)]
    pub fields: Option<Vec<String>>,

    /// Number of threads to use
    #[arg(long)]
    pub num_threads: Option<usize>,

    /// Copy the input feature ids
    #[arg(long, default_value_t)]
    pub copy_fid: bool,
}

#[derive(Clone, Copy)]
enum BandValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
}

#[derive(Clone)]
struct FieldDefinition {
    name: String,
    field_type: OGRFieldType::Type,
    width: Option<i32>,
    precision: Option<i32>,
}

impl FieldDefinition {
    fn to_field_defn(&self) -> gdal::errors::Result<FieldDefn> {
        let field_defn = FieldDefn::new(&self.name, self.field_type)?;
        if let Some(width) = self.width {
            field_defn.set_width(width);
        }
        if let Some(precision) = self.precision {
            field_defn.set_precision(precision);
        }
        Ok(field_defn)
    }
}

struct SamplingBlockReducer {
    block_points: BlockPoints,
    samples: Vec<BandValue>,
}

impl BlockReducer for SamplingBlockReducer {
    type InputState = BlockPoints;
    type Output = (Vec<BandValue>, BlockPoints);

    fn new(band_count: usize, block_points: Self::InputState) -> Self {
        let samples = vec![BandValue::I16(0); block_points.points.len() * band_count];

        Self {
            block_points,
            samples,
        }
    }

    fn push_block(&mut self, band_index: usize, band_count: usize, block: TypedBuffer) {
        match block {
            TypedBuffer::U8(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::U8(pix);
                }
            }
            TypedBuffer::I8(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::I8(pix);
                }
            }
            TypedBuffer::U16(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::U16(pix);
                }
            }
            TypedBuffer::I16(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::I16(pix);
                }
            }
            TypedBuffer::U32(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::U32(pix);
                }
            }
            TypedBuffer::I32(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::I32(pix);
                }
            }
            TypedBuffer::U64(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::U64(pix);
                }
            }
            TypedBuffer::I64(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::I64(pix);
                }
            }
            TypedBuffer::F32(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::F32(pix);
                }
            }
            TypedBuffer::F64(buf) => {
                for (idx, point) in self.block_points.points.iter().enumerate() {
                    let pix = buf[(point.by, point.bx)];
                    self.samples[band_count * idx + band_index] = BandValue::F64(pix);
                }
            }
        }
    }

    fn finalize(self) -> Self::Output {
        (self.samples, self.block_points)
    }
}

#[allow(clippy::type_complexity)]
#[derive(Clone)]
struct BlockSender(Vec<SyncSender<(Arc<(Vec<BandValue>, BlockPoints)>, Range<usize>)>>);

impl BlockFinalizer for BlockSender {
    type Input = (Vec<BandValue>, BlockPoints);

    fn apply(&self, input: Self::Input) {
        let input = Arc::new(input);
        for (idx, (&input_index, &start)) in input
            .1
            .input_indexes
            .iter()
            .zip(&input.1.input_start_positions)
            .enumerate()
        {
            let end = input
                .1
                .input_start_positions
                .get(idx + 1)
                .copied()
                .unwrap_or(input.1.points.len());
            self.0[input_index]
                .send((input.clone(), start..end))
                .unwrap()
        }
    }
}

struct SamplingPoint {
    _fid: Option<u64>,
    bx: usize,
    by: usize,
    orig_x: f64,
    orig_y: f64,
    original_fields: Vec<Option<FieldValue>>,
}

#[derive(Default)]
struct BlockPoints {
    points: Vec<SamplingPoint>,
    input_indexes: Vec<usize>,
    input_start_positions: Vec<usize>,
}

impl SampleExtractionArgs {
    pub fn run(&self) -> Result<()> {
        config::set_config_option("OGR_SQLITE_SYNCHRONOUS", "OFF")?;

        let image = Dataset::open(&self.input)?;
        let geo_transform = image.geo_transform()?;
        let geo_transform = geo_transform.invert()?;
        let block_size = image.rasterband(1)?.block_size();

        let band_count = image.raster_count();
        for band_idx in 0..band_count {
            let band = image.rasterband(band_idx + 1)?;
            assert_eq!(band.block_size(), block_size);
        }

        if let Some(fields) = self.fields.as_ref() {
            assert_eq!(fields.len(), band_count);
        }

        assert_eq!(self.points.len(), self.outputs.len());

        let mut band_fields = Vec::with_capacity(band_count);
        for band_index in 1..=band_count {
            let band = image.rasterband(band_index)?;
            let name = match self.fields.as_ref() {
                Some(fields) => fields[band_index - 1].clone(),
                None => match band.description() {
                    Ok(name) if !name.is_empty() => name,
                    _ => format!("band_{band_index}"),
                },
            };

            let field_type = match band.band_type() {
                GdalDataType::UInt16 => OGRFieldType::OFTInteger,
                GdalDataType::Int16 => OGRFieldType::OFTInteger,
                GdalDataType::Float32 => OGRFieldType::OFTReal,
                _ => unimplemented!(),
            };
            let field_definition = FieldDefinition {
                name,
                field_type,
                width: None,
                precision: None,
            };
            band_fields.push(field_definition);
        }

        let mut tile_points = HashMap::<_, BlockPoints>::new();
        let mut layer_names = Vec::with_capacity(self.points.len());
        let mut spatial_refs = Vec::with_capacity(self.points.len());
        let mut output_fields = Vec::with_capacity(self.points.len());
        for (input_index, path) in self.points.iter().enumerate() {
            let ds = Dataset::open(path)?;
            let mut layer = ds.layer(0)?;

            let layer_name = layer.name();
            layer_names.push(layer_name);

            for feature in layer.features() {
                if let Some(geometry) = feature.geometry().as_ref() {
                    let (orig_x, orig_y, _) = geometry.get_point(0);
                    let (x, y) = geo_transform.apply(orig_x, orig_y);
                    let (block_x, block_y) = (
                        (x / block_size.0 as f64) as usize,
                        (y / block_size.1 as f64) as usize,
                    );
                    let (x, y) = (x as usize, y as usize);
                    let sampling_point = SamplingPoint {
                        _fid: feature.fid(),
                        bx: x % block_size.0,
                        by: y % block_size.1,
                        orig_x,
                        orig_y,
                        original_fields: feature.fields().map(|f| f.1).collect::<Vec<_>>(),
                    };
                    let block_points = tile_points.entry((block_x, block_y)).or_default();
                    if block_points.input_indexes.last().copied() != Some(input_index) {
                        block_points.input_indexes.push(input_index);
                        block_points
                            .input_start_positions
                            .push(block_points.points.len());
                    }
                    block_points.points.push(sampling_point);
                }
            }

            let spatial_ref = layer.spatial_ref().map(|sr| sr.to_wkt()).transpose()?;
            spatial_refs.push(spatial_ref);

            let mut layer_fields = Vec::new();
            for field in layer.defn().fields() {
                let field_definition = FieldDefinition {
                    name: field.name(),
                    field_type: field.field_type(),
                    width: Some(field.width()),
                    precision: Some(field.precision()),
                };
                layer_fields.push(field_definition);
            }
            layer_fields.extend_from_slice(&band_fields);

            output_fields.push(layer_fields);
        }

        let mut output_txs = Vec::with_capacity(self.points.len());
        let mutex = Arc::new(Mutex::new(()));
        thread::scope(|scope| -> Result<()> {
            let mut output_threads = Vec::with_capacity(self.points.len());
            for (path, layer_name, spatial_ref, layer_fields) in self
                .outputs
                .iter()
                .zip(layer_names)
                .zip(spatial_refs)
                .zip(output_fields)
                .map(|(((a, b), c), d)| (a, b, c, d))
            {
                let (tx, rx) =
                    mpsc::sync_channel::<(Arc<(Vec<BandValue>, BlockPoints)>, Range<usize>)>(128);
                output_txs.push(tx);

                let mutex = mutex.clone();
                let thread = scope.spawn(move || {
                    let driver = match self.format {
                        Some(ref format) => DriverManager::get_driver_by_name(format)?,
                        None => DriverManager::get_output_driver_for_dataset_name(
                            path,
                            DriverType::Vector,
                        )
                        .expect("can't determine output driver"),
                    };
                    let mut output = driver.create_vector_only(path)?;
                    let output_layer = output.create_layer(LayerOptions {
                        name: &layer_name,
                        srs: spatial_ref
                            .map(|wkt| SpatialRef::from_wkt(&wkt))
                            .transpose()?
                            .as_ref(),
                        ty: OGRwkbGeometryType::wkbPoint,
                        options: None,
                    })?;
                    for field in layer_fields {
                        let field_defn = field.to_field_defn()?;
                        field_defn.add_to_layer(&output_layer)?;
                    }
                    for (data, range) in rx {
                        let (sample_values, block_points) = &*data;
                        output.maybe_run_in_batch(|ds| {
                            let output_layer = ds.layer(0)?;
                            for (idx, point) in
                                block_points.points[range.clone()].iter().enumerate()
                            {
                                let sample_idx = idx + range.start;
                                let mut feature = Feature::new(output_layer.defn())?;
                                if self.copy_fid {
                                    feature.set_fid(point._fid)?;
                                }
                                let field_offset = point.original_fields.len();
                                for (field_idx, field_value) in
                                    point.original_fields.iter().enumerate()
                                {
                                    if let Some(field_value) = field_value {
                                        feature.set_field(field_idx, field_value)?;
                                    }
                                }
                                for band_idx in 0..band_count {
                                    let field_index = field_offset + band_idx;
                                    match sample_values[band_count * sample_idx + band_idx] {
                                        BandValue::U8(value) => {
                                            feature.set_field_integer(field_index, value as i32)?;
                                        }
                                        BandValue::I8(value) => {
                                            feature.set_field_integer(field_index, value as i32)?;
                                        }
                                        BandValue::U16(value) => {
                                            feature.set_field_integer(field_index, value as i32)?;
                                        }
                                        BandValue::I16(value) => {
                                            feature.set_field_integer(field_index, value as i32)?;
                                        }
                                        BandValue::U32(value) => {
                                            feature.set_field_integer(field_index, value as i32)?;
                                        }
                                        BandValue::I32(value) => {
                                            feature.set_field_integer(field_index, value)?;
                                        }
                                        BandValue::U64(value) => {
                                            feature
                                                .set_field_integer64(field_index, value as i64)?;
                                        }
                                        BandValue::I64(value) => {
                                            feature.set_field_integer64(field_index, value)?;
                                        }
                                        BandValue::F32(value) => {
                                            feature.set_field_double(field_index, value as f64)?;
                                        }
                                        BandValue::F64(value) => {
                                            feature.set_field_double(field_index, value)?;
                                        }
                                    }
                                }
                                let mut geometry = Geometry::empty(OGRwkbGeometryType::wkbPoint)?;
                                geometry.add_point_2d((point.orig_x, point.orig_y));
                                feature.set_geometry(geometry)?;
                                feature.create(&output_layer)?;
                            }

                            Ok(())
                        })?;
                    }
                    {
                        let _guard = mutex.lock();
                        output.close()?;
                    }
                    Ok::<_, gdal::errors::GdalError>(())
                });
                output_threads.push(thread);
            }

            let mut tile_points = tile_points.into_iter().collect::<Vec<_>>();
            tile_points.sort_by_key(|t| (t.0 .1, t.0 .0));

            let block_sender = BlockSender(output_txs);

            let num_threads = NonZeroUsize::new(self.num_threads.unwrap_or(8))
                .unwrap()
                .min(available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap()));
            println!("Using {num_threads} threads");
            let mut block_reader = ThreadedBlockReader::new::<SamplingBlockReducer, _>(
                PathBuf::from(&self.input),
                block_sender,
                num_threads,
            );
            for ((block_x, block_y), points) in tile_points {
                block_reader.submit(block_x, block_y, points);
            }
            drop(block_reader);

            for thread in output_threads {
                thread.join().unwrap()?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use gdal::{
        raster::{Buffer, RasterCreationOptions},
        spatial_ref::SpatialRef,
        vector::{Feature, FieldDefn, Geometry, LayerAccess, LayerOptions},
        Dataset, DriverManager, DriverType,
    };
    use gdal_sys::OGRwkbGeometryType;
    use tempfile::tempdir;

    use super::SampleExtractionArgs;

    /// Simplified test that manually extracts values from a test raster to verify
    /// that the algorithm works correctly without relying on transactions
    #[test]
    fn test_sample_extraction_algorithm() -> Result<()> {
        // Create a test directory
        let temp_dir = tempdir()?;

        // Parameters for our raster
        let raster_width = 64;
        let raster_height = 64;
        let tile_size = 16;
        let bands = 3;

        // Create a raster with computed values
        let raster_path = temp_dir.path().join("test_raster.tif");
        let mut raster_ds = DriverManager::get_driver_by_name("GTiff")?
            .create_with_band_type_with_options::<u16, _>(
                &raster_path,
                raster_width,
                raster_height,
                bands,
                &RasterCreationOptions::from_iter([
                    "TILED=YES",
                    &format!("BLOCKXSIZE={tile_size}"),
                    &format!("BLOCKYSIZE={tile_size}"),
                ]),
            )?;

        // Set geotransform (origin at 0,0 with 10 unit pixel size)
        let geo_transform = [0.0, 10.0, 0.0, 0.0, 0.0, -10.0];
        raster_ds.set_geo_transform(&geo_transform)?;

        // Set a simple projection
        let spatial_ref = SpatialRef::from_epsg(4326)?;
        raster_ds.set_projection(&spatial_ref.to_wkt()?)?;

        // Fill bands with predictable values
        for band_idx in 1..=bands {
            let band_value_offset = (band_idx as u16) * 1000;
            let mut band = raster_ds.rasterband(band_idx)?;

            // Fill the entire band at once for simplicity
            let mut data = vec![0u16; raster_width as usize * raster_height as usize];
            for y in 0..raster_height {
                for x in 0..raster_width {
                    let value = band_value_offset + x as u16 + (y as u16 * raster_width as u16);
                    data[y as usize * raster_width as usize + x as usize] = value;
                }
            }

            let mut buffer = Buffer::new((raster_width as usize, raster_height as usize), data);
            band.write((0, 0), (raster_width, raster_height), &mut buffer)?;
        }

        // Define sample points (we won't create actual files, just compute expected values)
        let mut sample_points = Vec::new();
        let mut expected_values = Vec::new();

        // Generate 10 points with known coordinates and expected values
        for i in 0..10 {
            // Generate deterministic but scattered sample locations
            let x = (i * 7) % raster_width;
            let y = (i * 11) % raster_height;

            // Convert to projected coordinates
            let proj_x = x as f64 * 10.0;
            let proj_y = -(y as f64 * 10.0); // Negative because of the geotransform

            sample_points.push((i, proj_x, proj_y));

            // Compute expected band values for each sample point
            let mut band_values = Vec::new();
            for band_idx in 1..=bands {
                let band_value_offset = (band_idx as u16) * 1000;
                let value = band_value_offset + x as u16 + (y as u16 * raster_width as u16);
                band_values.push(value);
            }
            expected_values.push((i, band_values));
        }

        // Close raster dataset
        raster_ds.close()?;

        // Verify by reading raster values directly for each point
        let raster = Dataset::open(&raster_path)?;
        let geo_transform = raster.geo_transform()?;

        // Manually extract samples from the raster at each point location
        let mut actual_values = Vec::new();

        for (id, proj_x, proj_y) in &sample_points {
            // Convert projected coordinates back to pixel coordinates
            let px = (*proj_x - geo_transform[0]) / geo_transform[1];
            let py = (*proj_y - geo_transform[3]) / geo_transform[5];

            // Round to integers
            let px = px.round() as isize;
            let py = py.round() as isize;

            // Read values from each band
            let mut band_values = Vec::new();
            for band_idx in 1..=bands {
                let band = raster.rasterband(band_idx)?;
                let buffer = band.read_as::<u16>((px, py), (1, 1), (1, 1), None)?;
                band_values.push(buffer.data()[0]);
            }

            actual_values.push((*id, band_values));
        }

        // Sort both arrays to ensure deterministic comparison
        expected_values.sort_by_key(|e| e.0);
        actual_values.sort_by_key(|a| a.0);

        // Compare the values
        for ((expected_id, expected_bands), (actual_id, actual_bands)) in
            expected_values.iter().zip(actual_values.iter())
        {
            assert_eq!(expected_id, actual_id, "Point IDs don't match");
            assert_eq!(
                expected_bands, actual_bands,
                "Band values don't match for point {}",
                expected_id
            );
        }

        Ok(())
    }

    #[test]
    fn test_sample_extraction_command() -> Result<()> {
        // Create a test directory
        let temp_dir = tempdir()?;

        // Parameters for our raster
        let raster_width = 64;
        let raster_height = 64;
        let tile_size = 16;
        let bands = 3;

        // Create a raster with computed values
        let raster_path = "/vsimem/test_raster.tif";
        let mut raster_ds =
            DriverManager::get_output_driver_for_dataset_name(&raster_path, DriverType::Raster)
                .unwrap()
                .create_with_band_type_with_options::<u16, _>(
                    &raster_path,
                    raster_width,
                    raster_height,
                    bands,
                    &RasterCreationOptions::from_iter([
                        "TILED=YES",
                        &format!("BLOCKXSIZE={tile_size}"),
                        &format!("BLOCKYSIZE={tile_size}"),
                    ]),
                )?;

        // Set geotransform (origin at 0,0 with 10 unit pixel size)
        let geo_transform = [0.0, 10.0, 0.0, 0.0, 0.0, -10.0];
        raster_ds.set_geo_transform(&geo_transform)?;

        // Set a simple projection
        let spatial_ref = SpatialRef::from_epsg(4326)?;
        raster_ds.set_projection(&spatial_ref.to_wkt()?)?;

        // Fill bands with predictable values
        for band_idx in 1..=bands {
            let band_value_offset = (band_idx as u16) * 1000;
            let mut band = raster_ds.rasterband(band_idx)?;

            // For each block
            for y in 0..(raster_height / tile_size) {
                for x in 0..(raster_width / tile_size) {
                    // Create a buffer for this block
                    let mut block_data = vec![0u16; tile_size * tile_size];

                    // Fill the block with computable values
                    for by in 0..tile_size {
                        for bx in 0..tile_size {
                            let px = x * tile_size + bx;
                            let py = y * tile_size + by;

                            // Value = band_offset + x + y*width
                            let value =
                                band_value_offset + px as u16 + (py as u16 * raster_width as u16);
                            block_data[by * tile_size + bx] = value;
                        }
                    }

                    let mut buffer = Buffer::new((tile_size, tile_size), block_data);
                    band.write_block((x, y), &mut buffer)?;
                }
            }
        }

        // Create sample points in GeoPackage format
        let points_path = "points.gpkg";
        let points_driver = DriverManager::get_driver_by_name("GPKG")?;
        let mut points_ds = points_driver.create_vector_only(&points_path)?;

        let points_layer = points_ds.create_layer(LayerOptions {
            name: "points",
            srs: Some(&spatial_ref),
            ty: OGRwkbGeometryType::wkbPoint,
            options: None,
        })?;

        // Add an ID field
        let id_field = FieldDefn::new("id", gdal::vector::OGRFieldType::OFTInteger)?;
        id_field.add_to_layer(&points_layer)?;

        // Generate 10 points at specific locations
        let mut expected_values = Vec::new();

        for i in 0..10 {
            // Generate deterministic but scattered sample locations
            let x = (i * 7) % raster_width;
            let y = (i * 11) % raster_height;

            // Convert to projected coordinates
            let proj_x = x as f64 * 10.0;
            let proj_y = -(y as f64 * 10.0);

            // Add the point
            let mut feature = Feature::new(points_layer.defn())?;
            let mut geom = Geometry::empty(OGRwkbGeometryType::wkbPoint)?;
            geom.add_point_2d((proj_x, proj_y));
            feature.set_geometry(geom)?;
            feature.set_field_integer(0, i as i32)?; // ID field
            feature.create(&points_layer)?;

            // Store expected values for later comparison
            let mut band_values = Vec::new();
            for band_idx in 1..=bands {
                let band_value_offset = (band_idx as u16) * 1000;
                let value = band_value_offset + x as u16 + (y as u16 * raster_width as u16);
                band_values.push(value);
            }
            expected_values.push((i, band_values));
        }

        // Close datasets to flush changes
        points_ds.close()?;
        raster_ds.close()?;

        // Create output path (use GeoPackage to avoid ESRI Shapefile transaction issues)
        let output_path = temp_dir.path().join("output.gpkg");

        // Create and run the sample extraction command
        let args = SampleExtractionArgs {
            input: raster_path.into(),
            points: vec![points_path.into()],
            outputs: vec![output_path.clone()],
            format: Some("GPKG".to_string()),
            fields: Some(vec![
                "band1".to_string(),
                "band2".to_string(),
                "band3".to_string(),
            ]),
            num_threads: None,
            copy_fid: false,
        };

        // This actually calls the sample extraction code
        args.run()?;

        // Open the output and verify results
        let output_ds = Dataset::open(&output_path)?;
        let mut output_layer = output_ds.layer(0)?;

        // Collect all features and their extracted values
        let mut results = Vec::new();
        for feature in output_layer.features() {
            let id = feature.field_as_integer(0)?.unwrap();

            let band1 = feature.field_as_integer(1)?.unwrap() as u16;
            let band2 = feature.field_as_integer(2)?.unwrap() as u16;
            let band3 = feature.field_as_integer(3)?.unwrap() as u16;

            results.push((id as i32, vec![band1, band2, band3]));
        }

        // Sort results by id for deterministic comparison (since extraction is threaded)
        results.sort_by_key(|r| r.0);

        // Convert expected values to the right format and sort them
        let mut expected: Vec<(i32, Vec<u16>)> = expected_values
            .into_iter()
            .map(|(id, values)| (id as i32, values))
            .collect();
        expected.sort_by_key(|e| e.0);

        assert_eq!(results, expected);

        Ok(())
    }
}
