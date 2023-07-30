mod feature_ext;

use std::{collections::HashMap, sync::mpsc, thread};

use anyhow::Result;
use gdal::{
    raster::{GdalDataType, RasterBand},
    spatial_ref::SpatialRef,
    vector::{
        Feature, FieldDefn, FieldValue, Geometry, LayerAccess, OGRFieldType, OGRwkbGeometryType,
    },
    Dataset, DriverManager, GeoTransformEx, LayerOptions, Metadata,
};
use ndarray::Array2;

use crate::feature_ext::FeatureExt;

#[derive(Clone, Copy)]
enum BandValue {
    U16(u16),
    I16(i16),
    F32(f32),
}

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

#[derive(Debug)]
enum TypedBlock {
    U16(Array2<u16>),
    I16(Array2<i16>),
    F32(Array2<f32>),
}

fn read_typed_block<'d>(
    band: &RasterBand<'d>,
    x: usize,
    y: usize,
) -> gdal::errors::Result<TypedBlock> {
    match band.band_type() {
        GdalDataType::UInt16 => {
            let buf = band.read_block::<u16>((x, y))?;
            Ok(TypedBlock::U16(buf))
        }
        GdalDataType::Int16 => {
            let buf = band.read_block::<i16>((x, y))?;
            Ok(TypedBlock::I16(buf))
        }
        GdalDataType::Float32 => {
            let buf = band.read_block::<f32>((x, y))?;
            Ok(TypedBlock::F32(buf))
        }
        _ => unimplemented!(),
    }
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let image = Dataset::open(&args[1])?;
    let geo_transform = image.geo_transform()?;
    let geo_transform = geo_transform.invert()?;
    let block_size = image.rasterband(1)?.block_size();
    dbg!(block_size);

    let band_count = image.raster_count() as usize;
    for band_idx in 0..band_count {
        let band = image.rasterband(band_idx as isize + 1)?;
        assert!(band.block_size() == block_size);
    }
    let mut tile_points = HashMap::<_, Vec<_>>::new();
    let point_ds = Dataset::open(&args[2])?;
    let mut layer = point_ds.layer(0)?;
    dbg!(layer.feature_count());
    for feature in layer.features() {
        let (orig_x, orig_y, _) = feature.geometry().as_ref().unwrap().get_point(0);
        let (x, y) = geo_transform.apply(orig_x, orig_y);
        let (block_x, block_y) = (
            (x / block_size.0 as f64) as usize,
            (y / block_size.1 as f64) as usize,
        );
        let (x, y) = (x as usize, y as usize);
        tile_points.entry((block_x, block_y)).or_default().push((
            x % block_size.0,
            y % block_size.1,
            orig_x,
            orig_y,
            feature.fields().map(|f| f.1).collect::<Vec<_>>(),
        ));
    }
    let spatial_ref = layer.spatial_ref().map(|sr| sr.to_wkt()).transpose()?;
    let mut output_fields = Vec::new();
    for field in layer.defn().fields() {
        let field_definition = FieldDefinition {
            name: field.name(),
            field_type: field.field_type(),
            width: Some(field.width()),
            precision: Some(field.precision()),
        };
        output_fields.push(field_definition);
    }
    for band_idx in 1..=band_count {
        let band = image.rasterband(band_idx as isize)?;
        let name = match band.description() {
            Ok(name) if !name.is_empty() => name,
            _ => format!("band_{band_idx}"),
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
        output_fields.push(field_definition);
    }

    let (tx, rx) =
        mpsc::sync_channel::<(Vec<BandValue>, Vec<(f64, f64, Vec<Option<FieldValue>>)>)>(128);
    let output_thread = thread::spawn(move || {
        let mut output =
            DriverManager::get_driver_by_name("GPKG")?.create_vector_only("output.gpkg")?;
        let output_layer = output.create_layer(LayerOptions {
            name: "output",
            srs: spatial_ref
                .map(|wkt| SpatialRef::from_wkt(&wkt))
                .transpose()?
                .as_ref(),
            ty: OGRwkbGeometryType::wkbPoint,
            options: None,
        })?;
        for field in output_fields {
            let field_defn = field.to_field_defn()?;
            field_defn.add_to_layer(&output_layer)?;
        }
        for (sample_values, points) in rx {
            let tx = output.start_transaction()?;
            let output_layer = tx.layer(0)?;
            for (idx, (x, y, fields)) in points.into_iter().enumerate() {
                let mut feature = Feature::new(output_layer.defn())?;
                let field_offset = fields.len();
                for (field_idx, field_value) in fields.into_iter().enumerate() {
                    if let Some(field_value) = field_value {
                        feature.set_field_by_index(field_idx, &field_value);
                    }
                }
                for band_idx in 0..band_count {
                    match sample_values[band_count * idx + band_idx] {
                        BandValue::U16(value) => {
                            feature
                                .set_field_integer_by_index(band_idx + field_offset, value as i32);
                        }
                        BandValue::I16(value) => {
                            feature
                                .set_field_integer_by_index(band_idx + field_offset, value as i32);
                        }
                        BandValue::F32(value) => {
                            feature
                                .set_field_double_by_index(band_idx + field_offset, value as f64);
                        }
                    }
                }
                let mut geometry = Geometry::empty(OGRwkbGeometryType::wkbPoint)?;
                geometry.add_point_2d((x, y));
                feature.set_geometry(geometry)?;
                feature.create(&output_layer)?;
            }
            tx.commit()?;
        }
        Ok::<_, gdal::errors::GdalError>(())
    });

    let mut tile_points = tile_points.into_iter().collect::<Vec<_>>();
    tile_points.sort_by_key(|t| (t.0 .1, t.0 .0));

    for ((block_x, block_y), points) in tile_points {
        println!("{block_x} {block_y}");
        let mut sample_values = vec![BandValue::I16(0); points.len() * band_count];
        for band_idx in 0..band_count {
            let band = image.rasterband(band_idx as isize + 1)?;
            let block = read_typed_block(&band, block_x, block_y)?;
            match block {
                TypedBlock::U16(buf) => {
                    for (idx, &(bx, by, ..)) in points.iter().enumerate() {
                        let pix = buf[(by, bx)];
                        sample_values[band_count * idx + band_idx] = BandValue::U16(pix);
                    }
                }
                TypedBlock::I16(buf) => {
                    for (idx, &(bx, by, ..)) in points.iter().enumerate() {
                        let pix = buf[(by, bx)];
                        sample_values[band_count * idx + band_idx] = BandValue::I16(pix);
                    }
                }
                TypedBlock::F32(buf) => {
                    for (idx, &(bx, by, ..)) in points.iter().enumerate() {
                        let pix = buf[(by, bx)];
                        sample_values[band_count * idx + band_idx] = BandValue::F32(pix);
                    }
                }
            }
        }
        let field_values = points
            .into_iter()
            .map(|p| (p.2, p.3, p.4))
            .collect::<Vec<_>>();
        tx.send((sample_values, field_values))?;
    }

    drop(tx);
    output_thread.join().unwrap()?;

    Ok(())
}
