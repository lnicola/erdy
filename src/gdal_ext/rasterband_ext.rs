use gdal::raster::{GdalDataType, RasterBand};
use ndarray::Array2;

#[derive(Debug)]
pub enum TypedBlock {
    U8(Array2<u8>),
    U16(Array2<u16>),
    I16(Array2<i16>),
    F32(Array2<f32>),
}

pub trait RasterBandExt {
    fn read_typed_block(&self, x: usize, y: usize) -> gdal::errors::Result<TypedBlock>;
}

impl<'d> RasterBandExt for RasterBand<'d> {
    fn read_typed_block(&self, x: usize, y: usize) -> gdal::errors::Result<TypedBlock> {
        match self.band_type() {
            GdalDataType::UInt8 => {
                let buf = self.read_block::<u8>((x, y))?;
                Ok(TypedBlock::U8(buf))
            }
            GdalDataType::UInt16 => {
                let buf = self.read_block::<u16>((x, y))?;
                Ok(TypedBlock::U16(buf))
            }
            GdalDataType::Int16 => {
                let buf = self.read_block::<i16>((x, y))?;
                Ok(TypedBlock::I16(buf))
            }
            GdalDataType::Float32 => {
                let buf = self.read_block::<f32>((x, y))?;
                Ok(TypedBlock::F32(buf))
            }
            _ => unimplemented!(),
        }
    }
}
