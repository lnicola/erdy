use gdal::raster::{GdalDataType, RasterBand};
use ndarray::Array2;

#[derive(Debug)]
pub enum TypedBlock {
    U8(Array2<u8>),
    U16(Array2<u16>),
    I16(Array2<i16>),
    F32(Array2<f32>),
}

impl TypedBlock {
    /// Returns `true` if the typed block is [`U8`].
    ///
    /// [`U8`]: TypedBlock::U8
    #[must_use]
    pub fn is_u8(&self) -> bool {
        matches!(self, Self::U8(..))
    }

    /// Returns `true` if the typed block is [`U16`].
    ///
    /// [`U16`]: TypedBlock::U16
    #[must_use]
    pub fn is_u16(&self) -> bool {
        matches!(self, Self::U16(..))
    }

    /// Returns `true` if the typed block is [`I16`].
    ///
    /// [`I16`]: TypedBlock::I16
    #[must_use]
    pub fn is_i16(&self) -> bool {
        matches!(self, Self::I16(..))
    }

    /// Returns `true` if the typed block is [`F32`].
    ///
    /// [`F32`]: TypedBlock::F32
    #[must_use]
    pub fn is_f32(&self) -> bool {
        matches!(self, Self::F32(..))
    }

    pub fn as_u8(&self) -> Option<&Array2<u8>> {
        if let Self::U8(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_u16(&self) -> Option<&Array2<u16>> {
        if let Self::U16(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_i16(&self) -> Option<&Array2<i16>> {
        if let Self::I16(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_f32(&self) -> Option<&Array2<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn try_into_u8(self) -> Result<Array2<u8>, Self> {
        if let Self::U8(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_u16(self) -> Result<Array2<u16>, Self> {
        if let Self::U16(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_i16(self) -> Result<Array2<i16>, Self> {
        if let Self::I16(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_f32(self) -> Result<Array2<f32>, Self> {
        if let Self::F32(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }
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
