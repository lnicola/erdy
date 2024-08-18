#![allow(dead_code)]

use gdal::{
    errors,
    raster::{Buffer, GdalDataType, RasterBand},
};

#[derive(Debug)]
pub enum TypedBuffer {
    U8(Buffer<u8>),
    I8(Buffer<i8>),
    U16(Buffer<u16>),
    I16(Buffer<i16>),
    U32(Buffer<u32>),
    I32(Buffer<i32>),
    F32(Buffer<f32>),
}

impl TypedBuffer {
    /// Returns `true` if the typed buffer is [`U8`].
    ///
    /// [`U8`]: TypedBuffer::U8
    #[must_use]
    pub fn is_u8(&self) -> bool {
        matches!(self, Self::U8(..))
    }

    /// Returns `true` if the typed buffer is [`I8`].
    ///
    /// [`I8`]: TypedBuffer::I8
    #[must_use]
    pub fn is_i8(&self) -> bool {
        matches!(self, Self::I8(..))
    }

    /// Returns `true` if the typed buffer is [`U16`].
    ///
    /// [`U16`]: TypedBuffer::U16
    #[must_use]
    pub fn is_u16(&self) -> bool {
        matches!(self, Self::U16(..))
    }

    /// Returns `true` if the typed buffer is [`I16`].
    ///
    /// [`I16`]: TypedBuffer::I16
    #[must_use]
    pub fn is_i16(&self) -> bool {
        matches!(self, Self::I16(..))
    }

    /// Returns `true` if the typed buffer is [`U32`].
    ///
    /// [`U32`]: TypedBuffer::U32
    #[must_use]
    pub fn is_u32(&self) -> bool {
        matches!(self, Self::U32(..))
    }

    /// Returns `true` if the typed buffer is [`I32`].
    ///
    /// [`I32`]: TypedBuffer::I32
    #[must_use]
    pub fn is_i32(&self) -> bool {
        matches!(self, Self::I32(..))
    }

    /// Returns `true` if the typed buffer is [`F32`].
    ///
    /// [`F32`]: TypedBuffer::F32
    #[must_use]
    pub fn is_f32(&self) -> bool {
        matches!(self, Self::F32(..))
    }

    pub fn as_u8(&self) -> Option<&Buffer<u8>> {
        if let Self::U8(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_i8(&self) -> Option<&Buffer<i8>> {
        if let Self::I8(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_u16(&self) -> Option<&Buffer<u16>> {
        if let Self::U16(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_i16(&self) -> Option<&Buffer<i16>> {
        if let Self::I16(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_u32(&self) -> Option<&Buffer<u32>> {
        if let Self::U32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_i32(&self) -> Option<&Buffer<i32>> {
        if let Self::I32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_f32(&self) -> Option<&Buffer<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn try_into_u8(self) -> Result<Buffer<u8>, Self> {
        if let Self::U8(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_i8(self) -> Result<Buffer<i8>, Self> {
        if let Self::I8(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_u16(self) -> Result<Buffer<u16>, Self> {
        if let Self::U16(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_i16(self) -> Result<Buffer<i16>, Self> {
        if let Self::I16(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_u32(self) -> Result<Buffer<u32>, Self> {
        if let Self::U32(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_i32(self) -> Result<Buffer<i32>, Self> {
        if let Self::I32(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn try_into_f32(self) -> Result<Buffer<f32>, Self> {
        if let Self::F32(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            TypedBuffer::U8(buf) => buf.shape(),
            TypedBuffer::I8(buf) => buf.shape(),
            TypedBuffer::U16(buf) => buf.shape(),
            TypedBuffer::I16(buf) => buf.shape(),
            TypedBuffer::U32(buf) => buf.shape(),
            TypedBuffer::I32(buf) => buf.shape(),
            TypedBuffer::F32(buf) => buf.shape(),
        }
    }
}

pub trait RasterBandExt {
    fn read_typed_block(&self, x: usize, y: usize) -> errors::Result<TypedBuffer>;
}

impl<'d> RasterBandExt for RasterBand<'d> {
    fn read_typed_block(&self, x: usize, y: usize) -> errors::Result<TypedBuffer> {
        match self.band_type() {
            GdalDataType::Unknown => unimplemented!(),
            GdalDataType::UInt8 => {
                let buf = self.read_block((x, y))?;
                Ok(TypedBuffer::U8(buf))
            }
            GdalDataType::Int8 => {
                let buf = self.read_block((x, y))?;
                Ok(TypedBuffer::I8(buf))
            }
            GdalDataType::UInt16 => {
                let buf = self.read_block((x, y))?;
                Ok(TypedBuffer::U16(buf))
            }
            GdalDataType::Int16 => {
                let buf = self.read_block((x, y))?;
                Ok(TypedBuffer::I16(buf))
            }
            GdalDataType::UInt32 => {
                let buf = self.read_block((x, y))?;
                Ok(TypedBuffer::U32(buf))
            }
            GdalDataType::Int32 => {
                let buf = self.read_block((x, y))?;
                Ok(TypedBuffer::I32(buf))
            }
            GdalDataType::Float32 => {
                let buf = self.read_block((x, y))?;
                Ok(TypedBuffer::F32(buf))
            }
            GdalDataType::UInt64 => todo!(),
            GdalDataType::Int64 => todo!(),
            GdalDataType::Float64 => todo!(),
        }
    }
}
