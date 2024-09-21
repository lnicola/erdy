use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use gdal::Dataset;
use quick_xml::{events::BytesText, Writer};

#[derive(Debug, Parser)]
pub struct BuildVrtArgs {
    /// Input path
    #[arg(long)]
    input: PathBuf,

    /// Output path
    #[arg(long)]
    output: PathBuf,

    /// Mode
    #[arg(long)]
    mode: Mode,

    /// Resampler
    #[arg(long)]
    resampler: Option<String>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode {
    MsiL1C,
}

impl BuildVrtArgs {
    pub fn run(&self) -> Result<()> {
        match self.mode {
            Mode::MsiL1C => self.run_msi_l1c(),
        }
    }

    fn run_msi_l1c(&self) -> Result<()> {
        let granule_dir = self.input.join("GRANULE");
        let entry = fs::read_dir(granule_dir)?
            .next()
            .expect("missing granule")?;
        let mut granule = entry.path();
        let granule_name = self.input.file_name().expect("missing product name");
        let granule_name = granule_name.to_str().expect("invalid granule name");
        let prefix = String::from(&granule_name[38..45]) + &granule_name[11..27];
        granule.push("IMG_DATA");
        granule = granule.canonicalize()?;

        let b02 = granule.join(prefix.clone() + "B02.jp2");

        let ds = Dataset::open(&b02)?;
        let wkt = ds.spatial_ref()?.to_wkt()?;
        let geo_transform = ds.geo_transform()?;

        let file = File::create(&self.output)?;
        let writer = BufWriter::new(file);
        let mut writer = Writer::new_with_indent(writer, b' ', 2);

        struct VrtBandWriter<'a, W> {
            writer: &'a mut Writer<W>,
            resampler: &'a Option<String>,
            index: usize,
        }

        impl<'a, W> VrtBandWriter<'a, W> {
            fn new(writer: &'a mut Writer<W>, resampler: &'a Option<String>) -> Self {
                Self {
                    writer,
                    resampler,
                    index: 1,
                }
            }
        }

        impl<'a, W: Write> VrtBandWriter<'a, W> {
            fn write_band(&mut self, path: &Path, resolution: usize) -> quick_xml::Result<()> {
                let source_size = (10980 * 10 / resolution).to_string();
                self.writer
                    .create_element("VRTRasterBand")
                    .with_attributes([("dataType", "UInt16"), ("band", &self.index.to_string())])
                    .write_inner_content::<_, quick_xml::Error>(|writer| {
                        let mut element_writer = writer.create_element("SimpleSource");
                        if resolution != 10 {
                            if let Some(resampler) = self.resampler {
                                element_writer = element_writer
                                    .with_attribute(("resampling", resampler.as_str()));
                            }
                        }
                        element_writer.write_inner_content::<_, quick_xml::Error>(|writer| {
                            writer
                                .create_element("SourceFilename")
                                .with_attribute(("relativeToVRT", "0"))
                                .write_text_content(BytesText::new(
                                    path.to_str().expect("invalid path"),
                                ))?;
                            writer
                                .create_element("SourceProperties")
                                .with_attributes([
                                    ("RasterXSize", source_size.as_str()),
                                    ("RasterYSize", &source_size),
                                    ("DataType", "UInt16"),
                                    // ("BlockXSize", "1024"),
                                    // ("BlockYSize", "1024"),
                                ])
                                .write_empty()?;
                            writer
                                .create_element("SrcRect")
                                .with_attributes([
                                    ("xOff", "0"),
                                    ("yOff", "0"),
                                    ("xSize", &source_size),
                                    ("ySize", &source_size),
                                ])
                                .write_empty()?;
                            writer
                                .create_element("DstRect")
                                .with_attributes([
                                    ("xOff", "0"),
                                    ("yOff", "0"),
                                    ("xSize", "10980"),
                                    ("ySize", "10980"),
                                ])
                                .write_empty()?;
                            Ok(())
                        })?;
                        Ok(())
                    })?;

                self.index += 1;
                Ok(())
            }
        }

        writer
            .create_element("VRTDataset")
            .with_attributes([("rasterXSize", "10980"), ("rasterYSize", "10980")])
            .write_inner_content::<_, quick_xml::Error>(|writer| {
                writer
                    .create_element("SRS")
                    .with_attribute(("dataAxisToSRSAxisMapping", "1,2"))
                    .write_text_content(BytesText::from_escaped(&wkt))?
                    .create_element("GeoTransform")
                    .write_text_content(BytesText::new(&format!(
                        "{}, {}, {}, {}, {}, {}",
                        geo_transform[0],
                        geo_transform[1],
                        geo_transform[2],
                        geo_transform[3],
                        geo_transform[4],
                        geo_transform[5]
                    )))?;

                static BANDS: [(&str, usize); 13] = [
                    ("B01", 60),
                    ("B02", 10),
                    ("B03", 10),
                    ("B04", 10),
                    ("B05", 20),
                    ("B06", 20),
                    ("B07", 20),
                    ("B08", 10),
                    ("B8A", 20),
                    ("B09", 60),
                    ("B10", 60),
                    ("B11", 20),
                    ("B12", 20),
                ];
                let mut band_writer = VrtBandWriter::new(writer, &self.resampler);
                for (name, resolution) in BANDS {
                    let path = granule.join(format!("{prefix}{name}.jp2"));
                    band_writer.write_band(&path, resolution)?;
                }

                Ok(())
            })?;
        Ok(())
    }
}
