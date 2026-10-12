//! 8- and 16-bit PNG frames.
//!
//! The synthesis search works on 8-bit images. Output can still be written as 16-bit PNG, so
//! that a pipeline keeps full precision downstream; with 16-bit paintings the video pipeline
//! votes its final result from them, so the extra bits carry real information. Writes are
//! atomic: a frame is written to a hidden temporary file and renamed into place, so a reader
//! (or an interrupted render) never sees half a file.

use std::fs;
use std::path::{Path, PathBuf};

use freesynth_core::Image;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ExtendedColorType, ImageEncoder};
use serde::{Deserialize, Serialize};

use crate::error::{MediaError, io_error};

/// Bits per channel of written frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitDepth {
    #[default]
    Eight,
    Sixteen,
}

impl BitDepth {
    /// 8 or 16.
    pub fn bits(self) -> u32 {
        match self {
            BitDepth::Eight => 8,
            BitDepth::Sixteen => 16,
        }
    }

    /// The depth for 8 or 16 bits.
    pub fn from_bits(bits: u32) -> Result<Self, MediaError> {
        match bits {
            8 => Ok(BitDepth::Eight),
            16 => Ok(BitDepth::Sixteen),
            _ => Err(MediaError::Invalid(format!("unsupported bit depth {bits} (expected 8 or 16)"))),
        }
    }
}

/// `image` as RGBA bytes (grey is replicated, missing alpha is opaque).
pub fn to_rgba8(image: &Image) -> Vec<u8> {
    let c = image.channels();
    if c == 4 {
        return image.data().to_vec();
    }
    image
        .data()
        .chunks_exact(c)
        .flat_map(|p| match c {
            1 => [p[0], p[0], p[0], 255],
            2 => [p[0], p[0], p[0], p[1]],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        })
        .collect()
}

/// A hidden temporary name next to `path`, unique to this process and thread.
fn temp_path(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let thread = format!("{:?}", std::thread::current().id());
    let thread: String = thread.chars().filter(char::is_ascii_digit).collect();
    path.with_file_name(format!(".{name}.{}-{thread}.tmp", std::process::id()))
}

/// Encodes PNG bytes for 1-4 channel samples.
fn encode_png(
    path: &Path,
    width: usize,
    height: usize,
    channels: usize,
    depth: BitDepth,
    bytes: &[u8],
) -> Result<Vec<u8>, MediaError> {
    let color = match (channels, depth) {
        (1, BitDepth::Eight) => ExtendedColorType::L8,
        (2, BitDepth::Eight) => ExtendedColorType::La8,
        (3, BitDepth::Eight) => ExtendedColorType::Rgb8,
        (4, BitDepth::Eight) => ExtendedColorType::Rgba8,
        (1, BitDepth::Sixteen) => ExtendedColorType::L16,
        (2, BitDepth::Sixteen) => ExtendedColorType::La16,
        (3, BitDepth::Sixteen) => ExtendedColorType::Rgb16,
        (4, BitDepth::Sixteen) => ExtendedColorType::Rgba16,
        _ => return Err(MediaError::Invalid(format!("cannot write a {channels}-channel PNG"))),
    };
    let mut out = Vec::with_capacity(bytes.len() / 2);
    PngEncoder::new_with_quality(&mut out, CompressionType::Fast, FilterType::Adaptive)
        .write_image(bytes, width as u32, height as u32, color)
        .map_err(|e| MediaError::Image { path: path.to_owned(), message: e.to_string() })?;
    Ok(out)
}

/// Writes `bytes` to `path` through a temporary file and a rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), MediaError> {
    let tmp = temp_path(path);
    fs::write(&tmp, bytes).map_err(io_error(&tmp))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        MediaError::Io { path: path.to_owned(), source: e }
    })
}

/// Saves an 8-bit image as a PNG of the given depth (16-bit values are `v * 257`, the exact
/// widening), with fast compression, atomically. The channel count is kept.
pub fn save_png(image: &Image, path: &Path, depth: BitDepth) -> Result<(), MediaError> {
    let bytes = match depth {
        BitDepth::Eight => encode_png(path, image.width(), image.height(), image.channels(), depth, image.data())?,
        BitDepth::Sixteen => {
            // PNG stores 16-bit samples big-endian; the encoder takes native-endian bytes.
            let wide: Vec<u8> = image.data().iter().flat_map(|&v| (u16::from(v) * 257).to_ne_bytes()).collect();
            encode_png(path, image.width(), image.height(), image.channels(), depth, &wide)?
        }
    };
    write_atomic(path, &bytes)
}

/// Saves 16-bit samples (`channels` per pixel, 1-4) as a 16-bit PNG, atomically.
pub fn save_png16(width: usize, height: usize, channels: usize, data: &[u16], path: &Path) -> Result<(), MediaError> {
    if data.len() != width * height * channels {
        return Err(MediaError::Invalid("16-bit image data does not match its size".into()));
    }
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let png = encode_png(path, width, height, channels, BitDepth::Sixteen, &bytes)?;
    write_atomic(path, &png)
}

/// Saves `image` by the extension of `path`: PNG at `depth` (fast compression), other formats
/// (JPEG, BMP, TGA) at 8 bits through the engine's writer. Atomic for PNG.
pub fn save_frame(image: &Image, path: &Path, depth: BitDepth) -> Result<(), MediaError> {
    let is_png = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("png"));
    if is_png {
        return save_png(image, path, depth);
    }
    if depth == BitDepth::Sixteen {
        return Err(MediaError::Invalid(format!("16-bit output needs PNG files, not '{}'", path.display())));
    }
    let tmp = temp_path(path).with_extension(path.extension().unwrap_or_default());
    freesynth_core::io::save(image, &tmp)
        .map_err(|e| MediaError::Image { path: path.to_owned(), message: e.to_string() })?;
    fs::rename(&tmp, path).map_err(io_error(path))
}

/// Loads an image file as 16-bit RGBA (8-bit files are widened exactly). Returns width,
/// height and the samples.
pub fn load_rgba16(path: &Path) -> Result<(usize, usize, Vec<u16>), MediaError> {
    let img = image::ImageReader::open(path)
        .map_err(io_error(path))?
        .with_guessed_format()
        .map_err(io_error(path))?
        .decode()
        .map_err(|e| MediaError::Image { path: path.to_owned(), message: e.to_string() })?;
    let rgba = img.into_rgba16();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    Ok((w, h, rgba.into_raw()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/tmp")
            .join(format!("freesynth-media-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn png_round_trips_at_both_depths() {
        let dir = scratch("png");
        let rgb = Image::from_raw(3, 2, 3, (0..18).map(|v| (v * 14) as u8).collect());
        let p8 = dir.join("a.png");
        save_png(&rgb, &p8, BitDepth::Eight).unwrap();
        assert_eq!(freesynth_core::io::load_rgba(&p8).unwrap().data(), to_rgba8(&rgb));
        let p16 = dir.join("b.png");
        save_png(&rgb, &p16, BitDepth::Sixteen).unwrap();
        let (w, h, data) = load_rgba16(&p16).unwrap();
        assert_eq!((w, h), (3, 2));
        let expected: Vec<u16> = to_rgba8(&rgb).iter().map(|&v| u16::from(v) * 257).collect();
        assert_eq!(data, expected);
        let raw = image::open(&p16).unwrap();
        assert_eq!(raw.color(), image::ColorType::Rgb16, "16-bit RGB on disk");
        // No temporary files are left behind.
        let names: Vec<_> = fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(save_frame(&rgb, &dir.join("c.jpg"), BitDepth::Sixteen).is_err());
        save_frame(&rgb, &dir.join("c.jpg"), BitDepth::Eight).unwrap();
        save_png16(1, 1, 4, &[1, 2, 3, 65535], &dir.join("d.png")).unwrap();
        assert_eq!(load_rgba16(&dir.join("d.png")).unwrap().2, [1, 2, 3, 65535]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rgba_expansion() {
        let grey = Image::from_raw(2, 1, 1, vec![7, 9]);
        assert_eq!(to_rgba8(&grey), [7, 7, 7, 255, 9, 9, 9, 255]);
        assert_eq!(BitDepth::from_bits(16).unwrap(), BitDepth::Sixteen);
        assert!(BitDepth::from_bits(12).is_err());
    }
}
