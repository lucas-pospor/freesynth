//! Image file I/O and channel selection, following the 2019 command-line tool.
//!
//! The original loads every file as 8-bit RGBA (stb_image) and then keeps only the channels that
//! carry information: 1 for grey images, 3 for colour, plus 1 if any pixel is not fully opaque.
//! It always writes PNG.
//!
//! Decoding uses the `image` crate. Lossless formats (PNG, BMP, TGA) give the same pixels as
//! stb_image and 16-bit images are reduced the same way; JPEG pixels can differ slightly because
//! decoders use different IDCTs and chroma upsampling.
//!
//! Nearest-neighbour fields can be stored in a small binary format (see [`write_nnf`]), and
//! per-pixel float maps such as patch errors as greyscale PFM (see [`write_pfm`]).

use std::io::{Cursor, Read, Write};
use std::path::Path;

use image::{DynamicImage, ExtendedColorType, ImageError, ImageFormat, ImageReader};

use crate::error::{Error, invalid};
use crate::image::{Image, Nnf};

/// Loads an image file as 8-bit RGBA, like `stbi_load(path, ..., 4)`.
///
/// As in stb_image, the format is detected from the file contents (the extension is only used
/// for formats without a signature, such as TGA), 16-bit images are reduced to 8 bits by
/// dropping the low byte, and grey or grey+alpha images are expanded to RGBA.
pub fn load_rgba(path: impl AsRef<Path>) -> Result<Image, Error> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error::Image(ImageError::IoError(e)))?;
    decode_rgba(&bytes, ImageFormat::from_path(path).ok())
}

/// Decodes an in-memory image file as 8-bit RGBA with the same rules as [`load_rgba`].
/// `fallback` is the format to assume when the contents have no recognisable signature.
pub fn decode_rgba(bytes: &[u8], fallback: Option<ImageFormat>) -> Result<Image, Error> {
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(ImageError::IoError)?;
    if reader.format().is_none()
        && let Some(f) = fallback
    {
        reader.set_format(f);
    }
    Ok(dynamic_to_rgba(reader.decode()?))
}

/// Converts a decoded image to 8-bit RGBA with the same rules as [`load_rgba`].
pub fn dynamic_to_rgba(img: DynamicImage) -> Image {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data = match img {
        DynamicImage::ImageLuma16(_) | DynamicImage::ImageLumaA16(_) | DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_) => {
            img.to_rgba16().into_raw().into_iter().map(|v| (v >> 8) as u8).collect()
        }
        _ => img.to_rgba8().into_raw(),
    };
    Image::from_raw(w, h, 4, data)
}

/// Number of informative channels of an RGBA image (`evalNumChannels`): 1 if every pixel is grey
/// (`r == g == b`), else 3; plus 1 if any pixel has alpha < 255.
pub fn eval_num_channels(rgba: &Image) -> usize {
    assert_eq!(rgba.channels(), 4, "eval_num_channels expects RGBA");
    let mut is_gray = true;
    let mut has_alpha = false;
    for px in rgba.data().as_chunks::<4>().0 {
        if !(px[0] == px[1] && px[1] == px[2]) {
            is_gray = false;
        }
        if px[3] < 255 {
            has_alpha = true;
        }
        if !is_gray && has_alpha {
            break;
        }
    }
    (if is_gray { 1 } else { 3 }) + usize::from(has_alpha)
}

/// The RGBA channels kept for a given channel count: 1 = R, 2 = R+A, 3 = RGB, 4 = RGBA.
pub fn channel_map(num_channels: usize) -> &'static [usize] {
    match num_channels {
        1 => &[0],
        2 => &[0, 3],
        3 => &[0, 1, 2],
        4 => &[0, 1, 2, 3],
        _ => panic!("unsupported channel count {num_channels}"),
    }
}

/// Extracts `num_channels` channels (see [`channel_map`]) from an RGBA image.
pub fn select_channels(rgba: &Image, num_channels: usize) -> Image {
    assert_eq!(rgba.channels(), 4, "select_channels expects RGBA");
    let map = channel_map(num_channels);
    let mut data = Vec::with_capacity(rgba.width() * rgba.height() * map.len());
    for px in rgba.data().as_chunks::<4>().0 {
        data.extend(map.iter().map(|&c| px[c]));
    }
    Image::from_raw(rgba.width(), rgba.height(), map.len(), data)
}

fn color_type(img: &Image) -> Result<ExtendedColorType, Error> {
    Ok(match img.channels() {
        1 => ExtendedColorType::L8,
        2 => ExtendedColorType::La8,
        3 => ExtendedColorType::Rgb8,
        4 => ExtendedColorType::Rgba8,
        n => return invalid(format!("cannot save an image with {n} channels")),
    })
}

/// Saves a 1- to 4-channel image (grey, grey+alpha, RGB, RGBA) as PNG, whatever the file name,
/// like the 2019 tool (`stbi_write_png`).
pub fn save_png(img: &Image, path: impl AsRef<Path>) -> Result<(), Error> {
    let color = color_type(img)?;
    image::save_buffer_with_format(path, img.data(), img.width() as u32, img.height() as u32, color, ImageFormat::Png)?;
    Ok(())
}

/// Saves a 1- to 4-channel image (grey, grey+alpha, RGB, RGBA) in the format given by the file
/// extension (PNG, JPEG, BMP or TGA). Use [`save_png`] for the 2019 tool's behaviour.
pub fn save(img: &Image, path: impl AsRef<Path>) -> Result<(), Error> {
    let color = color_type(img)?;
    image::save_buffer(path, img.data(), img.width() as u32, img.height() as u32, color)?;
    Ok(())
}

/// Magic bytes at the start of an NNF file.
pub const NNF_MAGIC: &[u8; 6] = b"FSNNF1";

/// Writes a nearest-neighbour field in the FreeSynth NNF format.
///
/// The format is little-endian and has no padding:
///
/// | Offset | Size | Contents |
/// |---|---|---|
/// | 0 | 6 | magic `FSNNF1` (ASCII) |
/// | 6 | 4 | width, `u32` |
/// | 10 | 4 | height, `u32` |
/// | 14 | 8 * width * height | for every target pixel in scan-line order, the source patch centre as `i32` x then `i32` y |
pub fn write_nnf(nnf: &Nnf, mut w: impl Write) -> Result<(), Error> {
    let (width, height) = (u32::try_from(nnf.width()), u32::try_from(nnf.height()));
    let (Ok(width), Ok(height)) = (width, height) else {
        return invalid("NNF is too large for the file format");
    };
    let mut buf = Vec::with_capacity(14 + nnf.data().len() * 8);
    buf.extend_from_slice(NNF_MAGIC);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    for &[x, y] in nnf.data() {
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
    }
    w.write_all(&buf)?;
    Ok(())
}

/// Reads a nearest-neighbour field written by [`write_nnf`].
pub fn read_nnf(mut r: impl Read) -> Result<Nnf, Error> {
    let mut header = [0u8; 14];
    r.read_exact(&mut header).map_err(|_| Error::InvalidInput("not an NNF file (too short)".into()))?;
    if &header[..6] != NNF_MAGIC {
        return invalid("not an NNF file (bad magic)");
    }
    let width = u32::from_le_bytes(header[6..10].try_into().expect("4 bytes")) as usize;
    let height = u32::from_le_bytes(header[10..14].try_into().expect("4 bytes")) as usize;
    let Some(len) = width.checked_mul(height).and_then(|n| n.checked_mul(8)) else {
        return invalid("NNF size overflows");
    };
    let mut body = Vec::new();
    r.take(len as u64 + 1).read_to_end(&mut body)?;
    if body.len() != len {
        return invalid(format!("NNF file holds {} data bytes, expected {len} for {width}x{height}", body.len()));
    }
    let data = body
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            let x = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let y = i32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            [x, y]
        })
        .collect();
    Ok(Nnf::from_raw(width, height, data))
}

/// Saves a nearest-neighbour field to a file (see [`write_nnf`]).
pub fn save_nnf(nnf: &Nnf, path: impl AsRef<Path>) -> Result<(), Error> {
    // `write_nnf` assembles the whole file in memory and writes it at once, so no buffering is
    // needed (and no write error can be lost in a buffer dropped without flushing).
    write_nnf(nnf, std::fs::File::create(path)?)
}

/// Loads a nearest-neighbour field from a file (see [`write_nnf`]).
pub fn load_nnf(path: impl AsRef<Path>) -> Result<Nnf, Error> {
    let file = std::fs::File::open(path)?;
    read_nnf(std::io::BufReader::new(file))
}

/// Writes a single-channel float map (for example the per-pixel patch errors of a synthesis) as a
/// greyscale Portable Float Map (`Pf`), which many image tools read.
///
/// `values` holds `width * height` values in scan-line order, top row first. The file stores
/// little-endian `f32` rows bottom row first, as the format requires.
pub fn write_pfm(values: &[f32], width: usize, height: usize, mut w: impl Write) -> Result<(), Error> {
    if values.len() != width * height {
        return invalid(format!("{} values do not fill a {width}x{height} map", values.len()));
    }
    let header = format!("Pf\n{width} {height}\n-1.0\n");
    let mut buf = Vec::with_capacity(header.len() + values.len() * 4);
    buf.extend_from_slice(header.as_bytes());
    if width > 0 {
        for row in values.chunks_exact(width).rev() {
            for v in row {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    w.write_all(&buf)?;
    Ok(())
}

/// Reads a greyscale Portable Float Map (either byte order). Returns `(width, height, values)`
/// with the values in scan-line order, top row first.
pub fn read_pfm(mut r: impl Read) -> Result<(usize, usize, Vec<f32>), Error> {
    // The header is three whitespace-separated tokens after the magic, ended by one whitespace
    // byte.
    let mut tokens: Vec<String> = Vec::new();
    let mut token = String::new();
    let mut byte = [0u8; 1];
    while tokens.len() < 4 {
        if r.read(&mut byte)? == 0 {
            return invalid("not a PFM file (truncated header)");
        }
        let c = byte[0] as char;
        if c.is_ascii_whitespace() {
            if !token.is_empty() {
                tokens.push(std::mem::take(&mut token));
            }
        } else if token.len() < 32 {
            token.push(c);
        } else {
            return invalid("not a PFM file (bad header)");
        }
    }
    if tokens[0] != "Pf" {
        return invalid("not a greyscale PFM file (expected magic 'Pf')");
    }
    let parse = |t: &str| t.parse::<usize>().map_err(|_| Error::InvalidInput(format!("bad PFM size '{t}'")));
    let (width, height) = (parse(&tokens[1])?, parse(&tokens[2])?);
    let scale: f32 = tokens[3].parse().map_err(|_| Error::InvalidInput(format!("bad PFM scale '{}'", tokens[3])))?;
    let little = scale < 0.0;
    let Some(len) = width.checked_mul(height).and_then(|n| n.checked_mul(4)) else {
        return invalid("PFM size overflows");
    };
    let mut body = Vec::new();
    r.take(len as u64 + 1).read_to_end(&mut body)?;
    if body.len() != len {
        return invalid(format!("PFM file holds {} data bytes, expected {len} for {width}x{height}", body.len()));
    }
    let mut values: Vec<f32> = body
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| if little { f32::from_le_bytes(b) } else { f32::from_be_bytes(b) })
        .collect();
    if width > 0 {
        // Bottom-up rows to top-down.
        let rows: Vec<Vec<f32>> = values.chunks_exact(width).rev().map(<[f32]>::to_vec).collect();
        values = rows.concat();
    }
    Ok((width, height, values))
}

/// Saves a single-channel float map as greyscale PFM (see [`write_pfm`]).
pub fn save_pfm(values: &[f32], width: usize, height: usize, path: impl AsRef<Path>) -> Result<(), Error> {
    write_pfm(values, width, height, std::fs::File::create(path)?)
}

/// Loads a greyscale PFM file (see [`read_pfm`]).
pub fn load_pfm(path: impl AsRef<Path>) -> Result<(usize, usize, Vec<f32>), Error> {
    read_pfm(std::io::BufReader::new(std::fs::File::open(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory on disk under the build directory (the system temp directory may be
    /// RAM-backed).
    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"));
        target.join("tmp").join(format!("{name}-{}", std::process::id()))
    }


    fn rgba(px: &[[u8; 4]]) -> Image {
        Image::from_raw(px.len(), 1, 4, px.iter().flatten().copied().collect())
    }

    #[test]
    fn channel_counts() {
        assert_eq!(eval_num_channels(&rgba(&[[5, 5, 5, 255], [9, 9, 9, 255]])), 1);
        assert_eq!(eval_num_channels(&rgba(&[[5, 5, 5, 255], [9, 9, 9, 254]])), 2);
        assert_eq!(eval_num_channels(&rgba(&[[5, 6, 5, 255]])), 3);
        assert_eq!(eval_num_channels(&rgba(&[[5, 6, 5, 0]])), 4);
    }

    #[test]
    fn selection_order() {
        let img = rgba(&[[1, 2, 3, 4]]);
        assert_eq!(select_channels(&img, 1).data(), &[1]);
        assert_eq!(select_channels(&img, 2).data(), &[1, 4]);
        assert_eq!(select_channels(&img, 3).data(), &[1, 2, 3]);
        assert_eq!(select_channels(&img, 4).data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn png_round_trip() {
        let dir = scratch_dir("freesynth-io");
        std::fs::create_dir_all(&dir).unwrap();
        for nc in 1..=4 {
            let img = Image::from_raw(3, 2, nc, (0..6 * nc).map(|v| (v * 40) as u8).collect());
            let path = dir.join(format!("t{nc}.png"));
            save(&img, &path).unwrap();
            let back = load_rgba(&path).unwrap();
            assert_eq!(select_channels(&back, nc), img);
            // save_png writes PNG whatever the name; loading sniffs the contents.
            let odd = dir.join(format!("t{nc}.jpg"));
            save_png(&img, &odd).unwrap();
            assert_eq!(&std::fs::read(&odd).unwrap()[..4], b"\x89PNG");
            assert_eq!(select_channels(&load_rgba(&odd).unwrap(), nc), img);
            let bare = dir.join(format!("t{nc}"));
            save_png(&img, &bare).unwrap();
            assert_eq!(select_channels(&load_rgba(&bare).unwrap(), nc), img);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pfm_round_trip() {
        let values = [0.0f32, 1.5, -2.0, 1e9, 7.25, f32::MAX];
        let mut buf = Vec::new();
        write_pfm(&values, 3, 2, &mut buf).unwrap();
        assert!(buf.starts_with(b"Pf\n3 2\n-1.0\n"));
        // Bottom row first.
        let data = &buf[buf.len() - 24..];
        assert_eq!(&data[..4], &1e9f32.to_le_bytes());
        assert_eq!(read_pfm(buf.as_slice()).unwrap(), (3, 2, values.to_vec()));

        // Big-endian files (positive scale) read too.
        let mut be = b"Pf 2 1 1.0\n".to_vec();
        be.extend_from_slice(&3.0f32.to_be_bytes());
        be.extend_from_slice(&4.0f32.to_be_bytes());
        assert_eq!(read_pfm(be.as_slice()).unwrap(), (2, 1, vec![3.0, 4.0]));

        assert!(write_pfm(&values, 4, 2, Vec::new()).is_err());
        assert!(read_pfm(&buf[..buf.len() - 1]).is_err());
        assert!(read_pfm(&b"PF\n1 1\n-1.0\n\0\0\0\0"[..]).is_err());
        assert!(read_pfm(&b"Pf\n1"[..]).is_err());

        let dir = scratch_dir("freesynth-pfm");
        std::fs::create_dir_all(&dir).unwrap();
        save_pfm(&values, 2, 3, dir.join("e.pfm")).unwrap();
        assert_eq!(load_pfm(dir.join("e.pfm")).unwrap(), (2, 3, values.to_vec()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nnf_round_trip() {
        let nnf = Nnf::from_raw(3, 2, vec![[0, 0], [1, -2], [i32::MAX, i32::MIN], [7, 8], [9, 10], [11, 12]]);
        let mut buf = Vec::new();
        write_nnf(&nnf, &mut buf).unwrap();
        assert_eq!(buf.len(), 14 + 6 * 8);
        assert_eq!(&buf[..6], b"FSNNF1");
        assert_eq!(&buf[6..14], &[3, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(&buf[22..30], &[1, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff]);
        assert_eq!(read_nnf(buf.as_slice()).unwrap(), nnf);

        // Truncated, padded and mislabelled files are rejected.
        assert!(read_nnf(&buf[..buf.len() - 1]).is_err());
        let mut long = buf.clone();
        long.push(0);
        assert!(read_nnf(long.as_slice()).is_err());
        let mut bad = buf.clone();
        bad[0] = b'X';
        assert!(read_nnf(bad.as_slice()).is_err());
        assert!(read_nnf(&buf[..10]).is_err());

        let empty = Nnf::new(0, 0);
        let mut buf = Vec::new();
        write_nnf(&empty, &mut buf).unwrap();
        assert_eq!(read_nnf(buf.as_slice()).unwrap(), empty);
    }
}
