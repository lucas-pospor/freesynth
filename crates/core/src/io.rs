//! Image file I/O and channel selection, following the 2019 command-line tool.
//!
//! The original loads every file as 8-bit RGBA (stb_image) and then keeps only the channels that
//! carry information: 1 for grey images, 3 for colour, plus 1 if any pixel is not fully opaque.
//! It always writes PNG.
//!
//! Loading matches stb_image byte for byte: JPEG files go through a translation of stb_image's
//! own decoder (used by [`decode_rgba`]); PNG, BMP and TGA are lossless, so any
//! correct decoder gives stb_image's pixels, and 16-bit images are reduced like stb_image does.

use std::io::Cursor;
use std::path::Path;

use image::{DynamicImage, ExtendedColorType, ImageError, ImageFormat, ImageReader};

use crate::error::{Error, invalid};
use crate::image::Image;
use crate::jpeg;

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
    // stb_image tries JPEG first, with the same test (the first marker is SOI).
    if jpeg::is_jpeg(bytes) {
        return jpeg::decode_rgba(bytes);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let dir = std::env::temp_dir().join(format!("freesynth-io-{}", std::process::id()));
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
}
