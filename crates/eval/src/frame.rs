//! Floating-point RGB frames for the image and video metrics.

use std::path::Path;

use freesynth_core::{Image, io};

use crate::error::{Error, invalid};

/// An RGB image with channels in `[0, 1]`, in scan-line order.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImage {
    width: usize,
    height: usize,
    data: Vec<[f32; 3]>,
}

impl RgbImage {
    /// Wraps existing pixels (`width * height`, scan-line order).
    ///
    /// # Panics
    /// If `data.len()` does not match the dimensions.
    pub fn from_raw(width: usize, height: usize, data: Vec<[f32; 3]>) -> Self {
        assert_eq!(data.len(), width * height, "image length does not match {width}x{height}");
        Self { width, height, data }
    }

    /// Converts an 8-bit image: 1 channel is grey, 2 is grey + alpha, 3 is RGB, 4 is RGBA.
    /// Alpha is ignored.
    pub fn from_image(img: &Image) -> Result<Self, Error> {
        let nc = img.channels();
        let px = |p: &[u8]| -> [f32; 3] {
            let f = |v: u8| v as f32 / 255.0;
            match nc {
                1 | 2 => [f(p[0]); 3],
                _ => [f(p[0]), f(p[1]), f(p[2])],
            }
        };
        if !(1..=4).contains(&nc) {
            return invalid(format!("cannot use an image with {nc} channels as RGB"));
        }
        let data = img.data().chunks_exact(nc).map(px).collect();
        Ok(Self { width: img.width(), height: img.height(), data })
    }

    /// Loads an image file (any format [`io::load_rgba`] reads).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let img = io::load_rgba(path).map_err(|e| Error::Invalid(format!("cannot load '{}': {e}", path.display())))?;
        Self::from_image(&img)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// `(width, height)`.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }

    /// Rec. 601 luma (`0.299 R + 0.587 G + 0.114 B`) of every pixel, in `[0, 1]`.
    pub fn luma(&self) -> Vec<f32> {
        self.data.iter().map(|&[r, g, b]| luma(r, g, b)).collect()
    }

    /// Mean luma over the image.
    pub fn mean_luma(&self) -> f64 {
        self.data.iter().map(|&[r, g, b]| luma(r, g, b) as f64).sum::<f64>() / self.data.len() as f64
    }
}

#[inline]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion() {
        let grey = Image::from_raw(2, 1, 1, vec![0, 255]);
        let g = RgbImage::from_image(&grey).unwrap();
        assert_eq!(g.data(), &[[0.0; 3], [1.0; 3]]);
        let rgba = Image::from_raw(1, 1, 4, vec![255, 0, 0, 7]);
        let c = RgbImage::from_image(&rgba).unwrap();
        assert_eq!(c.data(), &[[1.0, 0.0, 0.0]]);
        assert!((c.mean_luma() - 0.299).abs() < 1e-6);
        assert!(RgbImage::from_image(&Image::new(1, 1, 5)).is_err());
    }
}
