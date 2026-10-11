//! Single-channel input frames.

use crate::error::{FlowError, invalid};

/// Rec. 601 luma weights, the usual RGB to grey conversion for flow estimation.
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

/// A single-channel image with `f32` samples, row-major. Flow estimation works on intensities
/// in the 0-255 range (the conversions below produce that range); other ranges work too, but
/// the default parameters assume it.
#[derive(Clone, Debug, PartialEq)]
pub struct GrayImage {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl GrayImage {
    /// A black image.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height] }
    }

    /// Wraps existing samples (`width * height` of them, row-major).
    pub fn from_vec(width: usize, height: usize, data: Vec<f32>) -> Result<Self, FlowError> {
        check_len(width, height, 1, data.len())?;
        Ok(Self { width, height, data })
    }

    /// Converts 8-bit grey samples.
    pub fn from_luma8(width: usize, height: usize, data: &[u8]) -> Result<Self, FlowError> {
        check_len(width, height, 1, data.len())?;
        Ok(Self { width, height, data: data.iter().map(|&v| f32::from(v)).collect() })
    }

    /// Converts interleaved 8-bit RGB to Rec. 601 luma.
    pub fn from_rgb8(width: usize, height: usize, data: &[u8]) -> Result<Self, FlowError> {
        Self::from_interleaved(width, height, 3, data)
    }

    /// Converts interleaved 8-bit RGBA to Rec. 601 luma (alpha is ignored).
    pub fn from_rgba8(width: usize, height: usize, data: &[u8]) -> Result<Self, FlowError> {
        Self::from_interleaved(width, height, 4, data)
    }

    fn from_interleaved(width: usize, height: usize, channels: usize, data: &[u8]) -> Result<Self, FlowError> {
        check_len(width, height, channels, data.len())?;
        let data = data.chunks_exact(channels).map(|p| luma([p[0], p[1], p[2]])).collect();
        Ok(Self { width, height, data })
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

    /// Samples, row-major.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable samples, row-major.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// The sample at pixel `(x, y)`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

/// Rec. 601 luma of an 8-bit RGB triple, in 0-255.
pub(crate) fn luma(rgb: [u8; 3]) -> f32 {
    LUMA[0] * f32::from(rgb[0]) + LUMA[1] * f32::from(rgb[1]) + LUMA[2] * f32::from(rgb[2])
}

fn check_len(width: usize, height: usize, channels: usize, len: usize) -> Result<(), FlowError> {
    if width.checked_mul(height).and_then(|n| n.checked_mul(channels)) != Some(len) {
        return invalid(format!("{len} samples do not make a {width}x{height} image with {channels} channel(s)"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        let g = GrayImage::from_rgb8(2, 1, &[255, 255, 255, 0, 255, 0]).unwrap();
        assert!((g.get(0, 0) - 255.0).abs() < 1e-3);
        assert!((g.get(1, 0) - 0.587 * 255.0).abs() < 1e-3);
        let g = GrayImage::from_rgba8(1, 1, &[10, 10, 10, 0]).unwrap();
        assert!((g.get(0, 0) - 10.0).abs() < 1e-4);
        assert_eq!(GrayImage::from_luma8(2, 1, &[3, 4]).unwrap().data(), &[3.0, 4.0]);
        assert!(GrayImage::from_rgb8(2, 2, &[0; 11]).is_err());
        assert!(GrayImage::from_vec(3, 3, vec![0.0; 8]).is_err());
    }
}
