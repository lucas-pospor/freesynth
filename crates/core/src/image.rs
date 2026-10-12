//! Plain 2D containers: multi-channel 8- and 16-bit images and nearest-neighbour fields.

/// A 2D image of `channels` interleaved 8-bit channels per pixel, stored in scan-line order.
///
/// This is the Rust counterpart of the original `Array2<Vec<N,unsigned char>>`, with the channel
/// count chosen at run time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u8>,
}

impl Image {
    /// A zero-filled image.
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self { width, height, channels, data: vec![0; width * height * channels] }
    }

    /// Wraps existing pixel data (`width * height * channels` bytes, scan-line order).
    ///
    /// # Panics
    /// If `data.len()` does not match the dimensions.
    pub fn from_raw(width: usize, height: usize, channels: usize, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            width * height * channels,
            "image data length does not match {width}x{height}x{channels}"
        );
        Self { width, height, channels, data }
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

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }

    /// The channels of pixel `(x, y)`.
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> &[u8] {
        let i = (y * self.width + x) * self.channels;
        &self.data[i..i + self.channels]
    }

    /// Mutable channels of pixel `(x, y)`.
    #[inline]
    pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut [u8] {
        let i = (y * self.width + x) * self.channels;
        &mut self.data[i..i + self.channels]
    }
}

/// A 2D image of `channels` interleaved 16-bit channels per pixel, stored in scan-line order.
///
/// The engine searches on 8-bit images ([`Image`]); a 16-bit copy of the style lets it produce a
/// 16-bit result by voting the final field from the full-precision style
/// ([`SynthesisExtras::style16`](crate::SynthesisExtras::style16)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image16 {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u16>,
}

impl Image16 {
    /// A zero-filled image.
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self { width, height, channels, data: vec![0; width * height * channels] }
    }

    /// Wraps existing samples (`width * height * channels`, scan-line order).
    ///
    /// # Panics
    /// If `data.len()` does not match the dimensions.
    pub fn from_raw(width: usize, height: usize, channels: usize, data: Vec<u16>) -> Self {
        assert_eq!(
            data.len(),
            width * height * channels,
            "image data length does not match {width}x{height}x{channels}"
        );
        Self { width, height, channels, data }
    }

    /// The exact 16-bit version of an 8-bit image (`v * 257`, so 255 becomes 65535).
    pub fn from_image(image: &Image) -> Self {
        let data = image.data().iter().map(|&v| u16::from(v) * 257).collect();
        Self::from_raw(image.width(), image.height(), image.channels(), data)
    }

    /// The image rounded to 8 bits (`round(v / 257)`); the inverse of [`from_image`](Self::from_image).
    pub fn to_image(&self) -> Image {
        let data = self.data.iter().map(|&v| to_u8(v)).collect();
        Image::from_raw(self.width, self.height, self.channels, data)
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

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn data(&self) -> &[u16] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u16] {
        &mut self.data
    }

    pub fn into_raw(self) -> Vec<u16> {
        self.data
    }

    /// The channels of pixel `(x, y)`.
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> &[u16] {
        let i = (y * self.width + x) * self.channels;
        &self.data[i..i + self.channels]
    }
}

/// A 16-bit sample rounded to 8 bits: `round(v / 257)`.
#[inline]
pub(crate) fn to_u8(v: u16) -> u8 {
    ((u32::from(v) + 128) / 257) as u8
}

/// A nearest-neighbour field: for every target pixel, the `[x, y]` centre of its source patch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nnf {
    width: usize,
    height: usize,
    data: Vec<[i32; 2]>,
}

impl Nnf {
    /// A field with every entry set to `[0, 0]`.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![[0, 0]; width * height] }
    }

    /// Wraps existing entries (`width * height`, scan-line order).
    ///
    /// # Panics
    /// If `data.len()` does not match the dimensions.
    pub fn from_raw(width: usize, height: usize, data: Vec<[i32; 2]>) -> Self {
        assert_eq!(data.len(), width * height, "NNF length does not match {width}x{height}");
        Self { width, height, data }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> &[[i32; 2]] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [[i32; 2]] {
        &mut self.data
    }

    pub fn into_raw(self) -> Vec<[i32; 2]> {
        self.data
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [i32; 2] {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: [i32; 2]) {
        self.data[y * self.width + x] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixteen_bit_conversions_round_trip() {
        let img = Image::from_raw(2, 1, 2, vec![0, 1, 128, 255]);
        let wide = Image16::from_image(&img);
        assert_eq!(wide.data(), &[0, 257, 128 * 257, 65535]);
        assert_eq!(wide.to_image(), img);
        // Rounds to the nearest 8-bit level.
        let mid = Image16::from_raw(1, 1, 3, vec![128, 129, 65535 - 128]);
        assert_eq!(mid.to_image().data(), &[0, 1, 255]);
        for v in 0..=u16::MAX {
            let r = to_u8(v);
            assert!((f64::from(v) / 257.0 - f64::from(r)).abs() <= 0.5, "{v} -> {r}");
        }
    }
}
