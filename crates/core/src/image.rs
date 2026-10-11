//! Plain 2D containers: multi-channel 8-bit images and nearest-neighbour fields.

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
