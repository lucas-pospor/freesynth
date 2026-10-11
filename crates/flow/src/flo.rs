//! Dense flow fields and the Middlebury `.flo` file format.
//!
//! A `.flo` file is little-endian:
//!
//! | Offset | Type | Content |
//! |---|---|---|
//! | 0 | `f32` | magic number `202021.25` (the bytes spell `PIEH`) |
//! | 4 | `i32` | width |
//! | 8 | `i32` | height |
//! | 12 | `f32` pairs | `(u, v)` for each pixel, row-major, top row first |
//!
//! The flow at pixel `(x, y)` of frame A says that the content at that pixel centre is at
//! `(x + u, y + v)` in frame B. Pixel centres are at integer coordinates. Values above `1e9`
//! mark unknown flow in the Middlebury data sets; they are kept as they are.
//!
//! Reference: Baker, Scharstein, Lewis, Roth, Black, Szeliski, "A Database and Evaluation
//! Methodology for Optical Flow", IJCV 2011.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic number at the start of every `.flo` file.
pub const FLO_MAGIC: f32 = 202_021.25;

/// Largest width or height accepted when reading, to reject corrupt headers early.
const MAX_DIM: i32 = 1 << 16;

/// A dense 2-D flow field: one `[u, v]` displacement per pixel, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowField {
    width: usize,
    height: usize,
    data: Vec<[f32; 2]>,
}

impl FlowField {
    /// A zero field.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![[0.0; 2]; width * height] }
    }

    /// Wraps existing displacements (`width * height` of them, row-major).
    ///
    /// # Panics
    /// If `data.len()` does not match the dimensions.
    pub fn from_vec(width: usize, height: usize, data: Vec<[f32; 2]>) -> Self {
        assert_eq!(data.len(), width * height, "flow length does not match {width}x{height}");
        Self { width, height, data }
    }

    /// A field with the same displacement everywhere.
    pub fn constant(width: usize, height: usize, uv: [f32; 2]) -> Self {
        Self { width, height, data: vec![uv; width * height] }
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

    /// Displacements, row-major.
    pub fn data(&self) -> &[[f32; 2]] {
        &self.data
    }

    /// Mutable displacements, row-major.
    pub fn data_mut(&mut self) -> &mut [[f32; 2]] {
        &mut self.data
    }

    /// Consumes the field and returns its displacements.
    pub fn into_vec(self) -> Vec<[f32; 2]> {
        self.data
    }

    /// The displacement at pixel `(x, y)`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 2] {
        self.data[y * self.width + x]
    }

    /// Bilinear interpolation at a continuous position, clamped to the field.
    ///
    /// # Panics
    /// If the field is empty.
    pub fn sample(&self, x: f32, y: f32) -> [f32; 2] {
        let xc = x.clamp(0.0, (self.width - 1) as f32);
        let yc = y.clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (xc.floor() as usize, yc.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (xc - x0 as f32, yc - y0 as f32);
        let lerp = |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let top = lerp(self.get(x0, y0), self.get(x1, y0), fx);
        let bottom = lerp(self.get(x0, y1), self.get(x1, y1), fx);
        lerp(top, bottom, fy)
    }

    /// Writes the field in `.flo` format. The whole file is assembled in memory and written
    /// with one call, so `w` needs no buffering.
    pub fn write_flo(&self, mut w: impl Write) -> io::Result<()> {
        let (Ok(width), Ok(height)) = (i32::try_from(self.width), i32::try_from(self.height)) else {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "flow field is too large for the .flo format"));
        };
        let mut buf = Vec::with_capacity(12 + self.data.len() * 8);
        buf.extend_from_slice(&FLO_MAGIC.to_le_bytes());
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        for &[u, v] in &self.data {
            buf.extend_from_slice(&u.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&buf)
    }

    /// Reads a field in `.flo` format. Trailing bytes after the data are an error.
    pub fn read_flo(mut r: impl Read) -> io::Result<Self> {
        let mut header = [0u8; 12];
        r.read_exact(&mut header).map_err(|_| bad_data("not a .flo file (too short)"))?;
        let word = |i: usize| [header[i], header[i + 1], header[i + 2], header[i + 3]];
        if f32::from_le_bytes(word(0)) != FLO_MAGIC {
            return Err(bad_data("not a .flo file (wrong magic number)"));
        }
        let (w, h) = (i32::from_le_bytes(word(4)), i32::from_le_bytes(word(8)));
        if !(0..=MAX_DIM).contains(&w) || !(0..=MAX_DIM).contains(&h) {
            return Err(bad_data(format!(".flo size {w}x{h} out of range")));
        }
        let (w, h) = (w as usize, h as usize);
        // Read at most one byte more than the header promises, so that a corrupt header cannot
        // make us allocate gigabytes up front, and trailing bytes are detected.
        let len = w * h * 8;
        let mut bytes = Vec::new();
        r.take(len as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() != len {
            return Err(bad_data(format!(".flo file holds {} data bytes, expected {len} for {w}x{h}", bytes.len())));
        }
        let data = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| [f32::from_le_bytes([c[0], c[1], c[2], c[3]]), f32::from_le_bytes([c[4], c[5], c[6], c[7]])])
            .collect();
        Ok(Self { width: w, height: h, data })
    }

    /// Writes a `.flo` file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        self.write_flo(&mut w)?;
        w.flush()
    }

    /// Reads a `.flo` file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::read_flo(BufReader::new(File::open(path)?))
    }

    /// Mean end-point error against `reference` over the pixels where `mask` is true (all
    /// pixels when `mask` is `None`). Returns `None` when no pixel is selected.
    ///
    /// # Panics
    /// If the sizes differ.
    pub fn mean_epe(&self, reference: &FlowField, mask: Option<&[bool]>) -> Option<f64> {
        assert_eq!(self.size(), reference.size(), "flow fields differ in size");
        if let Some(m) = mask {
            assert_eq!(m.len(), self.data.len(), "mask does not match the flow size");
        }
        let (mut sum, mut n) = (0.0f64, 0usize);
        for (i, (a, b)) in self.data.iter().zip(&reference.data).enumerate() {
            if mask.is_none_or(|m| m[i]) {
                sum += f64::from((a[0] - b[0]).hypot(a[1] - b[1]));
                n += 1;
            }
        }
        (n > 0).then(|| sum / n as f64)
    }
}

fn bad_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_the_documented_layout() {
        let flow = FlowField::from_vec(2, 1, vec![[1.5, -2.0], [0.25, 3.0]]);
        let mut bytes = Vec::new();
        flow.write_flo(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 12 + 2 * 8);
        assert_eq!(&bytes[0..4], b"PIEH");
        assert_eq!(&bytes[0..4], &202_021.25f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &2i32.to_le_bytes());
        assert_eq!(&bytes[8..12], &1i32.to_le_bytes());
        let floats: Vec<f32> = bytes[12..].chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(floats, [1.5, -2.0, 0.25, 3.0]);
    }

    #[test]
    fn round_trips() {
        let flow = FlowField::from_vec(
            3,
            2,
            vec![[0.0, 0.0], [1.5, -2.25], [-0.125, 1e10], [3.0, 4.0], [f32::MIN_POSITIVE, 0.0], [7.0, -8.0]],
        );
        let mut bytes = Vec::new();
        flow.write_flo(&mut bytes).unwrap();
        assert_eq!(FlowField::read_flo(bytes.as_slice()).unwrap(), flow);
        let empty = FlowField::new(0, 0);
        bytes.clear();
        empty.write_flo(&mut bytes).unwrap();
        assert_eq!(FlowField::read_flo(bytes.as_slice()).unwrap(), empty);
    }

    #[test]
    fn rejects_bad_headers_and_truncation() {
        let mut bytes = Vec::new();
        FlowField::new(3, 2).write_flo(&mut bytes).unwrap();
        let mut wrong_magic = bytes.clone();
        wrong_magic[0] ^= 1;
        assert!(FlowField::read_flo(wrong_magic.as_slice()).is_err());
        assert!(FlowField::read_flo(&bytes[..bytes.len() - 1]).is_err());
        assert!(FlowField::read_flo(&bytes[..8]).is_err());
        let mut long = bytes.clone();
        long.push(0);
        assert!(FlowField::read_flo(long.as_slice()).is_err());
        let mut huge = bytes.clone();
        huge[4..8].copy_from_slice(&i32::MAX.to_le_bytes());
        assert!(FlowField::read_flo(huge.as_slice()).is_err());
        let mut negative = bytes.clone();
        negative[8..12].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(FlowField::read_flo(negative.as_slice()).is_err());
    }

    #[test]
    fn file_round_trip() {
        let dir = crate::test_util::scratch_dir("flo");
        let path = dir.join("00000.flo");
        let flow = FlowField::constant(4, 3, [0.5, -1.0]);
        flow.save(&path).unwrap();
        assert_eq!(FlowField::load(&path).unwrap(), flow);
        assert!(FlowField::load(dir.join("missing.flo")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bilinear_sampling() {
        let flow = FlowField::from_vec(2, 2, vec![[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [2.0, 4.0]]);
        assert_eq!(flow.sample(0.5, 0.5), [1.0, 2.0]);
        assert_eq!(flow.sample(-3.0, 9.0), [0.0, 4.0]);
    }

    #[test]
    fn end_point_error() {
        let a = FlowField::constant(2, 1, [3.0, 4.0]);
        let b = FlowField::from_vec(2, 1, vec![[0.0, 0.0], [3.0, 4.0]]);
        assert_eq!(a.mean_epe(&b, None), Some(2.5));
        assert_eq!(a.mean_epe(&b, Some(&[true, false])), Some(5.0));
        assert_eq!(a.mean_epe(&b, Some(&[false, false])), None);
    }
}
