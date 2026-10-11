//! Dense optical flow fields and the Middlebury `.flo` file format.
//!
//! A `.flo` file is little-endian: the magic float `202021.25` (the bytes `PIEH`), the width and
//! height as `i32`, then `width * height` pairs of `f32` `(u, v)` in scan-line order. A flow
//! vector `(u, v)` at pixel `(x, y)` of frame A says that the point is at `(x + u, y + v)` in
//! frame B. Values above `1e9` mark unknown flow in the Middlebury data sets; they are kept as
//! they are here.
//!
//! Reference: Baker, Scharstein, Lewis, Roth, Black, Szeliski, "A Database and Evaluation
//! Methodology for Optical Flow", IJCV 2011.

use std::io::{Read, Write};
use std::path::Path;

use crate::error::{Error, invalid};

/// The magic number at the start of a `.flo` file.
pub const FLO_MAGIC: f32 = 202021.25;

/// A dense 2D flow field: one `[u, v]` displacement per pixel, in scan-line order.
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

    /// Wraps existing vectors (`width * height`, scan-line order).
    ///
    /// # Panics
    /// If `data.len()` does not match the dimensions.
    pub fn from_raw(width: usize, height: usize, data: Vec<[f32; 2]>) -> Self {
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

    pub fn data(&self) -> &[[f32; 2]] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [[f32; 2]] {
        &mut self.data
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 2] {
        self.data[y * self.width + x]
    }
}

/// Writes a flow field in the `.flo` format.
pub fn write_flo(flow: &FlowField, mut w: impl Write) -> Result<(), Error> {
    let (Ok(width), Ok(height)) = (i32::try_from(flow.width), i32::try_from(flow.height)) else {
        return invalid("flow field is too large for the .flo format");
    };
    let mut buf = Vec::with_capacity(12 + flow.data.len() * 8);
    buf.extend_from_slice(&FLO_MAGIC.to_le_bytes());
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    for &[u, v] in &flow.data {
        buf.extend_from_slice(&u.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&buf)?;
    Ok(())
}

/// Reads a flow field in the `.flo` format.
pub fn read_flo(mut r: impl Read) -> Result<FlowField, Error> {
    let mut header = [0u8; 12];
    r.read_exact(&mut header).map_err(|_| Error::Invalid("not a .flo file (too short)".into()))?;
    let le = |i: usize| [header[i], header[i + 1], header[i + 2], header[i + 3]];
    if f32::from_le_bytes(le(0)) != FLO_MAGIC {
        return invalid("not a .flo file (bad magic)");
    }
    let (width, height) = (i32::from_le_bytes(le(4)), i32::from_le_bytes(le(8)));
    if width < 0 || height < 0 {
        return invalid(format!("bad .flo size {width}x{height}"));
    }
    let (width, height) = (width as usize, height as usize);
    let Some(len) = width.checked_mul(height).and_then(|n| n.checked_mul(8)) else {
        return invalid(format!("bad .flo size {width}x{height}"));
    };
    let mut body = Vec::new();
    r.take(len as u64 + 1).read_to_end(&mut body)?;
    if body.len() != len {
        return invalid(format!(".flo file holds {} data bytes, expected {len} for {width}x{height}", body.len()));
    }
    let data = body
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| [f32::from_le_bytes([c[0], c[1], c[2], c[3]]), f32::from_le_bytes([c[4], c[5], c[6], c[7]])])
        .collect();
    Ok(FlowField { width, height, data })
}

/// Saves a flow field as a `.flo` file.
pub fn save_flo(flow: &FlowField, path: impl AsRef<Path>) -> Result<(), Error> {
    // `write_flo` writes the whole file at once, so it needs no buffering.
    write_flo(flow, std::fs::File::create(path)?)
}

/// Loads a `.flo` file.
pub fn load_flo(path: impl AsRef<Path>) -> Result<FlowField, Error> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| Error::Invalid(format!("cannot open '{}': {e}", path.display())))?;
    read_flo(std::io::BufReader::new(file)).map_err(|e| Error::Invalid(format!("'{}': {e}", path.display())))
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


    #[test]
    fn round_trip() {
        let flow = FlowField::from_raw(3, 2, vec![[0.0, 0.0], [1.5, -2.25], [-0.125, 1e10], [3.0, 4.0], [f32::MIN_POSITIVE, 0.0], [7.0, -8.0]]);
        let mut buf = Vec::new();
        write_flo(&flow, &mut buf).unwrap();
        assert_eq!(buf.len(), 12 + 6 * 8);
        assert_eq!(&buf[..4], b"PIEH");
        assert_eq!(&buf[4..12], &[3, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(read_flo(buf.as_slice()).unwrap(), flow);

        assert!(read_flo(&buf[..buf.len() - 4]).is_err());
        let mut long = buf.clone();
        long.extend_from_slice(&[0; 8]);
        assert!(read_flo(long.as_slice()).is_err());
        let mut bad = buf.clone();
        bad[0] ^= 1;
        assert!(read_flo(bad.as_slice()).is_err());
    }

    #[test]
    fn file_round_trip() {
        let dir = scratch_dir("freesynth-eval-flo");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("00000.flo");
        let flow = FlowField::constant(4, 3, [0.5, -1.0]);
        save_flo(&flow, &path).unwrap();
        assert_eq!(load_flo(&path).unwrap(), flow);
        assert!(load_flo(dir.join("missing.flo")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
