//! Dense optical flow fields and Middlebury `.flo` files, with this crate's error type.
//!
//! The field type and the file format live in `freesynth-flow` ([`FlowField`]); this module
//! re-exports them and wraps the file functions so that errors name the file and convert into
//! [`Error`]. A flow vector `(u, v)` at pixel `(x, y)` of frame A says that the point is at
//! `(x + u, y + v)` in frame B.

use std::io::{Read, Write};
use std::path::Path;

pub use freesynth_flow::{FLO_MAGIC, FlowField};

use crate::error::Error;

/// Writes a flow field in the `.flo` format.
pub fn write_flo(flow: &FlowField, w: impl Write) -> Result<(), Error> {
    flow.write_flo(w).map_err(io_error)
}

/// Reads a flow field in the `.flo` format.
pub fn read_flo(r: impl Read) -> Result<FlowField, Error> {
    FlowField::read_flo(r).map_err(io_error)
}

/// Saves a flow field as a `.flo` file.
pub fn save_flo(flow: &FlowField, path: impl AsRef<Path>) -> Result<(), Error> {
    flow.save(path).map_err(Error::Io)
}

/// Loads a `.flo` file.
pub fn load_flo(path: impl AsRef<Path>) -> Result<FlowField, Error> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).map_err(|e| Error::Invalid(format!("cannot open '{}': {e}", path.display())))?;
    read_flo(std::io::BufReader::new(file)).map_err(|e| Error::Invalid(format!("'{}': {e}", path.display())))
}

/// Malformed data becomes [`Error::Invalid`]; other I/O errors stay [`Error::Io`].
fn io_error(e: std::io::Error) -> Error {
    match e.kind() {
        std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput | std::io::ErrorKind::UnexpectedEof => {
            Error::Invalid(e.to_string())
        }
        _ => Error::Io(e),
    }
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
        let flow = FlowField::from_vec(
            3,
            2,
            vec![[0.0, 0.0], [1.5, -2.25], [-0.125, 1e10], [3.0, 4.0], [f32::MIN_POSITIVE, 0.0], [7.0, -8.0]],
        );
        let mut buf = Vec::new();
        write_flo(&flow, &mut buf).unwrap();
        assert_eq!(buf.len(), 12 + 6 * 8);
        assert_eq!(&buf[..4], b"PIEH");
        assert_eq!(&buf[4..12], &[3, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(read_flo(buf.as_slice()).unwrap(), flow);

        assert!(matches!(read_flo(&buf[..buf.len() - 4]), Err(Error::Invalid(_))));
        let mut long = buf.clone();
        long.extend_from_slice(&[0; 8]);
        assert!(matches!(read_flo(long.as_slice()), Err(Error::Invalid(_))));
        let mut bad = buf.clone();
        bad[0] ^= 1;
        assert!(matches!(read_flo(bad.as_slice()), Err(Error::Invalid(_))));
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
