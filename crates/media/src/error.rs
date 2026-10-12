use std::fmt;
use std::path::PathBuf;

/// Errors of the media layer. New kinds of errors may be added.
#[derive(Debug)]
#[non_exhaustive]
pub enum MediaError {
    /// No usable ffmpeg (or ffprobe) was found. The message says where we looked and how to fix
    /// it.
    FfmpegMissing(String),
    /// ffmpeg or ffprobe ran but failed; holds what it printed on stderr.
    Ffmpeg { what: String, stderr: String },
    /// ffmpeg lacks an encoder the output needs.
    EncoderMissing { encoder: String, hint: String },
    /// The file holds no video stream, or one we cannot use.
    Unsupported(String),
    /// Reading or writing a file (or a pipe) failed.
    Io { path: PathBuf, source: std::io::Error },
    /// An image could not be encoded or decoded.
    Image { path: PathBuf, message: String },
    /// Invalid settings or arguments.
    Invalid(String),
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaError::FfmpegMissing(msg) => f.write_str(msg),
            MediaError::Ffmpeg { what, stderr } => {
                let stderr = stderr.trim();
                if stderr.is_empty() { write!(f, "{what} failed") } else { write!(f, "{what} failed: {stderr}") }
            }
            MediaError::EncoderMissing { encoder, hint } => {
                write!(f, "this ffmpeg has no '{encoder}' encoder; {hint}")
            }
            MediaError::Unsupported(msg) => f.write_str(msg),
            MediaError::Io { path, source } => write!(f, "'{}': {source}", path.display()),
            MediaError::Image { path, message } => write!(f, "'{}': {message}", path.display()),
            MediaError::Invalid(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for MediaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MediaError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Attaches a path to an I/O error.
pub(crate) fn io_error(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> MediaError {
    let path = path.into();
    move |source| MediaError::Io { path, source }
}
