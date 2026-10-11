use std::fmt;
use std::path::PathBuf;

use freesynth_video::VideoError;

/// Errors of the FreeSynth library.
#[derive(Debug)]
pub enum Error {
    /// A project file could not be parsed, or holds inconsistent settings.
    Project(String),
    /// Reading or writing a file failed.
    Io { path: PathBuf, source: std::io::Error },
    /// Rendering failed (or was cancelled: [`Error::is_cancelled`]).
    Render(VideoError),
}

impl Error {
    /// True when the render stopped because it was cancelled.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Error::Render(VideoError::Cancelled))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Project(msg) => write!(f, "project: {msg}"),
            Error::Io { path, source } => write!(f, "'{}': {source}", path.display()),
            Error::Render(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            Error::Render(e) => Some(e),
            Error::Project(_) => None,
        }
    }
}

impl From<VideoError> for Error {
    fn from(e: VideoError) -> Self {
        Error::Render(e)
    }
}
