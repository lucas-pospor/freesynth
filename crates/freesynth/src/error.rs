use std::fmt;
use std::path::PathBuf;

use freesynth_media::MediaError;
use freesynth_video::VideoError;

/// Errors of the FreeSynth library. New kinds of errors may be added.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A project file could not be parsed, or holds inconsistent settings.
    Project(String),
    /// An imported project file (such as an EbSynth `.ebs` file) is malformed.
    Import(String),
    /// Reading or writing a file failed.
    Io { path: PathBuf, source: std::io::Error },
    /// Video or image I/O failed (ffmpeg missing or failing, an unreadable video, ...).
    Media(MediaError),
    /// Rendering failed (or was cancelled: [`Error::is_cancelled`]).
    Render(VideoError),
}

impl Error {
    /// True when the render stopped because it was cancelled.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Error::Render(VideoError::Cancelled))
    }

    /// True when no usable GPU was found or the GPU failed.
    pub fn is_device_error(&self) -> bool {
        matches!(self, Error::Render(VideoError::Device(_)))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Project(msg) => write!(f, "project: {msg}"),
            Error::Import(msg) => write!(f, "import: {msg}"),
            Error::Io { path, source } => write!(f, "'{}': {source}", path.display()),
            Error::Media(e) => write!(f, "{e}"),
            Error::Render(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            Error::Media(e) => Some(e),
            Error::Render(e) => Some(e),
            Error::Project(_) | Error::Import(_) => None,
        }
    }
}

impl From<VideoError> for Error {
    fn from(e: VideoError) -> Self {
        Error::Render(e)
    }
}

impl From<MediaError> for Error {
    fn from(e: MediaError) -> Self {
        Error::Media(e)
    }
}
