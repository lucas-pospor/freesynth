use std::fmt;

/// Errors from scene generation and scene file I/O.
#[derive(Debug)]
pub enum Error {
    /// The scene parameters are inconsistent (sizes, frame counts, keyframes, ...).
    InvalidParams(String),
    /// A scene description or manifest could not be parsed.
    Parse(String),
    /// Encoding an image failed.
    Image(image::ImageError),
    /// Reading or writing a file failed.
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParams(msg) | Error::Parse(msg) => f.write_str(msg),
            Error::Image(e) => write!(f, "{e}"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Image(e) => Some(e),
            Error::Io(e) => Some(e),
            Error::InvalidParams(_) | Error::Parse(_) => None,
        }
    }
}

impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Error::Image(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
