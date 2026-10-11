use std::fmt;

/// Errors returned by the synthesis API and the image I/O helpers.
#[derive(Debug)]
pub enum Error {
    /// The inputs or parameters are inconsistent (sizes, channel counts, weights, ...).
    InvalidInput(String),
    /// Decoding or encoding an image failed.
    Image(image::ImageError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => f.write_str(msg),
            Error::Image(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Image(e) => Some(e),
            Error::InvalidInput(_) => None,
        }
    }
}

impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Error::Image(e)
    }
}

/// Shorthand for an [`Error::InvalidInput`] result.
pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error::InvalidInput(msg.into()))
}
