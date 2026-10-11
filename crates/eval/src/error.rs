use std::fmt;

/// Errors returned by the evaluation library.
#[derive(Debug)]
pub enum Error {
    /// Inputs are missing, inconsistent or malformed.
    Invalid(String),
    /// Reading or writing a file failed.
    Io(std::io::Error),
    /// An error from the synthesis library (image decoding, inconsistent synthesis inputs).
    Core(freesynth_core::Error),
    /// A report could not be parsed or written as JSON.
    Json(serde_json::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(msg) => f.write_str(msg),
            Error::Io(e) => write!(f, "{e}"),
            Error::Core(e) => write!(f, "{e}"),
            Error::Json(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Invalid(_) => None,
            Error::Io(e) => Some(e),
            Error::Core(e) => Some(e),
            Error::Json(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<freesynth_core::Error> for Error {
    fn from(e: freesynth_core::Error) -> Self {
        Error::Core(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

/// Shorthand for an [`Error::Invalid`] result.
pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(msg.into()))
}
