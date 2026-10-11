use std::fmt;

/// Errors of the flow estimators.
#[derive(Debug)]
pub enum FlowError {
    /// The inputs or parameters are inconsistent.
    InvalidInput(String),
    /// No suitable GPU adapter was found.
    NoAdapter,
    /// The adapter refused to create a device, or lacks a feature or limit that is needed.
    Device(String),
    /// wgpu reported an error (validation, out of memory, lost device, failed readback, ...).
    Gpu(String),
}

impl fmt::Display for FlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlowError::InvalidInput(msg) => f.write_str(msg),
            FlowError::NoAdapter => f.write_str("no GPU adapter found"),
            FlowError::Device(msg) => write!(f, "cannot use GPU device: {msg}"),
            FlowError::Gpu(msg) => write!(f, "GPU error: {msg}"),
        }
    }
}

impl std::error::Error for FlowError {}

/// Shorthand for an [`FlowError::InvalidInput`] result.
pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T, FlowError> {
    Err(FlowError::InvalidInput(msg.into()))
}
