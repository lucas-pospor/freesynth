use std::fmt;

/// Errors of the GPU backend.
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU adapter was found.
    NoAdapter,
    /// The adapter refused to create a device (e.g. it lacks a required limit).
    Device(String),
    /// The inputs or parameters are inconsistent.
    InvalidInput(String),
    /// The images do not fit into the device's buffer limits.
    TooLarge(String),
    /// wgpu reported an error (validation, out of memory, lost device, ...).
    Wgpu(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoAdapter => f.write_str("no GPU adapter found"),
            GpuError::Device(msg) => write!(f, "cannot create GPU device: {msg}"),
            GpuError::InvalidInput(msg) => f.write_str(msg),
            GpuError::TooLarge(msg) => write!(f, "images too large for the GPU: {msg}"),
            GpuError::Wgpu(msg) => write!(f, "GPU error: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Shorthand for an [`GpuError::InvalidInput`] result.
pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T, GpuError> {
    Err(GpuError::InvalidInput(msg.into()))
}
