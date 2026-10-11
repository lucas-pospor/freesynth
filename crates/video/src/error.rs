use std::fmt;
use std::path::PathBuf;

/// Errors of the video pipeline.
#[derive(Debug)]
pub enum VideoError {
    /// The shot, the keyframes or the parameters are inconsistent.
    InvalidInput(String),
    /// A frame or keyframe file could not be read or decoded.
    Load { path: PathBuf, source: freesynth_core::Error },
    /// An output frame could not be encoded or written.
    Save { path: PathBuf, source: freesynth_core::Error },
    /// Reading or writing a cache, spill or output file failed.
    Io { path: PathBuf, source: std::io::Error },
    /// The synthesis engine failed.
    Synthesis(String),
    /// The optical flow engine failed.
    Flow(freesynth_flow::FlowError),
    /// No usable GPU, or the GPU device could not be created.
    Device(String),
    /// The render was cancelled through its [`CancelToken`](crate::CancelToken).
    Cancelled,
    /// A frame sink, a progress callback or an engine panicked during the render. The render
    /// stops and returns this instead of unwinding into the caller.
    Panicked(String),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::InvalidInput(msg) => f.write_str(msg),
            VideoError::Load { path, source } => write!(f, "cannot load '{}': {source}", path.display()),
            VideoError::Save { path, source } => write!(f, "cannot write '{}': {source}", path.display()),
            VideoError::Io { path, source } => write!(f, "'{}': {source}", path.display()),
            VideoError::Synthesis(msg) => write!(f, "synthesis failed: {msg}"),
            VideoError::Flow(e) => write!(f, "optical flow failed: {e}"),
            VideoError::Device(msg) => write!(f, "GPU device: {msg}"),
            VideoError::Cancelled => f.write_str("cancelled"),
            VideoError::Panicked(msg) => write!(f, "panicked: {msg}"),
        }
    }
}

impl std::error::Error for VideoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VideoError::Load { source, .. } | VideoError::Save { source, .. } => Some(source),
            VideoError::Io { source, .. } => Some(source),
            VideoError::Flow(e) => Some(e),
            _ => None,
        }
    }
}

impl From<freesynth_flow::FlowError> for VideoError {
    fn from(e: freesynth_flow::FlowError) -> Self {
        VideoError::Flow(e)
    }
}

/// A [`VideoError::Panicked`] describing a caught panic: `what` panicked, with the panic's
/// message when it has one.
pub(crate) fn panicked(what: &str, payload: &(dyn std::any::Any + Send)) -> VideoError {
    let msg = payload.downcast_ref::<&str>().copied().or_else(|| payload.downcast_ref::<String>().map(String::as_str));
    VideoError::Panicked(match msg {
        Some(m) => format!("{what}: {m}"),
        None => what.to_owned(),
    })
}

/// Shorthand for an [`VideoError::InvalidInput`] result.
pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T, VideoError> {
    Err(VideoError::InvalidInput(msg.into()))
}

/// Attaches a path to an I/O error.
pub(crate) fn io_error(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> VideoError {
    let path = path.into();
    move |source| VideoError::Io { path, source }
}
