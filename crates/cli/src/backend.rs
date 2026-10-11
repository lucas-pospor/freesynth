//! Backend selection and dispatch.

use std::fmt;

use freesynth_core::{Engine, Params, SynthesisInput, SynthesisOutput};

/// A synthesis backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Multithreaded CPU engine (`freesynth-core`), the reference implementation.
    Cpu,
    /// wgpu compute engine (`freesynth-gpu`).
    Gpu,
}

impl Backend {
    /// Parses a `-backend` argument. `cuda` is accepted as an alias of `gpu` for compatibility
    /// with ebsynth command lines.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cpu" => Some(Backend::Cpu),
            "gpu" | "cuda" => Some(Backend::Gpu),
            _ => None,
        }
    }

    /// The backend used when `-backend` is not given.
    pub fn default_backend() -> Self {
        Backend::Cpu
    }

    /// Whether the backend can be used on this machine.
    pub fn is_available(self) -> bool {
        match self {
            Backend::Cpu => true,
            Backend::Gpu => !freesynth_gpu::list_adapters().is_empty(),
        }
    }

    /// The engine kind presets are resolved for.
    pub fn engine(self) -> Engine {
        match self {
            Backend::Cpu => Engine::Cpu,
            Backend::Gpu => Engine::Gpu,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }

    /// Runs the synthesis on this backend.
    pub fn run(self, input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, BackendError> {
        match self {
            Backend::Cpu => freesynth_core::synthesize(input, params).map_err(BackendError::Core),
            Backend::Gpu => freesynth_gpu::synthesize(input, params).map_err(BackendError::Gpu),
        }
    }
}

/// Failure of a backend run.
#[derive(Debug)]
pub enum BackendError {
    Core(freesynth_core::Error),
    Gpu(freesynth_gpu::GpuError),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::Core(e) => write!(f, "{e}"),
            BackendError::Gpu(e) => write!(f, "{e}"),
        }
    }
}
