//! The compute device: synthesis and optical flow on one shared GPU device, or on the CPU.

use freesynth_core::{Engine, Params, SynthesisInput, SynthesisOutput};
use freesynth_flow::cpu::CpuDis;
use freesynth_flow::{DisParams, FlowEstimator};

use crate::error::VideoError;

/// Which hardware to run on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// The GPU when one is usable, the CPU otherwise.
    #[default]
    Auto,
    /// The GPU; fails without one.
    Gpu,
    /// The CPU (rayon threads).
    Cpu,
}

impl Backend {
    /// The backend's lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Gpu => "gpu",
            Backend::Cpu => "cpu",
        }
    }
}

impl std::str::FromStr for Backend {
    type Err = VideoError;

    fn from_str(s: &str) -> Result<Self, VideoError> {
        match s {
            "auto" => Ok(Backend::Auto),
            "gpu" => Ok(Backend::Gpu),
            "cpu" => Ok(Backend::Cpu),
            _ => Err(VideoError::InvalidInput(format!("unknown backend '{s}' (expected auto, gpu or cpu)"))),
        }
    }
}

/// Runs one synthesis. Implemented by [`ComputeDevice`]; other implementations can wrap a
/// different engine.
pub trait Synthesizer: Sync {
    /// The engine kind presets are resolved for.
    fn engine(&self) -> Engine;

    /// Synthesizes `input`. Called from several threads at once.
    fn synthesize(&self, input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, VideoError>;
}

/// Creates optical flow estimators. Implemented by [`ComputeDevice`].
pub trait FlowProvider: Sync {
    /// A new estimator for `width` x `height` frames.
    fn flow_estimator(&self, width: usize, height: usize, params: &DisParams)
    -> Result<Box<dyn FlowEstimator + Send>, VideoError>;
}

/// The two engines' handles on one shared wgpu device.
#[cfg(feature = "gpu")]
struct GpuContexts {
    synth: freesynth_gpu::GpuContext,
    flow: freesynth_flow::gpu::GpuContext,
}

enum Kind {
    #[cfg(feature = "gpu")]
    Gpu(Box<GpuContexts>),
    Cpu,
}

/// The device a render runs on: one wgpu device shared by the synthesis engine and the flow
/// estimator (so that the engine's pipelines and both crates' work share one queue), or the CPU.
///
/// Create it once and reuse it: opening a GPU device and compiling the kernels takes a while.
pub struct ComputeDevice {
    kind: Kind,
    name: String,
    /// Why [`Backend::Auto`] fell back to the CPU.
    fallback_reason: Option<String>,
}

impl ComputeDevice {
    /// Opens the device for `backend`.
    pub fn new(backend: Backend) -> Result<Self, VideoError> {
        match backend {
            Backend::Cpu => Ok(Self::cpu()),
            Backend::Gpu => Self::gpu(),
            Backend::Auto => Ok(Self::gpu().unwrap_or_else(|e| Self { fallback_reason: Some(e.to_string()), ..Self::cpu() })),
        }
    }

    /// The CPU.
    pub fn cpu() -> Self {
        let threads = rayon::current_num_threads();
        Self { kind: Kind::Cpu, name: format!("CPU ({threads} threads)"), fallback_reason: None }
    }

    /// The default (high-performance) GPU adapter, with one device for synthesis and flow.
    #[cfg(feature = "gpu")]
    pub fn gpu() -> Result<Self, VideoError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|_| VideoError::Device("no GPU adapter found".into()))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("freesynth-video"),
            required_limits: freesynth_gpu::GpuContext::required_limits(&adapter.limits()),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| VideoError::Device(e.to_string()))?;
        let mut this = Self::from_wgpu(device, queue)?;
        this.name = format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend);
        Ok(this)
    }

    /// Without the `gpu` feature there is no GPU.
    #[cfg(not(feature = "gpu"))]
    pub fn gpu() -> Result<Self, VideoError> {
        Err(VideoError::Device("built without GPU support".into()))
    }

    /// Uses a device created elsewhere (a GUI's, for example). It must have the synthesis
    /// engine's limits: create it with `freesynth_gpu::GpuContext::required_limits`.
    #[cfg(feature = "gpu")]
    pub fn from_wgpu(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, VideoError> {
        let synth = freesynth_gpu::GpuContext::from_device(device.clone(), queue.clone())
            .map_err(|e| VideoError::Device(e.to_string()))?;
        let name = synth.adapter_name();
        let flow = freesynth_flow::gpu::GpuContext::from_device(device, queue);
        Ok(Self { kind: Kind::Gpu(Box::new(GpuContexts { synth, flow })), name, fallback_reason: None })
    }

    /// True when synthesis and flow run on a GPU.
    pub fn is_gpu(&self) -> bool {
        !matches!(self.kind, Kind::Cpu)
    }

    /// A description of the device.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// When the device was opened with [`Backend::Auto`] and no GPU could be used: why (for
    /// example "no GPU adapter found"), so that a host can tell the user. `None` otherwise.
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }
}

impl Synthesizer for ComputeDevice {
    fn engine(&self) -> Engine {
        if self.is_gpu() { Engine::Gpu } else { Engine::Cpu }
    }

    fn synthesize(&self, input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, VideoError> {
        match &self.kind {
            #[cfg(feature = "gpu")]
            Kind::Gpu(g) => g.synth.synthesize(input, params).map_err(|e| VideoError::Synthesis(e.to_string())),
            Kind::Cpu => freesynth_core::synthesize(input, params).map_err(|e| VideoError::Synthesis(e.to_string())),
        }
    }
}

impl FlowProvider for ComputeDevice {
    fn flow_estimator(
        &self,
        width: usize,
        height: usize,
        params: &DisParams,
    ) -> Result<Box<dyn FlowEstimator + Send>, VideoError> {
        match &self.kind {
            #[cfg(feature = "gpu")]
            Kind::Gpu(g) => Ok(Box::new(freesynth_flow::gpu::GpuDis::new(&g.flow, width, height, params)?)),
            Kind::Cpu => Ok(Box::new(CpuDis::new(width, height, params)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_round_trip() {
        for b in [Backend::Auto, Backend::Gpu, Backend::Cpu] {
            assert_eq!(b.name().parse::<Backend>().unwrap(), b);
        }
        assert!("cuda".parse::<Backend>().is_err());
    }

    #[test]
    fn cpu_device() {
        let d = ComputeDevice::new(Backend::Cpu).unwrap();
        assert!(!d.is_gpu());
        assert_eq!(d.fallback_reason(), None);
        assert_eq!(d.engine(), Engine::Cpu);
        assert!(d.flow_estimator(32, 32, &DisParams::default()).is_ok());
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn auto_explains_a_cpu_fallback() {
        let d = ComputeDevice::new(Backend::Auto).unwrap();
        assert!(!d.is_gpu());
        assert_eq!(d.fallback_reason(), Some("GPU device: built without GPU support"));
    }
}
