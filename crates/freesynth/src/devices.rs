//! What this machine offers: GPU adapters, CPU threads and ffmpeg.

/// A GPU adapter as seen by wgpu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuAdapter {
    /// The adapter's name, such as `AMD Radeon RX 9060 XT (RADV GFX1200)`.
    pub name: String,
    /// `DiscreteGpu`, `IntegratedGpu`, `VirtualGpu`, `Cpu` (a software rasterizer) or `Other`.
    pub device_type: String,
    /// The graphics API: `Vulkan`, `Metal`, `Dx12` or `Gl`.
    pub backend: String,
    /// Driver name and version.
    pub driver: String,
    /// True when the synthesis kernels can run on it.
    pub usable: bool,
    /// Why it cannot be used, when it cannot.
    pub problem: Option<String>,
    /// True for the adapter a GPU render picks (the high-performance default).
    pub default: bool,
}

/// The GPU adapters wgpu can see, on every backend. Empty without the `gpu` feature.
pub fn gpu_adapters() -> Vec<GpuAdapter> {
    #[cfg(feature = "gpu")]
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let default = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()
        .map(|a| a.get_info());
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .map(|a| {
                let info = a.get_info();
                let limits = a.limits();
                let compute = a.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);
                let need = freesynth_gpu::REQUIRED_STORAGE_BUFFERS;
                let problem = if !compute {
                    Some("no compute shaders".to_owned())
                } else if limits.max_storage_buffers_per_shader_stage < need {
                    Some(format!(
                        "{} storage buffers per shader stage, {need} needed",
                        limits.max_storage_buffers_per_shader_stage
                    ))
                } else {
                    None
                };
                let driver = [info.driver.as_str(), info.driver_info.as_str()]
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                GpuAdapter {
                    name: info.name.clone(),
                    device_type: format!("{:?}", info.device_type),
                    backend: format!("{:?}", info.backend),
                    driver,
                    usable: problem.is_none(),
                    problem,
                    default: default.as_ref() == Some(&info),
                }
            })
            .collect()
    }
    #[cfg(not(feature = "gpu"))]
    {
        Vec::new()
    }
}

/// True when this build can render on a GPU.
pub fn gpu_support() -> bool {
    cfg!(feature = "gpu")
}

/// Number of CPU threads the CPU backend uses.
pub fn cpu_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}
