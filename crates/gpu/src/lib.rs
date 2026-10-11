//! FreeSynth GPU backend: example-based synthesis with wgpu compute shaders.
//!
//! A WGSL port of the public-domain 2019 ebsynth CUDA engine (`ebsynth_cuda.cu`). It runs on any
//! adapter wgpu supports (Vulkan, Metal, DX12) and takes the same [`SynthesisInput`] and
//! [`Params`] as the CPU engine in `freesynth-core`.
//!
//! The algorithm is the CUDA engine's: a coarse-to-fine pyramid where every level alternates
//! parallel PatchMatch (jump-flood-like propagation at offsets 4, 2, 1 followed by random search
//! at radii 1, 2, 4, ...) with patch voting. Every target pixel is processed by its own GPU
//! thread; the uniformity term's occupancy map is shared through atomics, so (like the original)
//! results vary slightly from run to run.
//!
//! Differences from the CUDA engine, chosen for consistency with the CPU engine:
//! - The initial random field comes from the same generator as the CPU engine (glibc `rand()`
//!   seeded with [`Params::seed`]), so both backends start from the same field.
//! - The stop-threshold mask is only used when [`Params::use_stop_threshold`] is set (the CUDA
//!   engine always used it).
//! - Plain voting skips patches that fall outside the target or source, like the CPU engine,
//!   instead of reading clamped texels. (Weighted voting, which only the CUDA engine had, clamps
//!   like the CUDA engine, in both backends.)
//! - The per-pixel random generator is a 32-bit PCG seeded from [`Params::seed`].
//!
//! [`Params::num_threads`], [`Params::num_bands`], [`Params::omega_sync`] and
//! [`Params::bit_exact`] only affect the CPU engine.
//!
//! ```no_run
//! use freesynth_core::{io, GuideLayer, Params, SynthesisInput};
//!
//! let style = io::load_rgba("style.png")?;
//! let guide = GuideLayer::new(io::load_rgba("src.png")?, io::load_rgba("tgt.png")?);
//! let input = SynthesisInput::from_rgba_layers(&style, None, &[guide])?;
//! let ctx = freesynth_gpu::GpuContext::new()?;
//! let out = ctx.synthesize(&input, &Params::default())?;
//! io::save_png(&out.image, "out.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod engine;
mod error;

use freesynth_core::{Params, SynthesisInput, SynthesisOutput};

pub use crate::error::GpuError;

/// Number of storage buffers the synthesis kernels bind at once.
const REQUIRED_STORAGE_BUFFERS: u32 = 12;

fn instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
}

/// Human-readable descriptions of all GPU adapters wgpu can see.
pub fn list_adapters() -> Vec<String> {
    pollster::block_on(instance().enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .map(|a| describe(&a.get_info()))
        .collect()
}

fn describe(i: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?}, {:?}, driver {})", i.name, i.device_type, i.backend, i.driver_info)
}

/// A GPU device ready to run syntheses. Creating one takes some time (adapter and device
/// initialisation), so reuse it for several syntheses.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Opens the default (high-performance) adapter.
    pub fn new() -> Result<Self, GpuError> {
        let instance = instance();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|_| GpuError::NoAdapter)?;
        let info = adapter.get_info();
        let available = adapter.limits();
        if available.max_storage_buffers_per_shader_stage < REQUIRED_STORAGE_BUFFERS {
            return Err(GpuError::Device(format!(
                "adapter '{}' supports only {} storage buffers per shader stage, {} are needed",
                info.name, available.max_storage_buffers_per_shader_stage, REQUIRED_STORAGE_BUFFERS
            )));
        }
        // Default limits, raised where the kernels need more: storage buffer count, the largest
        // buffers the adapter allows (for large images) and all of its workgroup memory (for the
        // target tile cache).
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: REQUIRED_STORAGE_BUFFERS,
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
            max_buffer_size: available.max_buffer_size,
            max_compute_workgroup_storage_size: available.max_compute_workgroup_storage_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("freesynth"),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| GpuError::Device(e.to_string()))?;
        Ok(Self { device, queue, info })
    }

    /// Description of the adapter in use.
    pub fn adapter_name(&self) -> String {
        describe(&self.info)
    }

    /// Runs example-based synthesis on the GPU (`ebsynthRun` with the CUDA backend).
    pub fn synthesize(&self, input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, GpuError> {
        engine::synthesize(self, input, params)
    }
}

/// Convenience wrapper: opens a [`GpuContext`] and runs one synthesis.
pub fn synthesize(input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, GpuError> {
    GpuContext::new()?.synthesize(input, params)
}

#[cfg(test)]
mod tests {
    use freesynth_core::{Image, Params, SynthesisInput, VoteMode};

    use super::*;

    #[test]
    fn lists_adapters() {
        for a in super::list_adapters() {
            println!("{a}");
        }
    }

    /// The GPU context, or `None` (test skipped) when the machine has no usable adapter.
    fn context() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(c) => {
                println!("adapter: {}", c.adapter_name());
                Some(c)
            }
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    /// A smooth but non-repetitive colour image.
    fn test_image(w: usize, h: usize, phase: f32) -> Image {
        let mut img = Image::new(w, h, 3);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 + phase, y as f32);
                let p = img.pixel_mut(x, y);
                p[0] = (128.0 + 100.0 * (fx * 0.11).sin() * (fy * 0.07).cos()) as u8;
                p[1] = (128.0 + 100.0 * (fx * 0.05 + fy * 0.09).sin()) as u8;
                p[2] = ((x * 3 + y * 5) % 256) as u8;
            }
        }
        img
    }

    fn psnr(a: &Image, b: &Image) -> f64 {
        assert_eq!((a.size(), a.channels()), (b.size(), b.channels()));
        let mse = a.data().iter().zip(b.data()).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>()
            / a.data().len() as f64;
        if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
    }

    fn identity_input(img: &Image) -> SynthesisInput {
        SynthesisInput {
            source_style: img.clone(),
            source_guide: img.clone(),
            target_guide: img.clone(),
            target_modulation: None,
            style_weights: vec![1.0 / 3.0; 3],
            guide_weights: vec![1.0 / 3.0; 3],
        }
    }

    #[test]
    fn identity_synthesis_reproduces_style() {
        let Some(ctx) = context() else { return };
        let img = test_image(96, 80, 0.0);
        let input = identity_input(&img);
        let out = ctx.synthesize(&input, &Params::default()).unwrap();
        assert_eq!(out.image.size(), (96, 80));
        assert_eq!((out.nnf.width(), out.nnf.height()), (96, 80));
        let p = psnr(&out.image, &img);
        assert!(p > 35.0, "PSNR {p:.2} dB");
    }

    #[test]
    fn other_modes_and_channel_layouts() {
        let Some(ctx) = context() else { return };
        let img = test_image(64, 64, 0.0);
        let mut input = identity_input(&img);
        let params = Params {
            vote_mode: VoteMode::Weighted,
            use_stop_threshold: true,
            extra_pass_3x3: true,
            ..Params::default()
        };
        let out = ctx.synthesize(&input, &params).unwrap();
        assert!(psnr(&out.image, &img) > 35.0);

        input.target_modulation = Some(Image::from_raw(64, 64, 3, vec![255; 64 * 64 * 3]));
        let out = ctx.synthesize(&input, &Params::default()).unwrap();
        assert!(psnr(&out.image, &img) > 35.0);

        // Multi-word and partial-word layouts: 5 style + 3 guide channels, 1 style + 9 guide.
        let five = Image::from_raw(64, 64, 5, img.data().chunks(3).flat_map(|p| [p[0], p[1], p[2], p[0], p[1]]).collect());
        let one = Image::from_raw(64, 64, 1, img.data().chunks(3).map(|p| p[1]).collect());
        let nine = Image::from_raw(64, 64, 9, img.data().chunks(3).flat_map(|p| [p[0], p[1], p[2]].repeat(3)).collect());
        for (style, guide) in [(&five, &img), (&one, &nine)] {
            let input = SynthesisInput {
                source_style: style.clone(),
                source_guide: guide.clone(),
                target_guide: guide.clone(),
                target_modulation: None,
                style_weights: vec![1.0 / style.channels() as f32; style.channels()],
                guide_weights: vec![1.0 / guide.channels() as f32; guide.channels()],
            };
            let out = ctx.synthesize(&input, &Params { patch_size: 3, ..Params::default() }).unwrap();
            assert_eq!(out.image.channels(), style.channels());
            let p = psnr(&out.image, style);
            assert!(p > 35.0, "{} style + {} guide channels: PSNR {p:.2} dB", style.channels(), guide.channels());
        }
    }

    /// A small non-trivial synthesis (target differs from the source) must stay close to the CPU
    /// engine's result.
    #[test]
    fn matches_cpu_backend() {
        let Some(ctx) = context() else { return };
        let src = test_image(120, 90, 0.0);
        let tgt = test_image(120, 90, 17.0);
        let input = SynthesisInput {
            source_style: src.clone(),
            source_guide: src,
            target_guide: tgt,
            target_modulation: None,
            style_weights: vec![1.0 / 3.0; 3],
            guide_weights: vec![1.0 / 3.0; 3],
        };
        for vote_mode in [VoteMode::Plain, VoteMode::Weighted] {
            let params = Params { vote_mode, ..Params::default() };
            let gpu = ctx.synthesize(&input, &params).unwrap();
            let cpu = freesynth_core::synthesize(&input, &params).unwrap();
            let p = psnr(&gpu.image, &cpu.image);
            println!("GPU vs CPU ({vote_mode:?}): {p:.2} dB");
            assert!(p > 25.0, "GPU vs CPU PSNR {p:.2} dB ({vote_mode:?})");
            for &[x, y] in gpu.nnf.data() {
                assert!((2..118).contains(&x) && (2..88).contains(&y), "NNF entry ({x}, {y}) out of range");
            }
        }
    }

    /// Source and target of different, odd sizes (partial workgroups, clamped tiles).
    #[test]
    fn different_source_and_target_sizes() {
        let Some(ctx) = context() else { return };
        let src = test_image(53, 41, 0.0);
        let tgt = test_image(31, 67, 5.0);
        let input = SynthesisInput {
            source_style: src.clone(),
            source_guide: src,
            target_guide: tgt,
            target_modulation: None,
            style_weights: vec![1.0; 3],
            guide_weights: vec![1.0; 3],
        };
        for patch_size in [3, 5, 7] {
            let params = Params { patch_size, extra_pass_3x3: patch_size == 7, ..Params::default() };
            let out = ctx.synthesize(&input, &params).unwrap();
            assert_eq!(out.image.size(), (31, 67));
            assert_eq!((out.nnf.width(), out.nnf.height()), (31, 67));
            let r = if params.extra_pass_3x3 { 1 } else { patch_size as i32 / 2 };
            for &[x, y] in out.nnf.data() {
                assert!((r..53 - r).contains(&x) && (r..41 - r).contains(&y), "({x}, {y}), patch {patch_size}");
            }
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let Some(ctx) = context() else { return };
        let img = test_image(16, 16, 0.0);
        let input = identity_input(&img);
        assert!(ctx.synthesize(&input, &Params { patch_size: 4, ..Params::default() }).is_err());
        assert!(ctx.synthesize(&input, &Params { patch_size: 9, ..Params::default() }).is_err());
        let mut bad = input.clone();
        bad.guide_weights.pop();
        assert!(ctx.synthesize(&bad, &Params::default()).is_err());
        let mut bad = input.clone();
        bad.guide_weights[0] = -1.0;
        assert!(ctx.synthesize(&bad, &Params::default()).is_err());
    }
}
