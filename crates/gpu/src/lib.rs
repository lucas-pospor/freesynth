//! FreeSynth GPU backend: example-based synthesis with wgpu compute shaders.
//!
//! A WGSL port of the public-domain 2019 ebsynth CUDA engine (`ebsynth_cuda.cu`). It runs on any
//! adapter wgpu supports (Vulkan, Metal, DX12) and takes the same [`SynthesisInput`] and
//! [`Params`] as the CPU engine in `freesynth-core`.
//!
//! The algorithm is the CUDA engine's: a coarse-to-fine pyramid where every level alternates
//! parallel PatchMatch with patch voting. Every target pixel is processed by its own GPU thread;
//! the uniformity term's occupancy map is shared through atomics, so (like the original) results
//! vary slightly from run to run. Two PatchMatch variants are available
//! ([`SearchMode`](freesynth_core::SearchMode)):
//! - the compat search (the CUDA engine's): jump-flood-like propagation at offsets 4, 2, 1,
//!   followed by random search at radii 1, 2, 4, ... below half the source size;
//! - the improved search (every preset except compat): two checkerboard (red-black) half-passes
//!   per PatchMatch iteration, in which half of the pixels try their four direct neighbours'
//!   matches and then do their random search. The neighbours have the other colour, so they are
//!   stable during the pass and carry the other half-pass's fresh results, random search
//!   included (Gauss-Seidel instead of Jacobi). It reaches the same patch error with a third of
//!   the propagation work and half the dispatches.
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
//! [`Params::bit_exact`] only affect the CPU engine (`bit_exact` changes how the CPU sums patch
//! errors and nothing else; vote rounding comes from the preset or
//! [`Params::vote_rounding`]). Presets resolve with
//! [`Engine::Gpu`](freesynth_core::Engine::Gpu) (see [`Params::schedule`]), which differs from the
//! CPU only in [`Preset::Fast`](freesynth_core::Preset::Fast).
//!
//! # Sharing a device
//!
//! [`GpuContext::new`] opens its own device. To run on a device that other GPU work also uses
//! (an optical-flow engine, a GUI), create it with [`GpuContext::required_limits`] (or
//! [`GpuContext::required_limits_from`] to keep other raised limits) and wrap it with
//! [`GpuContext::from_device`]. Work is submitted in command buffers of bounded size, so a
//! long synthesis does not trip driver timeouts and leaves gaps for other work on the queue.
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

use std::sync::Mutex;

use freesynth_core::{Params, SynthesisInput, SynthesisOutput};

pub use crate::error::GpuError;

/// Number of storage buffers the synthesis kernels bind at once. wgpu's default limit is 8.
pub const REQUIRED_STORAGE_BUFFERS: u32 = 12;

/// Invocations per workgroup of the largest kernel (16 x 16, the resampling kernel).
const REQUIRED_INVOCATIONS: u32 = 256;
/// Workgroup edge length of the largest kernel.
const REQUIRED_WORKGROUP_SIZE: u32 = 16;

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
/// initialisation), so reuse it for several syntheses; it also keeps the compiled shader
/// pipelines, so only the first synthesis with a given channel layout and patch size compiles
/// them.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    /// Estimated work per command buffer (see `engine::Submitter`).
    submission_budget: u64,
    /// Pipelines compiled by earlier syntheses.
    pipelines: Mutex<engine::PipelineCache>,
}

/// Checks that `limits` (of a device) allow the synthesis kernels to run. Returns every shortfall
/// in one message.
fn check_limits(limits: &wgpu::Limits) -> Result<(), String> {
    let checks = [
        ("max_storage_buffers_per_shader_stage", limits.max_storage_buffers_per_shader_stage, REQUIRED_STORAGE_BUFFERS),
        ("max_compute_invocations_per_workgroup", limits.max_compute_invocations_per_workgroup, REQUIRED_INVOCATIONS),
        ("max_compute_workgroup_size_x", limits.max_compute_workgroup_size_x, REQUIRED_WORKGROUP_SIZE),
        ("max_compute_workgroup_size_y", limits.max_compute_workgroup_size_y, REQUIRED_WORKGROUP_SIZE),
        ("max_compute_workgroups_per_dimension", limits.max_compute_workgroups_per_dimension, 1),
        ("max_bind_groups", limits.max_bind_groups, 1),
        ("max_dynamic_uniform_buffers_per_pipeline_layout", limits.max_dynamic_uniform_buffers_per_pipeline_layout, 1),
    ];
    let missing: Vec<String> =
        checks.iter().filter(|(_, have, need)| have < need).map(|(name, have, need)| format!("{name} is {have}, needs {need}")).collect();
    if missing.is_empty() { Ok(()) } else { Err(missing.join("; ")) }
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
        check_limits(&available).map_err(|m| GpuError::Device(format!("adapter '{}' is too limited: {m}", info.name)))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("freesynth"),
            required_limits: Self::required_limits(&available),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| GpuError::Device(e.to_string()))?;
        Ok(Self::wrap(device, queue, info))
    }

    fn wrap(device: wgpu::Device, queue: wgpu::Queue, info: wgpu::AdapterInfo) -> Self {
        Self {
            device,
            queue,
            info,
            submission_budget: engine::DEFAULT_SUBMISSION_BUDGET,
            pipelines: Mutex::new(engine::PipelineCache::default()),
        }
    }

    /// The device limits to request for a device the engine will use, given the adapter's limits
    /// (`adapter.limits()`). Pass them, or larger ones, as `required_limits` when creating a
    /// device for [`from_device`](Self::from_device).
    ///
    /// These are wgpu's defaults raised where the engine needs more; see
    /// [`required_limits_from`](Self::required_limits_from) to raise other base limits instead.
    pub fn required_limits(adapter_limits: &wgpu::Limits) -> wgpu::Limits {
        Self::required_limits_from(&wgpu::Limits::default(), adapter_limits)
    }

    /// `base` with every limit the engine depends on raised to what it needs, given the adapter's
    /// limits (`adapter.limits()`). Use it when the device also serves other work with its own
    /// limits (a GUI that needs larger textures), or to start from
    /// `wgpu::Limits::downlevel_defaults()` on adapters that do not reach wgpu's defaults. Each
    /// raised field is the larger of `base` and the engine's need; all others are `base`'s.
    ///
    /// Raised fields:
    /// - `max_storage_buffers_per_shader_stage`: [`REQUIRED_STORAGE_BUFFERS`] (the default is 8,
    ///   too few for the kernels).
    /// - `max_compute_invocations_per_workgroup` (256), `max_compute_workgroup_size_x` and `_y`
    ///   (16), `max_bind_groups` and `max_dynamic_uniform_buffers_per_pipeline_layout` (1): the
    ///   kernels' fixed needs, all within wgpu's defaults.
    /// - `max_storage_buffer_binding_size` and `max_buffer_size`: the adapter's maximum. The
    ///   defaults (128 and 256 MiB) cap the image size: each buffer holds one image level, and
    ///   a larger image fails with [`GpuError::TooLarge`].
    /// - `max_compute_workgroup_storage_size`: the adapter's maximum. Kernels cache target
    ///   patches in workgroup memory when it fits and read global memory otherwise, so less
    ///   only costs speed.
    pub fn required_limits_from(base: &wgpu::Limits, adapter_limits: &wgpu::Limits) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: base.max_storage_buffers_per_shader_stage.max(REQUIRED_STORAGE_BUFFERS),
            max_compute_invocations_per_workgroup: base.max_compute_invocations_per_workgroup.max(REQUIRED_INVOCATIONS),
            max_compute_workgroup_size_x: base.max_compute_workgroup_size_x.max(REQUIRED_WORKGROUP_SIZE),
            max_compute_workgroup_size_y: base.max_compute_workgroup_size_y.max(REQUIRED_WORKGROUP_SIZE),
            max_compute_workgroups_per_dimension: base.max_compute_workgroups_per_dimension.max(1),
            max_bind_groups: base.max_bind_groups.max(1),
            max_dynamic_uniform_buffers_per_pipeline_layout: base.max_dynamic_uniform_buffers_per_pipeline_layout.max(1),
            max_storage_buffer_binding_size: base
                .max_storage_buffer_binding_size
                .max(adapter_limits.max_storage_buffer_binding_size),
            max_buffer_size: base.max_buffer_size.max(adapter_limits.max_buffer_size),
            max_compute_workgroup_storage_size: base
                .max_compute_workgroup_storage_size
                .max(adapter_limits.max_compute_workgroup_storage_size),
            ..base.clone()
        }
    }

    /// Uses a device and queue created elsewhere, so that the engine can share one device with
    /// other GPU work. Request the device with [`required_limits`](Self::required_limits).
    ///
    /// Fails with [`GpuError::Device`] if the device's limits are below what the kernels need
    /// (most often `max_storage_buffers_per_shader_stage` left at the default of 8). Buffer size
    /// limits are checked per synthesis, against the image size.
    ///
    /// Each synthesis pushes and pops its own error scopes and submits its own command buffers;
    /// it never waits for work other than its own readback, but the final readback polls the
    /// device, which also completes other users' callbacks.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, GpuError> {
        let info = device.adapter_info();
        check_limits(&device.limits()).map_err(|m| {
            GpuError::Device(format!(
                "the device lacks limits the engine needs ({m}); create it with GpuContext::required_limits"
            ))
        })?;
        Ok(Self::wrap(device, queue, info))
    }

    /// The device this context runs on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue this context submits to.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Description of the adapter in use.
    pub fn adapter_name(&self) -> String {
        describe(&self.info)
    }

    /// Runs example-based synthesis on the GPU (`ebsynthRun` with the CUDA backend).
    pub fn synthesize(&self, input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, GpuError> {
        engine::synthesize(self, input, params).map(|(out, _)| out)
    }
}

/// Convenience wrapper: opens a [`GpuContext`] and runs one synthesis.
pub fn synthesize(input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, GpuError> {
    GpuContext::new()?.synthesize(input, params)
}

#[cfg(test)]
mod tests {
    use freesynth_core::{
        GlibcRand, Image, Params, PerLevel, Preset, SearchMode, SynthesisInput, VoteMode, VoteRounding,
        nnf_init_random, nnf_patch_errors, vote_plain,
    };

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

    /// Without the uniformity term, a target equal to the source has the identity field as its
    /// exact solution. Both search variants must find it nearly everywhere in the interior (the
    /// edge rows and columns are covered by `every_pixel_is_searched`).
    #[test]
    fn both_search_modes_find_the_identity_field() {
        let Some(ctx) = context() else { return };
        let (w, h) = (75, 61);
        let input = identity_input(&test_image(w, h, 0.0));
        for search in [SearchMode::Compat, SearchMode::Improved] {
            let params = Params { uniformity_weight: 0.0, search: Some(search), ..Params::default() };
            let out = ctx.synthesize(&input, &params).unwrap();
            let interior: Vec<(usize, usize)> = (2..h - 2).flat_map(|y| (2..w - 2).map(move |x| (x, y))).collect();
            let exact = interior.iter().filter(|&&(x, y)| out.nnf.get(x, y) == [x as i32, y as i32]).count();
            assert!(exact * 100 >= interior.len() * 98, "{search}: {exact} of {} interior pixels", interior.len());
        }
    }

    /// Every pixel, the first and last row and column included, takes part in the search: after
    /// one level with a single search/vote and PatchMatch iteration, almost no pixel may still
    /// hold its random initial match. The sizes leave partial workgroups in both directions,
    /// an unpaired last column in the checkerboard half-passes, and rows of odd length.
    #[test]
    fn every_pixel_is_searched() {
        let Some(ctx) = context() else { return };
        for (w, h, patch_size) in [(75, 61, 5), (16, 17, 5), (33, 9, 3), (31, 47, 7)] {
            let input = identity_input(&test_image(w, h, 0.0));
            for search in [SearchMode::Compat, SearchMode::Improved] {
                let params = Params {
                    patch_size,
                    uniformity_weight: 0.0,
                    num_pyramid_levels: Some(1),
                    search_vote_iters: Some(PerLevel::Uniform(1)),
                    patch_match_iters: Some(PerLevel::Uniform(1)),
                    search: Some(search),
                    ..Params::default()
                };
                let out = ctx.synthesize(&input, &params).unwrap();
                let init = nnf_init_random((w, h), (w, h), patch_size, &mut GlibcRand::new(params.seed));
                let kept = |x: usize, y: usize| out.nnf.get(x, y) == init.get(x, y);
                // A pixel the search skipped keeps its initial match; a searched one rarely does
                // (at most 13% of a row or column, 3% of all pixels at these sizes).
                let case = format!("{w}x{h}, patch {patch_size}, {search} search");
                for x in 0..w {
                    let n = (0..h).filter(|&y| kept(x, y)).count();
                    assert!(4 * n <= h, "{case}: column {x} kept {n} of {h} initial matches");
                }
                for y in 0..h {
                    let n = (0..w).filter(|&x| kept(x, y)).count();
                    assert!(4 * n <= w, "{case}: row {y} kept {n} of {w} initial matches");
                }
                let total = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).filter(|&(x, y)| kept(x, y)).count();
                assert!(20 * total <= w * h, "{case}: {total} of {} initial matches kept", w * h);
            }
        }
    }

    #[test]
    fn other_modes_and_channel_layouts() {
        let Some(ctx) = context() else { return };
        let img = test_image(64, 64, 0.0);
        let mut input = identity_input(&img);
        for preset in Preset::ALL {
            let params = Params {
                vote_mode: Some(VoteMode::Weighted),
                use_stop_threshold: Some(true),
                extra_pass_3x3: true,
                ..Params::with_preset(preset)
            };
            let out = ctx.synthesize(&input, &params).unwrap();
            assert!(psnr(&out.image, &img) > 35.0, "{preset}");
            let out = ctx.synthesize(&input, &Params::with_preset(preset)).unwrap();
            assert!(psnr(&out.image, &img) > 35.0, "{preset}");
        }

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
            let params = Params { vote_mode: Some(vote_mode), ..Params::default() };
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

    fn shifted_input() -> SynthesisInput {
        let src = test_image(120, 90, 0.0);
        SynthesisInput {
            source_style: src.clone(),
            source_guide: src,
            target_guide: test_image(120, 90, 17.0),
            target_modulation: None,
            style_weights: vec![1.0 / 3.0; 3],
            guide_weights: vec![1.0 / 3.0; 3],
        }
    }

    /// Votes of the initial random field (no search): the GPU must reproduce the CPU's votes
    /// exactly, with either rounding.
    #[test]
    fn votes_match_cpu_in_both_rounding_modes() {
        let Some(ctx) = context() else { return };
        let input = shifted_input();
        for rounding in [VoteRounding::Truncate, VoteRounding::Nearest] {
            let params = Params {
                num_pyramid_levels: Some(1),
                search_vote_iters: Some(PerLevel::Uniform(0)),
                vote_rounding: Some(rounding),
                seed: 9,
                ..Params::default()
            };
            let out = ctx.synthesize(&input, &params).unwrap();
            let nnf = nnf_init_random((120, 90), (120, 90), 5, &mut GlibcRand::new(9));
            assert_eq!(out.nnf, nnf);
            let mut want = Image::new(120, 90, 3);
            vote_plain(&mut want, &input.source_style, &nnf, 5, rounding);
            assert_eq!(out.image, want, "{rounding:?}");
        }
    }

    /// One weighted vote after the initial field (no PatchMatch): the GPU's weights and rounding
    /// must agree with the CPU's up to float summation order, so values differ by at most 1 and
    /// only rarely.
    #[test]
    fn weighted_votes_match_cpu_in_both_rounding_modes() {
        let Some(ctx) = context() else { return };
        let input = shifted_input();
        for rounding in [VoteRounding::Truncate, VoteRounding::Nearest] {
            let params = Params {
                num_pyramid_levels: Some(1),
                search_vote_iters: Some(PerLevel::Uniform(1)),
                patch_match_iters: Some(PerLevel::Uniform(0)),
                vote_mode: Some(VoteMode::Weighted),
                vote_rounding: Some(rounding),
                seed: 9,
                num_threads: Some(1),
                ..Params::default()
            };
            let gpu = ctx.synthesize(&input, &params).unwrap();
            let cpu = freesynth_core::synthesize(&input, &params).unwrap();
            assert_eq!(gpu.nnf, cpu.nnf);
            let diffs: Vec<i32> =
                gpu.image.data().iter().zip(cpu.image.data()).map(|(&g, &c)| g as i32 - c as i32).collect();
            let max = diffs.iter().map(|d| d.abs()).max().unwrap();
            let differing = diffs.iter().filter(|&&d| d != 0).count() as f64 / diffs.len() as f64;
            let mean = diffs.iter().sum::<i32>() as f64 / diffs.len() as f64;
            println!("{rounding:?}: max |diff| {max}, {:.4}% differ, mean {mean:+.5}", 100.0 * differing);
            assert!(max <= 1, "{rounding:?}: max |diff| {max}");
            assert!(differing < 1e-3, "{rounding:?}: {:.4}% of values differ", 100.0 * differing);
            assert!(mean.abs() < 0.01, "{rounding:?}: mean diff {mean}");
        }
    }

    /// The per-pixel errors the GPU returns are the patch errors of its own output.
    #[test]
    fn returned_errors_match_the_cpu_evaluator() {
        let Some(ctx) = context() else { return };
        let input = shifted_input();
        for extra_pass_3x3 in [false, true] {
            let params = Params { return_errors: true, extra_pass_3x3, ..Params::default() };
            let out = ctx.synthesize(&input, &params).unwrap();
            let e = nnf_patch_errors(&input, &params, &out.image, &out.nnf).unwrap();
            let errors = out.errors.as_deref().expect("errors requested");
            assert_eq!(errors.len(), 120 * 90);
            for (i, &v) in errors.iter().enumerate() {
                let want = e.style[i] + e.guide[i];
                assert!((v - want).abs() <= 1e-3 * want.max(1.0), "pixel {i}: GPU {v}, CPU {want}");
            }
            assert!(ctx.synthesize(&input, &Params { return_errors: false, ..params }).unwrap().errors.is_none());
        }
    }

    /// A tiny budget splits the work into many command buffers, a huge one into very few; both
    /// give a valid synthesis. Without the uniformity term the GPU engine is deterministic (no
    /// races on the occupancy map), so there the split must not change a single value: a
    /// dropped or reordered dispatch would.
    #[test]
    fn submissions_follow_the_budget() {
        let Some(mut ctx) = context() else { return };
        let input = shifted_input();
        let params = Params::default();
        let mut counts = Vec::new();
        for budget in [1, u64::MAX] {
            ctx.submission_budget = budget;
            let (out, submissions) = engine::synthesize(&ctx, &input, &params).unwrap();
            let p = psnr(&out.image, &freesynth_core::synthesize(&input, &params).unwrap().image);
            assert!(p > 25.0, "budget {budget}: GPU vs CPU PSNR {p:.2} dB");
            counts.push(submissions);
        }
        println!("submissions: {counts:?}");
        assert_eq!(counts[1], 1);
        assert!(counts[0] > 100, "{counts:?}");

        for (search, extra_pass_3x3) in [(SearchMode::Improved, false), (SearchMode::Improved, true), (SearchMode::Compat, false)] {
            let params = Params { uniformity_weight: 0.0, extra_pass_3x3, search: Some(search), ..Params::default() };
            let mut outputs = Vec::new();
            for budget in [u64::MAX, 1, 2_000_000] {
                ctx.submission_budget = budget;
                let (out, submissions) = engine::synthesize(&ctx, &input, &params).unwrap();
                outputs.push((budget, submissions, out));
            }
            let (_, _, reference) = &outputs[0];
            for (budget, submissions, out) in &outputs[1..] {
                let case = format!("budget {budget} ({submissions} submissions), {search} search, extra pass {extra_pass_3x3}");
                assert_eq!(out.nnf, reference.nnf, "{case}");
                assert_eq!(out.image, reference.image, "{case}");
            }
            let n: Vec<usize> = outputs.iter().map(|o| o.1).collect();
            assert!(n[0] < n[2] && n[2] < n[1], "submissions per budget: {n:?}");
        }
    }

    #[test]
    fn shares_an_external_device() {
        let Some(ctx) = context() else { return };
        let adapter = pollster::block_on(instance().request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        // A device with wgpu's default limits has too few storage buffers.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        match GpuContext::from_device(device, queue) {
            Err(GpuError::Device(msg)) => assert!(msg.contains("max_storage_buffers_per_shader_stage"), "{msg}"),
            other => panic!("expected a limits error, got {:?}", other.err()),
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: GpuContext::required_limits(&adapter.limits()),
            ..Default::default()
        }))
        .unwrap();
        let shared = GpuContext::from_device(device.clone(), queue).unwrap();
        assert!(!shared.adapter_name().is_empty());
        let input = shifted_input();
        let out = shared.synthesize(&input, &Params::default()).unwrap();
        let reference = ctx.synthesize(&input, &Params::default()).unwrap();
        assert!(psnr(&out.image, &reference.image) > 25.0);
        // The caller's handle still works afterwards.
        let _ = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    }

    /// Later syntheses reuse the compiled pipelines; a new layout or patch size adds entries.
    #[test]
    fn pipelines_are_cached_per_layout_and_patch_size() {
        let Some(ctx) = context() else { return };
        let input = shifted_input();
        let first = ctx.synthesize(&input, &Params::default()).unwrap();
        let count = |ctx: &GpuContext| ctx.pipelines.lock().unwrap().len();
        assert_eq!(count(&ctx), 1);
        let again = ctx.synthesize(&input, &Params { seed: 2, ..Params::default() }).unwrap();
        assert_eq!(count(&ctx), 1);
        assert!(psnr(&first.image, &again.image) > 25.0);
        ctx.synthesize(&input, &Params { patch_size: 3, ..Params::default() }).unwrap();
        ctx.synthesize(&input, &Params { extra_pass_3x3: true, ..Params::default() }).unwrap();
        assert_eq!(count(&ctx), 2);
    }

    #[test]
    fn required_limits_raise_a_callers_base() {
        let adapter = wgpu::Limits { max_buffer_size: 1 << 32, ..wgpu::Limits::default() };
        let base = wgpu::Limits { max_texture_dimension_2d: 16384, ..wgpu::Limits::downlevel_defaults() };
        let l = GpuContext::required_limits_from(&base, &adapter);
        assert!(check_limits(&l).is_ok(), "{:?}", check_limits(&l));
        assert_eq!(l.max_texture_dimension_2d, 16384);
        assert_eq!(l.max_buffer_size, 1 << 32);
        assert_eq!(l.max_storage_buffers_per_shader_stage, REQUIRED_STORAGE_BUFFERS);
        // Fields the caller already set higher are kept.
        let big = wgpu::Limits { max_storage_buffers_per_shader_stage: 16, ..wgpu::Limits::default() };
        assert_eq!(GpuContext::required_limits_from(&big, &adapter).max_storage_buffers_per_shader_stage, 16);
    }

    #[test]
    fn limit_check_lists_every_shortfall() {
        assert!(check_limits(&GpuContext::required_limits(&wgpu::Limits::default())).is_ok());
        let msg = check_limits(&wgpu::Limits { max_bind_groups: 0, ..wgpu::Limits::default() }).unwrap_err();
        assert!(msg.contains("max_storage_buffers_per_shader_stage is 8, needs 12"), "{msg}");
        assert!(msg.contains("max_bind_groups is 0"), "{msg}");
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
