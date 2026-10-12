//! Buffers, pipelines and the per-level schedule (`ebsynthCuda` in the original).
//!
//! All buffers are allocated once at the finest level's size and reused by every pyramid level.
//! The schedule is recorded up front with a single readback at the end; the CPU never waits for
//! the GPU in between. Dispatches are grouped into command buffers by estimated cost (see
//! [`Submitter`]), so that small levels share a submission and no submission runs long enough to
//! trip a driver's GPU timeout. Per-level constants live in one uniform buffer, selected with
//! dynamic offsets.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, mpsc};
use std::time::Instant;

use freesynth_core::{
    Engine, GlibcRand, Image, Image16, Nnf, Params, Schedule, SearchMode, SynthesisExtras, SynthesisInput,
    SynthesisOutput, VoteMode, VoteRounding, nnf_init_random, pyramid_level_size, vote_plain16, vote_weighted16,
};
use wgpu::util::DeviceExt;

use crate::GpuContext;
use crate::error::GpuError;

/// Workgroup edge length of the synthesis kernels (WG x WG invocations).
const WG: u32 = 8;
/// Workgroup edge length of the resampling kernel (fixed in `resample.wgsl`).
const RESAMPLE_WG: u32 = 16;
/// Propagation offsets of one PatchMatch iteration of the compat search, as in the CUDA engine.
const PROP_RADII: [i32; 3] = [4, 2, 1];

const SYNTH_TEMPLATE: &str = include_str!("shaders/synth.wgsl");
const RESAMPLE_SHADER: &str = include_str!("shaders/resample.wgsl");

/// Size of the fixed part of the `Params` uniform (16 scalars).
const PARAMS_HEADER: usize = 64;
/// Size of the `ResampleParams` uniform.
const RESAMPLE_PARAMS: usize = 32;

/// Packed channel layout of a synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Layout {
    /// u32 words per style pixel.
    style_w: usize,
    /// u32 words per guide (and modulation) pixel.
    guide_w: usize,
    modulated: bool,
}

impl Layout {
    fn params_size(self) -> usize {
        PARAMS_HEADER + 16 * (self.style_w + self.guide_w)
    }
}

/// One pass over a pyramid level (the extra 3x3 pass repeats the finest level).
#[derive(Clone, Debug)]
struct Phase {
    level: usize,
    extra: bool,
    target: (usize, usize),
    source: (usize, usize),
    /// Target size of the previous (coarser) level, for NNF upscaling.
    prev_target: (usize, usize),
    patch_size: usize,
    lambda: f32,
}

impl Phase {
    fn use_omega(&self) -> bool {
        self.lambda != 0.0
    }
}

fn phases(params: &Params, num_levels: usize, source: (usize, usize), target: (usize, usize)) -> Vec<Phase> {
    debug_assert!(num_levels > 0);
    let mut out: Vec<Phase> = (0..num_levels)
        .map(|level| Phase {
            level,
            extra: false,
            target: pyramid_level_size(target, num_levels, level),
            source: pyramid_level_size(source, num_levels, level),
            prev_target: if level > 0 { pyramid_level_size(target, num_levels, level - 1) } else { (0, 0) },
            patch_size: params.patch_size,
            lambda: params.uniformity_weight,
        })
        .collect();
    if params.extra_pass_3x3 {
        let last = out[num_levels - 1].clone();
        out.push(Phase { extra: true, patch_size: 3, lambda: 0.0, ..last });
    }
    out
}

/// Packs an image's 8-bit channels four per u32 word (`words` words per pixel, zero-padded).
fn pack_words(img: &Image, words: usize) -> Vec<u32> {
    let nc = img.channels();
    let mut out = vec![0u32; img.width() * img.height() * words];
    for (px, o) in img.data().chunks_exact(nc).zip(out.chunks_exact_mut(words)) {
        for (c, &v) in px.iter().enumerate() {
            o[c / 4] |= (v as u32) << (8 * (c % 4));
        }
    }
    out
}

/// The source style and guide packed into one buffer: per pixel, the style words followed by
/// the guide words.
fn pack_source(input: &SynthesisInput, layout: Layout) -> Vec<u32> {
    let style = pack_words(&input.source_style, layout.style_w);
    let guide = pack_words(&input.source_guide, layout.guide_w);
    style
        .chunks_exact(layout.style_w)
        .zip(guide.chunks_exact(layout.guide_w))
        .flat_map(|(s, g)| s.iter().chain(g).copied())
        .collect()
}

/// Inverse of [`pack_words`].
fn unpack_words(words: &[u32], width: usize, height: usize, channels: usize, per_pixel: usize) -> Image {
    let mut img = Image::new(width, height, channels);
    for (px, w) in img.data_mut().chunks_exact_mut(channels).zip(words.chunks_exact(per_pixel)) {
        for (c, v) in px.iter_mut().enumerate() {
            *v = (w[c / 4] >> (8 * (c % 4))) as u8;
        }
    }
    img
}

/// The `Params` uniform of one phase (layout of `struct Params` in `synth.wgsl`).
fn params_bytes(ph: &Phase, layout: Layout, input: &SynthesisInput, params: &Params, sched: &Schedule) -> Vec<u8> {
    let (tw, th) = ph.target;
    let (sw, sh) = ph.source;
    let p = ph.patch_size;
    let ns = input.source_style.channels();
    let omega_best = ((tw * th) as f32 / (sw * sh) as f32) * (p * p) as f32;
    let mut b = Vec::with_capacity(layout.params_size());
    let mut int = |v: i64| b.extend_from_slice(&(v as i32).to_le_bytes());
    for v in [tw, th, sw, sh, ph.prev_target.0, ph.prev_target.1, p] {
        int(v as i64);
    }
    int(ph.use_omega() as i64);
    b.extend_from_slice(&ph.lambda.to_le_bytes());
    b.extend_from_slice(&omega_best.to_le_bytes());
    b.extend_from_slice(&sched.stop_threshold[ph.level].to_le_bytes());
    b.extend_from_slice(&params.seed.to_le_bytes());
    b.extend_from_slice(&u32::from(sched.vote_rounding == VoteRounding::Nearest).to_le_bytes());
    b.extend_from_slice(&((p * p * ns) as f32).to_le_bytes());
    b.extend_from_slice(&[0u8; 8]);
    debug_assert_eq!(b.len(), PARAMS_HEADER);
    for (weights, words) in [(&input.style_weights, layout.style_w), (&input.guide_weights, layout.guide_w)] {
        for i in 0..words * 4 {
            b.extend_from_slice(&weights.get(i).copied().unwrap_or(0.0).to_le_bytes());
        }
    }
    b
}

fn resample_bytes(input: (usize, usize), output: (usize, usize), words: usize) -> [u8; RESAMPLE_PARAMS] {
    let mut b = [0u8; RESAMPLE_PARAMS];
    for (i, v) in [input.0, input.1, output.0, output.1, words].into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&(v as u32).to_le_bytes());
    }
    // The same scale as the CPU resampler (an exactly rounded division).
    let scale = input.0 as f32 / output.0 as f32;
    b[20..24].copy_from_slice(&scale.to_le_bytes());
    b
}

/// Bind group and pipeline layout of the synthesis kernels for one channel layout.
#[derive(Clone)]
struct SynthLayout {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

/// The resampling pipeline and its bind group layout.
#[derive(Clone)]
struct ResamplePipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// Pipelines compiled by earlier syntheses on one device, so that later syntheses with the same
/// channel layout and patch size skip shader compilation. There are few distinct keys (channel
/// word counts, modulation, patch size), so entries are never evicted.
#[derive(Default)]
pub(crate) struct PipelineCache {
    layouts: HashMap<Layout, SynthLayout>,
    kernels: HashMap<(Layout, usize), Arc<Kernels>>,
    resample: Option<ResamplePipeline>,
}

impl PipelineCache {
    /// Number of cached kernel sets.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.kernels.len()
    }

    fn synth_layout(&mut self, device: &wgpu::Device, layout: Layout) -> SynthLayout {
        self.layouts
            .entry(layout)
            .or_insert_with(|| {
                let mut entries = vec![uniform_entry(layout.params_size())];
                entries.extend((1..=3).map(|b| storage_entry(b, true)));
                entries.extend((4..=12).map(|b| storage_entry(b, false)));
                let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("synth"),
                    entries: &entries,
                });
                let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("synth"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
                SynthLayout { bind_group_layout, pipeline_layout }
            })
            .clone()
    }

    fn kernels(
        &mut self,
        device: &wgpu::Device,
        synth: &SynthLayout,
        layout: Layout,
        patch_size: usize,
    ) -> Arc<Kernels> {
        Arc::clone(
            self.kernels
                .entry((layout, patch_size))
                .or_insert_with(|| Arc::new(Kernels::new(device, &synth.pipeline_layout, layout, patch_size))),
        )
    }

    fn resample(&mut self, device: &wgpu::Device) -> ResamplePipeline {
        self.resample
            .get_or_insert_with(|| {
                let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("resample"),
                    entries: &[uniform_entry(RESAMPLE_PARAMS), storage_entry(1, true), storage_entry(2, false)],
                });
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("freesynth resample"),
                    source: wgpu::ShaderSource::Wgsl(RESAMPLE_SHADER.into()),
                });
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("resample"),
                    layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("resample"),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    })),
                    module: &module,
                    entry_point: Some("resample"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
                ResamplePipeline { bind_group_layout, pipeline }
            })
            .clone()
    }
}

/// The synthesis pipelines for one patch size.
struct Kernels {
    eval_error: wgpu::ComputePipeline,
    propagate: Vec<wgpu::ComputePipeline>,
    random_search: wgpu::ComputePipeline,
    /// The two half-passes (colours 0 and 1) of the improved search.
    checkerboard: [wgpu::ComputePipeline; 2],
    vote_plain: wgpu::ComputePipeline,
    vote_weighted: wgpu::ComputePipeline,
    eval_mask: wgpu::ComputePipeline,
    dilate_mask: wgpu::ComputePipeline,
    clear_mask: wgpu::ComputePipeline,
    clear_omega: wgpu::ComputePipeline,
    init_omega: wgpu::ComputePipeline,
    upscale_nnf: wgpu::ComputePipeline,
    init_rng: wgpu::ComputePipeline,
}

/// Shape of the target tile a shader module caches in workgroup memory (see `load_tile_rect` in
/// `synth.wgsl`). The checkerboard kernel gets a module of its own, so its wider tile neither
/// takes workgroup memory from the other kernels nor makes them fall back to global reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TileShape {
    /// (WG + 2R)^2 pixels: every kernel except `checkerboard`.
    Square,
    /// (2 WG + 2R) x (WG + 2R) pixels: `checkerboard`, whose workgroups cover 2 WG x WG pixels.
    Checkerboard,
}

/// Number of u32 words of a target tile of the given shape.
fn tile_len(layout: Layout, patch_size: usize, shape: TileShape) -> usize {
    let apron = 2 * (patch_size / 2);
    let width = match shape {
        TileShape::Square => WG as usize,
        TileShape::Checkerboard => 2 * WG as usize,
    } + apron;
    width * (WG as usize + apron) * (layout.style_w + layout.guide_w * (1 + layout.modulated as usize))
}

/// Substitutes the template constants of `synth.wgsl`. The target tile is used when it fits
/// into `workgroup_storage` bytes.
fn synth_source(layout: Layout, patch_size: usize, shape: TileShape, workgroup_storage: usize) -> String {
    let tile = tile_len(layout, patch_size, shape);
    let use_tile = tile * 4 <= workgroup_storage;
    SYNTH_TEMPLATE
        .replace("{{STYLE_W}}", &layout.style_w.to_string())
        .replace("{{GUIDE_W}}", &layout.guide_w.to_string())
        .replace("{{R}}", &(patch_size / 2).to_string())
        .replace("{{MODULATED}}", if layout.modulated { "true" } else { "false" })
        .replace("{{WG}}", &WG.to_string())
        .replace("{{USE_TILE}}", if use_tile { "true" } else { "false" })
        .replace("{{TILE_LEN}}", &(if use_tile { tile } else { 1 }).to_string())
}

impl Kernels {
    fn new(device: &wgpu::Device, pipeline_layout: &wgpu::PipelineLayout, layout: Layout, patch_size: usize) -> Self {
        let storage = device.limits().max_compute_workgroup_storage_size as usize;
        let module = |shape: TileShape| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("freesynth synth"),
                source: wgpu::ShaderSource::Wgsl(synth_source(layout, patch_size, shape, storage).into()),
            })
        };
        let (square, wide) = (module(TileShape::Square), module(TileShape::Checkerboard));
        let make_in = |module: &wgpu::ShaderModule, entry: &str, constants: &[(&str, f64)]| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };
        let make = |entry: &str, constants: &[(&str, f64)]| make_in(&square, entry, constants);
        Self {
            eval_error: make("eval_error", &[]),
            propagate: PROP_RADII.iter().map(|&r| make("propagate", &[("PROP_R", r as f64)])).collect(),
            random_search: make("random_search", &[]),
            checkerboard: [0.0, 1.0].map(|colour| make_in(&wide, "checkerboard", &[("CB_COLOR", colour)])),
            vote_plain: make("vote_plain", &[]),
            vote_weighted: make("vote_weighted", &[]),
            eval_mask: make("eval_mask", &[]),
            dilate_mask: make("dilate_mask", &[]),
            clear_mask: make("clear_mask", &[]),
            clear_omega: make("clear_omega", &[]),
            init_omega: make("init_omega", &[]),
            upscale_nnf: make("upscale_nnf", &[]),
            init_rng: make("init_rng", &[]),
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(size: usize) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: NonZeroU64::new(size as u64),
        },
        count: None,
    }
}

/// A uniform window of `size` bytes at offset 0 (moved by the dynamic offset).
fn uniform_binding(buffer: &wgpu::Buffer, size: usize) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(size as u64),
        }),
    }
}

fn round_up(v: usize, align: usize) -> usize {
    v.div_ceil(align) * align
}

/// Default for [`GpuContext::submission_budget`]: the estimated work, in [`Cost`] units, after
/// which the current command buffer is submitted. About 30-40 ms on a current mid-range GPU, so
/// a GPU ten to twenty times slower still stays well below the usual two-second driver timeout.
pub(crate) const DEFAULT_SUBMISSION_BUDGET: u64 = 8_000_000_000;

/// Estimated GPU work of one dispatch, in units of one packed 32-bit word compared between the
/// target and the source (the inner step of the patch distance and the vote).
#[derive(Clone, Copy, Debug)]
struct Cost {
    /// Words read per patch-pixel comparison: style plus guide (plus modulation) words.
    words: u64,
    /// Style words (for votes).
    style_words: u64,
}

impl Cost {
    fn new(layout: Layout) -> Self {
        let words = (layout.style_w + layout.guide_w * (1 + layout.modulated as usize)) as u64;
        Self { words, style_words: layout.style_w as u64 }
    }

    fn pixels(size: (usize, usize)) -> u64 {
        (size.0 * size.1) as u64
    }

    /// `candidates` patch distances per target pixel.
    fn patches(self, size: (usize, usize), patch: usize, candidates: u64) -> u64 {
        Self::pixels(size) * (patch * patch) as u64 * self.words * candidates
    }

    /// One vote over a target of `size`.
    fn vote(self, size: (usize, usize), patch: usize) -> u64 {
        Self::pixels(size) * (patch * patch) as u64 * self.style_words.max(1)
    }

    /// A kernel doing a small constant amount of work per pixel.
    fn simple(size: (usize, usize), per_pixel: u64) -> u64 {
        Self::pixels(size) * per_pixel
    }
}

/// Records compute dispatches into command buffers and submits each buffer once its estimated
/// work reaches the budget (without waiting for the GPU).
///
/// One buffer per fixed step (for example per search/vote iteration) either wastes submissions
/// on the tiny coarse levels or, on large images, builds GPU jobs long enough to hit a driver
/// timeout (Windows resets the GPU after two seconds by default; Linux drivers have similar
/// watchdogs). A single dispatch is never split, so the budget bounds every submission to about
/// the budget plus one dispatch.
struct Submitter<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    budget: u64,
    encoder: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
    /// Estimated work recorded since the last submission.
    pending: u64,
    /// Number of command buffers submitted so far.
    submissions: usize,
}

impl<'a> Submitter<'a> {
    fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, budget: u64) -> Self {
        Self { device, queue, budget: budget.max(1), encoder: None, pass: None, pending: 0, submissions: 0 }
    }

    /// The open compute pass, after submitting the recorded work first if `cost` more would
    /// exceed the budget.
    fn pass_for(&mut self, cost: u64) -> &mut wgpu::ComputePass<'static> {
        if self.pending > 0 && self.pending.saturating_add(cost) > self.budget {
            self.flush();
        }
        self.pending = self.pending.saturating_add(cost);
        if self.pass.is_none() {
            let device = self.device;
            let encoder = self.encoder.get_or_insert_with(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("synthesis") })
            });
            let pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("synthesis"), timestamp_writes: None })
                .forget_lifetime();
            self.pass = Some(pass);
        }
        self.pass.as_mut().expect("pass was just opened")
    }

    /// Records one dispatch of a synthesis kernel over a `size` grid of pixels.
    fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        offset: u32,
        size: (usize, usize),
        cost: u64,
    ) {
        self.dispatch_groups(
            pipeline,
            group,
            offset,
            ((size.0 as u32).div_ceil(WG), (size.1 as u32).div_ceil(WG)),
            cost,
        );
    }

    /// Records one dispatch of `groups` workgroups.
    fn dispatch_groups(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        offset: u32,
        groups: (u32, u32),
        cost: u64,
    ) {
        let pass = self.pass_for(cost);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[offset]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    /// Records a buffer copy after the dispatches recorded so far.
    fn copy(&mut self, from: &wgpu::Buffer, to: &wgpu::Buffer, bytes: u64) {
        self.pass = None;
        let device = self.device;
        self.encoder
            .get_or_insert_with(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") })
            })
            .copy_buffer_to_buffer(from, 0, to, 0, bytes);
    }

    /// Submits everything recorded so far.
    fn flush(&mut self) {
        // The pass must end before its encoder can finish.
        self.pass = None;
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit([encoder.finish()]);
            self.submissions += 1;
        }
        self.pending = 0;
    }
}

fn timing_enabled() -> bool {
    std::env::var_os("FREESYNTH_GPU_TIMING").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Runs a synthesis, turning wgpu validation and out-of-memory errors into [`GpuError`]s.
/// Also returns the number of command buffers submitted.
pub(crate) fn synthesize(
    ctx: &GpuContext,
    input: &SynthesisInput,
    params: &Params,
    extras: &SynthesisExtras,
) -> Result<(SynthesisOutput, usize), GpuError> {
    let sched = params.schedule(input, Engine::Gpu).map_err(|e| GpuError::InvalidInput(e.to_string()))?;
    extras.validate(input).map_err(|e| GpuError::InvalidInput(e.to_string()))?;
    let oom = ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = run(ctx, input, params, &sched, extras);
    let validation = pollster::block_on(validation.pop());
    let oom = pollster::block_on(oom.pop());
    if let Some(e) = validation.or(oom) {
        return Err(GpuError::Wgpu(e.to_string()));
    }
    result
}

fn run(
    ctx: &GpuContext,
    input: &SynthesisInput,
    params: &Params,
    sched: &Schedule,
    extras: &SynthesisExtras,
) -> Result<(SynthesisOutput, usize), GpuError> {
    let t_start = Instant::now();
    let (device, queue) = (&ctx.device, &ctx.queue);
    let limits = device.limits();

    let ns = input.source_style.channels();
    let ng = input.source_guide.channels();
    let layout =
        Layout { style_w: ns.div_ceil(4), guide_w: ng.div_ceil(4), modulated: input.target_modulation.is_some() };
    let source_size = input.source_style.size();
    let target_size = input.target_guide.size();
    let num_levels = sched.num_levels;
    let phases = phases(params, num_levels, source_size, target_size);
    let use_mask = sched.use_stop_threshold;
    // With a warm start, the coarse levels are skipped.
    let first_level = extras.warm_start.as_ref().map_or(0, |w| num_levels.saturating_sub(w.levels));

    // --- Limits -------------------------------------------------------------------------------
    let src_px = (source_size.0 * source_size.1) as u64;
    let tgt_px = (target_size.0 * target_size.1) as u64;
    let src_words = (layout.style_w + layout.guide_w) as u64;
    let tgt_words = layout.style_w.max(layout.guide_w) as u64;
    let largest = (src_px * src_words * 4).max(tgt_px * (tgt_words * 4).max(8));
    if largest > limits.max_storage_buffer_binding_size || largest > limits.max_buffer_size {
        return Err(GpuError::TooLarge(format!(
            "a {largest}-byte buffer is needed, the device allows {} (max_storage_buffer_binding_size \
             and max_buffer_size; a device created with GpuContext::required_limits gets the adapter's maximum)",
            limits.max_storage_buffer_binding_size.min(limits.max_buffer_size)
        )));
    }
    let max_groups = limits.max_compute_workgroups_per_dimension;
    let max_dim = source_size.0.max(source_size.1).max(target_size.0).max(target_size.1) as u32;
    if max_dim.div_ceil(WG.min(RESAMPLE_WG)) > max_groups {
        return Err(GpuError::TooLarge(format!("{max_dim} pixels exceed the dispatch size limit")));
    }

    // --- Buffers ------------------------------------------------------------------------------
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let storage = |label: &str, size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage,
            mapped_at_creation: false,
        })
    };
    let upload = |label: &str, words: &[u32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let (sw4, gw4) = (layout.style_w as u64 * 4, layout.guide_w as u64 * 4);

    // Full-resolution inputs, resampled into the level buffers at the start of every level.
    let full_src = upload("full source", &pack_source(input, layout));
    let full_tgt_guide = upload("full target guide", &pack_words(&input.target_guide, layout.guide_w));
    let full_tgt_mod = match &input.target_modulation {
        Some(m) => upload("full target modulation", &pack_words(m, layout.guide_w)),
        None => upload("no modulation", &[0]),
    };
    let src = storage("source", src_px * (sw4 + gw4));
    let tgt_guide = storage("target guide", tgt_px * gw4);
    let tgt_mod = storage("target modulation", if layout.modulated { tgt_px * gw4 } else { 4 });
    let style = [storage("target style A", tgt_px * sw4), storage("target style B", tgt_px * sw4)];
    let nnf = [storage("NNF A", tgt_px * 8), storage("NNF B", tgt_px * 8)];
    let err = storage("errors", tgt_px * 4);
    let omega = storage("omega", src_px * 4);
    let mask = storage("mask", tgt_px * 4);
    let mask_tmp = storage("mask scratch", tgt_px * 4);
    let rng = storage("rng", tgt_px * 4);

    // The initial field: the warm start's at its first level, or a random field at the coarsest
    // level from the CPU engine's generator.
    let init_nnf = match &extras.warm_start {
        Some(w) => {
            let ph = &phases[first_level];
            w.field_at_level(ph.target, source_size, ph.source, ph.patch_size)
        }
        None => {
            let mut rand = GlibcRand::new(params.seed);
            nnf_init_random(phases[0].target, phases[0].source, phases[0].patch_size, &mut rand)
        }
    };
    queue.write_buffer(&nnf[0], 0, bytemuck::cast_slice(init_nnf.data()));

    // The final vote's errors, for a 16-bit vote of the final field.
    let vote_err = extras.style16.as_ref().map(|_| storage("vote errors", tgt_px * 4));

    // Uniforms: one slot per phase, and one per (level, resampled image).
    let align = limits.min_uniform_buffer_offset_alignment as usize;
    let params_size = layout.params_size();
    let slot = round_up(params_size, align);
    let mut uniform = vec![0u8; slot * phases.len()];
    for (i, ph) in phases.iter().enumerate() {
        uniform[i * slot..i * slot + params_size].copy_from_slice(&params_bytes(ph, layout, input, params, sched));
    }
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: &uniform,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // (full input, level buffer, words per pixel, full size, is target side)
    let mut resample_jobs = vec![
        (&full_src, &src, layout.style_w + layout.guide_w, source_size, false),
        (&full_tgt_guide, &tgt_guide, layout.guide_w, target_size, true),
    ];
    if layout.modulated {
        resample_jobs.push((&full_tgt_mod, &tgt_mod, layout.guide_w, target_size, true));
    }
    let rslot = round_up(RESAMPLE_PARAMS, align);
    let mut runiform = vec![0u8; rslot * num_levels * resample_jobs.len()];
    for (level, ph) in phases.iter().take(num_levels).enumerate() {
        for (j, &(_, _, words, full, is_target)) in resample_jobs.iter().enumerate() {
            let out = if is_target { ph.target } else { ph.source };
            let o = (level * resample_jobs.len() + j) * rslot;
            runiform[o..o + RESAMPLE_PARAMS].copy_from_slice(&resample_bytes(full, out, words));
        }
    }
    let rparams_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("resample params"),
        contents: &runiform,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- Pipelines and bind groups ------------------------------------------------------------
    let (synth_layout, kernels, kernels_3x3, resample_pipeline) = {
        let mut cache = ctx.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        let synth_layout = cache.synth_layout(device, layout);
        let kernels = cache.kernels(device, &synth_layout, layout, params.patch_size);
        let kernels_3x3 =
            (params.extra_pass_3x3 && params.patch_size != 3).then(|| cache.kernels(device, &synth_layout, layout, 3));
        (synth_layout, kernels, kernels_3x3, cache.resample(device))
    };
    let bgl = &synth_layout.bind_group_layout;

    // groups[s][n]: style buffer `s` and NNF buffer `n` are the current ones.
    let group = |s: usize, n: usize| {
        let bufs: [&wgpu::Buffer; 12] = [
            &src,
            &tgt_guide,
            &tgt_mod,
            &style[s],
            &style[1 - s],
            &nnf[n],
            &nnf[1 - n],
            &err,
            &omega,
            &mask,
            &mask_tmp,
            &rng,
        ];
        let mut e = vec![uniform_binding(&params_buf, params_size)];
        e.extend(
            bufs.iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() }),
        );
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("synth"), layout: bgl, entries: &e })
    };
    let groups = [[group(0, 0), group(0, 1)], [group(1, 0), group(1, 1)]];

    let resample = &resample_pipeline.pipeline;
    let rgroups: Vec<wgpu::BindGroup> = resample_jobs
        .iter()
        .map(|&(full, level_buf, ..)| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resample"),
                layout: &resample_pipeline.bind_group_layout,
                entries: &[
                    uniform_binding(&rparams_buf, RESAMPLE_PARAMS),
                    wgpu::BindGroupEntry { binding: 1, resource: full.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: level_buf.as_entire_binding() },
                ],
            })
        })
        .collect();
    let t_setup = Instant::now();

    // --- Schedule -----------------------------------------------------------------------------
    // Nothing waits for the GPU until the final readback; the submitter cuts the recording into
    // command buffers of bounded work.
    let cost = Cost::new(layout);
    let mut sub = Submitter::new(device, queue, ctx.submission_budget);
    let (mut cs, mut cn) = (0usize, 0usize); // current style / NNF buffer
    let finest = num_levels - 1;
    sub.dispatch(
        &kernels.init_rng,
        &groups[cs][cn],
        (finest * slot) as u32,
        phases[finest].target,
        Cost::simple(phases[finest].target, 4),
    );
    let mut last_kernels = &*kernels;
    // Whether the latest vote was weighted (with the errors in `err`).
    let mut weighted_vote = false;
    for (pi, ph) in phases.iter().enumerate().filter(|(_, ph)| ph.level >= first_level) {
        let k = match &kernels_3x3 {
            Some(k3) if ph.patch_size != params.patch_size => &**k3,
            _ => &*kernels,
        };
        last_kernels = k;
        let off = (pi * slot) as u32;
        let (tsz, ssz) = (ph.target, ph.source);
        let p = ph.patch_size;

        if !ph.extra {
            for (j, (g, &(_, _, words, _, is_target))) in rgroups.iter().zip(&resample_jobs).enumerate() {
                let out = if is_target { tsz } else { ssz };
                let groups = ((out.0 as u32).div_ceil(RESAMPLE_WG), (out.1 as u32).div_ceil(RESAMPLE_WG));
                let roff = ((ph.level * resample_jobs.len() + j) * rslot) as u32;
                sub.dispatch_groups(resample, g, roff, groups, Cost::simple(out, 16 * words as u64));
            }
            if ph.level > first_level {
                sub.dispatch(&k.upscale_nnf, &groups[cs][cn], off, tsz, Cost::simple(tsz, 4));
                cn ^= 1;
            }
            if ph.use_omega() {
                sub.dispatch(&k.clear_omega, &groups[cs][cn], off, ssz, Cost::simple(ssz, 1));
                sub.dispatch(&k.init_omega, &groups[cs][cn], off, tsz, Cost::simple(tsz, (p * p) as u64));
            }
        }
        sub.dispatch(&k.vote_plain, &groups[cs][cn], off, tsz, cost.vote(tsz, p));
        cs ^= 1;
        weighted_vote = false;
        if use_mask {
            sub.dispatch(&k.clear_mask, &groups[cs][cn], off, tsz, Cost::simple(tsz, 1));
        }

        // Random search tries radii 1, 2, 4, ... below max(sw, sh) / 2.
        let radii = (ssz.0.max(ssz.1) / 2).max(1).next_power_of_two().trailing_zeros() as u64;
        let num_sv = sched.search_vote_iters[ph.level];
        let num_pm = sched.patch_match_iters[ph.level];
        // A checkerboard half-pass covers every other pixel: 4 neighbours and the radii each.
        let half = (tsz.0.div_ceil(2), tsz.1);
        for vote_iter in 0..num_sv {
            sub.dispatch(&k.eval_error, &groups[cs][cn], off, tsz, cost.patches(tsz, p, 1));
            for _ in 0..num_pm {
                match sched.search {
                    SearchMode::Compat => {
                        for prop in &k.propagate {
                            sub.dispatch(prop, &groups[cs][cn], off, tsz, cost.patches(tsz, p, 4));
                            cn ^= 1;
                        }
                        sub.dispatch(&k.random_search, &groups[cs][cn], off, tsz, cost.patches(tsz, p, radii));
                    }
                    SearchMode::Improved => {
                        for pass in &k.checkerboard {
                            sub.dispatch(pass, &groups[cs][cn], off, half, cost.patches(half, p, 4 + radii));
                        }
                    }
                }
            }
            let vote = match sched.vote_mode {
                VoteMode::Plain => &k.vote_plain,
                VoteMode::Weighted => &k.vote_weighted,
            };
            sub.dispatch(vote, &groups[cs][cn], off, tsz, cost.vote(tsz, p));
            cs ^= 1;
            weighted_vote = sched.vote_mode == VoteMode::Weighted;
            if use_mask && vote_iter + 1 < num_sv {
                sub.dispatch(&k.eval_mask, &groups[cs][cn], off, tsz, Cost::simple(tsz, cost.style_words));
                sub.dispatch(&k.dilate_mask, &groups[cs][cn], off, tsz, Cost::simple(tsz, (p * p) as u64));
            }
        }
    }
    let err_bytes = tgt_px * 4;
    if let Some(v) = &vote_err {
        // Kept before the error evaluation below overwrites them.
        sub.copy(&err, v, err_bytes);
    }
    if params.return_errors {
        // Patch errors of the final field against the final image (now the current style).
        let last = phases.len() - 1;
        let ph = &phases[last];
        sub.dispatch(
            &last_kernels.eval_error,
            &groups[cs][cn],
            (last * slot) as u32,
            ph.target,
            cost.patches(ph.target, ph.patch_size, 1),
        );
    }
    sub.flush();

    // --- Readback -----------------------------------------------------------------------------
    let (tw, th) = target_size;
    let style_bytes = tgt_px * sw4;
    let nnf_bytes = tgt_px * 8;
    let staging = |label: &str, size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let style_out = staging("style readback", style_bytes);
    let nnf_out = staging("NNF readback", nnf_bytes);
    let err_out = params.return_errors.then(|| staging("error readback", err_bytes));
    let vote_err_out = vote_err.as_ref().filter(|_| weighted_vote).map(|_| staging("vote error readback", err_bytes));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    encoder.copy_buffer_to_buffer(&style[cs], 0, &style_out, 0, style_bytes);
    encoder.copy_buffer_to_buffer(&nnf[cn], 0, &nnf_out, 0, nnf_bytes);
    if let Some(e) = &err_out {
        encoder.copy_buffer_to_buffer(&err, 0, e, 0, err_bytes);
    }
    if let (Some(v), Some(out)) = (&vote_err, &vote_err_out) {
        encoder.copy_buffer_to_buffer(v, 0, out, 0, err_bytes);
    }
    queue.submit([encoder.finish()]);

    let readbacks: Vec<&wgpu::Buffer> =
        [Some(&style_out), Some(&nnf_out), err_out.as_ref(), vote_err_out.as_ref()].into_iter().flatten().collect();
    let (tx, rx) = mpsc::channel();
    for b in &readbacks {
        let tx = tx.clone();
        b.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
    }
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| GpuError::Wgpu(e.to_string()))?;
    for _ in 0..readbacks.len() {
        rx.recv()
            .map_err(|_| GpuError::Wgpu("buffer mapping was cancelled".into()))?
            .map_err(|e| GpuError::Wgpu(format!("cannot read back results: {e}")))?;
    }
    let t_gpu = Instant::now();

    let image = {
        let view = style_out.get_mapped_range(..).map_err(|e| GpuError::Wgpu(e.to_string()))?;
        unpack_words(bytemuck::cast_slice(&view), tw, th, ns, layout.style_w)
    };
    let nnf = {
        let view = nnf_out.get_mapped_range(..).map_err(|e| GpuError::Wgpu(e.to_string()))?;
        Nnf::from_raw(tw, th, bytemuck::cast_slice::<u8, [i32; 2]>(&view).to_vec())
    };
    let errors = match &err_out {
        Some(b) => {
            let view = b.get_mapped_range(..).map_err(|e| GpuError::Wgpu(e.to_string()))?;
            Some(bytemuck::cast_slice::<u8, f32>(&view).to_vec())
        }
        None => None,
    };
    // The 16-bit result, voted on the CPU from the final field (and the final vote's errors).
    let image16 = match &extras.style16 {
        Some(s16) => {
            let mut out = Image16::new(tw, th, ns);
            let patch = phases[phases.len() - 1].patch_size;
            match &vote_err_out {
                Some(b) => {
                    let view = b.get_mapped_range(..).map_err(|e| GpuError::Wgpu(e.to_string()))?;
                    let errors: &[f32] = bytemuck::cast_slice(&view);
                    vote_weighted16(&mut out, s16, &nnf, errors, patch, VoteRounding::Nearest);
                }
                None => vote_plain16(&mut out, s16, &nnf, patch, VoteRounding::Nearest),
            }
            Some(out)
        }
        None => None,
    };
    if timing_enabled() {
        eprintln!(
            "gpu: setup {:.1} ms, synthesis {:.1} ms ({} phases, {} submissions)",
            (t_setup - t_start).as_secs_f64() * 1e3,
            (t_gpu - t_setup).as_secs_f64() * 1e3,
            phases.len(),
            sub.submissions
        );
    }
    Ok((SynthesisOutput { image, nnf, errors, image16 }, sub.submissions))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_packing_round_trips() {
        for nc in 1..=9 {
            let data: Vec<u8> = (0..5 * 3 * nc).map(|i| (i * 37 % 256) as u8).collect();
            let img = Image::from_raw(5, 3, nc, data);
            let words = nc.div_ceil(4);
            let packed = pack_words(&img, words);
            assert_eq!(packed.len(), 15 * words);
            assert_eq!(unpack_words(&packed, 5, 3, nc, words), img);
        }
    }

    #[test]
    fn params_layout_matches_shader() {
        let img = Image::new(40, 30, 3);
        let input = SynthesisInput {
            source_style: img.clone(),
            source_guide: Image::new(40, 30, 6),
            target_guide: Image::new(40, 30, 6),
            target_modulation: None,
            style_weights: vec![1.0; 3],
            guide_weights: vec![0.5; 6],
        };
        let layout = Layout { style_w: 1, guide_w: 2, modulated: false };
        let params = Params::default();
        let sched = params.schedule_for_levels(2, Engine::Gpu).unwrap();
        let ph = &phases(&params, 2, (40, 30), (40, 30))[1];
        let b = params_bytes(ph, layout, &input, &params, &sched);
        assert_eq!(b.len(), layout.params_size());
        let f = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let u = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((u(0), u(1), u(4), u(5), u(6)), (40, 30, 20, 15, 5));
        assert_eq!(f(8), 3500.0);
        assert_eq!(f(9), 25.0); // same size: omega_best = patch^2
        assert_eq!(u(12), 1); // round votes
        let compat = Params::compat();
        let b = params_bytes(ph, layout, &input, &compat, &compat.schedule_for_levels(2, Engine::Gpu).unwrap());
        assert_eq!(u32::from_le_bytes(b[48..52].try_into().unwrap()), 0);
        assert_eq!(f(13), 75.0);
        assert_eq!([f(16), f(17), f(18), f(19)], [1.0, 1.0, 1.0, 0.0]);
        assert_eq!([f(20), f(25), f(26), f(27)], [0.5, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn resample_params_carry_the_exact_scale() {
        let b = resample_bytes((48, 30), (48, 30), 2);
        assert_eq!(f32::from_le_bytes(b[20..24].try_into().unwrap()), 1.0);
        let b = resample_bytes((1200, 912), (75, 57), 4);
        assert_eq!(f32::from_le_bytes(b[20..24].try_into().unwrap()), 16.0);
        assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), 4);
    }

    #[test]
    fn shader_template_is_fully_substituted() {
        let layout = Layout { style_w: 2, guide_w: 6, modulated: true };
        let src = synth_source(layout, 7, TileShape::Square, 1 << 20);
        assert!(src.contains("const USE_TILE: bool = true;"));
        assert!(src.contains(&format!("const TILE_LEN: u32 = {}u;", tile_len(layout, 7, TileShape::Square))));
        assert!(synth_source(layout, 7, TileShape::Square, 1024).contains("const USE_TILE: bool = false;"));
        assert!(!src.contains("{{"), "unsubstituted placeholder");
        assert!(src.contains("const R: i32 = 3;") && src.contains("const MODULATED: bool = true;"));
    }

    #[test]
    fn checkerboard_tile_does_not_change_the_square_one() {
        // 2 style + 6 guide words + 6 modulation words per pixel, 7 x 7 patches.
        let layout = Layout { style_w: 2, guide_w: 6, modulated: true };
        assert_eq!(tile_len(layout, 7, TileShape::Square), 14 * 14 * 14);
        assert_eq!(tile_len(layout, 7, TileShape::Checkerboard), 22 * 14 * 14);
        // A layout whose square tile fits 16 KiB but whose checkerboard tile does not: only the
        // checkerboard module falls back to global reads.
        let layout = Layout { style_w: 3, guide_w: 12, modulated: false };
        let fits = |shape| synth_source(layout, 7, shape, 16384).contains("const USE_TILE: bool = true;");
        assert!(fits(TileShape::Square) && !fits(TileShape::Checkerboard));
    }
}
