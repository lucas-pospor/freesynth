//! Buffers, pipelines and the per-level schedule (`ebsynthCuda` in the original).
//!
//! All buffers are allocated once at the finest level's size and reused by every pyramid level.
//! The schedule is recorded up front (one command buffer per level setup and per search/vote
//! iteration), with a single readback at the end; the CPU never waits for the GPU in between.
//! Per-level constants live in one uniform buffer, selected with dynamic offsets.

use std::num::NonZeroU64;
use std::sync::mpsc;
use std::time::Instant;

use freesynth_core::{
    GlibcRand, Image, MAX_GUIDE_CHANNELS, MAX_STYLE_CHANNELS, Nnf, Params, PerLevel, SynthesisInput,
    SynthesisOutput, VoteMode, max_pyramid_levels, nnf_init_random, pyramid_level_size,
};
use wgpu::util::DeviceExt;

use crate::GpuContext;
use crate::error::{GpuError, invalid};

/// Workgroup edge length of the synthesis kernels (WG x WG invocations).
const WG: u32 = 8;
/// Workgroup edge length of the resampling kernel (fixed in `resample.wgsl`).
const RESAMPLE_WG: u32 = 16;
/// Propagation offsets of one PatchMatch iteration, as in the CUDA engine.
const PROP_RADII: [i32; 3] = [4, 2, 1];

const SYNTH_TEMPLATE: &str = include_str!("shaders/synth.wgsl");
const RESAMPLE_SHADER: &str = include_str!("shaders/resample.wgsl");

/// Size of the fixed part of the `Params` uniform (16 scalars).
const PARAMS_HEADER: usize = 64;
/// Size of the `ResampleParams` uniform.
const RESAMPLE_PARAMS: usize = 32;

/// Packed channel layout of a synthesis.
#[derive(Clone, Copy, Debug)]
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

fn covers<T>(v: &PerLevel<T>, n: usize) -> bool {
    match v {
        PerLevel::Uniform(_) => true,
        PerLevel::Levels(l) => l.len() >= n,
    }
}

/// Checks the input and parameters (same rules as the CPU engine) and returns the number of
/// pyramid levels.
fn validate(input: &SynthesisInput, params: &Params) -> Result<usize, GpuError> {
    let ns = input.source_style.channels();
    let ng = input.source_guide.channels();
    if !(1..=MAX_STYLE_CHANNELS).contains(&ns) {
        return invalid(format!("number of style channels must be 1..={MAX_STYLE_CHANNELS}, got {ns}"));
    }
    if !(1..=MAX_GUIDE_CHANNELS).contains(&ng) {
        return invalid(format!("number of guide channels must be 1..={MAX_GUIDE_CHANNELS}, got {ng}"));
    }
    if input.source_guide.size() != input.source_style.size() {
        return invalid("source guide and source style differ in size");
    }
    if input.target_guide.channels() != ng {
        return invalid("source and target guides differ in channel count");
    }
    if let Some(m) = &input.target_modulation
        && (m.size() != input.target_guide.size() || m.channels() != ng)
    {
        return invalid("target modulation must have the target guide's size and channels");
    }
    if input.style_weights.len() != ns || input.guide_weights.len() != ng {
        return invalid("need one weight per style channel and one per guide channel");
    }
    if input.style_weights.iter().chain(&input.guide_weights).any(|w| !w.is_finite() || *w < 0.0) {
        return invalid("weights must be finite and non-negative");
    }
    if params.patch_size < 3 || params.patch_size.is_multiple_of(2) {
        return invalid("patch size must be an odd number >= 3");
    }
    if !params.uniformity_weight.is_finite() {
        return invalid("uniformity weight must be finite");
    }
    let max_levels = max_pyramid_levels(input.source_style.size(), input.target_guide.size(), params.patch_size);
    if max_levels == 0 {
        return invalid(format!(
            "source and target must be at least {0}x{0} pixels for patch size {1}",
            2 * params.patch_size + 1,
            params.patch_size
        ));
    }
    let num_levels = match params.num_pyramid_levels {
        Some(0) => return invalid("number of pyramid levels must be at least 1"),
        Some(n) => n.min(max_levels),
        None => max_levels,
    };
    if !covers(&params.search_vote_iters, num_levels)
        || !covers(&params.patch_match_iters, num_levels)
        || !covers(&params.stop_threshold, num_levels)
    {
        return invalid(format!("per-level settings must cover all {num_levels} pyramid levels"));
    }
    Ok(num_levels)
}

fn phases(params: &Params, num_levels: usize, source: (usize, usize), target: (usize, usize)) -> Vec<Phase> {
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
fn params_bytes(ph: &Phase, layout: Layout, input: &SynthesisInput, params: &Params) -> Vec<u8> {
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
    b.extend_from_slice(&params.stop_threshold.get(ph.level).to_le_bytes());
    b.extend_from_slice(&params.seed.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
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
    b
}

/// The synthesis pipelines for one patch size.
struct Kernels {
    eval_error: wgpu::ComputePipeline,
    propagate: Vec<wgpu::ComputePipeline>,
    random_search: wgpu::ComputePipeline,
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

/// Number of u32 words of the target tile cached in workgroup memory (see `load_tile` in
/// `synth.wgsl`).
fn tile_len(layout: Layout, patch_size: usize) -> usize {
    let side = WG as usize + 2 * (patch_size / 2);
    side * side * (layout.style_w + layout.guide_w * (1 + layout.modulated as usize))
}

/// Substitutes the template constants of `synth.wgsl`. The target tile is used when it fits
/// into `workgroup_storage` bytes.
fn synth_source(layout: Layout, patch_size: usize, workgroup_storage: usize) -> String {
    let tile = tile_len(layout, patch_size);
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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("freesynth synth"),
            source: wgpu::ShaderSource::Wgsl(
                synth_source(layout, patch_size, device.limits().max_compute_workgroup_storage_size as usize).into(),
            ),
        });
        let make = |entry: &str, constants: &[(&str, f64)]| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };
        Self {
            eval_error: make("eval_error", &[]),
            propagate: PROP_RADII.iter().map(|&r| make("propagate", &[("PROP_R", r as f64)])).collect(),
            random_search: make("random_search", &[]),
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

/// Records one dispatch of a synthesis kernel over a `size` grid.
fn dispatch(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    offset: u32,
    size: (usize, usize),
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[offset]);
    pass.dispatch_workgroups((size.0 as u32).div_ceil(WG), (size.1 as u32).div_ceil(WG), 1);
}

/// Records one compute pass with `record` and submits it (without waiting for it).
fn submit_pass(device: &wgpu::Device, queue: &wgpu::Queue, record: impl FnOnce(&mut wgpu::ComputePass<'_>)) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("synthesis") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("synthesis"), timestamp_writes: None });
        record(&mut pass);
    }
    queue.submit([encoder.finish()]);
}

fn timing_enabled() -> bool {
    std::env::var_os("FREESYNTH_GPU_TIMING").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Runs a synthesis, turning wgpu validation and out-of-memory errors into [`GpuError`]s.
pub(crate) fn synthesize(ctx: &GpuContext, input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, GpuError> {
    let num_levels = validate(input, params)?;
    let oom = ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = run(ctx, input, params, num_levels);
    let validation = pollster::block_on(validation.pop());
    let oom = pollster::block_on(oom.pop());
    if let Some(e) = validation.or(oom) {
        return Err(GpuError::Wgpu(e.to_string()));
    }
    result
}

fn run(ctx: &GpuContext, input: &SynthesisInput, params: &Params, num_levels: usize) -> Result<SynthesisOutput, GpuError> {
    let t_start = Instant::now();
    let (device, queue) = (&ctx.device, &ctx.queue);
    let limits = device.limits();

    let ns = input.source_style.channels();
    let ng = input.source_guide.channels();
    let layout = Layout { style_w: ns.div_ceil(4), guide_w: ng.div_ceil(4), modulated: input.target_modulation.is_some() };
    let source_size = input.source_style.size();
    let target_size = input.target_guide.size();
    let phases = phases(params, num_levels, source_size, target_size);
    let use_mask = params.use_stop_threshold;

    // --- Limits -------------------------------------------------------------------------------
    let src_px = (source_size.0 * source_size.1) as u64;
    let tgt_px = (target_size.0 * target_size.1) as u64;
    let src_words = (layout.style_w + layout.guide_w) as u64;
    let tgt_words = layout.style_w.max(layout.guide_w) as u64;
    let largest = (src_px * src_words * 4).max(tgt_px * (tgt_words * 4).max(8));
    if largest > limits.max_storage_buffer_binding_size || largest > limits.max_buffer_size {
        return Err(GpuError::TooLarge(format!(
            "a {largest}-byte buffer is needed, the device allows {}",
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
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(4), usage, mapped_at_creation: false })
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

    // Random initial field at the coarsest level, from the CPU engine's generator.
    let mut rand = GlibcRand::new(params.seed);
    let init_nnf = nnf_init_random(phases[0].target, phases[0].source, phases[0].patch_size, &mut rand);
    queue.write_buffer(&nnf[0], 0, bytemuck::cast_slice(init_nnf.data()));

    // Uniforms: one slot per phase, and one per (level, resampled image).
    let align = limits.min_uniform_buffer_offset_alignment as usize;
    let params_size = layout.params_size();
    let slot = round_up(params_size, align);
    let mut uniform = vec![0u8; slot * phases.len()];
    for (i, ph) in phases.iter().enumerate() {
        uniform[i * slot..i * slot + params_size].copy_from_slice(&params_bytes(ph, layout, input, params));
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
    let mut entries = vec![uniform_entry(params_size)];
    entries.extend((1..=3).map(|b| storage_entry(b, true)));
    entries.extend((4..=12).map(|b| storage_entry(b, false)));
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("synth"), entries: &entries });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("synth"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let kernels = Kernels::new(device, &pipeline_layout, layout, params.patch_size);
    let kernels_3x3 = (params.extra_pass_3x3 && params.patch_size != 3).then(|| Kernels::new(device, &pipeline_layout, layout, 3));

    // groups[s][n]: style buffer `s` and NNF buffer `n` are the current ones.
    let group = |s: usize, n: usize| {
        let bufs: [&wgpu::Buffer; 12] = [
            &src, &tgt_guide, &tgt_mod, &style[s], &style[1 - s], &nnf[n], &nnf[1 - n], &err, &omega,
            &mask, &mask_tmp, &rng,
        ];
        let mut e = vec![uniform_binding(&params_buf, params_size)];
        e.extend(bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() }));
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("synth"), layout: &bgl, entries: &e })
    };
    let groups = [[group(0, 0), group(0, 1)], [group(1, 0), group(1, 1)]];

    let rbgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("resample"),
        entries: &[uniform_entry(RESAMPLE_PARAMS), storage_entry(1, true), storage_entry(2, false)],
    });
    let rmodule = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("freesynth resample"),
        source: wgpu::ShaderSource::Wgsl(RESAMPLE_SHADER.into()),
    });
    let resample = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resample"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resample"),
            bind_group_layouts: &[Some(&rbgl)],
            immediate_size: 0,
        })),
        module: &rmodule,
        entry_point: Some("resample"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let rgroups: Vec<wgpu::BindGroup> = resample_jobs
        .iter()
        .map(|&(full, level_buf, ..)| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resample"),
                layout: &rbgl,
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
    // Every level's setup and every search/vote iteration is submitted as its own command
    // buffer, so that no single GPU job runs long enough to trigger a driver timeout on large
    // images. Nothing waits for the GPU until the final readback.
    let (mut cs, mut cn) = (0usize, 0usize); // current style / NNF buffer
    let finest = num_levels - 1;
    submit_pass(device, queue, |pass| {
        dispatch(pass, &kernels.init_rng, &groups[cs][cn], (finest * slot) as u32, phases[finest].target);
    });
    for (pi, ph) in phases.iter().enumerate() {
        let k = match &kernels_3x3 {
            Some(k3) if ph.patch_size != params.patch_size => k3,
            _ => &kernels,
        };
        let off = (pi * slot) as u32;
        let (tsz, ssz) = (ph.target, ph.source);

        submit_pass(device, queue, |pass| {
            if !ph.extra {
                pass.set_pipeline(&resample);
                for (j, (g, &(_, _, _, _, is_target))) in rgroups.iter().zip(&resample_jobs).enumerate() {
                    let out = if is_target { tsz } else { ssz };
                    pass.set_bind_group(0, g, &[((ph.level * resample_jobs.len() + j) * rslot) as u32]);
                    pass.dispatch_workgroups((out.0 as u32).div_ceil(RESAMPLE_WG), (out.1 as u32).div_ceil(RESAMPLE_WG), 1);
                }
                if ph.level > 0 {
                    dispatch(pass, &k.upscale_nnf, &groups[cs][cn], off, tsz);
                    cn ^= 1;
                }
                if ph.use_omega() {
                    dispatch(pass, &k.clear_omega, &groups[cs][cn], off, ssz);
                    dispatch(pass, &k.init_omega, &groups[cs][cn], off, tsz);
                }
            }
            dispatch(pass, &k.vote_plain, &groups[cs][cn], off, tsz);
            cs ^= 1;
            if use_mask {
                dispatch(pass, &k.clear_mask, &groups[cs][cn], off, tsz);
            }
        });

        let num_sv = params.search_vote_iters.get(ph.level);
        let num_pm = params.patch_match_iters.get(ph.level);
        for vote_iter in 0..num_sv {
            submit_pass(device, queue, |pass| {
                dispatch(pass, &k.eval_error, &groups[cs][cn], off, tsz);
                for _ in 0..num_pm {
                    for p in &k.propagate {
                        dispatch(pass, p, &groups[cs][cn], off, tsz);
                        cn ^= 1;
                    }
                    dispatch(pass, &k.random_search, &groups[cs][cn], off, tsz);
                }
                let vote = match params.vote_mode {
                    VoteMode::Plain => &k.vote_plain,
                    VoteMode::Weighted => &k.vote_weighted,
                };
                dispatch(pass, vote, &groups[cs][cn], off, tsz);
                cs ^= 1;
                if use_mask && vote_iter + 1 < num_sv {
                    dispatch(pass, &k.eval_mask, &groups[cs][cn], off, tsz);
                    dispatch(pass, &k.dilate_mask, &groups[cs][cn], off, tsz);
                }
            });
        }
    }

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
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    encoder.copy_buffer_to_buffer(&style[cs], 0, &style_out, 0, style_bytes);
    encoder.copy_buffer_to_buffer(&nnf[cn], 0, &nnf_out, 0, nnf_bytes);
    queue.submit([encoder.finish()]);

    let (tx, rx) = mpsc::channel();
    for b in [&style_out, &nnf_out] {
        let tx = tx.clone();
        b.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
    }
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| GpuError::Wgpu(e.to_string()))?;
    for _ in 0..2 {
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
    if timing_enabled() {
        eprintln!(
            "gpu: setup {:.1} ms, synthesis {:.1} ms ({} phases)",
            (t_setup - t_start).as_secs_f64() * 1e3,
            (t_gpu - t_setup).as_secs_f64() * 1e3,
            phases.len()
        );
    }
    Ok(SynthesisOutput { image, nnf })
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
        let ph = &phases(&params, 2, (40, 30), (40, 30))[1];
        let b = params_bytes(ph, layout, &input, &params);
        assert_eq!(b.len(), layout.params_size());
        let f = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let u = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((u(0), u(1), u(4), u(5), u(6)), (40, 30, 20, 15, 5));
        assert_eq!(f(8), 3500.0);
        assert_eq!(f(9), 25.0); // same size: omega_best = patch^2
        assert_eq!(f(13), 75.0);
        assert_eq!([f(16), f(17), f(18), f(19)], [1.0, 1.0, 1.0, 0.0]);
        assert_eq!([f(20), f(25), f(26), f(27)], [0.5, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn shader_template_is_fully_substituted() {
        let layout = Layout { style_w: 2, guide_w: 6, modulated: true };
        let src = synth_source(layout, 7, 1 << 20);
        assert!(src.contains("const USE_TILE: bool = true;"));
        assert!(src.contains(&format!("const TILE_LEN: u32 = {}u;", tile_len(layout, 7))));
        assert!(synth_source(layout, 7, 1024).contains("const USE_TILE: bool = false;"));
        assert!(!src.contains("{{"), "unsubstituted placeholder");
        assert!(src.contains("const R: i32 = 3;") && src.contains("const MODULATED: bool = true;"));
    }
}
