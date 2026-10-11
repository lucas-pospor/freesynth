//! Buffers, pipelines and the schedule of the GPU DIS.
//!
//! All buffers are allocated when a [`GpuDis`] is created, for one frame size. A flow is one
//! compute pass: input conversion, the pyramid, then for every level from the coarsest down
//! to the finest searched one the inverse search passes, densification and the variational
//! refinement, and finally the upsampling into the caller's output buffer. When the finest
//! level computed is the full resolution, its last dispatch writes the caller's buffer
//! directly instead.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{FrameFormat, GpuContext, GpuFlow, GpuFrame, Staging};
use crate::error::{FlowError, invalid};
use crate::flo::FlowField;
use crate::image::GrayImage;
use crate::params::{DisParams, Plan};

const COMMON: &str = include_str!("shaders/common.wgsl");
const SAMPLE: &str = include_str!("shaders/sample.wgsl");
const LOAD: &str = include_str!("shaders/load.wgsl");
const DOWN: &str = include_str!("shaders/down.wgsl");
const UPSAMPLE: &str = include_str!("shaders/upsample.wgsl");
const SEARCH: &str = include_str!("shaders/search.wgsl");
const DENSIFY: &str = include_str!("shaders/densify.wgsl");
const REFINE: &str = include_str!("shaders/vr.wgsl");

/// Workgroup edge of the per-pixel kernels.
const WG: u32 = 16;
/// Patches per search workgroup (eight threads each).
const PATCHES_PER_WG: u32 = 8;
/// Largest workgroup count per dispatch dimension guaranteed by wgpu's default limits.
const MAX_GROUPS: u32 = 65535;
/// Refinement tile size (two pixels per thread along x).
const VR_TILE: (u32, u32) = (32, 16);
/// Tile grid shift of every other refinement dispatch.
const VR_SHIFT: (u32, u32) = (VR_TILE.0 / 2, VR_TILE.1 / 2);
/// Fixed-point iterations per refinement dispatch.
const VR_PER_DISPATCH: usize = 2;

/// Per-level constants, laid out like `Level` in `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct LevelUniform {
    w: u32,
    h: u32,
    pw: u32,
    ph: u32,
    stride: u32,
    iters0: u32,
    iters1: u32,
    mean_norm: u32,
    cw: u32,
    ch: u32,
    inner: u32,
    tol2: f32,
    alpha: f32,
    delta: f32,
    gamma: f32,
    omega: f32,
    scale: f32,
    zeta2: f32,
    eps2: f32,
    margin: f32,
}

/// Binding kinds of a bind group layout (binding 0 is always the level uniform).
#[derive(Clone, Copy)]
enum Slot {
    Read,
    Write,
}

/// Buffers and bind groups of one pyramid level.
#[derive(Debug)]
struct Level {
    w: u32,
    h: u32,
    pw: u32,
    ph: u32,
    i0: wgpu::Buffer,
    i1: wgpu::Buffer,
    down: Option<wgpu::BindGroup>,
    /// Search passes: the first pass, then one per propagation pass.
    search: Vec<wgpu::BindGroup>,
    densify: Option<wgpu::BindGroup>,
    /// True on levels below the finest searched one: no patches; the densification kernel
    /// upsamples the coarser flow, which the refinement then improves.
    from_coarse: bool,
    /// Refinement dispatches, with the number of fixed-point iterations each runs.
    refine: Vec<(wgpu::BindGroup, usize)>,
    /// The buffer that holds this level's final flow.
    flow: Option<wgpu::Buffer>,
    /// On a full-resolution output level: the bindings of its last dispatch except the
    /// output, so that [`GpuDis::encode`] can direct that dispatch to the caller's buffer.
    last: Option<LastDispatch>,
}

/// The uniform and input buffers of a level's last dispatch (the densification, or the last
/// refinement dispatch).
#[derive(Debug)]
struct LastDispatch {
    uniform: wgpu::Buffer,
    inputs: Vec<wgpu::Buffer>,
    refine: bool,
}

/// Compute pipelines and their bind group layouts. Pipelines that differ only in an override
/// constant share a layout.
struct Pipelines {
    /// By input format: [`FrameFormat::Gray32F`], [`FrameFormat::Rgba8`].
    load: [wgpu::ComputePipeline; 2],
    load_layout: wgpu::BindGroupLayout,
    down: wgpu::ComputePipeline,
    down_layout: wgpu::BindGroupLayout,
    /// First pass, propagation pass.
    search: [wgpu::ComputePipeline; 2],
    search_layout: wgpu::BindGroupLayout,
    /// From patches, from the coarser level's flow.
    densify: [wgpu::ComputePipeline; 2],
    densify_layout: wgpu::BindGroupLayout,
    /// By tile shift (none, half a tile) and by fixed-point iterations per dispatch
    /// (1..=VR_PER_DISPATCH).
    refine: [Vec<wgpu::ComputePipeline>; 2],
    refine_layout: wgpu::BindGroupLayout,
    upsample: wgpu::ComputePipeline,
    upsample_layout: wgpu::BindGroupLayout,
}

/// Host-side buffers of [`FlowEstimator::compute`](crate::FlowEstimator::compute), allocated
/// on first use.
struct HostIo {
    frames: [GpuFrame; 2],
    out: GpuFlow,
    staging: Staging,
}

/// Dense Inverse Search on the GPU, for frames of one size.
pub struct GpuDis {
    ctx: GpuContext,
    params: DisParams,
    plan: Plan,
    levels: Vec<Level>,
    pipelines: Pipelines,
    load_uniform: wgpu::Buffer,
    upsample_uniform: wgpu::Buffer,
    host: Option<HostIo>,
}

impl std::fmt::Debug for GpuDis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDis").field("params", &self.params).field("plan", &self.plan).finish_non_exhaustive()
    }
}

impl GpuDis {
    /// Allocates buffers and compiles the kernels for `width` x `height` frames.
    pub fn new(ctx: &GpuContext, width: usize, height: usize, params: &DisParams) -> Result<Self, FlowError> {
        let plan = params.plan(width, height)?;
        let limits = ctx.device().limits();
        // The largest binding is the full-resolution flow.
        let largest = (width * height * 8) as u64;
        if largest > limits.max_storage_buffer_binding_size || largest > limits.max_buffer_size {
            return Err(FlowError::Device(format!(
                "{width}x{height} frames need {largest} byte buffers, the device allows {}",
                limits.max_storage_buffer_binding_size
            )));
        }
        ctx.scoped(|| Ok(Self::build(ctx, params, plan)))
    }

    fn build(ctx: &GpuContext, params: &DisParams, plan: Plan) -> Self {
        let device = ctx.device();
        let pipelines = Pipelines::new(device);
        let storage = |label: &str, bytes: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let uniform = |label: &str, u: &LevelUniform| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(u),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let dummy = storage("flow dummy", 16);
        let passes = params.propagation_passes;
        let base = LevelUniform {
            stride: params.patch_stride as u32,
            iters0: params.gd_iterations as u32,
            iters1: params.propagation_iterations as u32,
            mean_norm: u32::from(params.mean_normalization),
            inner: params.vr_inner_iterations as u32,
            tol2: params.gd_tolerance * params.gd_tolerance,
            alpha: params.vr_alpha,
            delta: params.vr_delta,
            gamma: params.vr_gamma,
            omega: params.vr_omega,
            scale: 2.0,
            zeta2: params.vr_zeta * params.vr_zeta,
            eps2: params.vr_epsilon * params.vr_epsilon,
            margin: params.propagation_margin,
            ..LevelUniform::default()
        };

        // Image pyramid for every level up to the coarsest.
        let mut levels: Vec<Level> = (0..=plan.coarsest)
            .map(|l| {
                let (w, h) = plan.level_size(l);
                Level {
                    w: w as u32,
                    h: h as u32,
                    pw: 0,
                    ph: 0,
                    i0: storage("frame 0 level", w * h * 4),
                    i1: storage("frame 1 level", w * h * 4),
                    down: None,
                    search: Vec::new(),
                    densify: None,
                    from_coarse: false,
                    refine: Vec::new(),
                    flow: None,
                    last: None,
                }
            })
            .collect();
        for l in 1..levels.len() {
            let (fine, coarse) = levels.split_at_mut(l);
            let (src, dst) = (&fine[l - 1], &mut coarse[0]);
            let u = LevelUniform { w: dst.w, h: dst.h, cw: src.w, ch: src.h, ..base };
            dst.down = Some(bind(
                device,
                &pipelines.down_layout,
                &uniform("down uniform", &u),
                &[&src.i0, &src.i1, &dst.i0, &dst.i1],
            ));
        }

        // Search, densification and refinement, coarsest level first. Levels below the finest
        // searched one only upsample the coarser flow and refine it.
        for l in (plan.output_level()..=plan.coarsest).rev() {
            let searched = plan.is_searched(l);
            let refined = plan.is_refined(l);
            let (pw, ph) = if searched { plan.patch_grid(l, params.patch_stride) } else { (0, 0) };
            let coarser = levels.get(l + 1).and_then(|c| c.flow.clone().map(|f| (c.w, c.h, f)));
            let level = &mut levels[l];
            level.pw = pw as u32;
            level.ph = ph as u32;
            level.from_coarse = !searched;
            let (cw, ch, coarse_flow) = coarser.unwrap_or((0, 0, dummy.clone()));
            let ub = uniform("level uniform", &LevelUniform { w: level.w, h: level.h, pw: level.pw, ph: level.ph, cw, ch, ..base });

            let patches = [storage("patch flow A", pw * ph * 8), storage("patch flow B", pw * ph * 8)];
            let search_passes = if searched { passes + 1 } else { 0 };
            level.search = (0..search_passes)
                .map(|k| {
                    let input = if k == 0 { &dummy } else { &patches[(k - 1) % 2] };
                    bind(device, &pipelines.search_layout, &ub, &[&level.i0, &level.i1, &coarse_flow, input, &patches[k % 2]])
                })
                .collect();

            // The densified (or upsampled) flow, then the refinement's iterates in two buffers
            // used in turn.
            let pixels = (level.w * level.h) as usize;
            let dense = storage("level flow", pixels * 8);
            let source = if searched { &patches[passes % 2] } else { &coarse_flow };
            level.densify = Some(bind(device, &pipelines.densify_layout, &ub, &[&level.i0, &level.i1, source, &dense]));
            let iterations = if refined { params.vr_iterations } else { 0 };
            let dispatches = iterations.div_ceil(VR_PER_DISPATCH);
            let iterates: Vec<wgpu::Buffer> = (0..dispatches.min(2)).map(|_| storage("refined flow", pixels * 8)).collect();
            level.refine = (0..dispatches)
                .map(|k| {
                    let input = if k == 0 { &dense } else { &iterates[(k - 1) % 2] };
                    let group =
                        bind(device, &pipelines.refine_layout, &ub, &[&level.i0, &level.i1, &dense, input, &iterates[k % 2]]);
                    (group, VR_PER_DISPATCH.min(iterations - k * VR_PER_DISPATCH))
                })
                .collect();
            if l == 0 {
                let inputs = if dispatches == 0 {
                    vec![level.i0.clone(), level.i1.clone(), source.clone()]
                } else {
                    let input = if dispatches == 1 { &dense } else { &iterates[dispatches % 2] };
                    vec![level.i0.clone(), level.i1.clone(), dense.clone(), input.clone()]
                };
                level.last = Some(LastDispatch { uniform: ub.clone(), inputs, refine: dispatches > 0 });
            }
            level.flow = Some(if dispatches == 0 { dense } else { iterates[(dispatches - 1) % 2].clone() });
        }

        let output = plan.output_level();
        let (ow, oh) = plan.level_size(output);
        let load_uniform = uniform("load uniform", &LevelUniform { w: plan.width as u32, h: plan.height as u32, ..base });
        let upsample_uniform = uniform(
            "upsample uniform",
            &LevelUniform {
                w: plan.width as u32,
                h: plan.height as u32,
                cw: ow as u32,
                ch: oh as u32,
                scale: (1u32 << output) as f32,
                ..base
            },
        );
        Self { ctx: ctx.clone(), params: params.clone(), plan, levels, pipelines, load_uniform, upsample_uniform, host: None }
    }

    pub fn params(&self) -> &DisParams {
        &self.params
    }

    /// The pyramid levels in use.
    pub fn plan(&self) -> Plan {
        self.plan
    }

    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Frame size this instance was built for.
    pub fn size(&self) -> (usize, usize) {
        (self.plan.width, self.plan.height)
    }

    /// Records the flow from `frame0` to `frame1` into `encoder`, written to `out`:
    /// `frame0(x) ~ frame1(x + out(x))`. Everything runs in one compute pass, which writes
    /// `timestamps` if given. The frames and the output must have this instance's size.
    ///
    /// Several flows can be recorded into one encoder; they run one after another (they share
    /// this instance's scratch buffers). For flows that should run concurrently, use one
    /// `GpuDis` per stream.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame0: &GpuFrame,
        frame1: &GpuFrame,
        out: &GpuFlow,
        timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<(), FlowError> {
        let size = self.size();
        if (frame0.width, frame0.height) != size || (frame1.width, frame1.height) != size {
            return invalid(format!("frames must be {}x{}", size.0, size.1));
        }
        if frame0.format != frame1.format {
            return invalid("both frames must have the same format");
        }
        if (out.width, out.height) != size {
            return invalid(format!("the output flow must be {}x{}", size.0, size.1));
        }
        let device = self.ctx.device();
        let p = &self.pipelines;
        let format = usize::from(frame0.format == FrameFormat::Rgba8);
        let level0 = &self.levels[0];
        let load = bind(device, &p.load_layout, &self.load_uniform, &[&frame0.buffer, &frame1.buffer, &level0.i0, &level0.i1]);
        let output_level = self.plan.output_level();
        let finest = &self.levels[output_level];
        // A full-resolution output level writes `out` in its last dispatch; otherwise the
        // output level's flow is upsampled into `out`.
        let (last, upsample) = match &finest.last {
            Some(last) => {
                let mut buffers: Vec<&wgpu::Buffer> = last.inputs.iter().collect();
                buffers.push(&out.buffer);
                let layout = if last.refine { &p.refine_layout } else { &p.densify_layout };
                (Some(bind(device, layout, &last.uniform, &buffers)), None)
            }
            None => {
                let flow = finest.flow.as_ref().expect("the output level is computed");
                (None, Some(bind(device, &p.upsample_layout, &self.upsample_uniform, &[flow, &out.buffer])))
            }
        };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dis"), timestamp_writes: timestamps });
        let per_pixel = |w: u32, h: u32| (w.div_ceil(WG), h.div_ceil(WG));
        pass.set_pipeline(&p.load[format]);
        pass.set_bind_group(0, &load, &[]);
        let (gx, gy) = per_pixel(level0.w, level0.h);
        pass.dispatch_workgroups(gx, gy, 1);
        pass.set_pipeline(&p.down);
        for level in &self.levels[1..] {
            pass.set_bind_group(0, level.down.as_ref().expect("coarser levels downsample"), &[]);
            let (gx, gy) = per_pixel(level.w, level.h);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        for level in self.levels[output_level..].iter().rev() {
            // The bind group of this level's last dispatch if it writes `out`.
            let last = last.as_ref().filter(|_| level.last.is_some());
            let groups = (level.pw * level.ph).div_ceil(PATCHES_PER_WG);
            let (sx, sy) = (groups.min(MAX_GROUPS), groups.div_ceil(MAX_GROUPS));
            for (k, group) in level.search.iter().enumerate() {
                pass.set_pipeline(&p.search[usize::from(k > 0)]);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(sx, sy, 1);
            }
            let (gx, gy) = per_pixel(level.w, level.h);
            pass.set_pipeline(&p.densify[usize::from(level.from_coarse)]);
            let densify = level.densify.as_ref().expect("every computed level densifies");
            pass.set_bind_group(0, last.filter(|_| level.refine.is_empty()).unwrap_or(densify), &[]);
            pass.dispatch_workgroups(gx, gy, 1);
            for (k, (group, outer)) in level.refine.iter().enumerate() {
                let (sx, sy) = if k % 2 == 1 { VR_SHIFT } else { (0, 0) };
                pass.set_pipeline(&p.refine[k % 2][outer - 1]);
                pass.set_bind_group(0, last.filter(|_| k + 1 == level.refine.len()).unwrap_or(group), &[]);
                pass.dispatch_workgroups((level.w + sx).div_ceil(VR_TILE.0), (level.h + sy).div_ceil(VR_TILE.1), 1);
            }
        }
        if let Some(upsample) = &upsample {
            pass.set_pipeline(&p.upsample);
            pass.set_bind_group(0, upsample, &[]);
            let (gx, gy) = per_pixel(level0.w, level0.h);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        Ok(())
    }
}

impl crate::FlowEstimator for GpuDis {
    fn compute(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<FlowField, FlowError> {
        let (w, h) = self.size();
        if frame0.size() != (w, h) || frame1.size() != (w, h) {
            return invalid(format!("frames must be {w}x{h}"));
        }
        let ctx = self.ctx.clone();
        let host = self.host.get_or_insert_with(|| HostIo {
            frames: [GpuFrame::new(&ctx, w, h, FrameFormat::Gray32F), GpuFrame::new(&ctx, w, h, FrameFormat::Gray32F)],
            out: GpuFlow::new(&ctx, w, h),
            staging: Staging::new(&ctx, w, h),
        });
        host.frames[0].write_gray(&ctx, frame0)?;
        host.frames[1].write_gray(&ctx, frame1)?;
        let host = self.host.as_ref().expect("allocated above");
        ctx.scoped(|| {
            let mut encoder = ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dis") });
            self.encode(&mut encoder, &host.frames[0], &host.frames[1], &host.out, None)?;
            ctx.queue().submit([encoder.finish()]);
            host.staging.read(&ctx, &host.out)
        })
    }
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        use Slot::{Read, Write};
        let module = |label: &str, parts: &[&str]| {
            let source: String = std::iter::once(COMMON).chain(parts.iter().copied()).collect::<Vec<_>>().join("\n");
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(source.into()) })
        };
        let pipeline = |module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, entry: &str, constants: &[(&str, f64)]| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
                cache: None,
            })
        };
        let load_layout = layout(device, &[Read, Read, Write, Write]);
        let load_module = module("dis load", &[LOAD]);
        let load = [0.0, 1.0].map(|f| pipeline(&load_module, &load_layout, "load", &[("FORMAT", f)]));
        let down_layout = layout(device, &[Read, Read, Write, Write]);
        let down = pipeline(&module("dis down", &[DOWN]), &down_layout, "down", &[]);
        let search_layout = layout(device, &[Read, Read, Read, Read, Write]);
        let search_module = module("dis search", &[SAMPLE, SEARCH]);
        let search = [0.0, 1.0].map(|p| pipeline(&search_module, &search_layout, "search", &[("PROPAGATE", p)]));
        let densify_layout = layout(device, &[Read, Read, Read, Write]);
        let densify_module = module("dis densify", &[SAMPLE, DENSIFY]);
        let densify = [0.0, 1.0].map(|c| pipeline(&densify_module, &densify_layout, "densify", &[("FROM_COARSE", c)]));
        let refine_layout = layout(device, &[Read, Read, Read, Read, Write]);
        let refine_module = module("dis refine", &[REFINE]);
        let refine = [(0, 0), VR_SHIFT].map(|(x, y)| {
            (1..=VR_PER_DISPATCH)
                .map(|outer| {
                    let constants = [("SHIFT_X", f64::from(x)), ("SHIFT_Y", f64::from(y)), ("OUTER", outer as f64)];
                    pipeline(&refine_module, &refine_layout, "refine", &constants)
                })
                .collect()
        });
        let upsample_layout = layout(device, &[Read, Write]);
        let upsample = pipeline(&module("dis upsample", &[SAMPLE, UPSAMPLE]), &upsample_layout, "upsample", &[]);
        Self {
            load,
            load_layout,
            down,
            down_layout,
            search,
            search_layout,
            densify,
            densify_layout,
            refine,
            refine_layout,
            upsample,
            upsample_layout,
        }
    }
}

/// A bind group layout: a uniform at binding 0, then storage buffers.
fn layout(device: &wgpu::Device, slots: &[Slot]) -> wgpu::BindGroupLayout {
    let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let entries: Vec<_> = std::iter::once(entry(0, wgpu::BufferBindingType::Uniform))
        .chain(slots.iter().enumerate().map(|(i, s)| {
            entry(i as u32 + 1, wgpu::BufferBindingType::Storage { read_only: matches!(s, Slot::Read) })
        }))
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries })
}

/// A bind group with `uniform` at binding 0 and `buffers` at bindings 1, 2, ...
fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, uniform: &wgpu::Buffer, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<_> = std::iter::once(uniform)
        .chain(buffers.iter().copied())
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_layout_matches_the_shader() {
        // 20 four-byte fields; WGSL rounds uniform structs to 16 bytes.
        assert_eq!(std::mem::size_of::<LevelUniform>(), 80);
    }
}
