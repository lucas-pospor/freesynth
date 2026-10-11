//! GPU DIS against known motion. The tests skip themselves when no GPU adapter is available,
//! or when `FREESYNTH_NO_GPU` is set.

use freesynth_flow::cpu::CpuDis;
use freesynth_flow::gpu::{FrameFormat, GpuContext, GpuDis, GpuFlow, GpuFrame};
use freesynth_flow::{DisParams, FlowEstimator, FlowField, GrayImage, Preset};
use freesynth_scenes::{Scene, SceneParams, StyleFamily};

/// The GPU context, or `None` (test skipped) when the machine has no usable adapter.
fn context() -> Option<GpuContext> {
    if std::env::var_os("FREESYNTH_NO_GPU").is_some() {
        eprintln!("skipped: FREESYNTH_NO_GPU is set");
        return None;
    }
    match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

/// A smooth, non-repetitive texture: a sum of oriented sine waves, in 0-255.
fn texture(x: f32, y: f32) -> f32 {
    const WAVES: [(f32, f32, f32, f32); 6] = [
        (0.21, 0.07, 0.3, 30.0),
        (-0.05, 0.17, 1.1, 25.0),
        (0.11, -0.13, 2.0, 20.0),
        (0.33, 0.29, 0.7, 12.0),
        (-0.27, 0.41, 2.9, 10.0),
        (0.02, 0.05, 0.4, 20.0),
    ];
    128.0 + WAVES.iter().map(|&(fx, fy, ph, amp)| amp * (fx * x + fy * y + ph).sin()).sum::<f32>()
}

/// The texture moved by `shift`, so that `render(0)(x) = render(shift)(x + shift)`.
fn render(w: usize, h: usize, shift: [f32; 2]) -> GrayImage {
    let data = (0..w * h).map(|i| texture((i % w) as f32 - shift[0], (i / w) as f32 - shift[1])).collect();
    GrayImage::from_vec(w, h, data).unwrap()
}

/// Packed RGBA8 bytes of a grey image (rounded).
fn to_rgba8(img: &GrayImage) -> Vec<u8> {
    img.data().iter().flat_map(|&v| {
        let g = v.round().clamp(0.0, 255.0) as u8;
        [g, g, g, 255]
    }).collect()
}

/// Mean end-point error against a constant flow, ignoring a border of `margin` pixels.
fn interior_epe(flow: &FlowField, truth: [f32; 2], margin: usize) -> f64 {
    let (w, h) = flow.size();
    let mask: Vec<bool> =
        (0..w * h).map(|i| (margin..w - margin).contains(&(i % w)) && (margin..h - margin).contains(&(i / w))).collect();
    flow.mean_epe(&FlowField::constant(w, h, truth), Some(&mask)).unwrap()
}

fn max_difference(a: &FlowField, b: &FlowField) -> f32 {
    a.data().iter().zip(b.data()).map(|(p, q)| (p[0] - q[0]).abs().max((p[1] - q[1]).abs())).fold(0.0, f32::max)
}

#[test]
fn recovers_constant_translation() {
    let Some(ctx) = context() else { return };
    let (w, h) = (160, 120);
    let shift = [2.75, -1.5];
    let (a, b) = (render(w, h, [0.0, 0.0]), render(w, h, shift));
    // The quarter-resolution presets lose the texture's finest waves to aliasing.
    for (preset, bound) in [(Preset::UltraFast, 0.15), (Preset::Fast, 0.1), (Preset::Medium, 0.02)] {
        let mut dis = GpuDis::new(&ctx, w, h, &DisParams::preset(preset)).unwrap();
        let flow = dis.compute(&a, &b).unwrap();
        let epe = interior_epe(&flow, shift, 12);
        eprintln!("{preset}: EPE {epe:.4} px");
        assert!(epe < bound, "{preset}: EPE {epe:.4} px");
    }
}

/// Packed RGBA input, several flows recorded into one command buffer, and a device created
/// outside the crate.
#[test]
fn shared_device_and_batched_encoding() {
    if context().is_none() {
        return;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let ctx = GpuContext::from_device(device, queue);
    assert!(ctx.adapter_name().is_none());

    let (w, h) = (128, 96);
    let shifts = [[1.0, 0.5], [-3.25, 2.0]];
    let base = render(w, h, [0.0, 0.0]);
    let moved: Vec<GrayImage> = shifts.iter().map(|&s| render(w, h, s)).collect();
    let mut dis = GpuDis::new(&ctx, w, h, &DisParams::default()).unwrap();

    // Reference: one flow at a time from grey images (rounded like the RGBA data below).
    let round = |img: &GrayImage| GrayImage::from_vec(w, h, img.data().iter().map(|v| v.round().clamp(0.0, 255.0)).collect()).unwrap();
    let single: Vec<FlowField> = moved.iter().map(|m| dis.compute(&round(&base), &round(m)).unwrap()).collect();

    let frame = |img: &GrayImage| {
        let f = GpuFrame::new(&ctx, w, h, FrameFormat::Rgba8);
        f.write_rgba8(&ctx, &to_rgba8(img)).unwrap();
        f
    };
    let f0 = frame(&base);
    let others: Vec<GpuFrame> = moved.iter().map(frame).collect();
    let outs: Vec<GpuFlow> = shifts.iter().map(|_| GpuFlow::new(&ctx, w, h)).collect();
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    for (f1, out) in others.iter().zip(&outs) {
        dis.encode(&mut encoder, &f0, f1, out, None).unwrap();
    }
    ctx.queue().submit([encoder.finish()]);
    for ((out, reference), shift) in outs.iter().zip(&single).zip(shifts) {
        let flow = out.read(&ctx).unwrap();
        let diff = max_difference(&flow, reference);
        assert!(diff < 1e-3, "batched RGBA flow differs from the single grey flow by {diff}");
        let epe = interior_epe(&flow, shift, 12);
        assert!(epe < 0.05, "shift {shift:?}: EPE {epe:.4} px");
    }
}

/// End-point error against the exact flow of a generated scene (rigid objects moving over a
/// panning background, with occlusion), for frame gaps of 1 and 5.
#[test]
fn exact_flow_scene() {
    let Some(ctx) = context() else { return };
    let params = SceneParams { width: 320, height: 180, frames: 6, seed: 3, family: StyleFamily::Rigid, ..SceneParams::default() };
    let scene = Scene::new(&params).unwrap();
    let (w, h) = (params.width, params.height);
    let gray = |f: &freesynth_scenes::Frame| GrayImage::from_rgb8(w, h, f.image.as_raw()).unwrap();
    let frames: Vec<_> = [0, 1, 5].iter().map(|&t| scene.render(t)).collect();
    let mut dis = GpuDis::new(&ctx, w, h, &DisParams::default()).unwrap();
    for (k, gap, bound) in [(1, 1, 0.15), (2, 5, 0.6)] {
        let exact = scene.flow(0, gap, &frames[0].layers, &frames[k].layers);
        let visible: Vec<bool> = exact.occlusion.as_raw().iter().map(|&v| v == 0).collect();
        let flow = dis.compute(&gray(&frames[0]), &gray(&frames[k])).unwrap();
        let epe = flow.mean_epe(&exact.flow, Some(&visible)).unwrap();
        let zero = FlowField::new(w, h).mean_epe(&exact.flow, Some(&visible)).unwrap();
        eprintln!("gap {gap}: EPE {epe:.4} px (zero flow {zero:.3} px)");
        assert!(epe < bound && epe < 0.2 * zero, "gap {gap}: EPE {epe:.4} px, zero flow {zero:.3} px");
    }
}

/// Mean and 99th percentile of the per-pixel distance between two flows.
fn distance(a: &FlowField, b: &FlowField) -> (f64, f32) {
    let mut d: Vec<f32> = a.data().iter().zip(b.data()).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).collect();
    let mean = d.iter().map(|&v| f64::from(v)).sum::<f64>() / d.len() as f64;
    d.sort_by(f32::total_cmp);
    (mean, d[(d.len() - 1) * 99 / 100])
}

/// The CPU estimator computes the same flow as the GPU kernels, up to rounding: on a scene with
/// occlusion and at frame gaps 1 and 5, for every preset and for two settings whose last
/// full-resolution dispatch is the densification or a single refinement dispatch.
#[test]
fn cpu_and_gpu_agree() {
    let Some(ctx) = context() else { return };
    let params = SceneParams { width: 320, height: 180, frames: 6, seed: 3, family: StyleFamily::Rigid, ..SceneParams::default() };
    let scene = Scene::new(&params).unwrap();
    let (w, h) = (params.width, params.height);
    let gray = |t: usize| GrayImage::from_rgb8(w, h, scene.render(t).image.as_raw()).unwrap();
    let (f0, f1, f5) = (gray(0), gray(1), gray(5));
    let settings = Preset::ALL.map(|p| (p.name(), DisParams::preset(p))).into_iter().chain([
        ("full-resolution search", DisParams { finest_level: 0, vr_iterations: 0, ..DisParams::default() }),
        ("one refinement dispatch", DisParams { vr_iterations: 1, ..DisParams::default() }),
    ]);
    for (preset, p) in settings {
        let mut gpu = GpuDis::new(&ctx, w, h, &p).unwrap();
        let mut cpu = CpuDis::new(w, h, &p).unwrap();
        for (gap, other) in [(1, &f1), (5, &f5)] {
            let (g, c) = (gpu.compute(&f0, other).unwrap(), cpu.compute(&f0, other).unwrap());
            let (mean, p99) = distance(&g, &c);
            eprintln!("{preset}, gap {gap}: mean distance {mean:.6} px, 99th percentile {p99:.5} px");
            assert!(mean < 0.005 && p99 < 0.02, "{preset}, gap {gap}: mean distance {mean} px, 99th percentile {p99} px");
        }
    }
}

#[test]
fn odd_and_tiny_sizes() {
    let Some(ctx) = context() else { return };
    for (w, h) in [(101, 67), (8, 8), (9, 33)] {
        let shift = [0.5, 0.25];
        let (a, b) = (render(w, h, [0.0, 0.0]), render(w, h, shift));
        for preset in Preset::ALL {
            let flow = GpuDis::new(&ctx, w, h, &DisParams::preset(preset)).unwrap().compute(&a, &b).unwrap();
            assert_eq!(flow.size(), (w, h));
            assert!(flow.data().iter().all(|v| v[0].is_finite() && v[1].is_finite()), "{w}x{h} {preset}: non-finite flow");
            let epe = interior_epe(&flow, shift, 0);
            assert!(epe < 0.5, "{w}x{h} {preset}: EPE {epe:.4} px");
        }
    }
}

#[test]
fn rejects_bad_inputs() {
    let Some(ctx) = context() else { return };
    assert!(GpuDis::new(&ctx, 7, 64, &DisParams::default()).is_err());
    assert!(GpuDis::new(&ctx, 64, 64, &DisParams { patch_stride: 0, ..DisParams::default() }).is_err());
    let mut dis = GpuDis::new(&ctx, 64, 48, &DisParams::default()).unwrap();
    let (a, small) = (render(64, 48, [0.0, 0.0]), render(32, 48, [0.0, 0.0]));
    assert!(dis.compute(&a, &small).is_err());
    let f = GpuFrame::new(&ctx, 64, 48, FrameFormat::Gray32F);
    let g = GpuFrame::new(&ctx, 64, 48, FrameFormat::Rgba8);
    let wrong = GpuFrame::new(&ctx, 48, 64, FrameFormat::Gray32F);
    let out = GpuFlow::new(&ctx, 64, 48);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    assert!(dis.encode(&mut encoder, &f, &g, &out, None).is_err());
    assert!(dis.encode(&mut encoder, &f, &wrong, &out, None).is_err());
    assert!(dis.encode(&mut encoder, &f, &f, &GpuFlow::new(&ctx, 8, 8), None).is_err());
    assert!(f.write_rgba8(&ctx, &[0; 64 * 48 * 4]).is_err());
    assert!(g.write_gray(&ctx, &a).is_err());
    // A still-usable instance after the rejections.
    assert_eq!(dis.compute(&a, &a).unwrap().size(), (64, 48));
}
