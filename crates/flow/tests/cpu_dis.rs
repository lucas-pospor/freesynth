//! CPU DIS against known motion.

use freesynth_flow::cpu::CpuDis;
use freesynth_flow::{DisParams, FlowEstimator, FlowField, GrayImage, Preset};
use freesynth_scenes::{Scene, SceneParams, StyleFamily};

/// A smooth, non-repetitive texture moved by `shift`: `render(0)(x) = render(shift)(x + shift)`.
fn render(w: usize, h: usize, shift: [f32; 2]) -> GrayImage {
    let texture = |x: f32, y: f32| {
        128.0
            + 30.0 * (0.21 * x + 0.07 * y + 0.3).sin()
            + 25.0 * (-0.05 * x + 0.17 * y + 1.1).sin()
            + 20.0 * (0.11 * x - 0.13 * y + 2.0).sin()
            + 12.0 * (0.33 * x + 0.29 * y + 0.7).sin()
    };
    let data = (0..w * h).map(|i| texture((i % w) as f32 - shift[0], (i / w) as f32 - shift[1])).collect();
    GrayImage::from_vec(w, h, data).unwrap()
}

#[test]
fn recovers_constant_translation() {
    let (w, h) = (160, 120);
    let shift = [2.75, -1.5];
    let (a, b) = (render(w, h, [0.0, 0.0]), render(w, h, shift));
    let margin = 12;
    let mask: Vec<bool> = (0..w * h)
        .map(|i| (margin..w - margin).contains(&(i % w)) && (margin..h - margin).contains(&(i / w)))
        .collect();
    for (preset, bound) in [(Preset::UltraFast, 0.15), (Preset::Fast, 0.1), (Preset::Medium, 0.02)] {
        let flow = CpuDis::new(w, h, &DisParams::preset(preset)).unwrap().compute(&a, &b).unwrap();
        let epe = flow.mean_epe(&FlowField::constant(w, h, shift), Some(&mask)).unwrap();
        eprintln!("{preset}: EPE {epe:.4} px");
        assert!(epe < bound, "{preset}: EPE {epe:.4} px");
    }
}

/// End-point error against the exact flow of a generated scene (rigid objects moving over a
/// panning background, with occlusion), for frame gaps of 1 and 5.
#[test]
fn exact_flow_scene() {
    let params = SceneParams {
        width: 320,
        height: 180,
        frames: 6,
        seed: 3,
        family: StyleFamily::Rigid,
        ..SceneParams::default()
    };
    let scene = Scene::new(&params).unwrap();
    let (w, h) = (params.width, params.height);
    let gray = |f: &freesynth_scenes::Frame| GrayImage::from_rgb8(w, h, f.image.as_raw()).unwrap();
    let frames: Vec<_> = [0, 1, 5].iter().map(|&t| scene.render(t)).collect();
    let mut dis = CpuDis::new(w, h, &DisParams::default()).unwrap();
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

/// The estimator gives the same flow on any thread pool (the work split must not matter).
#[test]
fn independent_of_the_thread_count() {
    let (w, h) = (130, 70);
    let (a, b) = (render(w, h, [0.0, 0.0]), render(w, h, [-1.25, 0.75]));
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| CpuDis::new(w, h, &DisParams::default()).unwrap().compute(&a, &b).unwrap())
    };
    assert_eq!(run(1), run(5));
}
