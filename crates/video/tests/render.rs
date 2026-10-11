//! End-to-end renders of small generated scenes. GPU tests skip without an adapter or when
//! `FREESYNTH_NO_GPU` is set.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use freesynth_core::{Engine, Image, Nnf, Params, SynthesisInput, SynthesisOutput};
use freesynth_flow::{DisParams, Preset as FlowPreset};
use freesynth_scenes::{Scene, SceneParams, StyleFamily};
use freesynth_video::resources::MAX_DEFAULT_PARALLEL;
use freesynth_video::{
    Backend, CancelToken, ComputeDevice, FrameSink, Keyframe, KeyframeOutput, MemoryFrames, MemorySink, Progress,
    RenderOptions, RenderStats, Renderer, Shot, Synthesizer, VideoError, VideoParams,
};

fn rgba(rgb: &[u8], w: usize, h: usize) -> Image {
    let data = rgb.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    Image::from_raw(w, h, 4, data)
}

/// Video frames and ground truth of a generated scene, in memory.
fn scene(width: usize, height: usize, frames: usize) -> (MemoryFrames, Vec<Image>) {
    let params = SceneParams { width, height, frames, seed: 5, family: StyleFamily::Rigid, ..SceneParams::default() };
    let scene = Scene::new(&params).unwrap();
    let (video, truth): (Vec<_>, Vec<_>) = (0..frames)
        .map(|t| {
            let f = scene.render(t);
            (rgba(f.image.as_raw(), width, height), rgba(f.truth.as_raw(), width, height))
        })
        .unzip();
    (MemoryFrames::new(0, video).unwrap(), truth)
}

fn scratch(name: &str) -> PathBuf {
    let target = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp"));
    let dir = target.join(format!("video-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    let n = a.width() * a.height();
    let rgb = |img: &Image, i: usize, c: usize| f64::from(img.data()[i * img.channels() + c]);
    (0..n).map(|i| (0..3).map(|c| (rgb(a, i, c) - rgb(b, i, c)).abs()).sum::<f64>()).sum::<f64>() / (3 * n) as f64
}

fn fast_params() -> VideoParams {
    VideoParams { flow: DisParams::preset(FlowPreset::Fast), ..VideoParams::classic() }
}

/// A stand-in engine that paints every output with the style's first pixel, so that outputs
/// show which keyframe produced them. Counts its calls.
#[derive(Default)]
struct SolidColour {
    calls: AtomicUsize,
}

impl Synthesizer for SolidColour {
    fn engine(&self) -> Engine {
        Engine::Cpu
    }

    fn synthesize(&self, input: &SynthesisInput, _params: &Params) -> Result<SynthesisOutput, VideoError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (w, h) = input.target_guide.size();
        let c = input.source_style.channels();
        let colour = input.source_style.pixel(0, 0).to_vec();
        let data = std::iter::repeat_n(colour, w * h).flatten().collect();
        Ok(SynthesisOutput { image: Image::from_raw(w, h, c, data), nnf: Nnf::new(w, h), errors: None })
    }
}

fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Image {
    rgba(&rgb.repeat(w * h), w, h)
}

#[test]
fn renders_every_frame_once_and_blends_linearly() {
    let (w, h) = (64, 48);
    let (frames, _) = scene(w, h, 9);
    let keyframes =
        vec![Keyframe { frame: 8, style: solid(w, h, [0, 0, 250]) }, Keyframe { frame: 2, style: solid(w, h, [200, 0, 0]) }];
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let engine = SolidColour::default();
    let device = ComputeDevice::cpu();
    let sink = MemorySink::new();
    let stats = Renderer::with_engines(&engine, &device, fast_params()).render(&shot, &sink).unwrap();
    let out = sink.into_frames();
    assert_eq!(out.len(), 9);
    // Keyframes 2 and 8, a backward chain 1..0, and two passes of 5 frames between them.
    assert_eq!(stats.syntheses, 2 + 2 + 2 * 5);
    assert_eq!(engine.calls.load(Ordering::Relaxed), stats.syntheses);
    assert_eq!((stats.frames, stats.chains), (9, 3));
    for (t, img) in &out {
        let expect = match t {
            0..=2 => [200, 0, 0],
            8 => [0, 0, 250],
            _ => {
                let a = (*t - 2) as f32 / 6.0;
                [(200.0 * (1.0 - a)).round() as u8, 0, (250.0 * a).round() as u8]
            }
        };
        assert_eq!(img.pixel(5, 5), &expect, "frame {t}");
        assert_eq!(img.channels(), 3);
    }
    assert!(stats.flow.computed > 0);
    assert!(stats.flow.reused > 0, "the temporal guide reuses the tracker's flows");
}

#[test]
fn cpu_render_follows_the_scene() {
    let (w, h) = (96, 64);
    let (frames, truth) = scene(w, h, 7);
    let keyframes = vec![
        Keyframe { frame: 0, style: truth[0].clone() },
        Keyframe { frame: 6, style: truth[6].clone() },
    ];
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let device = ComputeDevice::new(Backend::Cpu).unwrap();
    let sink = MemorySink::new();
    let reports = Mutex::new(Vec::new());
    let on_progress = |p: &Progress| reports.lock().unwrap().push(p.clone());
    let stats = Renderer::new(&device, fast_params()).on_progress(&on_progress).render(&shot, &sink).unwrap();
    assert_eq!((stats.frames, stats.syntheses, stats.parallel), (7, 12, 2));
    let out = sink.into_frames();
    for t in 1..6 {
        // Closer to the ground truth than either keyframe left in place.
        let err = mean_abs_diff(&out[&t], &truth[t]);
        let still = mean_abs_diff(&truth[0], &truth[t]).min(mean_abs_diff(&truth[6], &truth[t]));
        assert!(err < still, "frame {t}: {err:.2} vs {still:.2} for an unmoved keyframe");
    }
    let reports = reports.into_inner().unwrap();
    assert!(matches!(reports[0], Progress::Started { frames: 7, syntheses: 12, chains: 2, parallel: 2 }));
    assert_eq!(reports.iter().filter(|p| matches!(p, Progress::Synthesized { .. })).count(), 12);
    assert_eq!(reports.iter().filter(|p| matches!(p, Progress::Written { .. })).count(), 7);
}

#[test]
fn painting_at_keyframes_and_masks() {
    let (w, h) = (64, 48);
    let (frames, truth) = scene(w, h, 4);
    let masks = MemoryFrames::new(0, (0..4).map(|_| solid(w, h, [255, 255, 255])).collect()).unwrap();
    let shot = Shot {
        frames: &frames,
        keyframes: vec![Keyframe { frame: 1, style: truth[1].clone() }],
        mask: Some(&masks),
        range: Some((0, 3)),
    };
    let device = ComputeDevice::cpu();
    let sink = MemorySink::new();
    let params = VideoParams { keyframe_output: KeyframeOutput::Painting, gate_mapping: true, ..fast_params() };
    Renderer::new(&device, params).options(RenderOptions { parallel_chains: Some(1), ..Default::default() }).render(&shot, &sink).unwrap();
    let out = sink.into_frames();
    assert_eq!(out.len(), 4);
    let painting: Vec<u8> = truth[1].data().as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    assert_eq!(out[&1].data(), &painting[..]);
}

#[test]
fn rejects_inconsistent_shots() {
    let (frames, truth) = scene(32, 32, 3);
    let device = ComputeDevice::cpu();
    let render = |keyframes: Vec<Keyframe>, range| {
        let shot = Shot { frames: &frames, keyframes, mask: None, range };
        Renderer::new(&device, fast_params()).render(&shot, &MemorySink::new())
    };
    assert!(render(vec![], None).is_err());
    assert!(render(vec![Keyframe { frame: 5, style: truth[0].clone() }], None).is_err());
    assert!(render(vec![Keyframe { frame: 0, style: solid(16, 16, [0; 3]) }], None).is_err());
    assert!(render(vec![Keyframe { frame: 0, style: truth[0].clone() }], Some((0, 7))).is_err());
}

#[test]
fn cancellation_stops_the_render() {
    let (frames, truth) = scene(48, 32, 12);
    let shot = Shot { frames: &frames, keyframes: vec![Keyframe { frame: 0, style: truth[0].clone() }], mask: None, range: None };
    let engine = SolidColour::default();
    let device = ComputeDevice::cpu();
    let token = CancelToken::new();
    let on_progress = |p: &Progress| {
        if let Progress::Synthesized { done: 3, .. } = p {
            token.cancel();
        }
    };
    let result = Renderer::with_engines(&engine, &device, fast_params())
        .cancel_token(token.clone())
        .on_progress(&on_progress)
        .render(&shot, &MemorySink::new());
    assert!(matches!(result, Err(VideoError::Cancelled)), "{result:?}");
    assert!(engine.calls.load(Ordering::Relaxed) < 12);
}

/// Renders a scene with the solid-colour engine and memory budgets of about one frame, checks
/// that the frame cache and the pass buffers spilled to disk, and that every output is right.
fn render_with_spills(w: usize, h: usize, frames: usize, name: &str) {
    let (video, _) = scene(w, h, frames);
    let last = frames - 1;
    let keyframes = vec![
        Keyframe { frame: 0, style: solid(w, h, [240, 10, 0]) },
        Keyframe { frame: last, style: solid(w, h, [0, 10, 240]) },
    ];
    let shot = Shot { frames: &video, keyframes, mask: None, range: None };
    let engine = SolidColour::default();
    let device = ComputeDevice::cpu();
    let spill = scratch(name);
    let options = RenderOptions {
        frame_cache_bytes: w * h * 4,
        buffer_bytes: w * h * 3,
        spill_dir: Some(spill.clone()),
        ..RenderOptions::default()
    };
    let params = VideoParams { flow: DisParams::preset(FlowPreset::UltraFast), ..VideoParams::classic() };
    let sink = MemorySink::new();
    let stats = Renderer::with_engines(&engine, &device, params).options(options).render(&shot, &sink).unwrap();
    println!("{stats:#?}");
    assert!(stats.frame_cache.spilled > 0, "{:?}", stats.frame_cache);
    // The budget plus about one frame per thread using the store (two chains here).
    assert!(stats.frame_cache.peak_bytes <= 3 * w * h * 4, "{:?}", stats.frame_cache);
    assert!(stats.buffers.spilled > 0, "{:?}", stats.buffers);
    assert!(stats.buffers.peak_bytes <= 3 * w * h * 3, "{:?}", stats.buffers);
    let out = sink.into_frames();
    assert_eq!(out.len(), frames);
    for (t, img) in &out {
        let a = *t as f32 / last as f32;
        let expect = [(240.0 * (1.0 - a)).round() as u8, 10, (240.0 * a).round() as u8];
        assert_eq!(img.pixel(w - 1, h - 1), &expect, "frame {t}");
    }
    assert!(!spill.exists(), "spill files are removed after the render");
}

#[test]
fn frames_and_passes_spill_to_disk() {
    render_with_spills(160, 96, 6, "spill");
}

/// The same with 4K frames, where one interval of frames is gigabytes. Generating the scene
/// takes minutes without optimizations: run with `cargo test --release -- --ignored`.
#[test]
#[ignore = "slow without optimizations; run with --release -- --ignored"]
fn large_frames_spill_to_disk() {
    render_with_spills(3840, 2160, 4, "spill4k");
}

#[test]
fn gpu_render_matches_cpu_quality() {
    if std::env::var_os("FREESYNTH_NO_GPU").is_some() {
        eprintln!("skipped: FREESYNTH_NO_GPU is set");
        return;
    }
    let device = match ComputeDevice::new(Backend::Gpu) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    println!("device: {}", device.name());
    let (w, h) = (96, 64);
    let (frames, truth) = scene(w, h, 7);
    let keyframes = vec![Keyframe { frame: 0, style: truth[0].clone() }, Keyframe { frame: 6, style: truth[6].clone() }];
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let run = |device: &ComputeDevice| {
        let sink = MemorySink::new();
        Renderer::new(device, VideoParams::classic()).render(&shot, &sink).unwrap();
        let out = sink.into_frames();
        (1..6).map(|t| mean_abs_diff(&out[&t], &truth[t])).sum::<f64>() / 5.0
    };
    let gpu = run(&device);
    let cpu = run(&ComputeDevice::cpu());
    println!("mean error against the ground truth: GPU {gpu:.2}, CPU {cpu:.2}");
    assert!(gpu < 1.5 * cpu + 1.0 && cpu < 1.5 * gpu + 1.0, "GPU {gpu:.2} vs CPU {cpu:.2}");
}

#[test]
fn concurrent_renders_can_share_a_spill_directory() {
    let (w, h, n) = (64, 48, 12);
    let (video, _) = scene(w, h, n);
    let spill = scratch("shared-spill");
    let device = ComputeDevice::cpu();
    let run = |c0: [u8; 3], c1: [u8; 3]| {
        let keyframes = vec![Keyframe { frame: 0, style: solid(w, h, c0) }, Keyframe { frame: n - 1, style: solid(w, h, c1) }];
        let shot = Shot { frames: &video, keyframes, mask: None, range: None };
        let options = RenderOptions {
            frame_cache_bytes: w * h * 4,
            buffer_bytes: w * h * 3,
            spill_dir: Some(spill.clone()),
            ..RenderOptions::default()
        };
        let params = VideoParams { flow: DisParams::preset(FlowPreset::UltraFast), ..VideoParams::classic() };
        let sink = MemorySink::new();
        let stats = Renderer::with_engines(&SolidColour::default(), &device, params).options(options).render(&shot, &sink).unwrap();
        assert!(stats.frame_cache.spilled > 0 && stats.buffers.spilled > 0, "{stats:?}");
        for (t, img) in sink.into_frames() {
            let a = t as f32 / (n - 1) as f32;
            let expect: Vec<u8> = (0..3).map(|c| (f32::from(c0[c]) * (1.0 - a) + f32::from(c1[c]) * a).round() as u8).collect();
            assert_eq!(img.pixel(1, 1), &expect[..], "frame {t}");
        }
    };
    std::thread::scope(|s| {
        let runs = [([240, 0, 0], [0, 0, 240]), ([0, 240, 0], [240, 240, 0]), ([10, 20, 30], [200, 100, 50])];
        for (c0, c1) in runs {
            s.spawn(move || run(c0, c1));
        }
    });
    assert!(!spill.exists(), "the spill directory the renders created is removed");
    // A directory that existed before is kept.
    std::fs::create_dir_all(&spill).unwrap();
    run([1, 2, 3], [4, 5, 6]);
    assert!(spill.exists());
    assert_eq!(std::fs::read_dir(&spill).unwrap().count(), 0, "the render's own subdirectory is removed");
    std::fs::remove_dir(&spill).unwrap();
}

#[test]
fn default_parallelism_is_bounded_for_many_keyframes() {
    let (w, h, n) = (48, 32, 25);
    let (frames, _) = scene(w, h, n);
    let keyframes = (0..n).step_by(3).map(|k| Keyframe { frame: k, style: solid(w, h, [k as u8, 0, 0]) }).collect();
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let device = ComputeDevice::cpu();
    let stats = Renderer::with_engines(&SolidColour::default(), &device, fast_params()).render(&shot, &MemorySink::new()).unwrap();
    assert_eq!(stats.chains, 16);
    assert!(stats.parallel >= 1 && stats.parallel <= MAX_DEFAULT_PARALLEL, "{stats:?}");
    assert_eq!(stats.parallel, stats.resources.parallel);
    // A tight budget runs one chain at a time.
    let options = RenderOptions { memory_budget: 1, ..RenderOptions::default() };
    let stats = Renderer::with_engines(&SolidColour::default(), &device, fast_params())
        .options(options)
        .render(&shot, &MemorySink::new())
        .unwrap();
    assert_eq!(stats.parallel, 1);
}

/// Runs `render` on another thread and fails the test if it does not return within a minute
/// (a hang is the failure these tests guard against).
fn returns_in_time(render: impl FnOnce() -> Result<RenderStats, VideoError> + Send + 'static) -> Result<RenderStats, VideoError> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(render());
    });
    rx.recv_timeout(Duration::from_secs(60)).expect("the render hangs")
}

/// A sink that panics on every frame.
struct PanickingSink;

impl FrameSink for PanickingSink {
    fn write(&self, frame: usize, _image: &Image) -> Result<(), VideoError> {
        panic!("cannot store frame {frame}");
    }
}

/// An engine that panics on its third call.
#[derive(Default)]
struct PanickingEngine(SolidColour);

impl Synthesizer for PanickingEngine {
    fn engine(&self) -> Engine {
        Engine::Cpu
    }

    fn synthesize(&self, input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, VideoError> {
        if self.0.calls.load(Ordering::Relaxed) == 2 {
            panic!("engine failure");
        }
        self.0.synthesize(input, params)
    }
}

fn long_shot() -> (MemoryFrames, Vec<Keyframe>) {
    let (w, h, n) = (48, 32, 40);
    let (frames, _) = scene(w, h, n);
    (frames, vec![Keyframe { frame: 0, style: solid(w, h, [9, 9, 9]) }, Keyframe { frame: n - 1, style: solid(w, h, [90, 9, 9]) }])
}

#[test]
fn a_panicking_sink_fails_the_render_instead_of_hanging() {
    let result = returns_in_time(|| {
        let (frames, keyframes) = long_shot();
        let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
        Renderer::with_engines(&SolidColour::default(), &ComputeDevice::cpu(), fast_params()).render(&shot, &PanickingSink)
    });
    match result {
        Err(VideoError::Panicked(msg)) => assert!(msg.contains("frame sink") && msg.contains("cannot store frame"), "{msg}"),
        other => panic!("expected a panic error, got {other:?}"),
    }
}

#[test]
fn a_panicking_progress_callback_or_engine_fails_the_render() {
    let result = returns_in_time(|| {
        let (frames, keyframes) = long_shot();
        let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
        let on_progress = |p: &Progress| {
            if let Progress::Written { done: 5, .. } = p {
                panic!("progress bar broke");
            }
        };
        Renderer::with_engines(&SolidColour::default(), &ComputeDevice::cpu(), fast_params())
            .on_progress(&on_progress)
            .render(&shot, &MemorySink::new())
    });
    assert!(matches!(&result, Err(VideoError::Panicked(m)) if m.contains("progress bar broke")), "{result:?}");

    let result = returns_in_time(|| {
        let (frames, keyframes) = long_shot();
        let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
        Renderer::with_engines(&PanickingEngine::default(), &ComputeDevice::cpu(), fast_params()).render(&shot, &MemorySink::new())
    });
    assert!(matches!(&result, Err(VideoError::Panicked(m)) if m.contains("engine failure")), "{result:?}");
}
