//! End-to-end renders of small generated scenes. GPU tests skip without an adapter or when
//! `FREESYNTH_NO_GPU` is set.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use freesynth_core::{Engine, Image, Image16, Nnf, OmegaSync, Params, SynthesisInput, SynthesisOutput};
use freesynth_flow::{DisParams, Preset as FlowPreset};
use freesynth_scenes::{Scene, SceneParams, StyleFamily};
use freesynth_video::resources::MAX_DEFAULT_PARALLEL;
use freesynth_video::{
    Backend, BlendMode, Blender, CancelToken, ComputeDevice, FrameSink, Keyframe, Keyframe16, KeyframeOutput,
    MemoryFrames, MemorySink, PassFrame, Progress, RenderOptions, RenderStats, Renderer, Shot, Synthesizer, VideoError,
    VideoParams, WarmStartSettings,
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
        Ok(SynthesisOutput { image: Image::from_raw(w, h, c, data), nnf: Nnf::new(w, h), errors: None, image16: None })
    }
}

fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Image {
    rgba(&rgb.repeat(w * h), w, h)
}

#[test]
fn renders_every_frame_once_and_blends_linearly() {
    let (w, h) = (64, 48);
    let (frames, _) = scene(w, h, 9);
    let keyframes = vec![
        Keyframe { frame: 8, style: solid(w, h, [0, 0, 250]) },
        Keyframe { frame: 2, style: solid(w, h, [200, 0, 0]) },
    ];
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

/// A stand-in engine that returns the style itself and, when asked, a constant patch error that
/// depends on the keyframe (its first pixel's red level / 100 as root-mean-square error).
/// Records whether each call asked for errors.
#[derive(Default)]
struct StyleCopy {
    asked: Mutex<Vec<(u8, bool)>>,
}

impl Synthesizer for StyleCopy {
    fn engine(&self) -> Engine {
        Engine::Cpu
    }

    fn synthesize(&self, input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, VideoError> {
        let style = &input.source_style;
        let red = style.pixel(0, 0)[0];
        self.asked.lock().unwrap().push((red, params.return_errors));
        let (w, h) = style.size();
        let total: f32 = input.style_weights.iter().chain(&input.guide_weights).sum();
        let rms = f32::from(red) / 100.0;
        let errors = params.return_errors.then(|| vec![rms * rms * 25.0 * total; w * h]);
        Ok(SynthesisOutput { image: style.clone(), nnf: Nnf::new(w, h), errors, image16: None })
    }
}

/// What a [`Recorder`] saw for one frame: the position in the interval and the mean error of
/// each pass, if it got errors.
type Seen = (f32, Option<f32>, Option<f32>);

/// A blender that records what it is given and returns pass A.
struct Recorder {
    errors: bool,
    seen: Mutex<Vec<Seen>>,
}

impl Blender for Recorder {
    fn needs_errors(&self) -> bool {
        self.errors
    }

    fn blend(&self, a: PassFrame<'_>, b: PassFrame<'_>, alpha: f32) -> Result<Image, VideoError> {
        let mean = |e: Option<&[f32]>| e.map(|e| e.iter().sum::<f32>() / e.len() as f32);
        self.seen.lock().unwrap().push((alpha, mean(a.errors), mean(b.errors)));
        Ok(a.image.clone())
    }
}

#[test]
fn blenders_get_the_engine_errors_they_ask_for() {
    let (w, h) = (48, 32);
    let (frames, _) = scene(w, h, 8);
    let keyframes = vec![
        Keyframe { frame: 2, style: solid(w, h, [100, 0, 0]) },
        Keyframe { frame: 7, style: solid(w, h, [250, 0, 0]) },
    ];
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let device = ComputeDevice::cpu();
    for wanted in [true, false] {
        let engine = StyleCopy::default();
        let recorder = Recorder { errors: wanted, seen: Mutex::new(Vec::new()) };
        let sink = MemorySink::new();
        Renderer::with_engines(&engine, &device, fast_params()).blender(&recorder).render(&shot, &sink).unwrap();
        assert_eq!(sink.into_frames().len(), 8);
        let mut seen = recorder.seen.into_inner().unwrap();
        seen.sort_by(|x, y| x.0.total_cmp(&y.0));
        // Frames 3 to 6 are blended, at a = 1/5 .. 4/5.
        assert_eq!(seen.iter().map(|s| (s.0 * 5.0).round() as usize).collect::<Vec<_>>(), [1, 2, 3, 4]);
        for (_, ea, eb) in seen {
            if wanted {
                // Pass A comes from keyframe 2 (red 100: error 1.0), pass B from keyframe 7 (2.5).
                assert!((ea.unwrap() - 1.0).abs() < 0.01 && (eb.unwrap() - 2.5).abs() < 0.01, "{ea:?} {eb:?}");
            } else {
                assert!(ea.is_none() && eb.is_none());
            }
        }
        // Errors are asked for exactly for the blended passes: not for keyframes, not for the
        // chain before the first keyframe.
        let asked = engine.asked.into_inner().unwrap();
        assert_eq!(asked.len(), 2 + 2 + 2 * 4);
        assert_eq!(asked.iter().filter(|a| a.1).count(), if wanted { 8 } else { 0 });
    }
}

#[test]
fn adaptive_blend_keeps_more_detail_than_the_crossfade() {
    let (w, h) = (96, 64);
    let (frames, truth) = scene(w, h, 9);
    let keyframes =
        vec![Keyframe { frame: 0, style: truth[0].clone() }, Keyframe { frame: 8, style: truth[8].clone() }];
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let device = ComputeDevice::new(Backend::Cpu).unwrap();
    // Per-band occupancy maps and a fixed band count make the CPU engine deterministic, so that
    // the keyframe frames of the two renders can be compared exactly.
    let synthesis = Params { omega_sync: OmegaSync::PerTile, num_bands: Some(4), ..Params::default() };
    let render = |blend| {
        let sink = MemorySink::new();
        let params = VideoParams { blend, synthesis: synthesis.clone(), ..fast_params() };
        Renderer::new(&device, params).render(&shot, &sink).unwrap();
        sink.into_frames()
    };
    let (linear, adaptive) = (render(BlendMode::Linear), render(BlendMode::Adaptive));
    assert_eq!(adaptive.len(), 9);
    // Keyframe frames are not blended.
    assert_eq!(adaptive[&0], linear[&0]);
    assert_eq!(adaptive[&8], linear[&8]);
    let detail = |img: &Image| {
        let (mut e, nc) = (0.0, img.channels());
        for y in 0..h {
            for x in 0..w - 1 {
                for c in 0..3 {
                    let d =
                        f64::from(img.data()[(y * w + x + 1) * nc + c]) - f64::from(img.data()[(y * w + x) * nc + c]);
                    e += d * d;
                }
            }
        }
        e
    };
    let (dl, da) = (detail(&linear[&4]), detail(&adaptive[&4]));
    assert!(da > 1.1 * dl, "middle frame detail: adaptive {da:.0} vs linear {dl:.0}");
    // And it still follows the scene.
    for t in 1..8 {
        let err = mean_abs_diff(&adaptive[&t], &truth[t]);
        let still = mean_abs_diff(&truth[0], &truth[t]).min(mean_abs_diff(&truth[8], &truth[t]));
        assert!(err < still, "frame {t}: {err:.2} vs {still:.2} for an unmoved keyframe");
    }
}

#[test]
fn cpu_render_follows_the_scene() {
    let (w, h) = (96, 64);
    let (frames, truth) = scene(w, h, 7);
    let keyframes =
        vec![Keyframe { frame: 0, style: truth[0].clone() }, Keyframe { frame: 6, style: truth[6].clone() }];
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
    Renderer::new(&device, params)
        .options(RenderOptions { parallel_chains: Some(1), ..Default::default() })
        .render(&shot, &sink)
        .unwrap();
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
    let shot = Shot {
        frames: &frames,
        keyframes: vec![Keyframe { frame: 0, style: truth[0].clone() }],
        mask: None,
        range: None,
    };
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
    let keyframes =
        vec![Keyframe { frame: 0, style: truth[0].clone() }, Keyframe { frame: 6, style: truth[6].clone() }];
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
        let keyframes =
            vec![Keyframe { frame: 0, style: solid(w, h, c0) }, Keyframe { frame: n - 1, style: solid(w, h, c1) }];
        let shot = Shot { frames: &video, keyframes, mask: None, range: None };
        let options = RenderOptions {
            frame_cache_bytes: w * h * 4,
            buffer_bytes: w * h * 3,
            spill_dir: Some(spill.clone()),
            ..RenderOptions::default()
        };
        let params = VideoParams { flow: DisParams::preset(FlowPreset::UltraFast), ..VideoParams::classic() };
        let sink = MemorySink::new();
        let stats = Renderer::with_engines(&SolidColour::default(), &device, params)
            .options(options)
            .render(&shot, &sink)
            .unwrap();
        assert!(stats.frame_cache.spilled > 0 && stats.buffers.spilled > 0, "{stats:?}");
        for (t, img) in sink.into_frames() {
            let a = t as f32 / (n - 1) as f32;
            let expect: Vec<u8> =
                (0..3).map(|c| (f32::from(c0[c]) * (1.0 - a) + f32::from(c1[c]) * a).round() as u8).collect();
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
    let stats = Renderer::with_engines(&SolidColour::default(), &device, fast_params())
        .render(&shot, &MemorySink::new())
        .unwrap();
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
fn returns_in_time(
    render: impl FnOnce() -> Result<RenderStats, VideoError> + Send + 'static,
) -> Result<RenderStats, VideoError> {
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
    (
        frames,
        vec![
            Keyframe { frame: 0, style: solid(w, h, [9, 9, 9]) },
            Keyframe { frame: n - 1, style: solid(w, h, [90, 9, 9]) },
        ],
    )
}

#[test]
fn a_panicking_sink_fails_the_render_instead_of_hanging() {
    let result = returns_in_time(|| {
        let (frames, keyframes) = long_shot();
        let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
        Renderer::with_engines(&SolidColour::default(), &ComputeDevice::cpu(), fast_params())
            .render(&shot, &PanickingSink)
    });
    match result {
        Err(VideoError::Panicked(msg)) => {
            assert!(msg.contains("frame sink") && msg.contains("cannot store frame"), "{msg}")
        }
        other => panic!("expected a panic error, got {other:?}"),
    }
}

/// A blender whose third blend panics (or, with `fail_with_error`, returns an error).
#[derive(Default)]
struct BrokenBlender {
    calls: AtomicUsize,
    fail_with_error: bool,
}

impl Blender for BrokenBlender {
    fn needs_errors(&self) -> bool {
        false
    }

    fn blend(&self, a: PassFrame<'_>, _: PassFrame<'_>, _: f32) -> Result<Image, VideoError> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        match (n, self.fail_with_error) {
            (3, false) => panic!("blend exploded"),
            (3, true) => Err(VideoError::InvalidInput("blend refused".into())),
            _ => Ok(a.image.clone()),
        }
    }
}

#[test]
fn a_failing_blender_fails_the_render_instead_of_hanging() {
    for fail_with_error in [false, true] {
        let result = returns_in_time(move || {
            let (frames, keyframes) = long_shot();
            let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
            let blender = BrokenBlender { fail_with_error, ..BrokenBlender::default() };
            Renderer::with_engines(&SolidColour::default(), &ComputeDevice::cpu(), fast_params())
                .blender(&blender)
                .render(&shot, &MemorySink::new())
        });
        match (result, fail_with_error) {
            (Err(VideoError::Panicked(msg)), false) => assert!(msg.contains("blend exploded"), "{msg}"),
            (Err(VideoError::InvalidInput(msg)), true) => assert!(msg.contains("blend refused"), "{msg}"),
            (other, _) => panic!("expected the blender's failure, got {other:?}"),
        }
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
        Renderer::with_engines(&PanickingEngine::default(), &ComputeDevice::cpu(), fast_params())
            .render(&shot, &MemorySink::new())
    });
    assert!(matches!(&result, Err(VideoError::Panicked(m)) if m.contains("engine failure")), "{result:?}");
}

/// Deterministic CPU synthesis (per-band occupancy maps, fixed band count).
fn deterministic() -> Params {
    Params { omega_sync: OmegaSync::PerTile, num_bands: Some(4), ..Params::default() }
}

#[test]
fn warm_start_follows_the_scene_like_a_cold_start() {
    let (w, h) = (96, 64);
    let (frames, truth) = scene(w, h, 9);
    let keyframes =
        vec![Keyframe { frame: 0, style: truth[0].clone() }, Keyframe { frame: 8, style: truth[8].clone() }];
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let device = ComputeDevice::cpu();
    let render = |warm_start| {
        let sink = MemorySink::new();
        let params = VideoParams { warm_start, synthesis: deterministic(), ..fast_params() };
        Renderer::new(&device, params).render(&shot, &sink).unwrap();
        sink.into_frames()
    };
    let cold = render(None);
    let warm = render(Some(WarmStartSettings::default()));
    // Keyframes run cold either way.
    assert_eq!(warm[&0], cold[&0]);
    assert_eq!(warm[&8], cold[&8]);
    for t in 1..8 {
        let (ew, ec) = (mean_abs_diff(&warm[&t], &truth[t]), mean_abs_diff(&cold[&t], &truth[t]));
        assert!(ew < 1.25 * ec + 1.0, "frame {t}: warm {ew:.2} vs cold {ec:.2}");
        assert_ne!(warm[&t], cold[&t], "frame {t} did not change");
    }
}

#[test]
fn sixteen_bit_keyframes_give_16_bit_frames() {
    let (w, h) = (80, 56);
    let (frames, truth) = scene(w, h, 7);
    // 16-bit paintings with detail below one 8-bit level; the 8-bit keyframes are them rounded.
    let paint16 = |img: &Image, seed: u16| {
        let data = img
            .data()
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 4 == 3 { 65535 } else { (u16::from(v) * 257).saturating_add((i as u16 ^ seed) % 97) })
            .collect();
        Image16::from_raw(w, h, 4, data)
    };
    let k16 = vec![
        Keyframe16 { frame: 0, style: paint16(&truth[0], 3) },
        Keyframe16 { frame: 6, style: paint16(&truth[6], 7) },
    ];
    let keyframes: Vec<Keyframe> = k16.iter().map(|k| Keyframe { frame: k.frame, style: k.style.to_image() }).collect();
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let device = ComputeDevice::cpu();
    for blend in [BlendMode::Linear, BlendMode::Adaptive] {
        let params = VideoParams { blend, synthesis: deterministic(), ..fast_params() };
        let sink = MemorySink::new();
        Renderer::new(&device, params.clone()).keyframes16(&k16).render(&shot, &sink).unwrap();
        let out16 = sink.into_frames16();
        assert_eq!(out16.len(), 7, "{blend:?}");
        let eight = MemorySink::new();
        Renderer::new(&device, params).render(&shot, &eight).unwrap();
        let eight = eight.into_frames();
        for (t, img) in &out16 {
            assert_eq!(img.channels(), 3);
            assert!(img.data().iter().any(|v| v % 257 != 0), "{blend:?}, frame {t}: no extra precision");
            // The same render at more precision.
            let d = mean_abs_diff(&img.to_image(), &eight[t]);
            assert!(d < 0.6, "{blend:?}, frame {t}: {d:.3} from the 8-bit render");
        }
    }
    // The painting at keyframes, at 16 bits.
    let params = VideoParams { keyframe_output: KeyframeOutput::Painting, ..fast_params() };
    let sink = MemorySink::new();
    Renderer::new(&device, params).keyframes16(&k16).render(&shot, &sink).unwrap();
    let out16 = sink.into_frames16();
    let rgb: Vec<u16> = k16[1].style.data().as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    assert_eq!(out16[&6].data(), &rgb[..]);
    // Every keyframe needs its 16-bit painting.
    let err = Renderer::new(&device, fast_params()).keyframes16(&k16[..1]).render(&shot, &MemorySink::new());
    assert!(matches!(err, Err(VideoError::InvalidInput(_))), "{err:?}");
    // An engine without 16-bit support still gives 16-bit frames: its results widened (the
    // blended frames in between then carry the blend's fractions).
    let sink = MemorySink::new();
    Renderer::with_engines(&SolidColour::default(), &device, fast_params())
        .keyframes16(&k16)
        .render(&shot, &sink)
        .unwrap();
    let out16 = sink.into_frames16();
    assert_eq!(out16.len(), 7);
    assert!([0, 6].iter().all(|t| out16[t].data().iter().all(|v| v % 257 == 0)));
}
