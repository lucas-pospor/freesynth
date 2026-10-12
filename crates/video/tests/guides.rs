//! Checks the guides the renderer hands to the synthesis engine against the exact motion of
//! generated scenes: a capturing stand-in engine records every synthesis input, and the test
//! decodes the guide layers from it.

use std::ops::Range;
use std::sync::Mutex;

use freesynth_core::{Engine, Image, Nnf, Params, SynthesisInput, SynthesisOutput};
use freesynth_flow::{DisParams, FlowField, Preset as FlowPreset};
use freesynth_scenes::{Scene, SceneParams, StyleFamily};
use freesynth_video::guides::{to_rgba, warp_rgba};
use freesynth_video::{
    ComputeDevice, GuideWeights, Keyframe, MemoryFrames, MemorySink, RenderOptions, Renderer, Shot, Synthesizer,
    VideoError, VideoParams,
};

/// Layer weights chosen so that every layer has its own per-channel weight, which is how the
/// test finds the layers in the concatenated guide channels.
const WEIGHTS: GuideWeights = GuideWeights { style: 1.0, video: 6.0, temporal: 1.0, mapping: 9.0, mask: 0.5 };

/// A stand-in engine that returns the target video frame (the first guide layer) as its output
/// and records every input.
#[derive(Default)]
struct Capture {
    inputs: Mutex<Vec<SynthesisInput>>,
}

impl Synthesizer for Capture {
    fn engine(&self) -> Engine {
        Engine::Cpu
    }

    fn synthesize(&self, input: &SynthesisInput, _params: &Params) -> Result<SynthesisOutput, VideoError> {
        let (w, h) = input.target_guide.size();
        let video = &layers(input)[0];
        assert_eq!(video.weight, WEIGHTS.video);
        let image = Image::from_raw(w, h, video.channels.len(), channels(&input.target_guide, &video.channels));
        self.inputs.lock().unwrap().push(input.clone());
        Ok(SynthesisOutput { image, nnf: Nnf::new(w, h), errors: None, image16: None })
    }
}

/// One guide layer inside a synthesis input.
struct Layer {
    /// Total weight of the layer.
    weight: f32,
    /// Its guide channels.
    channels: Range<usize>,
}

/// Splits the guide channels into layers: runs of equal per-channel weight.
fn layers(input: &SynthesisInput) -> Vec<Layer> {
    let w = &input.guide_weights;
    let mut out: Vec<Layer> = Vec::new();
    for (i, &v) in w.iter().enumerate() {
        match out.last_mut() {
            Some(l) if w[l.channels.start] == v => l.channels.end = i + 1,
            _ => out.push(Layer { weight: 0.0, channels: i..i + 1 }),
        }
    }
    for l in &mut out {
        l.weight = (w[l.channels.start] * l.channels.len() as f32 * 1000.0).round() / 1000.0;
    }
    out
}

/// The layer of `input` with total weight `weight`.
fn layer(input: &SynthesisInput, weight: f32) -> Option<Range<usize>> {
    layers(input).into_iter().find(|l| l.weight == weight).map(|l| l.channels)
}

/// Channels `range` of every pixel of `image`.
fn channels(image: &Image, range: &Range<usize>) -> Vec<u8> {
    image.data().chunks_exact(image.channels()).flat_map(|px| px[range.clone()].iter().copied()).collect()
}

fn rgba(rgb: &[u8], w: usize, h: usize) -> Image {
    let data = rgb.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    Image::from_raw(w, h, 4, data)
}

struct TestScene {
    scene: Scene,
    video: MemoryFrames,
    truth: Vec<Image>,
}

impl TestScene {
    fn new(width: usize, height: usize, frames: usize, motion: f64) -> Self {
        let params = SceneParams {
            width,
            height,
            frames,
            seed: 7,
            motion,
            family: StyleFamily::Rigid,
            ..SceneParams::default()
        };
        let scene = Scene::new(&params).unwrap();
        let (video, truth): (Vec<_>, Vec<_>) = (0..frames)
            .map(|t| {
                let f = scene.render(t);
                (rgba(f.image.as_raw(), width, height), rgba(f.truth.as_raw(), width, height))
            })
            .unzip();
        Self { scene, video: MemoryFrames::new(0, video).unwrap(), truth }
    }

    /// Exact flow from frame `from` to frame `to` and its occlusion mask (255 = unusable).
    fn flow(&self, from: usize, to: usize) -> (FlowField, Vec<u8>) {
        let exact = self.scene.flow(from, to, &self.scene.render(from).layers, &self.scene.render(to).layers);
        (exact.flow, exact.occlusion.into_raw())
    }

    fn size(&self) -> (usize, usize) {
        let p = self.scene.params();
        (p.width, p.height)
    }
}

fn params(gate: bool) -> VideoParams {
    VideoParams {
        weights: WEIGHTS,
        flow: DisParams::preset(FlowPreset::Medium),
        gate_mapping: gate,
        ..VideoParams::classic()
    }
}

/// Renders the shot with a single keyframe with the capturing engine and returns the inputs in
/// synthesis order: the keyframe's frame first, then the chain walking away from it.
fn capture(
    scene: &TestScene,
    key: usize,
    style: Image,
    mask: Option<&MemoryFrames>,
    gate: bool,
) -> Vec<SynthesisInput> {
    capture_with(scene, key, style, mask, params(gate))
}

/// [`capture`] with explicit parameters.
fn capture_with(
    scene: &TestScene,
    key: usize,
    style: Image,
    mask: Option<&MemoryFrames>,
    params: VideoParams,
) -> Vec<SynthesisInput> {
    let shot = Shot {
        frames: &scene.video,
        keyframes: vec![Keyframe { frame: key, style }],
        mask: mask.map(|m| m as _),
        range: None,
    };
    let engine = Capture::default();
    let device = ComputeDevice::cpu();
    Renderer::with_engines(&engine, &device, params)
        .options(RenderOptions { parallel_chains: Some(1), ..RenderOptions::default() })
        .render(&shot, &MemorySink::new())
        .unwrap();
    engine.inputs.into_inner().unwrap()
}

/// Mean absolute difference of two RGBA images over the first three channels, on pixels where
/// `use_pixel` holds.
fn masked_mae(a: &Image, b: &Image, use_pixel: impl Fn(usize) -> bool) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for (i, (pa, pb)) in a.data().as_chunks::<4>().0.iter().zip(b.data().as_chunks::<4>().0).enumerate() {
        if use_pixel(i) {
            sum += (0..3).map(|c| (f64::from(pa[c]) - f64::from(pb[c])).abs()).sum::<f64>();
            n += 3;
        }
    }
    assert!(n > 0);
    sum / n as f64
}

/// The temporal target of frame `t` is the previous output warped to `t` along the true
/// motion: it matches the exact-flow warp of the recorded output of frame `prev`.
fn check_temporal(scene: &TestScene, key: usize, frames: impl Iterator<Item = usize>) {
    let (w, h) = scene.size();
    let inputs = capture(scene, key, scene.truth[key].clone(), None, false);
    assert!(layer(&inputs[0], WEIGHTS.temporal).is_none(), "the keyframe's frame has no temporal guide");
    let mut prev = key;
    for (input, t) in inputs[1..].iter().zip(frames) {
        let channels_of =
            |weight| layer(input, weight).unwrap_or_else(|| panic!("frame {t}: no layer of weight {weight}"));
        // The previous output is the previous frame's video (the engine echoes it).
        let previous_output = to_rgba(&scene.video_rgb(prev));
        let temporal = channels_of(WEIGHTS.temporal);
        assert_eq!(temporal.len(), 3);
        let target = rgba(&channels(&input.target_guide, &temporal), w, h);
        let (flow, occlusion) = scene.flow(t, prev);
        let expected = warp_rgba(&previous_output, &flow);
        let err = masked_mae(&target, &expected, |i| occlusion[i] == 0);
        let unwarped = masked_mae(&previous_output, &expected, |i| occlusion[i] == 0);
        println!("frame {t} (from {prev}): temporal target vs exact warp {err:.2}, unwarped {unwarped:.2}");
        assert!(err < 3.0 && err < 0.3 * unwarped, "frame {t}: {err:.2} vs {unwarped:.2} unwarped");
        prev = t;
    }
}

impl TestScene {
    /// Frame `t` of the video as an RGB image (what the capturing engine outputs).
    fn video_rgb(&self, t: usize) -> Image {
        let (w, h) = self.size();
        let data = freesynth_video::FrameSource::load(&self.video, t)
            .unwrap()
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        Image::from_raw(w, h, 3, data)
    }
}

#[test]
fn temporal_guide_follows_the_motion_forward() {
    let scene = TestScene::new(128, 80, 5, 1.5);
    check_temporal(&scene, 0, 1..5);
}

#[test]
fn temporal_guide_follows_the_motion_backward() {
    let scene = TestScene::new(128, 80, 5, 1.5);
    check_temporal(&scene, 4, (0..4).rev());
}

#[test]
fn mapping_guide_holds_the_true_keyframe_positions() {
    let scene = TestScene::new(160, 96, 6, 1.5);
    let (w, h) = scene.size();
    let step = (w.max(h) - 1) as f32 / 255.0;
    // Gating adds the tracker's confidence (as modulation) without changing the positions.
    let inputs = capture(&scene, 0, scene.truth[0].clone(), None, true);
    for (t, input) in inputs.iter().enumerate() {
        let mapping = layer(input, WEIGHTS.mapping).expect("mapping layer");
        assert_eq!(mapping.len(), 3, "R = x, G = y, B = 0");
        let codes = channels(&input.target_guide, &mapping);
        let confidence = input.target_modulation.as_ref().map(|m| channels(m, &(mapping.start..mapping.start + 1)));
        let (flow, occlusion) = scene.flow(t, 0);
        let (mut within, mut total, mut moved) = (0usize, 0usize, 0.0f64);
        // Pixels visible at the keyframe that the tracker is confident about.
        let checked = |i: usize| occlusion[i] == 0 && confidence.as_ref().is_none_or(|c| c[i] == 255);
        for i in (0..w * h).filter(|&i| checked(i)) {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let d = flow.data()[i];
            let truth = [x + d[0], y + d[1]];
            // The code is the truncated position: the position lies in [code, code + 1) steps.
            let decoded = [(f32::from(codes[3 * i]) + 0.5) * step, (f32::from(codes[3 * i + 1]) + 0.5) * step];
            let err = (decoded[0] - truth[0]).abs().max((decoded[1] - truth[1]).abs());
            within += usize::from(err <= step);
            total += 1;
            moved += f64::from(d[0].hypot(d[1]));
            assert_eq!(codes[3 * i + 2], 0);
        }
        let share = within as f64 / total as f64;
        let checked_share = total as f64 / (w * h) as f64;
        println!(
            "frame {t}: {:.1}% of {:.1}% checked pixels within one step ({step:.2} px), mean motion {:.2} px",
            100.0 * share,
            100.0 * checked_share,
            moved / total as f64
        );
        assert!(checked_share > 0.5, "frame {t}: most pixels are tracked");
        // Tracking errors grow with the distance from the keyframe (87% at 4.6 px here).
        assert!(share > 0.8, "frame {t}: only {:.1}% of the positions within one step", 100.0 * share);
    }
}

#[test]
fn mask_guide_pairs_the_painting_alpha_with_the_mask() {
    let scene = TestScene::new(64, 48, 4, 1.0);
    let (w, h) = scene.size();
    // A painting transparent on its top quarter, and masks black on the left half.
    let mut painting = scene.truth[1].clone();
    for (i, px) in painting.data_mut().as_chunks_mut::<4>().0.iter_mut().enumerate() {
        px[3] = if i / w < h / 4 { 0 } else { 255 };
    }
    let mask_frames: Vec<Image> = (0..4)
        .map(|_| {
            let data = (0..w * h).flat_map(|i| if i % w < w / 2 { [0, 0, 0, 255] } else { [255; 4] }).collect();
            Image::from_raw(w, h, 4, data)
        })
        .collect();
    let masks = MemoryFrames::new(0, mask_frames).unwrap();
    let inputs = capture(&scene, 1, painting.clone(), Some(&masks), false);
    assert_eq!(inputs.len(), 4);
    for input in &inputs {
        let mask = layer(input, WEIGHTS.mask).expect("mask layer");
        assert_eq!(mask.len(), 1);
        let target = channels(&input.target_guide, &mask);
        let expected: Vec<u8> = (0..w * h).map(|i| if i % w < w / 2 { 0 } else { 255 }).collect();
        assert_eq!(target, expected, "the target side is the frame's mask");
        let source = channels(&input.source_guide, &mask);
        let alpha: Vec<u8> = painting.data().as_chunks::<4>().0.iter().map(|px| px[3]).collect();
        assert_eq!(source, alpha, "the source side is the painting's alpha");
    }
}

#[test]
fn gating_switches_the_mapping_guide_off_where_tracking_is_lost() {
    let scene = TestScene::new(160, 96, 6, 2.0);
    let (w, h) = scene.size();
    let inputs = capture(&scene, 0, scene.truth[0].clone(), None, true);
    let key_layers = scene.scene.render(0).layers;
    let mut checked = 0;
    for (t, input) in inputs.iter().enumerate().skip(1) {
        let mapping = layer(input, WEIGHTS.mapping).expect("mapping layer");
        let modulation = input.target_modulation.as_ref().expect("gating modulates the mapping guide");
        let weights = channels(modulation, &mapping);
        // The other layers keep their full weight.
        let video = layer(input, WEIGHTS.video).unwrap();
        assert!(channels(modulation, &video).iter().all(|&v| v == 255));
        // Disoccluded: the content seen at t lies, at the keyframe, behind another layer.
        let layers = scene.scene.render(t).layers;
        let (flow, _) = scene.flow(t, 0);
        let disoccluded = |x: usize, y: usize| {
            let d = flow.data()[y * w + x];
            let (kx, ky) = ((x as f32 + d[0]).round(), (y as f32 + d[1]).round());
            if kx < 0.0 || ky < 0.0 || kx >= w as f32 || ky >= h as f32 {
                return true;
            }
            key_layers.layer(kx as usize, ky as usize) != layers.layer(x, y)
        };
        // Pixels well inside disoccluded regions (their 7x7 neighbourhood is disoccluded too).
        let r = 3;
        let (mut off, mut core) = (0usize, 0usize);
        for y in r..h - r {
            for x in r..w - r {
                if (y - r..=y + r).all(|yy| (x - r..=x + r).all(|xx| disoccluded(xx, yy))) {
                    core += 1;
                    let m = weights[(y * w + x) * mapping.len()];
                    off += usize::from(m == 0);
                }
            }
        }
        let confident = weights.iter().filter(|&&v| v == 255).count() as f64 / weights.len() as f64;
        println!("frame {t}: {core} core disoccluded pixels, {off} switched off; {:.1}% fully on", 100.0 * confident);
        assert!(off as f64 >= 0.9 * core as f64, "frame {t}: {off} of {core} disoccluded pixels switched off");
        assert!(confident > 0.5, "frame {t}: most of the frame keeps the mapping guide");
        checked += core;
    }
    assert!(checked > 30, "the scene has disocclusions to check ({checked} pixels)");
}

#[test]
fn temporal_gating_switches_the_temporal_guide_off_where_the_warp_is_wrong() {
    let scene = TestScene::new(160, 96, 6, 5.0);
    let (w, h) = scene.size();
    let gated = VideoParams { gate_temporal: true, temporal_gate_erosion: 1, ..params(false) };
    let inputs = capture_with(&scene, 0, scene.truth[0].clone(), None, gated);
    let mut checked = 0;
    for (t, input) in inputs.iter().enumerate().skip(1) {
        let temporal = layer(input, WEIGHTS.temporal).expect("temporal layer");
        let modulation = input.target_modulation.as_ref().expect("gating modulates the temporal guide");
        let weights = channels(modulation, &temporal);
        for other in [WEIGHTS.video, WEIGHTS.mapping] {
            let range = layer(input, other).unwrap();
            assert!(
                channels(modulation, &range).iter().all(|&v| v == 255),
                "frame {t}: only the temporal guide is gated"
            );
        }
        // Disoccluded since the previous frame: the content seen at t was behind another layer
        // (or outside the frame) at t - 1, so the warped previous output is wrong there.
        let (now, before) = (scene.scene.render(t).layers, scene.scene.render(t - 1).layers);
        let (flow, _) = scene.flow(t, t - 1);
        let disoccluded = |x: usize, y: usize| {
            let d = flow.data()[y * w + x];
            let (px, py) = ((x as f32 + d[0]).round(), (y as f32 + d[1]).round());
            if px < 0.0 || py < 0.0 || px >= w as f32 || py >= h as f32 {
                return true;
            }
            before.layer(px as usize, py as usize) != now.layer(x, y)
        };
        let r = 1;
        let (mut off, mut core) = (0usize, 0usize);
        for y in r..h - r {
            for x in r..w - r {
                if (y - r..=y + r).all(|yy| (x - r..=x + r).all(|xx| disoccluded(xx, yy))) {
                    core += 1;
                    off += usize::from(weights[(y * w + x) * temporal.len()] == 0);
                }
            }
        }
        let on = weights.iter().filter(|&&v| v == 255).count() as f64 / weights.len() as f64;
        println!("frame {t}: {core} core disoccluded pixels, {off} switched off; {:.1}% fully on", 100.0 * on);
        assert!(off as f64 >= 0.65 * core as f64, "frame {t}: {off} of {core} disoccluded pixels switched off");
        assert!(on > 0.8, "frame {t}: most of the frame keeps the temporal guide");
        checked += core;
    }
    assert!(checked > 20, "the scene has disocclusions to check ({checked} pixels)");
    // Without gating there is no modulation at all.
    let plain = capture_with(&scene, 0, scene.truth[0].clone(), None, params(false));
    assert!(plain.iter().all(|i| i.target_modulation.is_none()));
}
