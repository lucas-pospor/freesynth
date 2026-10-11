//! Runs a position-field walk and measures how well the confidence separates right from wrong
//! keyframe positions.
//!
//! ```text
//! cargo run --release -p freesynth-flow --example track_eval -- [options] --scene
//! cargo run --release -p freesynth-flow --example track_eval -- [options] --list FRAMES.txt --out DIR
//! ```
//!
//! With `--scene`, the frames come from a generated scene (`--width`, `--height`, `--frames`,
//! `--seed`, `--objects`, `--motion`; defaults 480x270, 100 frames, seed 4), and the exact
//! keyframe position of every pixel comes from the scene's geometry. At each reported frame
//! (`--at 10,30,99` or `--at all`; default 25, 50, 75 and the last frame) the example prints,
//! over the pixels whose content is visible in the keyframe (anti-aliased object edges
//! included):
//!
//! - the share of those pixels that are confident (confidence above 0.5);
//! - the mean and median position error of the confident ones;
//! - `bad3`: the share of confident pixels whose position is off by more than 3 px, overall and
//!   separately for the background and the moving objects;
//! - recall: the share of pixels off by more than 3 px that are flagged (not confident);
//! - precision: the share of flagged pixels that are off by more than 3 px;
//! - the same position errors for all of them (as if there were no confidence);
//! - `hidden`: confident pixels whose content is hidden in (or outside) the keyframe, which
//!   have no right position, and `bad3` with them counted as failures.
//!
//! A last line gives the worst `bad3` (with and without hidden pixels) and the lowest precision
//! over the reported frames, since errors do not grow steadily along a walk.
//!
//! With `--list`, the frames are read from a file (one path per line, the keyframe first, in
//! walk order), and for each reported frame the position field is written as the flow to the
//! keyframe, `DIR/pos_NNNNN.flo`, and the confidence as a 16-bit grey PNG, `DIR/conf_NNNNN.png`
//! (65535 = 1).
//!
//! Options: `--cpu` (the GPU is used by default), `--preset P`, `--stride N` (default 10),
//! `--fixed` (no adaptive re-anchoring), `--max-motion PX`, `--max-jump F` (failure jump),
//! `--erode R`, `--max-jump-px F` (position jump factor, `inf` disables), `--verbose` (one line
//! per frame: anchor, median motion, share of inconsistent pixels).

use std::path::PathBuf;

use freesynth_flow::cpu::CpuDis;
use freesynth_flow::gpu::{GpuContext, GpuDis};
use freesynth_flow::track::{ConfidenceMap, PositionField, TrackParams, Tracker};
use freesynth_flow::{DisParams, FlowEstimator, GrayImage, Preset};
use freesynth_scenes::{Scene, SceneParams, StyleFamily};

/// `--at all`: report every frame.
const ALL_FRAMES: usize = usize::MAX;

struct Options {
    cpu: bool,
    preset: Preset,
    track: TrackParams,
    at: Vec<usize>,
    scene: Option<SceneParams>,
    verbose: bool,
    list: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn parse() -> Result<Options, String> {
    let mut o = Options {
        cpu: false,
        preset: Preset::Medium,
        track: TrackParams::default(),
        at: Vec::new(),
        scene: None,
        verbose: false,
        list: None,
        out: None,
    };
    let mut scene = SceneParams {
        width: 480,
        height: 270,
        frames: 100,
        seed: 4,
        family: StyleFamily::Rigid,
        ..SceneParams::default()
    };
    let mut use_scene = false;
    let mut fixed = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or(format!("{a} needs a value"));
        let err = |e: &dyn std::fmt::Display| format!("{a}: {e}");
        match a.as_str() {
            "--cpu" => o.cpu = true,
            "--scene" => use_scene = true,
            "--fixed" => fixed = true,
            "--verbose" => o.verbose = true,
            "--preset" => o.preset = value()?.parse().map_err(|e| err(&e))?,
            "--stride" => o.track.max_stride = value()?.parse().map_err(|e| err(&e))?,
            "--max-motion" => o.track.max_anchor_motion = value()?.parse().map_err(|e| err(&e))?,
            "--max-jump" => o.track.max_failure_jump = value()?.parse().map_err(|e| err(&e))?,
            "--erode" => o.track.erode_radius = value()?.parse().map_err(|e| err(&e))?,
            "--max-jump-px" => o.track.max_position_jump = value()?.parse().map_err(|e| err(&e))?,
            "--at" => {
                let v = value()?;
                o.at = if v == "all" {
                    vec![ALL_FRAMES]
                } else {
                    v.split(',').map(|v| v.parse().map_err(|e| err(&e))).collect::<Result<_, _>>()?
                };
            }
            "--width" => scene.width = value()?.parse().map_err(|e| err(&e))?,
            "--height" => scene.height = value()?.parse().map_err(|e| err(&e))?,
            "--frames" => scene.frames = value()?.parse().map_err(|e| err(&e))?,
            "--seed" => scene.seed = value()?.parse().map_err(|e| err(&e))?,
            "--objects" => scene.objects = value()?.parse().map_err(|e| err(&e))?,
            "--motion" => scene.motion = value()?.parse().map_err(|e| err(&e))?,
            "--list" => o.list = Some(PathBuf::from(value()?)),
            "--out" => o.out = Some(PathBuf::from(value()?)),
            _ => return Err(format!("unknown option {a}")),
        }
    }
    if fixed {
        o.track.max_anchor_motion = f32::INFINITY;
        o.track.max_failure_jump = f32::INFINITY;
    }
    if use_scene {
        scene.keyframes = vec![0];
        o.scene = Some(scene);
    } else if o.list.is_none() || o.out.is_none() {
        return Err("usage: track_eval [options] (--scene | --list FRAMES.txt --out DIR)".into());
    }
    Ok(o)
}

fn estimator(
    cpu: bool,
    w: usize,
    h: usize,
    params: &DisParams,
) -> Result<Box<dyn FlowEstimator>, Box<dyn std::error::Error>> {
    Ok(if cpu {
        Box::new(CpuDis::new(w, h, params)?)
    } else {
        Box::new(GpuDis::new(&GpuContext::new()?, w, h, params)?)
    })
}

fn load_gray(path: &str) -> Result<GrayImage, Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_rgb8();
    Ok(GrayImage::from_rgb8(img.width() as usize, img.height() as usize, img.as_raw())?)
}

/// Confidence statistics of one frame.
struct Score {
    bad3: f64,
    bad3_hidden: f64,
    precision: f64,
}

/// Position errors and confidence statistics over the pixels that have a keyframe position
/// (`layer` is `Some`: 0 for the background, more for objects); confident pixels without one
/// are counted as `hidden` failures.
fn report(
    t: usize,
    positions: &PositionField,
    confidence: &ConfidenceMap,
    truth: &PositionField,
    layer: &[Option<u8>],
) -> Score {
    let mut confident_errors = Vec::new();
    let mut all_errors = Vec::new();
    let (mut bad, mut bad_flagged, mut flagged, mut flagged_bad, mut hidden) = (0usize, 0usize, 0usize, 0usize, 0usize);
    // Confident and confident-bad pixels of the background and of the objects.
    let mut by_kind = [[0usize; 2]; 2];
    for (i, ((p, q), &l)) in positions.data().iter().zip(truth.data()).zip(layer).enumerate() {
        let is_confident = confidence.data()[i] > 0.5;
        let Some(l) = l else {
            hidden += usize::from(is_confident);
            continue;
        };
        let e = (p[0] - q[0]).hypot(p[1] - q[1]);
        let is_bad = e > 3.0;
        all_errors.push(e);
        if is_confident {
            confident_errors.push(e);
            let kind = &mut by_kind[usize::from(l > 0)];
            kind[0] += 1;
            kind[1] += usize::from(is_bad);
        } else {
            flagged += 1;
            flagged_bad += usize::from(is_bad);
        }
        bad += usize::from(is_bad);
        bad_flagged += usize::from(is_bad && !is_confident);
    }
    let stats = |e: &mut Vec<f32>| {
        if e.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let mean = e.iter().map(|&v| f64::from(v)).sum::<f64>() / e.len() as f64;
        let mid = e.len() / 2;
        (mean, f64::from(*e.select_nth_unstable_by(mid, f32::total_cmp).1))
    };
    let share = |part: usize, of: usize| part as f64 / of.max(1) as f64;
    let n = all_errors.len();
    let confident_bad = confident_errors.iter().filter(|&&e| e > 3.0).count();
    let score = Score {
        bad3: share(confident_bad, confident_errors.len()),
        bad3_hidden: share(confident_bad + hidden, confident_errors.len() + hidden),
        precision: share(flagged_bad, flagged),
    };
    let (conf_mean, conf_median) = stats(&mut confident_errors);
    let (all_mean, all_median) = stats(&mut all_errors);
    println!(
        "frame {t:3}: confident {:5.1}% of {n} px | confident error mean {conf_mean:.3} median {conf_median:.3} px, bad3 {:.3}% \
         (background {:.3}% of {}, objects {:.3}% of {}) | recall {:.1}% precision {:.1}% | all: mean {all_mean:.2} median \
         {all_median:.3} px, bad3 {:.2}% | hidden {hidden} px, bad3 with them {:.3}%",
        100.0 * share(n - flagged, n),
        100.0 * score.bad3,
        100.0 * share(by_kind[0][1], by_kind[0][0]),
        by_kind[0][0],
        100.0 * share(by_kind[1][1], by_kind[1][0]),
        by_kind[1][0],
        100.0 * share(bad_flagged, bad),
        100.0 * score.precision,
        100.0 * share(bad, n),
        100.0 * score.bad3_hidden,
    );
    score
}

/// The worst scores over the reported frames.
fn summary(scores: &[(usize, Score)]) {
    let worst = |key: fn(&Score) -> f64, lowest: bool| {
        scores.iter().map(|(t, s)| (*t, key(s))).reduce(|a, b| if (b.1 < a.1) == lowest { b } else { a })
    };
    if let (Some(b), Some(bh), Some(p)) =
        (worst(|s| s.bad3, false), worst(|s| s.bad3_hidden, false), worst(|s| s.precision, true))
    {
        println!(
            "over {} reported frames: max bad3 {:.3}% (frame {}), max bad3 with hidden {:.3}% (frame {}), min precision {:.1}% (frame {})",
            scores.len(),
            100.0 * b.1,
            b.0,
            100.0 * bh.1,
            bh.0,
            100.0 * p.1,
            p.0
        );
    }
}

/// The frames to report out of `frames`: `at` (`ALL_FRAMES` for all), or `default`.
fn frames_to_report(at: &[usize], default: &[usize], frames: usize) -> Vec<usize> {
    let last = frames - 1;
    match at {
        [ALL_FRAMES] => (1..frames).collect(),
        [] => default.iter().copied().filter(|&t| t < last).chain([last]).collect(),
        at => at.to_vec(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let o = parse()?;
    let params = DisParams::preset(o.preset);
    if let Some(sp) = &o.scene {
        let scene = Scene::new(sp)?;
        let (w, h) = (sp.width, sp.height);
        let at = frames_to_report(&o.at, &[25, 50, 75], sp.frames);
        let key = scene.render(0);
        let gray = |f: &freesynth_scenes::Frame| GrayImage::from_rgb8(w, h, f.image.as_raw());
        let mut tracker = Tracker::new(estimator(o.cpu, w, h, &params)?, gray(&key)?, o.track)?;
        let mut anchors = Vec::new();
        let mut scores = Vec::new();
        for t in 1..sp.frames {
            let frame = scene.render(t);
            let info = tracker.step(&gray(&frame)?)?;
            if o.verbose {
                println!("{info:?}");
            }
            if info.anchored || info.reanchored {
                anchors.push(if info.reanchored { format!("{}*", info.anchor) } else { info.index.to_string() });
            }
            if at.contains(&t) {
                // Exact keyframe positions. A pixel's content is visible in the keyframe when its
                // position is inside the keyframe and the keyframe shows the same layer there.
                let truth = PositionField::from_displacement(&scene.flow(t, 0, &frame.layers, &key.layers).flow);
                let layer: Vec<Option<u8>> = truth
                    .data()
                    .iter()
                    .enumerate()
                    .map(|(i, q)| {
                        let l = frame.layers.layer(i % w, i / w);
                        let inside = q[0] >= 0.0 && q[1] >= 0.0 && q[0] <= (w - 1) as f32 && q[1] <= (h - 1) as f32;
                        let visible = inside && key.layers.layer(q[0].round() as usize, q[1].round() as usize) == l;
                        visible.then_some(l)
                    })
                    .collect();
                let score = report(t, tracker.positions(), tracker.confidence(), &truth, &layer);
                scores.push((t, score));
            }
        }
        summary(&scores);
        println!("anchors (* = moved back to the previous frame): {}", anchors.join(" "));
        return Ok(());
    }
    let list = std::fs::read_to_string(o.list.as_ref().expect("checked"))?;
    let paths: Vec<&str> = list.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let out = o.out.as_ref().expect("checked");
    std::fs::create_dir_all(out)?;
    let key = load_gray(paths[0])?;
    let (w, h) = key.size();
    let at = frames_to_report(&o.at, &[], paths.len());
    let mut tracker = Tracker::new(estimator(o.cpu, w, h, &params)?, key, o.track)?;
    for (t, path) in paths.iter().enumerate().skip(1) {
        let info = tracker.step(&load_gray(path)?)?;
        if o.verbose {
            println!("{info:?}");
        }
        if at.contains(&t) {
            tracker.positions().displacement().save(out.join(format!("pos_{t:05}.flo")))?;
            let conf: Vec<u16> =
                tracker.confidence().data().iter().map(|&c| (c.clamp(0.0, 1.0) * 65535.0).round() as u16).collect();
            image::ImageBuffer::<image::Luma<u16>, _>::from_raw(w as u32, h as u32, conf)
                .ok_or("confidence size")?
                .save(out.join(format!("conf_{t:05}.png")))?;
            println!(
                "frame {t}: anchor {}, {:.1}% confident",
                info.anchor,
                100.0 * tracker.confidence().fraction_confident(0.5)
            );
        }
    }
    Ok(())
}
