//! CI quick tier: the whole evaluation pipeline on tiny generated scenes.
//!
//! Generates small synthetic scenes with `freesynth-scenes` (exact flow and ground truth),
//! stylizes every frame from the first keyframe with the CPU engine, then runs each metric of
//! this crate and checks that the numbers fall inside sanity bounds: exact energy from the
//! engine's field, sampled exact energy, occupancy, sharpness, the temporal metrics through the
//! `freesynth-eval temporal --scene` command line, and paired bootstrap comparisons.
//!
//! It needs no external data and no GPU. The GPU engine is checked against the CPU engine with
//! the same evaluator in `freesynth-gpu`'s own tests, so this tier builds without wgpu.

use std::path::{Path, PathBuf};
use std::process::Command;

use freesynth_core::{GuideLayer, Image, Nnf, Params, PatchErrorEvaluator, SynthesisInput, io, synthesize};
use freesynth_eval::bootstrap::BootstrapConfig;
use freesynth_eval::compare::{Level, compare_reports};
use freesynth_eval::energy::{nnf_energy, sample_pixels, sampled_energy};
use freesynth_eval::frame::RgbImage;
use freesynth_eval::occupancy::occupancy;
use freesynth_eval::sharpness::{DEFAULT_SIGMA, Sharpness};
use freesynth_eval::{PixelSet, Report};
use freesynth_scenes::{SceneParams, StyleFamily, write_scene};

const PATCH: usize = 5;
/// Weight of the video guide, as in per-frame keyframe stylization.
const GUIDE_WEIGHT: f32 = 4.0;
/// Pixels for the sampled exact energy (enough for a clear paired comparison, fast in debug).
const SAMPLES: usize = 400;

/// A scratch directory under the cargo target directory, removed when the test passes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("freesynth-eval-ci").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn frame_file(dir: &Path, t: usize) -> PathBuf {
    dir.join(format!("{t:05}.png"))
}

fn arg(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Runs the `freesynth-eval` binary and returns its standard output.
fn eval_cli(args: &[String]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_freesynth-eval")).args(args).output().expect("binary runs");
    assert!(out.status.success(), "freesynth-eval {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

/// The keyframe stylization problem for frame `t`: style = key 0, guide = frame 0 -> frame t.
fn problem(scene: &Path, t: usize) -> SynthesisInput {
    let key = io::load_rgba(scene.join("keys/00000.png")).unwrap();
    let source = io::load_rgba(frame_file(&scene.join("frames"), 0)).unwrap();
    let target = io::load_rgba(frame_file(&scene.join("frames"), t)).unwrap();
    SynthesisInput::from_rgba_layers(&key, None, &[GuideLayer::new(source, target).with_weight(GUIDE_WEIGHT)]).unwrap()
}

fn params(seed: u32) -> Params {
    Params { seed, num_threads: Some(2), ..Params::default() }
}

/// A 3x3 box blur of every colour channel (alpha untouched).
fn blur(img: &Image) -> Image {
    let (w, h, nc) = (img.width(), img.height(), img.channels());
    let src = img.data();
    let mut out = src.to_vec();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc.min(3) {
                let mut sum = 0u32;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                        sum += src[(yy * w + xx) * nc + c] as u32;
                    }
                }
                out[(y * w + x) * nc + c] = ((sum + 4) / 9) as u8;
            }
        }
    }
    Image::from_raw(w, h, nc, out)
}

/// A sampled-energy report with per-pixel samples, as `freesynth-eval sampled` writes it.
fn sampled_report(label: &str, input: &SynthesisInput, image: &Image, pixels: &PixelSet, hint: Option<&Nnf>) -> Report {
    let evaluator = PatchErrorEvaluator::new(input, image, PATCH).unwrap();
    let s = sampled_energy(&evaluator, pixels, hint).unwrap();
    for m in &s.matches {
        assert!(m.style >= 0.0 && m.style <= m.total, "{label}: style {} vs total {}", m.style, m.total);
        if let Some(h) = m.hint_total {
            // Exhaustive search can never be worse than the engine's match.
            assert!(m.total <= h * (1.0 + 1e-5) + 1e-3, "{label}: exact {} above the engine's {h}", m.total);
        }
    }
    let mut r = Report::new("sampled");
    r.label = Some(label.to_string());
    r.metric("total", s.total).metric("style", s.style).metric("total_se", s.total_se);
    if let Some(h) = s.hint_total {
        r.metric("nnf_total", h);
    }
    r.samples("total", pixels.key.as_str(), s.matches.iter().map(|m| m.total as f64).collect());
    r
}

/// The `total` row of a comparison.
fn total_row(c: &freesynth_eval::compare::Comparison) -> &freesynth_eval::bootstrap::PairedComparison {
    &c.rows.iter().find(|r| r.metric == "total").expect("a total row").result
}

#[test]
fn quick_tier() {
    let s = Scratch::new("quick");
    let (frames, last) = (8usize, 7usize);
    let scene = s.join("rigid");
    let info = write_scene(
        &SceneParams { width: 64, height: 48, frames, seed: 3, family: StyleFamily::Rigid, flow_gaps: vec![5], ..SceneParams::default() },
        "ci-rigid",
        &scene,
    )
    .unwrap();
    assert_eq!(info.flow_gaps, vec![1, 5]);

    // Stylize every frame from key 0 (seed 1), and the last frame again with seed 2.
    let out_dir = s.join("v");
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut outputs = Vec::with_capacity(frames);
    for t in 0..frames {
        let out = synthesize(&problem(&scene, t), &params(1)).unwrap();
        io::save_png(&out.image, frame_file(&out_dir, t)).unwrap();
        outputs.push(out);
    }
    let input = problem(&scene, last);
    let seed2 = synthesize(&input, &params(2)).unwrap();
    let out = &outputs[last];
    let (w, h) = out.image.size();

    // Exact energy of the engine's own field, and occupancy.
    let evaluator = PatchErrorEvaluator::new(&input, &out.image, PATCH).unwrap();
    let e = nnf_energy(&evaluator, &out.nnf).unwrap();
    assert!(e.total.is_finite() && e.total > 0.0, "{e:?}");
    assert!((e.total - e.style - e.guide).abs() <= 1e-6 * e.total);
    let occ = occupancy(&out.nnf, input.source_style.size(), PATCH).unwrap();
    assert!((0.0..0.9).contains(&occ.unused_frac), "unused fraction {}", occ.unused_frac);
    // At the keyframe the problem is an identity: almost zero energy, almost every patch used.
    let input0 = problem(&scene, 0);
    let e0 = nnf_energy(&PatchErrorEvaluator::new(&input0, &outputs[0].image, PATCH).unwrap(), &outputs[0].nnf).unwrap();
    assert!(e0.total < 0.25 * e.total, "keyframe energy {} vs last frame {}", e0.total, e.total);

    // Sampled exact energy on one shared pixel set: below the engine's field, and a blurred
    // output scores measurably worse on the same pixels.
    let pixels = sample_pixels(w, h, SAMPLES, 1);
    let ref_v = sampled_report("v", &input, &out.image, &pixels, Some(&out.nnf));
    let (exact, nnf_px) = (ref_v.metrics["total"], ref_v.metrics["nnf_total"]);
    assert!(exact <= nnf_px && exact > 0.3 * nnf_px, "exact {exact} vs engine field {nnf_px}");
    let blurred = sampled_report("blur", &input, &blur(&out.image), &pixels, None);
    let cfg = BootstrapConfig { resamples: 2000, ..BootstrapConfig::default() };
    let c = compare_reports(&[("v".into(), ref_v.clone())], &[("blur".into(), blurred)], &["total".into()], None, &cfg).unwrap();
    assert_eq!(c.rows[0].level, Level::Samples);
    assert!(total_row(&c).diff_lo > 0.0, "blur must raise the exact energy: {:?}", total_row(&c));

    // A second seed is an equally valid synthesis: within a few percent.
    let ref_s2 = sampled_report("v seed 2", &input, &seed2.image, &pixels, Some(&seed2.nnf));
    let rel = ref_s2.metrics["total"] / exact - 1.0;
    assert!(rel.abs() < 0.15, "seed 2 differs by {:.1}%", 100.0 * rel);

    // Sharpness against the ground truth: the synthesis is in a sane range, the blur is softer.
    let truth = RgbImage::load(frame_file(&scene.join("truth"), last)).unwrap();
    let sharp_truth = Sharpness::of(&truth, DEFAULT_SIGMA);
    let sharp_v = Sharpness::of(&RgbImage::from_image(&out.image).unwrap(), DEFAULT_SIGMA).ratio(&sharp_truth).unwrap();
    let sharp_b = Sharpness::of(&RgbImage::from_image(&blur(&out.image)).unwrap(), DEFAULT_SIGMA).ratio(&sharp_truth).unwrap();
    assert!((0.2..3.0).contains(&sharp_v.hf), "hf ratio {}", sharp_v.hf);
    assert!((0.2..3.0).contains(&sharp_v.band), "band ratio {}", sharp_v.band);
    assert!(sharp_b.hf < 0.8 * sharp_v.hf, "blur {} vs synthesis {}", sharp_b.hf, sharp_v.hf);
    assert!(sharp_b.band < 0.8 * sharp_v.band, "blur band {} vs synthesis {}", sharp_b.band, sharp_v.band);

    // Temporal metrics through the command line, with the scene's exact flow. A copy of the
    // ground truth with alternating brightness stands in for a flickering method.
    let static_dir = s.join("static");
    let flicker_dir = s.join("flicker");
    std::fs::create_dir_all(&static_dir).unwrap();
    std::fs::create_dir_all(&flicker_dir).unwrap();
    for t in 0..frames {
        std::fs::copy(scene.join("keys/00000.png"), frame_file(&static_dir, t)).unwrap();
        let mut img = io::load_rgba(frame_file(&scene.join("truth"), t)).unwrap();
        if t % 2 == 1 {
            img.data_mut().iter_mut().enumerate().filter(|(i, _)| i % 4 != 3).for_each(|(_, v)| *v = (*v as f32 * 0.8) as u8);
        }
        io::save_png(&img, frame_file(&flicker_dir, t)).unwrap();
    }
    let report_dir = s.join("temporal");
    let seqs = [("input", scene.join("frames")), ("truth", scene.join("truth")), ("static", static_dir), ("flicker", flicker_dir), ("v", out_dir)];
    let mut args: Vec<String> = vec!["temporal".into(), "--scene".into(), arg(&scene), "--out-dir".into(), arg(&report_dir)];
    for (name, dir) in &seqs {
        args.extend(["--frames".into(), format!("{name}={}", arg(dir))]);
    }
    let table = eval_cli(&args);
    assert!(table.contains("| flicker |"), "{table}");
    let t = |name: &str| Report::load(report_dir.join(format!("{name}.json"))).unwrap();
    let (input_t, truth_t, static_t, flicker_t, v_t) = (t("input"), t("truth"), t("static"), t("flicker"), t("v"));
    assert_eq!(input_t.inputs["flow_bw_convention"], "pair");
    assert_eq!(input_t.inputs["direct_flow_gaps"], serde_json::json!([5]));
    let m = |r: &Report, k: &str| *r.metrics.get(k).unwrap_or_else(|| panic!("{k} missing in {:?}", r.metrics));
    // The exact flow warps the input video onto itself (this also checks that both crates
    // agree on the flow file format and indexing).
    assert!(m(&input_t, "warp1") < 2e-3 && m(&input_t, "warp5") < 2e-3, "{:?}", input_t.metrics);
    assert!((0.6..=1.0).contains(&m(&input_t, "mask_frac1")), "{:?}", input_t.metrics);
    assert!((0.4..=1.0).contains(&m(&input_t, "mask_frac5")), "{:?}", input_t.metrics);
    // Rigid ground truth moves with the flow; a repeated key does not (clearest over 5 frames).
    assert!(m(&truth_t, "warp5") < 0.25 * m(&static_t, "warp5"), "truth {:?} static {:?}", truth_t.metrics, static_t.metrics);
    assert!(m(&flicker_t, "warp1") > 3.0 * m(&truth_t, "warp1"));
    assert!(m(&flicker_t, "second_diff") > 3.0 * m(&truth_t, "second_diff"));
    assert!(m(&flicker_t, "global_flicker") > m(&truth_t, "global_flicker") + 5.0);
    // Per-frame synthesis: errors are in [0, 3] per pixel (RGB in [0, 1]), flicker in 8-bit levels.
    for k in ["warp1", "warp5", "second_diff"] {
        let v = m(&v_t, k);
        assert!((0.0..0.2).contains(&v), "v {k} = {v}");
    }
    assert!((0.0..20.0).contains(&m(&v_t, "global_flicker")), "{:?}", v_t.metrics);
    assert_eq!(v_t.samples["warp1"].values.len(), frames - 1);
    // Frame-pair paired comparison through the command line: the flicker is worse than truth.
    let cmp = s.join("cmp.json");
    eval_cli(&[
        "compare".into(),
        arg(&report_dir.join("truth.json")),
        arg(&report_dir.join("flicker.json")),
        "--metric".into(),
        "warp1".into(),
        "--resamples".into(),
        "1000".into(),
        "--json".into(),
        arg(&cmp),
        "--markdown".into(),
        arg(&s.join("cmp.md")),
    ]);
    let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cmp).unwrap()).unwrap();
    assert_eq!(c["rows"][0]["level"], "samples");
    assert_eq!(c["rows"][0]["significant"], true);

    eprintln!(
        "quick tier: energy {:.1} (keyframe {:.1}), exact {exact:.1} / engine {nnf_px:.1} on {SAMPLES} px, seed 2 {:+.1}%, unused {:.3}, hf {:.3} (blur {:.3}), band {:.3} (blur {:.3})",
        e.total,
        e0.total,
        100.0 * rel,
        occ.unused_frac,
        sharp_v.hf,
        sharp_b.hf,
        sharp_v.band,
        sharp_b.band
    );
    for (name, r) in [("input", &input_t), ("truth", &truth_t), ("static", &static_t), ("flicker", &flicker_t), ("v", &v_t)] {
        eprintln!("  {name}: {:?}", r.metrics);
    }
}
