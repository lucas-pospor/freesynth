//! End-to-end tests of the `freesynth-eval` binary on tiny generated inputs.

use std::path::PathBuf;
use std::process::Command;

use freesynth_core::{GuideLayer, Image, Params, SynthesisInput, io, synthesize};
use freesynth_eval::Report;
use freesynth_eval::flo::{FlowField, save_flo};

/// A per-test scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("freesynth-eval-cli-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// A smooth colour pattern with some texture, as RGBA.
fn pattern(w: usize, h: usize, phase: f32) -> Image {
    let mut data = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 + phase, y as f32);
            data.push((128.0 + 100.0 * (fx * 0.31).sin() * (fy * 0.23).cos()) as u8);
            data.push((128.0 + 90.0 * (fx * 0.11 + fy * 0.19).sin()) as u8);
            data.push(((fx * 7.0 + fy * 13.0) as i32).rem_euclid(200) as u8 + 20);
            data.push(255);
        }
    }
    Image::from_raw(w, h, 4, data)
}

/// A grey ramp guide, as RGBA.
fn ramp(w: usize, h: usize, scale: f32) -> Image {
    let data = (0..w * h).flat_map(|i| {
        let v = (((i % w) as f32 * scale + (i / w) as f32 * 2.0) as usize % 256) as u8;
        [v, v, v, 255]
    });
    Image::from_raw(w, h, 4, data.collect())
}

fn run<S: AsRef<str>>(args: &[S]) -> String {
    let args: Vec<&str> = args.iter().map(AsRef::as_ref).collect();
    let out = Command::new(env!("CARGO_BIN_EXE_freesynth-eval")).args(&args).output().expect("binary runs");
    assert!(out.status.success(), "freesynth-eval {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn run_fails(args: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_freesynth-eval")).args(args).output().expect("binary runs");
    assert!(!out.status.success(), "freesynth-eval {args:?} should fail");
}

fn report(path: &str) -> Report {
    Report::load(path).unwrap()
}

/// Writes a small synthesis problem and its result; returns the layer arguments.
fn problem(s: &Scratch) -> Vec<String> {
    let style = pattern(40, 32, 0.0);
    let (src, tgt) = (ramp(40, 32, 6.0), ramp(36, 30, 6.5));
    io::save_png(&style, s.path("style.png")).unwrap();
    io::save_png(&src, s.path("src.png")).unwrap();
    io::save_png(&tgt, s.path("tgt.png")).unwrap();
    let input = SynthesisInput::from_rgba_layers(&style, None, &[GuideLayer::new(src, tgt).with_weight(2.0)]).unwrap();
    let out = synthesize(&input, &Params { num_threads: Some(2), ..Params::default() }).unwrap();
    io::save_png(&out.image, s.path("out.png")).unwrap();
    io::save_nnf(&out.nnf, s.path("out.nnf")).unwrap();
    ["-style", &s.path("style.png"), "-guide", &s.path("src.png"), &s.path("tgt.png"), "-weight", "2"].map(String::from).to_vec()
}

#[test]
fn energy_sampled_and_occupancy() {
    let s = Scratch::new("energy");
    let layers = problem(&s);
    // `cmd`, the layer arguments, then `extra`.
    let call = |cmd: &str, extra: &[&str]| {
        let all: Vec<&str> = std::iter::once(cmd).chain(layers.iter().map(String::as_str)).chain(extra.iter().copied()).collect();
        run(&all)
    };

    let (out, nnf) = (s.path("out.png"), s.path("out.nnf"));
    call("energy", &["--image", &out, "--nnf", &nnf, "--json", &s.path("e.json"), "--error-map", &s.path("err.png")]);
    let e = report(&s.path("e.json"));
    assert_eq!(e.kind, "energy");
    let m = &e.metrics;
    assert!((m["total"] - m["style"] - m["guide"]).abs() < 1e-6 * m["total"]);
    assert!((0.0..1.0).contains(&m["unused_frac"]));
    assert_eq!(io::load_rgba(s.path("err.png")).unwrap().size(), (36, 30));

    // All pixels: the exact search can never be worse than the engine's own field.
    call("sampled", &["--image", &out, "--nnf", &nnf, "--samples", "100000", "--json", &s.path("s.json")]);
    let sm = report(&s.path("s.json"));
    assert_eq!(sm.pixels.as_ref().unwrap().coords.len(), 36 * 30);
    assert!(sm.metrics["total"] <= m["total"] + 1e-6);
    assert!((sm.metrics["nnf_total"] - m["total"]).abs() < 1e-3 * m["total"], "{} vs {}", sm.metrics["nnf_total"], m["total"]);
    for (exact, total) in sm.samples["total"].values.iter().zip(&sm.samples["style"].values) {
        assert!(exact >= total);
    }

    // The same pixels again, read from the first report, on a damaged image: paired comparison.
    let mut img = io::load_rgba(&out).unwrap();
    for (i, v) in img.data_mut().iter_mut().enumerate() {
        if i % 4 != 3 {
            *v = v.saturating_add(((i * 37) % 23) as u8);
        }
    }
    io::save_png(&img, s.path("damaged.png")).unwrap();
    call("sampled", &["--image", &s.path("damaged.png"), "--pixels", &s.path("s.json"), "--json", &s.path("d.json"), "--label", "damaged"]);
    let d = report(&s.path("d.json"));
    assert_eq!(d.pixels, sm.pixels);
    assert!(d.metrics["total"] > sm.metrics["total"]);
    run(&["compare", &s.path("s.json"), &s.path("d.json"), "--resamples", "500", "--json", &s.path("c.json"), "--markdown", &s.path("c.md")]);
    let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(s.path("c.json")).unwrap()).unwrap();
    let total = c["rows"].as_array().unwrap().iter().find(|r| r["metric"] == "total").unwrap();
    assert_eq!(total["level"], "samples");
    assert_eq!(total["significant"], true);
    assert!(total["diff_lo"].as_f64().unwrap() > 0.0);
    assert!(std::fs::read_to_string(s.path("c.md")).unwrap().contains("| total * | samples |"));

    // Occupancy alone gives the same numbers as the energy report.
    run(&["occupancy", "--nnf", &nnf, "-style", &s.path("style.png"), "--json", &s.path("o.json")]);
    let o = report(&s.path("o.json"));
    assert_eq!(o.metrics["unused_frac"], m["unused_frac"]);
    run(&["occupancy", "--nnf", &nnf, "--source-size", "40x32", "--json", &s.path("o2.json")]);
    assert_eq!(report(&s.path("o2.json")).metrics, o.metrics);

    // Tables and run statistics.
    let table = run(&["table", &s.path("s.json"), &s.path("d.json"), "--metric", "total"]);
    assert!(table.contains("| damaged |"));
    run(&["stats", &s.path("s.json"), &s.path("d.json"), "--json", &s.path("st.json"), "--markdown", &s.path("st.md")]);
    let st = report(&s.path("st.json"));
    assert!(st.metrics["total_sd"] > 0.0);

    // Bad input is reported, not ignored.
    run_fails(&["energy", "--image", &out]);
    run_fails(&["sampled", "-style", &s.path("style.png"), "-guide", &s.path("src.png"), &s.path("tgt.png"), "--image", &s.path("style.png")]);
    run_fails(&["frobnicate"]);
}

#[test]
fn sharpness_and_temporal() {
    let s = Scratch::new("temporal");
    let (w, h, n) = (32usize, 24usize, 7usize);
    for dir in ["frames", "flicker", "flow_fw", "flow_bw"] {
        std::fs::create_dir_all(s.0.join(dir)).unwrap();
    }
    // A pattern moving right by 1 pixel per frame, and a flickering copy.
    for t in 0..n {
        let img = pattern(w, h, -(t as f32));
        io::save_png(&img, s.path(&format!("frames/{t:05}.png"))).unwrap();
        let mut f = img.clone();
        if t % 2 == 1 {
            f.data_mut().iter_mut().enumerate().filter(|(i, _)| i % 4 != 3).for_each(|(_, v)| *v = (*v as f32 * 0.85) as u8);
        }
        io::save_png(&f, s.path(&format!("flicker/{t:05}.png"))).unwrap();
        if t + 1 < n {
            save_flo(&FlowField::constant(w, h, [1.0, 0.0]), s.path(&format!("flow_fw/{t:05}.flo"))).unwrap();
            save_flo(&FlowField::constant(w, h, [-1.0, 0.0]), s.path(&format!("flow_bw/{t:05}.flo"))).unwrap();
        }
    }
    let (fw, bw) = (s.path("flow_fw"), s.path("flow_bw"));
    run(&["temporal", "--frames", &s.path("frames"), "--flow-fw", &fw, "--flow-bw", &bw, "--json", &s.path("t.json")]);
    let t = report(&s.path("t.json"));
    assert!(t.metrics["warp1"] < 1e-4, "{:?}", t.metrics);
    assert!(t.metrics["warp5"] < 1e-4);
    assert_eq!(t.samples["warp1"].values.len(), n - 1);
    assert_eq!(t.samples["warp5"].values.len(), n - 5);

    let out = s.path("multi");
    let md = run(&["temporal", "--frames", &format!("steady={}", s.path("frames")), "--frames", &format!("flicker={}", s.path("flicker")), "--flow-fw", &fw, "--flow-bw", &bw, "--out-dir", &out]);
    assert!(md.contains("| steady |") && md.contains("| flicker |"));
    let (a, b) = (report(&format!("{out}/steady.json")), report(&format!("{out}/flicker.json")));
    assert!(b.metrics["warp1"] > 20.0 * a.metrics["warp1"]);
    assert!(b.metrics["global_flicker"] > a.metrics["global_flicker"]);
    assert_eq!(a.samples["warp1"].key, b.samples["warp1"].key);
    run_fails(&["temporal", "--frames", &s.path("frames"), "--frames", &s.path("flicker"), "--flow-fw", &fw, "--flow-bw", &bw]);

    // Two runs in the usual <run>/frames layout are named after the run, not both 'frames'.
    for run_dir in ["runA/frames", "runB/frames"] {
        std::fs::create_dir_all(s.0.join(run_dir)).unwrap();
        for t in 0..n {
            std::fs::copy(s.path(&format!("frames/{t:05}.png")), s.path(&format!("{run_dir}/{t:05}.png"))).unwrap();
        }
    }
    let (ra, rb, rep) = (s.path("runA/frames"), s.path("runB/frames"), s.path("rep"));
    let md = run(&["temporal", "--frames", &ra, "--frames", &rb, "--flow-fw", &fw, "--flow-bw", &bw, "--out-dir", &rep]);
    assert!(md.contains("| runA |") && md.contains("| runB |"), "{md}");
    assert!(report(&format!("{rep}/runA.json")).metrics.contains_key("warp1"));
    assert!(report(&format!("{rep}/runB.json")).metrics.contains_key("warp1"));
    // Equal labels, and labels that are not plain file names, are rejected.
    let flows = ["--flow-fw", fw.as_str(), "--flow-bw", bw.as_str(), "--out-dir", rep.as_str()];
    let same = [format!("x={ra}"), format!("x={rb}")];
    run_fails(&[&["temporal", "--frames", &same[0], "--frames", &same[1]][..], &flows].concat());
    run_fails(&[&["temporal", "--frames", &s.path("runA/frames"), "--frames", &s.path("runA/frames")][..], &flows].concat());
    let odd = format!("a b={ra}");
    run_fails(&[&["temporal", "--frames", &odd][..], &flows].concat());
    // Frame ranges outside the frames present are errors, not silently clipped.
    let one = ["temporal", "--frames", ra.as_str(), "--flow-fw", fw.as_str(), "--flow-bw", bw.as_str()];
    run_fails(&[&one[..], &["--last", "99"]].concat());
    run_fails(&[&one[..], &["--first", "3", "--last", "2"]].concat());
    run_fails(&[&one[..], &["--first", "9"]].concat());
    run(&[&one[..], &["--last", "4", "--json", &s.path("t4.json")]].concat());
    assert_eq!(report(&s.path("t4.json")).samples["warp1"].values.len(), 4);

    // Sharpness of a single image and of a sequence.
    let f0 = s.path("frames/00000.png");
    run(&["sharpness", "--image", &f0, "--reference", &f0, "--json", &s.path("sh.json")]);
    let sh = report(&s.path("sh.json"));
    assert!((sh.metrics["hf_ratio"] - 1.0).abs() < 1e-9 && (sh.metrics["band_ratio"] - 1.0).abs() < 1e-9);
    run(&["sharpness", "--frames", &s.path("flicker"), "--reference-frames", &s.path("frames"), "--json", &s.path("shs.json")]);
    let shs = report(&s.path("shs.json"));
    assert_eq!(shs.samples["hf_ratio"].values.len(), n);
    assert_eq!(shs.samples["band_ratio"].values.len(), n);
    assert!(shs.metrics["hf_ratio"] < 1.0);
    // Invalid scales and a flat reference are errors, not missing metrics or huge kernels.
    for sigma in ["-1", "0", "nan", "1e7"] {
        run_fails(&["sharpness", "--image", &f0, "--reference", &f0, "--sigma", sigma]);
    }
    io::save_png(&Image::from_raw(w, h, 3, vec![128; w * h * 3]), s.path("flat.png")).unwrap();
    run_fails(&["sharpness", "--image", &f0, "--reference", &s.path("flat.png")]);
    // Commands reject common options they would ignore.
    run_fails(&["table", &s.path("sh.json"), "--json", &s.path("x.json")]);
    run_fails(&["table", &s.path("sh.json"), "--label", "x"]);
    run_fails(&["compare", &s.path("sh.json"), &s.path("sh.json"), "--label", "x"]);
}
