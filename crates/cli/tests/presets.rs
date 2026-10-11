//! End-to-end checks of the preset options, `-dumperror` and `-modulation`.

use std::path::{Path, PathBuf};
use std::process::Command;

use freesynth_core::{GuideLayer, Image, Params, SynthesisInput, io, nnf_patch_errors};

/// A textured RGBA test image.
fn pattern(w: usize, h: usize, shift: usize) -> Image {
    let data = (0..w * h).flat_map(|i| {
        let (x, y) = (i % w + shift, i / w);
        [(x * 9 + y * 4) as u8, ((x * y) % 251) as u8, ((x ^ y) * 13) as u8, 255]
    });
    Image::from_raw(w, h, 4, data.collect())
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let s = Scratch(dir);
        io::save_png(&pattern(40, 30, 0), s.path("style.png")).unwrap();
        io::save_png(&pattern(40, 30, 3), s.path("src.png")).unwrap();
        io::save_png(&pattern(36, 28, 7), s.path("tgt.png")).unwrap();
        s
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }

    /// Runs the binary on the test images with `extra` options; returns (success, stdout).
    fn run(&self, extra: &[&str]) -> (bool, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_freesynth"))
            .args(["-style", &self.path("style.png"), "-guide", &self.path("src.png"), &self.path("tgt.png")])
            .args(extra)
            .args(["-threads", "2", "-deterministic"])
            .output()
            .expect("binary runs");
        (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

#[test]
fn presets_print_their_schedule() {
    let s = Scratch::new("freesynth-cli-presets");
    let out = s.path("o.png");
    let (ok, text) = s.run(&["-output", &out]);
    assert!(ok, "{text}");
    assert!(text.contains("preset: default") && text.contains("patchmatchiters: 2"), "{text}");
    assert!(text.contains("stopmask: off\nvote: weighted (rounded)\n"), "{text}");
    let (ok, text) = s.run(&["-output", &out, "-compat"]);
    assert!(ok, "{text}");
    // The compat output is the 2019 tool's: no preset, stop mask or vote lines.
    assert!(!text.contains("preset:") && text.contains("patchmatchiters: 4"), "{text}");
    assert!(!text.contains("stopmask:") && !text.contains("vote:"), "{text}");
    // Overrides on top of compat no longer give the 2019 result, so the summary says so.
    let (ok, text) = s.run(&["-output", &out, "-compat", "-vote", "weighted", "-stopmask", "on"]);
    assert!(ok, "{text}");
    assert!(text.contains("preset: compat") && text.contains("stopmask: on\nvote: weighted (truncated)"), "{text}");
    let (ok, text) = s.run(&["-output", &out, "-preset", "fast", "-backend", "cpu", "-patchmatchiters", "3"]);
    assert!(ok, "{text}");
    assert!(text.contains("preset: fast") && text.contains("patchmatchiters: 3") && text.contains("searchvoteiters: 6"), "{text}");
    let (ok, text) = s.run(&["-output", &out, "-preset", "turbo"]);
    assert!(!ok && text.contains("unknown preset"), "{text}");
}

#[test]
fn error_map_holds_the_final_patch_errors() {
    let s = Scratch::new("freesynth-cli-errormap");
    let (out, nnf, err) = (s.path("o.png"), s.path("o.nnf"), s.path("o.pfm"));
    let (ok, text) = s.run(&["-output", &out, "-dumpnnf", &nnf, "-dumperror", &err]);
    assert!(ok && text.contains("error map was written to"), "{text}");

    let (w, h, errors) = io::load_pfm(&err).unwrap();
    assert_eq!((w, h), (36, 28));
    let style = io::load_rgba(s.path("style.png")).unwrap();
    let guide = GuideLayer::new(io::load_rgba(s.path("src.png")).unwrap(), io::load_rgba(s.path("tgt.png")).unwrap());
    let input = SynthesisInput::from_rgba_layers(&style, None, &[guide]).unwrap();
    let image = io::select_channels(&io::load_rgba(&out).unwrap(), input.source_style.channels());
    let want = nnf_patch_errors(&input, &Params::default(), &image, &io::load_nnf(&nnf).unwrap()).unwrap();
    for (i, &e) in errors.iter().enumerate() {
        let t = want.style[i] + want.guide[i];
        assert!((e - t).abs() <= 1e-3 * t.max(1.0), "pixel {i}: {e} vs {t}");
    }
}

#[test]
fn modulation_maps_are_checked_and_applied() {
    let s = Scratch::new("freesynth-cli-modulation");
    io::save_png(&Image::new(36, 28, 1), s.path("off.png")).unwrap();
    io::save_png(&Image::new(20, 28, 1), s.path("small.png")).unwrap();
    let (out, plain, off) = (s.path("o.png"), s.path("plain.pfm"), s.path("off.pfm"));
    let (ok, text) = s.run(&["-modulation", &s.path("small.png"), "-output", &out]);
    assert!(!ok && text.contains("doesn't match the resolution"), "{text}");

    // With the only guide switched off, the result changes and the written errors are those of
    // the modulated problem (style errors only).
    let (ok, text) = s.run(&["-output", &out, "-dumperror", &plain]);
    assert!(ok, "{text}");
    let guided = io::load_rgba(&out).unwrap();
    let nnf = s.path("off.nnf");
    let (ok, text) = s.run(&["-modulation", &s.path("off.png"), "-output", &out, "-dumperror", &off, "-dumpnnf", &nnf]);
    assert!(ok, "{text}");
    let unguided = io::load_rgba(&out).unwrap();
    assert_ne!(guided, unguided);

    let style = io::load_rgba(s.path("style.png")).unwrap();
    let guide = GuideLayer::new(io::load_rgba(s.path("src.png")).unwrap(), io::load_rgba(s.path("tgt.png")).unwrap())
        .with_modulation(Image::new(36, 28, 1));
    let input = SynthesisInput::from_rgba_layers(&style, None, &[guide]).unwrap();
    let image = io::select_channels(&unguided, input.source_style.channels());
    let want = nnf_patch_errors(&input, &Params::default(), &image, &io::load_nnf(&nnf).unwrap()).unwrap();
    assert!(want.guide.iter().all(|&g| g == 0.0));
    let (_, _, errors) = io::load_pfm(&off).unwrap();
    for (i, &e) in errors.iter().enumerate() {
        assert!((e - want.style[i]).abs() <= 1e-3 * want.style[i].max(1.0), "pixel {i}: {e} vs {}", want.style[i]);
    }
}
