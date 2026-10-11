//! End-to-end check of `-dumpnnf`: the field the binary writes can be read back and evaluated
//! against the image it wrote.

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

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("freesynth-cli-dumpnnf");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn dumped_field_matches_the_written_image() {
    let dir = scratch();
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let (style, src, tgt) = (pattern(40, 30, 0), pattern(40, 30, 3), pattern(36, 28, 7));
    io::save_png(&style, p("style.png")).unwrap();
    io::save_png(&src, p("src.png")).unwrap();
    io::save_png(&tgt, p("tgt.png")).unwrap();

    let run = |out: &str, nnf: &str| {
        let status = Command::new(env!("CARGO_BIN_EXE_freesynth"))
            .args(["-style", &p("style.png"), "-guide", &p("src.png"), &p("tgt.png"), "-weight", "2"])
            .args(["-threads", "2", "-deterministic", "-output", &p(out), "-dumpnnf", &p(nnf)])
            .output()
            .expect("binary runs");
        assert!(status.status.success(), "{}", String::from_utf8_lossy(&status.stdout));
        assert!(String::from_utf8_lossy(&status.stdout).contains("nnf was written to"));
    };
    run("out.png", "out.nnf");
    run("again.png", "again.nnf");

    let nnf = io::load_nnf(p("out.nnf")).unwrap();
    assert_eq!((nnf.width(), nnf.height()), (36, 28));
    // Deterministic runs write the same field.
    assert_eq!(std::fs::read(p("out.nnf")).unwrap(), std::fs::read(p("again.nnf")).unwrap());

    // The field is valid for the problem and explains the written image: its patch error is
    // that of a finished synthesis, far below the error of a field pointing at one patch.
    let input = SynthesisInput::from_rgba_layers(&style, None, &[GuideLayer::new(src, tgt).with_weight(2.0)]).unwrap();
    let image = io::select_channels(&io::load_rgba(p("out.png")).unwrap(), input.source_style.channels());
    let params = Params::default();
    let errors = nnf_patch_errors(&input, &params, &image, &nnf).unwrap();
    let mean = |v: &[f32]| v.iter().map(|&e| e as f64).sum::<f64>() / v.len() as f64;
    let mut constant = nnf.clone();
    constant.data_mut().iter_mut().for_each(|c| *c = [20, 15]);
    let worse = nnf_patch_errors(&input, &params, &image, &constant).unwrap();
    assert!(mean(&errors.style) < 0.5 * mean(&worse.style), "{} vs {}", mean(&errors.style), mean(&worse.style));
    std::fs::remove_dir_all(&dir).ok();
}
