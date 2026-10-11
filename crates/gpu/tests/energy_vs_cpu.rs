//! The GPU engine against the CPU engine under the exact evaluator.
//!
//! Both engines stylize the same small generated frame (style = keyframe 0, guide = frame 0 to
//! frame 5); the sampled exact coherence energy of the two outputs, on the same pixels, must
//! agree within 15%. The test skips itself when no GPU adapter is available, or when
//! `FREESYNTH_NO_GPU` is set.

use freesynth_core::{GuideLayer, Image, Params, PatchErrorEvaluator, SynthesisInput, synthesize};
use freesynth_eval::energy::{sample_pixels, sampled_energy};
use freesynth_scenes::{Scene, SceneParams, StyleFamily};

/// An opaque RGBA image from a generated RGB frame.
fn rgba(img: &image::RgbImage) -> Image {
    let (w, h) = (img.width() as usize, img.height() as usize);
    Image::from_raw(w, h, 4, img.pixels().flat_map(|p| [p[0], p[1], p[2], 255]).collect())
}

#[test]
fn gpu_energy_matches_cpu() {
    if std::env::var_os("FREESYNTH_NO_GPU").is_some() {
        eprintln!("skipped: FREESYNTH_NO_GPU is set");
        return;
    }
    let ctx = match freesynth_gpu::GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    let scene = Scene::new(&SceneParams { width: 64, height: 48, frames: 8, seed: 3, family: StyleFamily::Rigid, ..SceneParams::default() }).unwrap();
    let (f0, f5) = (scene.render(0), scene.render(5));
    let guide = GuideLayer::new(rgba(&f0.image), rgba(&f5.image)).with_weight(4.0);
    let input = SynthesisInput::from_rgba_layers(&rgba(&f0.truth), None, &[guide]).unwrap();
    let params = Params { seed: 1, num_threads: Some(2), ..Params::default() };

    let cpu = synthesize(&input, &params).unwrap();
    let gpu = ctx.synthesize(&input, &params).expect("GPU synthesis");
    let pixels = sample_pixels(64, 48, 400, 1);
    let energy = |img: &Image| sampled_energy(&PatchErrorEvaluator::new(&input, img, 5).unwrap(), &pixels, None).unwrap().total;
    let (e_cpu, e_gpu) = (energy(&cpu.image), energy(&gpu.image));
    let rel = e_gpu / e_cpu - 1.0;
    eprintln!("exact energy: CPU {e_cpu:.1}, GPU {e_gpu:.1} ({:+.1}%)", 100.0 * rel);
    assert!(e_cpu > 0.0 && rel.abs() < 0.15, "GPU differs from CPU by {:.1}%", 100.0 * rel);
}
