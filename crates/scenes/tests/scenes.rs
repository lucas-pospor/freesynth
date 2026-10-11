//! End-to-end checks of generated scenes: reproducibility, exactness of the flow, the
//! non-rigid families and the file formats.

use std::fs;
use std::path::{Path, PathBuf};

use freesynth_scenes::output::{flow_path, frame_path, key_path, occlusion_path};
use freesynth_scenes::{
    Direction, Flow, Frame, Manifest, Scene, SceneInfo, SceneParams, StyleFamily, Tier, write_scene,
};
use image::RgbImage;

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("freesynth-scenes").join(name);
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn small(family: StyleFamily) -> SceneParams {
    SceneParams { width: 160, height: 90, frames: 6, family, seed: 11, ..SceneParams::default() }
}

/// Every file under `dir`, relative path and contents, sorted by path.
fn tree(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push((path.strip_prefix(dir).unwrap().to_path_buf(), fs::read(&path).unwrap()));
            }
        }
    }
    out.sort();
    out
}

fn bilinear(img: &RgbImage, x: f64, y: f64) -> [f64; 3] {
    let x = x.clamp(0.0, (img.width() - 1) as f64);
    let y = y.clamp(0.0, (img.height() - 1) as f64);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(img.width() - 1), (y0 + 1).min(img.height() - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let px = |x, y| img.get_pixel(x, y).0.map(f64::from);
    let lerp = |a: [f64; 3], b: [f64; 3], t: f64| std::array::from_fn(|c| a[c] + (b[c] - a[c]) * t);
    lerp(lerp(px(x0, y0), px(x1, y0), fx), lerp(px(x0, y1), px(x1, y1), fx), fy)
}

/// Per-pixel errors (mean over channels, 8-bit levels) of `a` against `b` warped back with
/// `flow`, on the pixels not marked in `occlusion`.
fn warp_errors(a: &RgbImage, b: &RgbImage, flow: &Flow, occlusion: &image::GrayImage) -> Vec<f64> {
    let mut errs = Vec::new();
    for y in 0..a.height() {
        for x in 0..a.width() {
            if occlusion.get_pixel(x, y)[0] != 0 {
                continue;
            }
            let [u, v] = flow.get(x as usize, y as usize);
            let w = bilinear(b, x as f64 + u as f64, y as f64 + v as f64);
            let p = a.get_pixel(x, y).0;
            errs.push((0..3).map(|c| (p[c] as f64 - w[c]).abs()).sum::<f64>() / 3.0);
        }
    }
    errs
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn render_all(scene: &Scene) -> Vec<Frame> {
    (0..scene.params().frames).map(|t| scene.render(t)).collect()
}

#[test]
fn same_seed_gives_the_same_bytes() {
    for family in StyleFamily::ALL {
        let params =
            SceneParams { width: 64, height: 48, frames: 4, family, flow_gaps: vec![2], ..SceneParams::default() };
        let (a, b) = (scratch(&format!("det-a-{family}")), scratch(&format!("det-b-{family}")));
        write_scene(&params, "det", &a).unwrap();
        write_scene(&params, "det", &b).unwrap();
        let (ta, tb) = (tree(&a), tree(&b));
        // 4 frames + 4 truths + 2 keys + flow and masks for gaps 1 (3 pairs) and 2 (2 pairs).
        assert_eq!(ta.len(), 4 + 4 + 2 + 4 * (3 + 2) + 1);
        assert!(ta == tb, "{family}: two runs with the same seed differ");

        let other = scratch(&format!("det-c-{family}"));
        write_scene(&SceneParams { seed: params.seed + 1, ..params }, "det", &other).unwrap();
        let tc = tree(&other);
        assert_ne!(ta.iter().find(|f| f.0.ends_with("00000.png")), tc.iter().find(|f| f.0.ends_with("00000.png")));
        for d in [a, b, other] {
            fs::remove_dir_all(d).unwrap();
        }
    }
}

#[test]
fn exact_flow_reproduces_the_next_frame_except_at_occlusions() {
    let scene = Scene::new(&small(StyleFamily::Rigid)).unwrap();
    let frames = render_all(&scene);
    let (mut visible, mut occluded_fraction) = (Vec::new(), Vec::new());
    for t in 0..frames.len() - 1 {
        let (f0, f1) = (&frames[t], &frames[t + 1]);
        let fw = scene.flow(t, t + 1, &f0.layers, &f1.layers);
        let bw = scene.flow(t + 1, t, &f1.layers, &f0.layers);
        for (flow, src, dst) in [(&fw, f0, f1), (&bw, f1, f0)] {
            visible.extend(warp_errors(&src.image, &dst.image, &flow.flow, &flow.occlusion));
            let marked = flow.occlusion.pixels().filter(|p| p[0] != 0).count();
            occluded_fraction.push(marked as f64 / (160.0 * 90.0));
        }
        // Forward and backward flow are exact inverses wherever both are usable.
        for y in 0..90 {
            for x in 0..160 {
                if fw.occlusion.get_pixel(x, y)[0] != 0 {
                    continue;
                }
                let [u, v] = fw.flow.get(x as usize, y as usize);
                let [bu, bv] = bw.flow.sample(x as f32 + u, y as f32 + v);
                assert!((u + bu).abs() < 1e-3 && (v + bv).abs() < 1e-3, "fw-bw mismatch at {x},{y}");
            }
        }
    }
    let mut sorted = visible.clone();
    sorted.sort_by(f64::total_cmp);
    let p99 = sorted[sorted.len() * 99 / 100];
    // Measured: mean 0.66 and 99th percentile 2.1 levels at this size (mean 0.35 at 480x270);
    // what remains is bilinear interpolation error on the fine surface texture.
    assert!(mean(&visible) < 1.0, "mean warp error {:.3} on unoccluded pixels", mean(&visible));
    assert!(p99 < 4.0, "99th percentile warp error {p99:.3}");
    let occ = mean(&occluded_fraction);
    assert!((0.005..0.25).contains(&occ), "occluded fraction {occ:.3}");
}

#[test]
fn occlusion_masks_cover_disocclusions() {
    // Warping across occlusion boundaries must fail visibly where the masks say so,
    // otherwise the masks would be marking pixels at random.
    let scene = Scene::new(&SceneParams { frames: 6, ..small(StyleFamily::Rigid) }).unwrap();
    let (f0, f5) = (scene.render(0), scene.render(5));
    let flow = scene.flow(0, 5, &f0.layers, &f5.layers);
    let inverted = image::GrayImage::from_fn(160, 90, |x, y| image::Luma([255 - flow.occlusion.get_pixel(x, y)[0]]));
    let unmasked = mean(&warp_errors(&f0.image, &f5.image, &flow.flow, &flow.occlusion));
    let masked = mean(&warp_errors(&f0.image, &f5.image, &flow.flow, &inverted));
    assert!(masked > 5.0 * unmasked, "masked {masked:.2} vs unmasked {unmasked:.2}");
}

#[test]
fn non_rigid_families_are_not_rigid_transport() {
    // Carry the first frame's ground truth to the last frame with the exact flow (perfect
    // advection) and compare with the last frame's ground truth.
    let error_of = |family| {
        let params = SceneParams { frames: 16, ..small(family) };
        let scene = Scene::new(&params).unwrap();
        let (first, last) = (scene.render(0), scene.render(15));
        let back = scene.flow(15, 0, &last.layers, &first.layers);
        mean(&warp_errors(&last.truth, &first.truth, &back.flow, &back.occlusion))
    };
    // Measured: rigid 2.1, painterly 8.7, hatching 11.5 levels.
    let rigid = error_of(StyleFamily::Rigid);
    let painterly = error_of(StyleFamily::Painterly);
    let hatching = error_of(StyleFamily::Hatching);
    assert!(rigid < 3.0, "rigid advection error {rigid:.2}");
    assert!(painterly > 3.0 * rigid, "painterly {painterly:.2} vs rigid {rigid:.2}");
    assert!(hatching > 3.0 * rigid, "hatching {hatching:.2} vs rigid {rigid:.2}");
}

#[test]
fn flo_files_match_the_format_and_the_flow() {
    let params = SceneParams { width: 64, height: 48, frames: 3, ..SceneParams::default() };
    let dir = scratch("flo");
    write_scene(&params, "flo", &dir).unwrap();
    let scene = Scene::new(&params).unwrap();
    let (f0, f1) = (scene.render(0), scene.render(1));
    for (direction, expected) in [
        (Direction::Forward, scene.flow(0, 1, &f0.layers, &f1.layers)),
        (Direction::Backward, scene.flow(1, 0, &f1.layers, &f0.layers)),
    ] {
        let bytes = fs::read(flow_path(&dir, direction, 1, 0)).unwrap();
        assert_eq!(bytes.len(), 12 + 64 * 48 * 8);
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 202_021.25);
        assert_eq!(i32::from_le_bytes(bytes[4..8].try_into().unwrap()), 64);
        assert_eq!(i32::from_le_bytes(bytes[8..12].try_into().unwrap()), 48);
        // Pixel (x, y) lives at byte 12 + 8 * (y * width + x), u then v.
        let (x, y) = (37usize, 21usize);
        let at = 12 + 8 * (y * 64 + x);
        let u = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let v = f32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap());
        assert_eq!([u, v], expected.flow.get(x, y));
        assert_eq!(Flow::load(flow_path(&dir, direction, 1, 0)).unwrap(), expected.flow);
        let mask = image::open(occlusion_path(&dir, direction, 1, 0)).unwrap().to_luma8();
        assert_eq!(mask, expected.occlusion);
    }
    let frame = image::open(frame_path(&dir, 1)).unwrap().to_rgb8();
    assert_eq!(frame, f1.image);
    let key = image::open(key_path(&dir, 2)).unwrap().to_rgb8();
    assert_eq!(key, scene.render(2).truth);
    let info = SceneInfo::load(&dir).unwrap();
    assert_eq!((info.keyframes.as_slice(), info.flow_gaps.as_slice()), (&[0, 2][..], &[1][..]));
    assert_eq!(info.licence, "CC0-1.0");
    assert_eq!(info.generator.as_ref(), Some(&params));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ci_tier_of_the_corpus_generates() {
    // Only inside the FreeSynth repository (not in a packaged crate).
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/manifest.toml");
    if !path.exists() {
        return;
    }
    let manifest = Manifest::load(&path).unwrap();
    for entry in manifest.scenes.iter().filter(|s| s.tier == Tier::Ci) {
        let dir = scratch(&format!("ci-{}", entry.name));
        let info = write_scene(&entry.params, &entry.name, &dir).unwrap();
        assert_eq!(info.frames, entry.params.frames);
        assert!(frame_path(&dir, info.frames - 1).exists());
        fs::remove_dir_all(dir).unwrap();
    }
}
