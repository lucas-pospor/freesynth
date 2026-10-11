//! Position fields against the exact keyframe positions of a generated scene, with the CPU
//! estimator (no GPU needed).

use freesynth_flow::cpu::CpuDis;
use freesynth_flow::track::{PositionField, TrackParams, Tracker};
use freesynth_flow::{DisParams, GrayImage};
use freesynth_scenes::{Scene, SceneParams, StyleFamily};

/// Position errors over the pixels visible in the keyframe: (share confident, mean error of
/// confident pixels, share of confident pixels off by more than 3 px, share of the pixels off by
/// more than 3 px that are flagged).
fn score(positions: &PositionField, confident: &[bool], truth: &PositionField, valid: &[bool]) -> (f64, f64, f64, f64) {
    let (mut n, mut conf, mut conf_err, mut conf_bad, mut bad, mut flagged_bad) = (0, 0, 0.0, 0, 0, 0);
    for i in 0..valid.len() {
        if !valid[i] {
            continue;
        }
        let (p, q) = (positions.data()[i], truth.data()[i]);
        let e = f64::from((p[0] - q[0]).hypot(p[1] - q[1]));
        n += 1;
        bad += usize::from(e > 3.0);
        if confident[i] {
            conf += 1;
            conf_err += e;
            conf_bad += usize::from(e > 3.0);
        } else {
            flagged_bad += usize::from(e > 3.0);
        }
    }
    (
        conf as f64 / n as f64,
        conf_err / conf as f64,
        conf_bad as f64 / conf as f64,
        flagged_bad as f64 / bad.max(1) as f64,
    )
}

#[test]
fn confidence_marks_wrong_positions() {
    let params = SceneParams {
        width: 320,
        height: 180,
        frames: 25,
        seed: 4,
        family: StyleFamily::Rigid,
        keyframes: vec![0],
        ..SceneParams::default()
    };
    let scene = Scene::new(&params).unwrap();
    let (w, h) = (params.width, params.height);
    let gray = |f: &freesynth_scenes::Frame| GrayImage::from_rgb8(w, h, f.image.as_raw()).unwrap();
    let key = scene.render(0);
    let dis = CpuDis::new(w, h, &DisParams::default()).unwrap();
    let mut tracker = Tracker::new(dis, gray(&key), TrackParams::default()).unwrap();
    for t in 1..params.frames {
        let frame = scene.render(t);
        let info = tracker.step(&gray(&frame)).unwrap();
        assert_eq!(info.index, t);
        if t % 12 != 0 {
            continue;
        }
        let exact = scene.flow(t, 0, &frame.layers, &key.layers);
        let valid: Vec<bool> = exact.occlusion.as_raw().iter().map(|&v| v == 0).collect();
        let truth = PositionField::from_displacement(&exact.flow);
        let (share, error, bad, recall) =
            score(tracker.positions(), &tracker.confidence().confident(0.5), &truth, &valid);
        eprintln!(
            "frame {t}: {:.1}% confident, error {error:.3} px, bad3 {:.2}%, bad pixels flagged {:.1}%",
            100.0 * share,
            100.0 * bad,
            100.0 * recall
        );
        assert!(share > 0.5, "frame {t}: only {share} confident");
        assert!(error < 0.6 && bad < 0.04, "frame {t}: confident error {error} px, bad3 {bad}");
        assert!(recall > 0.65, "frame {t}: only {recall} of the wrong positions flagged");
    }
    // Anchors every 10 frames at most.
    assert!(tracker.index() - tracker.anchor_index() < 10);
}
