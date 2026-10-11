//! A project on disk, rendered through a session on the CPU.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use freesynth::{
    Backend, CancelToken, Keyframe, MemoryFrames, MemorySink, Progress, Project, Session, SessionOptions, Shot,
    VideoParams,
};
use freesynth_core::io;
use freesynth_scenes::{SceneParams, write_scene};

fn scratch(name: &str) -> PathBuf {
    let target = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp"));
    let dir = target.join(format!("freesynth-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A small generated scene written to disk, with keyframes 0 and 5, and a project for it.
fn project(name: &str) -> (PathBuf, Project) {
    let dir = scratch(name);
    let params = SceneParams { width: 64, height: 48, frames: 6, seed: 2, ..SceneParams::default() };
    write_scene(&params, name, &dir.join("scene")).unwrap();
    let mut project = Project::new("scene/frames/[#####].png", "out");
    project.add_keyframe(0, "scene/keys/00000.png").add_keyframe(5, "scene/keys/00005.png");
    project.engine.backend = Backend::Cpu;
    let path = dir.join("shot.fsproj");
    project.save(&path).unwrap();
    (dir, Project::load(&path).unwrap())
}

#[test]
fn renders_a_project_file() {
    let (dir, project) = project("render");
    let session = Session::new(project.backend()).unwrap();
    assert!(!session.device().is_gpu());
    let written = AtomicUsize::new(0);
    let report = |p: &Progress| {
        if matches!(p, Progress::Written { .. }) {
            written.fetch_add(1, Ordering::Relaxed);
        }
    };
    let stats = session.render(&project, Some(&report), &CancelToken::new()).unwrap();
    assert_eq!((stats.frames, written.load(Ordering::Relaxed)), (6, 6));
    for t in 0..6 {
        let out = io::load_rgba(dir.join("out").join(format!("{t:05}.png"))).unwrap();
        assert_eq!(out.size(), (64, 48));
    }
    assert!(!dir.join(".freesynth-cache").exists(), "the empty cache directory is removed");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn renders_a_range_into_a_custom_sink() {
    let (dir, mut project) = project("range");
    let mut session = Session::new(Backend::Cpu).unwrap();
    session.set_options(SessionOptions {
        parallel_chains: Some(1),
        cache_dir: Some(dir.join("cache")),
        ..SessionOptions::default()
    });
    let sink = MemorySink::new();
    let stats = session.render_to(&project, Some((1, 4)), &sink, None, &CancelToken::new());
    // The range must contain the keyframes.
    assert!(stats.is_err());
    project.keyframes.retain(|k| k.frame == 0);
    let stats = session.render_to(&project, Some((0, 3)), &sink, None, &CancelToken::new()).unwrap();
    assert_eq!(stats.frames, 4);
    assert_eq!(sink.into_frames().keys().copied().collect::<Vec<_>>(), [0, 1, 2, 3]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cancelled_render_reports_cancellation() {
    let (dir, project) = project("cancel");
    let session = Session::new(Backend::Cpu).unwrap();
    let token = CancelToken::new();
    token.cancel();
    let err = session.render(&project, None, &token).unwrap_err();
    assert!(err.is_cancelled(), "{err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn renders_a_shot_held_in_memory() {
    let (dir, project) = project("memory");
    let load = |path: &str| io::load_rgba(project.resolve(path)).unwrap();
    let frames = MemoryFrames::new(0, (0..6).map(|t| load(&format!("scene/frames/{t:05}.png"))).collect()).unwrap();
    let keyframes = vec![Keyframe { frame: 5, style: load("scene/keys/00005.png") }];
    let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
    let mut session = Session::new(Backend::Cpu).unwrap();
    session.set_options(SessionOptions { cache_dir: Some(dir.join("cache")), ..SessionOptions::default() });
    let sink = MemorySink::new();
    let stats = session.render_shot(&shot, VideoParams::classic(), &sink, None, &CancelToken::new()).unwrap();
    assert_eq!(stats.frames, 6);
    assert_eq!(sink.into_frames().len(), 6);
    assert!(!dir.join("cache").join("spill").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn one_session_renders_two_projects_of_one_folder_at_once() {
    let (dir, project) = project("concurrent");
    let mut other = project.clone();
    other.output.frames = "out2".into();
    // Small budgets, so that both renders spill into the shared cache directory.
    let mut session = Session::new(Backend::Cpu).unwrap();
    session.set_options(SessionOptions {
        frame_cache_bytes: 64 * 48 * 4,
        buffer_bytes: 64 * 48 * 3,
        ..SessionOptions::default()
    });
    let session = std::sync::Arc::new(session);
    std::thread::scope(|s| {
        for p in [&project, &other] {
            let session = std::sync::Arc::clone(&session);
            s.spawn(move || {
                let stats = session.render(p, None, &CancelToken::new()).unwrap();
                assert!(stats.frame_cache.spilled > 0, "{stats:?}");
            });
        }
    });
    for out in ["out", "out2"] {
        assert_eq!(std::fs::read_dir(dir.join(out)).unwrap().count(), 6);
    }
    assert!(!dir.join(".freesynth-cache").exists(), "the empty cache directory is removed");
    std::fs::remove_dir_all(dir).unwrap();
}
