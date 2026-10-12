//! Projects on disk, rendered through a session on the CPU: whole renders, incremental
//! re-renders, ranges, cancellation and shots held in memory.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use freesynth::{
    Backend, CancelToken, Event, Keyframe, MemoryFrames, MemorySink, Project, RenderRequest, Session, SessionOptions,
    Shot, VideoParams,
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
    project_with(name, 6, &[0, 5])
}

/// A generated scene of `frames` frames with the given keyframes, and a project for it.
fn project_with(name: &str, frames: usize, keys: &[usize]) -> (PathBuf, Project) {
    let dir = scratch(name);
    let params =
        SceneParams { width: 64, height: 48, frames, seed: 2, keyframes: keys.to_vec(), ..SceneParams::default() };
    write_scene(&params, name, &dir.join("scene")).unwrap();
    let mut project = Project::new("scene/frames/[#####].png", "out");
    for &k in keys {
        project.add_keyframe(k, format!("scene/keys/{k:05}.png"));
    }
    project.engine.backend = Backend::Cpu;
    let path = dir.join("shot.fsyn");
    project.save(&path).unwrap();
    (dir, Project::load(&path).unwrap())
}

/// Modification times of the output frames.
fn mtimes(dir: &std::path::Path, frames: std::ops::RangeInclusive<usize>) -> Vec<std::time::SystemTime> {
    frames.map(|t| std::fs::metadata(dir.join(format!("out/{t:05}.png"))).unwrap().modified().unwrap()).collect()
}

#[test]
fn renders_a_project_file() {
    let (dir, project) = project("render");
    let session = Session::new(project.backend()).unwrap();
    assert!(!session.device().is_gpu());
    let written = AtomicUsize::new(0);
    let report = |e: &Event| {
        if matches!(e, Event::Written { .. }) {
            written.fetch_add(1, Ordering::Relaxed);
        }
    };
    let stats = session.render(&project, Some(&report), &CancelToken::new()).unwrap();
    assert_eq!((stats.frames, stats.rendered, written.load(Ordering::Relaxed)), (6, 6, 6));
    assert_eq!(stats.runs, [(0, 5)]);
    for t in 0..6 {
        let out = io::load_rgba(dir.join("out").join(format!("{t:05}.png"))).unwrap();
        assert_eq!(out.size(), (64, 48));
    }
    // The cache keeps only its record (no spill files, no frames: the output holds them).
    let cache = dir.join(".freesynth-cache").join("shot");
    let names: Vec<_> = std::fs::read_dir(&cache).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(names, ["manifest.json"]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn editing_a_keyframe_re_renders_only_its_reach() {
    let (dir, project) = project_with("incremental", 13, &[0, 6, 12]);
    let session = Session::new(Backend::Cpu).unwrap();
    let first = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((first.rendered, first.reused), (13, 0));

    // Nothing changed: nothing renders.
    let again = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((again.rendered, again.reused, again.syntheses), (0, 13, 0));
    assert!(again.runs.is_empty());

    // Repaint the last keyframe: its chains (7-12) and nothing else render again.
    let before = mtimes(&dir, 0..=12);
    let key = project.resolve("scene/keys/00012.png");
    let mut style = io::load_rgba(&key).unwrap();
    let mut data = style.data().to_vec();
    data.iter_mut().step_by(4).for_each(|r| *r = r.saturating_add(40));
    style = freesynth_core::Image::from_raw(64, 48, 4, data);
    io::save_png(&style, &key).unwrap();
    let edit = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((edit.rendered, edit.reused), (6, 7), "{edit:?}");
    assert_eq!(edit.runs, [(6, 12)]);
    let after = mtimes(&dir, 0..=12);
    assert_eq!(before[..=6], after[..=6], "frames 0-6 are untouched");
    assert!(before[7..].iter().zip(&after[7..]).all(|(b, a)| a >= b));

    // Repaint the middle keyframe: the frames between its neighbours (1-11) render again.
    let key = project.resolve("scene/keys/00006.png");
    let mut data = io::load_rgba(&key).unwrap().data().to_vec();
    data.iter_mut().skip(1).step_by(4).for_each(|g| *g = g.saturating_add(30));
    io::save_png(&freesynth_core::Image::from_raw(64, 48, 4, data), &key).unwrap();
    let edit = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((edit.rendered, edit.runs.clone()), (11, vec![(0, 12)]));

    // A deleted output frame is rendered again, alone with its segment.
    std::fs::remove_file(dir.join("out/00003.png")).unwrap();
    let fix = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((fix.rendered, fix.runs.clone()), (1, vec![(0, 6)]));

    // Forcing a keyframe, a range or everything.
    let mut request = RenderRequest::default();
    request.keyframes = vec![0];
    let forced = session.render_with(&project, &request, None, &CancelToken::new()).unwrap();
    assert_eq!(forced.rendered, 6, "keyframe 0 reaches frames 0-5");
    let mut request = RenderRequest::default();
    (request.range, request.force) = (Some((2, 3)), true);
    assert_eq!(session.render_with(&project, &request, None, &CancelToken::new()).unwrap().rendered, 2);
    let mut request = RenderRequest::default();
    request.force = true;
    assert_eq!(session.render_with(&project, &request, None, &CancelToken::new()).unwrap().rendered, 13);

    // A changed setting renders everything.
    let mut changed = project.clone();
    changed.guides.mapping = 5.0;
    assert_eq!(session.render(&changed, None, &CancelToken::new()).unwrap().rendered, 13);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sixteen_bit_output() {
    let (dir, mut project) = project("sixteen");
    // Repaint keyframe 5 at 16 bits, with detail finer than 8 bits can hold.
    let key = dir.join("scene/keys/00005.png");
    let painting = io::load_rgba(&key).unwrap();
    let deep: Vec<u16> = painting
        .data()
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 4 == 3 { 65535 } else { (u16::from(v) * 257).saturating_add(100) })
        .collect();
    freesynth_media::save_png16(64, 48, 4, &deep, &key).unwrap();
    project.output.bit_depth = 16;
    project.output.frames = "deep/f_[####].png".into();
    let session = Session::new(Backend::Cpu).unwrap();
    session.render(&project, None, &CancelToken::new()).unwrap();
    let img = image_info(&dir.join("deep/f_0003.png"));
    assert_eq!(img, (64, 48, 16));
    // The frames carry the paintings' extra precision (not 8-bit values widened).
    for t in [3, 5] {
        let (_, _, data) = freesynth_media::load_rgba16(&dir.join(format!("deep/f_{t:04}.png"))).unwrap();
        let fine = data.iter().filter(|v| *v % 257 != 0).count();
        assert!(fine > data.len() / 4, "frame {t}: {fine} of {} samples between 8-bit levels", data.len());
    }
    project.output.frames = "deep/f_[####].jpg".into();
    assert!(project.validate().is_err(), "16-bit needs PNG");
    std::fs::remove_dir_all(dir).unwrap();
}

/// Width, height and bits per channel of a PNG file (from its header).
fn image_info(path: &std::path::Path) -> (u32, u32, u8) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[1..4], b"PNG");
    let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    (w, h, bytes[24])
}

#[test]
fn renders_a_range_into_a_custom_sink() {
    let (dir, mut project) = project("range");
    let mut session = Session::new(Backend::Cpu).unwrap();
    let mut options = SessionOptions::default();
    (options.parallel_chains, options.cache_dir) = (Some(1), Some(dir.join("cache")));
    session.set_options(options);
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
    let mut options = SessionOptions::default();
    options.cache_dir = Some(dir.join("cache"));
    session.set_options(options);
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
    other.set_name("other");
    // Small budgets, so that both renders spill.
    let mut session = Session::new(Backend::Cpu).unwrap();
    let mut options = SessionOptions::default();
    (options.frame_cache_bytes, options.buffer_bytes) = (64 * 48 * 4, 64 * 48 * 3);
    session.set_options(options);
    let session = std::sync::Arc::new(session);
    std::thread::scope(|s| {
        for p in [&project, &other] {
            let session = std::sync::Arc::clone(&session);
            s.spawn(move || {
                let stats = session.render(p, None, &CancelToken::new()).unwrap();
                assert!(stats.run_stats[0].frame_cache.spilled > 0, "{stats:?}");
            });
        }
    });
    for out in ["out", "out2"] {
        assert_eq!(std::fs::read_dir(dir.join(out)).unwrap().count(), 6);
    }
    for name in ["shot", "other"] {
        let cache = dir.join(".freesynth-cache").join(name);
        assert!(!cache.join("spill").exists(), "spill files are removed");
        assert!(!cache.join("lock").exists(), "the lock is released");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_project_renders_once_at_a_time() {
    let (dir, project) = project("locked");
    let cache = project.default_cache_dir();
    std::fs::create_dir_all(&cache).unwrap();
    // A lock held by this (live) process.
    std::fs::write(cache.join("lock"), std::process::id().to_string()).unwrap();
    let session = Session::new(Backend::Cpu).unwrap();
    let err = session.render(&project, None, &CancelToken::new()).unwrap_err().to_string();
    assert!(err.contains("another render of this project is running"), "{err}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// Moving optical flow onto the GPU changes the frames slightly, so it makes every frame stale.
/// Skipped without a usable GPU.
#[test]
fn switching_the_flow_device_renders_every_frame_again() {
    let (dir, project) = project("flow-device");
    let Ok(mut session) = Session::new(Backend::Gpu) else {
        eprintln!("skipped: no usable GPU");
        std::fs::remove_dir_all(dir).unwrap();
        return;
    };
    let cancel = CancelToken::new();
    assert_eq!(session.render(&project, None, &cancel).unwrap().rendered, 6);
    assert_eq!(session.render(&project, None, &cancel).unwrap().rendered, 0, "up to date");
    let mut options = SessionOptions::default();
    options.flow_backend = freesynth::FlowBackend::Device;
    session.set_options(options);
    assert_eq!(session.render(&project, None, &cancel).unwrap().rendered, 6, "every frame is stale");
    assert_eq!(session.render(&project, None, &cancel).unwrap().rendered, 0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn outputs_never_overwrite_inputs() {
    let (dir, mut project) = project("overwrite");
    let session = Session::new(Backend::Cpu).unwrap();
    let before = std::fs::read(dir.join("scene/frames/00002.png")).unwrap();
    project.output.frames = "scene/frames/[#####].png".into();
    let err = session.render(&project, None, &CancelToken::new()).unwrap_err().to_string();
    assert!(err.contains("would overwrite the input file"), "{err}");
    assert!(project.check_inputs().unwrap_err().to_string().contains("would overwrite"));
    project.output.frames = "scene/keys".into();
    let err = session.render(&project, None, &CancelToken::new()).unwrap_err().to_string();
    assert!(err.contains("00000.png"), "{err}");
    assert_eq!(std::fs::read(dir.join("scene/frames/00002.png")).unwrap(), before);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_cache_follows_the_project_not_the_working_directory() {
    let (dir, project) = project("cache-place");
    let session = Session::new(Backend::Cpu).unwrap();
    assert_eq!(session.render(&project, None, &CancelToken::new()).unwrap().rendered, 6);
    // The same project, located through a relative path.
    let cwd = std::env::current_dir().unwrap();
    let relative = freesynth::paths::relative_to(&dir.join("shot.fsyn"), &cwd);
    let again = Project::load(&relative).unwrap();
    assert_eq!(session.render(&again, None, &CancelToken::new()).unwrap().rendered, 0);
    // A project folder moved as a whole keeps its cache.
    let moved = dir.with_file_name(format!("{}-moved", dir.file_name().unwrap().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&moved);
    std::fs::rename(&dir, &moved).unwrap();
    let there = Project::load(moved.join("shot.fsyn")).unwrap();
    let stats = session.render(&there, None, &CancelToken::new()).unwrap();
    assert_eq!(stats.rendered, 0, "{stats:?}");
    std::fs::remove_dir_all(moved).unwrap();
}

#[test]
fn projects_share_a_cache_folder() {
    let (dir, project) = project("shared-cache");
    let mut other = project.clone();
    other.output.frames = "out2".into();
    other.set_name("other");
    let mut session = Session::new(Backend::Cpu).unwrap();
    let mut options = SessionOptions::default();
    options.cache_dir = Some(dir.join("shared"));
    session.set_options(options);
    let cancel = CancelToken::new();
    for expected in [6, 0] {
        for p in [&project, &other] {
            assert_eq!(session.render(p, None, &cancel).unwrap().rendered, expected);
        }
    }
    assert_eq!(std::fs::read_dir(dir.join("shared")).unwrap().count(), 2);
    std::fs::remove_dir_all(dir).unwrap();
}
