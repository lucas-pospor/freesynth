//! End-to-end checks of the video commands: `new`, `render` (incremental, JSON progress,
//! `--watch`), `video`, `import-ebs` and `devices`, plus the dispatch between them and the
//! ebsynth-compatible mode. Tests that need ffmpeg are skipped without it.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use freesynth_core::io;
use freesynth_scenes::{SceneParams, write_scene};
use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_freesynth");

struct Scratch(PathBuf);

impl Scratch {
    /// A directory with a generated 64x48 scene of `frames` frames and keyframes `keys`.
    fn new(name: &str, frames: usize, keys: &[usize]) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("freesynth-cli-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let params =
            SceneParams { width: 64, height: 48, frames, seed: 7, keyframes: keys.to_vec(), ..SceneParams::default() };
        write_scene(&params, name, &dir.join("scene")).unwrap();
        Scratch(dir)
    }

    fn path(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }

    /// Runs the binary in the scratch directory, with the user's cache directory inside it.
    fn run(&self, args: &[&str]) -> Output {
        self.command(BIN).args(args).output().expect("the binary runs")
    }

    /// `program` set up to run in the scratch directory, with the user's cache directory
    /// inside it (one-shot renders keep their temporary cache there).
    fn command(&self, program: &str) -> Command {
        let mut c = Command::new(program);
        c.current_dir(&self.0).env("XDG_CACHE_HOME", self.0.join("user-cache"));
        c
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// The JSON events printed on stdout.
fn events(out: &Output) -> Vec<Value> {
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON ({e}): {l}")))
        .collect()
}

fn done(events: &[Value]) -> &Value {
    events.iter().rev().find(|e| e["event"] == "done").expect("a done event")
}

fn has_ffmpeg() -> bool {
    let ok = freesynth_media::Ffmpeg::locate().is_ok();
    if !ok {
        eprintln!("skipped: no ffmpeg");
    }
    ok
}

fn repaint(path: &Path, by: u8) {
    let img = io::load_rgba(path).unwrap();
    let mut data = img.data().to_vec();
    data.iter_mut().step_by(4).for_each(|r| *r = r.wrapping_add(by));
    io::save_png(&freesynth_core::Image::from_raw(img.width(), img.height(), 4, data), path).unwrap();
}

#[test]
fn new_then_render_incrementally() {
    let s = Scratch::new("new", 9, &[0, 4, 8]);
    let out = s.run(&["new", "shot", "--input", "scene/frames", "--keys", "scene/keys", "--preset", "fast"]);
    assert!(out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("3 keyframes: 0, 4, 8"), "{}", text(&out));
    let project = std::fs::read_to_string(s.path("shot.fsyn")).unwrap();
    assert!(project.contains("frames = \"scene/frames\"") && project.contains("image = \"scene/keys/00004.png\""));
    assert!(project.contains("frames = \"shot-out/[#####].png\""), "{project}");
    // New projects get the recommended look, written out so that it can be edited.
    assert!(project.contains("blend = \"adaptive\"") && project.contains("gate_mapping = true"), "{project}");
    // Refuses to overwrite.
    let again = s.run(&["new", "shot.fsyn", "--input", "scene/frames", "--keys", "scene/keys"]);
    assert_eq!(again.status.code(), Some(2));

    let first = s.run(&["render", "shot.fsyn", "--backend", "cpu", "--progress", "json"]);
    assert!(first.status.success(), "{}", text(&first));
    let ev = events(&first);
    assert_eq!(ev[0]["event"], "device");
    let planned = ev.iter().find(|e| e["event"] == "planned").unwrap();
    assert_eq!((planned["frames"].as_u64(), planned["render"].as_u64()), (Some(9), Some(9)));
    assert_eq!(ev.iter().filter(|e| e["event"] == "written").count(), 9);
    assert_eq!(done(&ev)["rendered"], 9);
    assert!(s.path("shot-out/00008.png").is_file());

    let second = s.run(&["render", "shot.fsyn", "--backend", "cpu", "--progress", "json"]);
    assert_eq!(done(&events(&second))["rendered"], 0);

    repaint(&s.path("scene/keys/00008.png"), 60);
    let third = s.run(&["render", "shot.fsyn", "--backend", "cpu", "--progress", "json"]);
    let ev = events(&third);
    assert_eq!(done(&ev)["rendered"], 4, "frames 5-8");
    assert_eq!(done(&ev)["runs"], serde_json::json!([[4, 8]]));

    let forced = s.run(&["render", "shot.fsyn", "--backend", "cpu", "--progress", "none", "--keyframe", "4"]);
    assert!(forced.status.success(), "{}", text(&forced));
    let human = s.run(&["render", "shot.fsyn", "--backend", "cpu", "--frames", "2-3", "--force"]);
    assert!(text(&human).contains("rendering 2 of 9 frames"), "{}", text(&human));
}

#[test]
fn look_options_reach_the_render() {
    let s = Scratch::new("look", 7, &[0, 6]);
    let args = ["new", "look", "--input", "scene/frames", "--keys", "scene/keys", "--preset", "fast"];
    let out = s.run(&[&args[..], &["--blend", "linear", "--gate-temporal", "--warm-start"]].concat());
    assert!(out.status.success(), "{}", text(&out));
    let project = std::fs::read_to_string(s.path("look.fsyn")).unwrap();
    for field in ["blend = \"linear\"", "gate_temporal = true", "warm_start = true"] {
        assert!(project.contains(field), "{field} in {project}");
    }
    let render = s.run(&["render", "look.fsyn", "--backend", "cpu", "--progress", "json"]);
    assert!(render.status.success(), "{}", text(&render));
    assert_eq!(done(&events(&render))["rendered"], 7);
    let bad = s.run(&[&args[..], &["--blend", "poisson"]].concat());
    assert_eq!(bad.status.code(), Some(1), "{}", text(&bad));
}

#[test]
fn render_errors_have_exit_codes() {
    let s = Scratch::new("errors", 3, &[0]);
    let missing = s.run(&["render", "nothing.fsyn"]);
    assert_eq!(missing.status.code(), Some(2));
    let json = s.run(&["render", "nothing.fsyn", "--progress", "json"]);
    let ev = events(&json);
    assert_eq!((ev[0]["event"].as_str(), ev[0]["code"].as_u64()), (Some("error"), Some(2)));
    let usage = s.run(&["render"]);
    assert_eq!(usage.status.code(), Some(1));
    let help = s.run(&["video", "--help"]);
    assert!(help.status.success() && text(&help).contains("--key <N=IMAGE>"), "{}", text(&help));
    // A video input without ffmpeg names the way out.
    std::fs::write(
        s.path("p.fsyn"),
        "format_version = 1\nmedia.frames = 'in.mov'\noutput.frames = 'o'\n\
         [[keyframe]]\nframe = 0\nimage = 'scene/keys/00000.png'\n",
    )
    .unwrap();
    let out = s
        .command(BIN)
        .args(["render", "p.fsyn", "--backend", "cpu"])
        .env("FREESYNTH_FFMPEG", s.path("no-such-ffmpeg"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(text(&out).contains("FREESYNTH_FFMPEG"), "{}", text(&out));
}

#[test]
fn dispatch_keeps_the_ebsynth_mode() {
    let s = Scratch::new("dispatch", 2, &[0]);
    let usage = s.run(&[]);
    assert_eq!(usage.status.code(), Some(1));
    assert!(text(&usage).contains("usage:") && text(&usage).contains("-style <style.png>"));
    let help = s.run(&["--help"]);
    assert!(help.status.success() && text(&help).contains("import-ebs"), "{}", text(&help));
    let version = s.run(&["--version"]);
    assert!(text(&version).contains(env!("CARGO_PKG_VERSION")));
    // An unknown option still gets the original error.
    let legacy = s.run(&["-bogus"]);
    assert_eq!(legacy.status.code(), Some(1));
    let out = s.path("o.png");
    let single = s.run(&[
        "-style",
        "scene/keys/00000.png",
        "-guide",
        "scene/frames/00000.png",
        "scene/frames/00001.png",
        "-output",
        out.to_str().unwrap(),
        "-backend",
        "cpu",
    ]);
    assert!(single.status.success(), "{}", text(&single));
    assert!(out.is_file());
}

#[test]
fn devices_lists_cpu_and_ffmpeg() {
    let out = Command::new(BIN).args(["devices", "--json"]).output().unwrap();
    assert!(out.status.success());
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["cpu_threads"].as_u64().unwrap() >= 1);
    assert!(v["gpus"].is_array());
    assert!(v["ffmpeg"].is_object());
}

#[test]
fn one_shot_video_command() {
    let s = Scratch::new("oneshot", 6, &[0, 5]);
    let out = s.run(&[
        "video",
        "-i",
        "scene/frames/[#####].png",
        "--key",
        "0=scene/keys/00000.png",
        "--key",
        "5=scene/keys/00005.png",
        "-o",
        "out",
        "--backend",
        "cpu",
        "--save-project",
        "saved/one.fsyn",
        "--progress",
        "json",
    ]);
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(done(&events(&out))["rendered"], 6);
    assert!(s.path("out/00005.png").is_file());
    // The saved project points at the same files from its own directory.
    let saved = std::fs::read_to_string(s.path("saved/one.fsyn")).unwrap();
    assert!(saved.contains("frames = \"../scene/frames/[#####].png\""), "{saved}");
    assert!(!s.path(".freesynth-cache").exists(), "one-shot renders leave no cache behind");
    let user_cache: Vec<_> = std::fs::read_dir(s.path("user-cache/freesynth"))
        .map(|d| d.flatten().map(|e| e.file_name()).collect())
        .unwrap_or_default();
    assert!(user_cache.is_empty(), "{user_cache:?}");
}

/// File names that ffmpeg would read as a protocol (`10:30.mov`) or an option (`-x.mov`) are
/// plain files. Skipped without ffmpeg.
#[test]
fn video_names_with_colons_and_dashes() {
    if !has_ffmpeg() {
        return;
    }
    let s = Scratch::new("names", 3, &[0, 2]);
    let args = ["video", "-i", "scene/frames", "--keys", "scene/keys", "--backend", "cpu", "-o", "10:30.mkv"];
    let out = s.run(&args);
    assert!(out.status.success(), "{}", text(&out));
    assert!(s.path("10:30.mkv").is_file());
    let out = s.run(&["video", "-i", "10:30.mkv", "--keys", "scene/keys", "--backend", "cpu", "--output=-x.mkv"]);
    assert!(out.status.success(), "{}", text(&out));
    assert!(s.path("-x.mkv").is_file());
}

/// Writes a three-keyframe `.ebs` project for the scene.
fn write_ebs(path: &Path) {
    let mut out = b"EBS05\0".to_vec();
    let s = |out: &mut Vec<u8>, t: &str| {
        out.extend((t.len() as i32).to_le_bytes());
        out.extend(t.as_bytes());
    };
    s(&mut out, "scene\\frames\\[#####].png");
    s(&mut out, "mask\\[#####].png");
    s(&mut out, "scene\\keys\\[#####].png");
    out.push(0);
    for w in [1.0f32, 4.0, 1.0, 10.0, 1.0, 3500.0] {
        out.extend(w.to_le_bytes());
    }
    let keys: [(i32, Option<i32>, Option<i32>); 3] = [(0, None, Some(4)), (4, Some(0), Some(8)), (8, Some(4), None)];
    out.extend((keys.len() as i32).to_le_bytes());
    for (k, back, fwd) in keys {
        out.extend(k.to_le_bytes());
        out.push(u8::from(back.is_some()));
        out.push(u8::from(fwd.is_some()));
        out.extend(back.unwrap_or(1).to_le_bytes());
        out.extend(fwd.unwrap_or(1).to_le_bytes());
        s(&mut out, &format!("out_{k:03}\\[###].png"));
    }
    std::fs::write(path, out).unwrap();
}

#[test]
fn import_ebs_then_render_mov_to_mov() {
    if !has_ffmpeg() {
        return;
    }
    let s = Scratch::new("ebs", 9, &[0, 4, 8]);
    write_ebs(&s.path("shot.ebs"));
    // The shot as a video file (BT.709-tagged ProRes), standing in for camera footage.
    let ff = freesynth_media::Ffmpeg::locate().unwrap();
    let settings = freesynth_media::EncodeSettings {
        frame_rate: freesynth_media::Rational::new(25, 1),
        colour: freesynth_media::ColourTags::bt709(),
        ..Default::default()
    };
    let mut enc = freesynth_media::VideoEncoder::create(&ff, &s.path("shot.mov"), 64, 48, &settings).unwrap();
    for t in 0..9 {
        enc.write(&io::load_rgba(s.path(&format!("scene/frames/{t:05}.png"))).unwrap()).unwrap();
    }
    enc.finish().unwrap();

    let out =
        s.run(&["import-ebs", "shot.ebs", "--input", "shot.mov", "--out", "styled.mov", "--out", "png/[###].png"]);
    assert!(out.status.success(), "{}", text(&out));
    // The outputs are given, so the note about the .ebs file's output folders is moot.
    assert!(!text(&out).contains("output folders"), "{}", text(&out));
    let project = std::fs::read_to_string(s.path("shot.fsyn")).unwrap();
    assert!(project.contains("frames = \"shot.mov\"") && project.contains("video = \"styled.mov\""), "{project}");

    let render = s.run(&["render", "shot.fsyn", "--backend", "cpu", "--progress", "json"]);
    assert!(render.status.success(), "{}", text(&render));
    let video = done(&events(&render))["video"].as_str().unwrap().to_owned();
    assert_eq!(Path::new(&video), s.path("styled.mov"));
    let info = freesynth_media::probe(&ff, &s.path("styled.mov")).unwrap();
    assert_eq!((info.frame_count, info.codec.as_str()), (9, "prores"));
    assert_eq!(info.colour, freesynth_media::ColourTags::bt709());
    assert!(s.path("png/008.png").is_file());
}

/// The md5-like fingerprint of a file's bytes, to see that it was not touched.
fn contents(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap()
}

#[test]
fn outputs_never_overwrite_inputs() {
    let s = Scratch::new("inplace", 4, &[0, 3]);
    let frame = contents(&s.path("scene/frames/00002.png"));
    let base = ["video", "-i", "scene/frames", "--keys", "scene/keys", "--backend", "cpu"];
    let same = s.run(&[&base[..], &["-o", "scene/frames"]].concat());
    assert_eq!(same.status.code(), Some(2), "{}", text(&same));
    assert!(text(&same).contains("would overwrite the input file"), "{}", text(&same));
    let keys = s.run(&[&base[..], &["-o", "scene/keys/[#####].png"]].concat());
    assert_eq!(keys.status.code(), Some(2), "{}", text(&keys));
    assert_eq!(contents(&s.path("scene/frames/00002.png")), frame);
    // A project whose output is its input fails before the device opens.
    let new = s.run(&["new", "p", "-i", "scene/frames", "--keys", "scene/keys", "-o", "scene/frames"]);
    assert!(new.status.success(), "{}", text(&new));
    let render = s.run(&["render", "p.fsyn", "--backend", "cpu"]);
    assert_eq!(render.status.code(), Some(2));
    assert!(!text(&render).contains("device:"), "{}", text(&render));
    if !has_ffmpeg() {
        return;
    }
    let made = s.run(&[&base[..], &["-o", "x.mov"]].concat());
    assert!(made.status.success(), "{}", text(&made));
    let mov = contents(&s.path("x.mov"));
    let over = s.run(&["video", "-i", "x.mov", "--keys", "scene/keys", "--backend", "cpu", "-o", "x.mov"]);
    assert_eq!(over.status.code(), Some(2), "{}", text(&over));
    assert!(text(&over).contains("x.mov' is an input"), "{}", text(&over));
    assert_eq!(contents(&s.path("x.mov")), mov);
}

#[test]
fn the_cache_does_not_depend_on_the_working_directory() {
    let s = Scratch::new("cwd", 4, &[0, 3]);
    let new = s.run(&["new", "shot", "-i", "scene/frames", "--keys", "scene/keys", "--preset", "fast"]);
    assert!(new.status.success(), "{}", text(&new));
    let render = |dir: &Path, project: &str| {
        let out = Command::new(BIN)
            .current_dir(dir)
            .args(["render", project, "--backend", "cpu", "--progress", "json"])
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", text(&out));
        done(&events(&out))["rendered"].as_u64().unwrap()
    };
    assert_eq!(render(&s.0, "shot.fsyn"), 4);
    let parent = s.0.parent().unwrap();
    let name = s.0.file_name().unwrap().to_str().unwrap();
    assert_eq!(render(parent, &format!("{name}/shot.fsyn")), 0, "from another directory");
    assert_eq!(render(Path::new("/"), s.path("shot.fsyn").to_str().unwrap()), 0, "through an absolute path");
    assert_eq!(render(&s.0, "shot"), 0, "the extension may be left out");
}

#[test]
fn mistakes_fail_early_with_their_exit_codes() {
    let s = Scratch::new("mistakes", 4, &[0, 3]);
    let write = |text: &str| {
        let body = format!(
            "format_version = 1\nmedia.frames = 'scene/frames'\noutput.frames = 'out'\n{text}\n\
             [[keyframe]]\nframe = 0\nimage = 'scene/keys/00000.png'\n"
        );
        std::fs::write(s.path("p.fsyn"), body).unwrap();
    };
    write("engine.patch_size = 4");
    let out = s.run(&["render", "p.fsyn", "--backend", "cpu"]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("patch_size must be odd") && text(&out).contains("line 4"), "{}", text(&out));
    write("media.last = 2\n[[keyframe]]\nframe = 3\nimage = 'scene/keys/00003.png'");
    let out = s.run(&["render", "p.fsyn", "--backend", "cpu"]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("keyframe 3 is outside the frame range 0-2"), "{}", text(&out));
    assert!(!text(&out).contains("device:"), "fails before the device opens: {}", text(&out));
    // A painting of the wrong size is named, with its size.
    let small = freesynth_core::Image::from_raw(10, 8, 4, vec![128; 10 * 8 * 4]);
    io::save_png(&small, s.path("small.png")).unwrap();
    let out = s.run(&["video", "-i", "scene/frames", "--key", "0=small.png", "-o", "o", "--backend", "cpu"]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("small.png') is 10x8") && text(&out).contains("64x48"), "{}", text(&out));
    // Usage mistakes exit with 1.
    let out = s.run(&["video", "-i", "scene/frames", "--key", "0=small.png", "-o", "a", "-o", "b"]);
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
}

#[test]
fn watch_re_renders_when_a_keyframe_changes() {
    let s = Scratch::new("watch", 9, &[0, 4, 8]);
    let new = s.run(&["new", "shot", "-i", "scene/frames", "--keys", "scene/keys", "--preset", "fast"]);
    assert!(new.status.success(), "{}", text(&new));
    let mut child = s
        .command(BIN)
        .args(["render", "shot.fsyn", "--watch", "--backend", "cpu", "--progress", "json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let (tx, rx) = mpsc::channel::<Value>();
    let stdout = child.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<Value>(&line)
                && tx.send(v).is_err()
            {
                return;
            }
        }
    });
    let wait_for = |event: &str| -> Value {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(v) if v["event"] == event => return v,
                Ok(_) => {}
                Err(e) => panic!("no '{event}' event: {e}"),
            }
        }
    };
    let first = wait_for("done");
    assert_eq!(first["rendered"], 9);
    wait_for("watch_waiting");
    repaint(&s.path("scene/keys/00000.png"), 70);
    let reason = wait_for("watch_render");
    assert!(reason["reason"].as_str().unwrap().contains("00000.png"), "{reason}");
    let second = wait_for("done");
    assert_eq!(second["rendered"], 4, "keyframe 0 reaches frames 0-3");
    assert_eq!(second["runs"], serde_json::json!([[0, 4]]));
    let _ = child.kill();
    let _ = child.wait();
}

/// A whole render, video in and out, with no network at all: in a new network namespace with
/// only a loopback device that is down. Skipped where unprivileged namespaces are unavailable.
#[cfg(target_os = "linux")]
#[test]
fn renders_without_network() {
    let probe = Command::new("unshare").args(["-rn", "true"]).output();
    if !probe.is_ok_and(|o| o.status.success()) {
        eprintln!("skipped: unshare -rn is not available");
        return;
    }
    let with_video = has_ffmpeg();
    let s = Scratch::new("offline", 6, &[0, 5]);
    let mut args = vec!["-rn", BIN, "video", "-i", "scene/frames", "--key", "0=scene/keys/00000.png"];
    args.extend(["--key", "5=scene/keys/00005.png", "-o", "out", "--backend", "cpu", "--progress", "json"]);
    if with_video {
        args.extend(["-o", "out.mkv"]);
    }
    let out = s.command("unshare").args(&args).output().unwrap();
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(done(&events(&out))["rendered"], 6);
    if with_video {
        assert!(s.path("out.mkv").is_file());
    }
}

/// The first fetched corpus clip (`corpus/fetch-clips.sh`), if any: its frames directory.
fn fetched_clip() -> Option<PathBuf> {
    let external = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/external");
    let mut clips: Vec<PathBuf> = std::fs::read_dir(external)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path().join("frames")))
        .filter(|p| p.join("00000.png").is_file())
        .collect();
    clips.sort();
    clips.into_iter().next()
}

/// A real clip, end to end: the fetched frames become a BT.709 ProRes .mov, two keyframes are
/// "painted" (posterized and tinted), an `.ebs` project is imported with the .mov as input and
/// rendered to a .mov and a 16-bit PNG sequence. Slow in debug builds, so ignored by default:
/// `cargo test --release -p freesynth-cli -- --ignored`. Skipped without a fetched clip or ffmpeg.
#[test]
#[ignore]
fn fetched_clip_mov_round_trip() {
    let Some(frames) = fetched_clip() else {
        eprintln!("skipped: no clip in corpus/external (run corpus/fetch-clips.sh)");
        return;
    };
    if !has_ffmpeg() {
        return;
    }
    let s = Scratch::new("clip", 1, &[0]);
    let count = std::fs::read_dir(&frames).unwrap().count().min(30);
    let last = count - 1;
    let ff = freesynth_media::Ffmpeg::locate().unwrap();
    let first = io::load_rgba(frames.join("00000.png")).unwrap();
    let settings = freesynth_media::EncodeSettings {
        frame_rate: freesynth_media::Rational::new(24, 1),
        colour: freesynth_media::ColourTags::bt709(),
        ..Default::default()
    };
    let mut enc =
        freesynth_media::VideoEncoder::create(&ff, &s.path("clip.mov"), first.width(), first.height(), &settings)
            .unwrap();
    for t in 0..count {
        enc.write(&io::load_rgba(frames.join(format!("{t:05}.png"))).unwrap()).unwrap();
    }
    enc.finish().unwrap();
    std::fs::create_dir_all(s.path("clipkeys")).unwrap();
    for t in [0, last] {
        let img = io::load_rgba(frames.join(format!("{t:05}.png"))).unwrap();
        let painted: Vec<u8> = img
            .data()
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 4 == 3 { v } else { (v / 64) * 64 + if i % 4 == 2 { 40 } else { 10 } })
            .collect();
        let painted = freesynth_core::Image::from_raw(img.width(), img.height(), 4, painted);
        io::save_png(&painted, s.path(&format!("clipkeys/{t:05}.png"))).unwrap();
    }
    let mut ebs = b"EBS05\0".to_vec();
    let put = |out: &mut Vec<u8>, t: &str| {
        out.extend((t.len() as i32).to_le_bytes());
        out.extend(t.as_bytes());
    };
    put(&mut ebs, "frames\\[#####].png");
    put(&mut ebs, "mask\\[#####].png");
    put(&mut ebs, "clipkeys\\[#####].png");
    ebs.push(0);
    for w in [1.0f32, 4.0, 1.0, 10.0, 1.0, 3500.0] {
        ebs.extend(w.to_le_bytes());
    }
    ebs.extend(2i32.to_le_bytes());
    for (k, back, fwd) in [(0i32, None, Some(last as i32)), (last as i32, Some(0i32), None)] {
        ebs.extend(k.to_le_bytes());
        ebs.push(u8::from(back.is_some()));
        ebs.push(u8::from(fwd.is_some()));
        ebs.extend(back.unwrap_or(1).to_le_bytes());
        ebs.extend(fwd.unwrap_or(1).to_le_bytes());
        put(&mut ebs, &format!("out_{k}\\[#####].png"));
    }
    std::fs::write(s.path("clip.ebs"), ebs).unwrap();
    let out = s.run(&["import-ebs", "clip.ebs", "--input", "clip.mov", "--out", "styled.mov", "--out", "png16"]);
    assert!(out.status.success(), "{}", text(&out));
    let project = std::fs::read_to_string(s.path("clip.fsyn")).unwrap().replace("bit_depth = 8", "bit_depth = 16");
    std::fs::write(s.path("clip.fsyn"), project).unwrap();
    let render = s.run(&["render", "clip.fsyn", "--progress", "json"]);
    assert!(render.status.success(), "{}", text(&render));
    assert_eq!(done(&events(&render))["rendered"], count);
    let info = freesynth_media::probe(&ff, &s.path("styled.mov")).unwrap();
    assert_eq!((info.frame_count, info.colour.clone()), (count, freesynth_media::ColourTags::bt709()));
    let png = std::fs::read(s.path(&format!("png16/{last:05}.png"))).unwrap();
    assert_eq!(png[24], 16, "16-bit PNG");
}
