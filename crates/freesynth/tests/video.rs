//! Video files in and out, through the system ffmpeg, and importing `.ebs` projects. The
//! ffmpeg tests are skipped when ffmpeg is not installed.

use std::path::{Path, PathBuf};

use freesynth::ebs::import_ebs;
use freesynth::{Backend, CancelToken, Codec, ColourTags, Event, Project, Session};
use freesynth_core::io;
use freesynth_media::{EncodeSettings, Ffmpeg, Rational, VideoEncoder, encode_files, probe};
use freesynth_scenes::{SceneParams, write_scene};

fn ffmpeg() -> Option<Ffmpeg> {
    Ffmpeg::locate().map_err(|e| eprintln!("skipped: {e}")).ok()
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("freesynth-video-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const W: usize = 64;
const H: usize = 48;
const FRAMES: usize = 13;
const KEYS: [usize; 3] = [0, 6, 12];

/// A generated scene in `dir/scene`, and its frames encoded into `dir/input.mov` with the given
/// colour tags.
fn scene_with_video(dir: &Path, ff: &Ffmpeg, tags: &ColourTags) -> PathBuf {
    let params = SceneParams {
        width: W,
        height: H,
        frames: FRAMES,
        seed: 3,
        keyframes: KEYS.to_vec(),
        ..SceneParams::default()
    };
    write_scene(&params, "video", &dir.join("scene")).unwrap();
    let path = dir.join("input.mov");
    let settings = EncodeSettings {
        codec: Codec::ProRes4444,
        frame_rate: Rational::new(24, 1),
        colour: tags.clone(),
        ..EncodeSettings::default()
    };
    let mut enc = VideoEncoder::create(ff, &path, W, H, &settings).unwrap();
    for t in 0..FRAMES {
        enc.write(&io::load_rgba(dir.join(format!("scene/frames/{t:05}.png"))).unwrap()).unwrap();
    }
    enc.finish().unwrap();
    path
}

fn video_project(dir: &Path) -> Project {
    let mut project = Project::new("input.mov", "out.mov");
    for k in KEYS {
        project.add_keyframe(k, format!("scene/keys/{k:05}.png"));
    }
    project.output.frames = "png/[####].png".into();
    project.engine.backend = Backend::Cpu;
    let path = dir.join("shot.fsyn");
    project.save(&path).unwrap();
    Project::load(&path).unwrap()
}

fn bump_red(path: &Path, by: u8) {
    let img = io::load_rgba(path).unwrap();
    let mut data = img.data().to_vec();
    data.iter_mut().step_by(4).for_each(|r| *r = r.wrapping_add(by));
    io::save_png(&freesynth_core::Image::from_raw(img.width(), img.height(), 4, data), path).unwrap();
}

#[test]
fn mov_in_mov_out_with_colour_tags() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("mov");
    // SD-style tags, which differ from the HD defaults: they must survive the round trip.
    let tags = ColourTags::from_probe(Some("smpte170m"), Some("tv"), Some("smpte170m"), Some("bt709"));
    scene_with_video(&dir, &ff, &tags);
    let project = video_project(&dir);
    let session = Session::new(Backend::Cpu).unwrap();
    let stats = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((stats.frames, stats.rendered), (FRAMES, FRAMES));
    assert_eq!(stats.video.as_deref(), Some(dir.join("out.mov").as_path()));

    let out = probe(&ff, &dir.join("out.mov")).unwrap();
    assert_eq!((out.width, out.height, out.frame_count), (W, H, FRAMES));
    assert_eq!(out.codec, "prores");
    assert_eq!(out.frame_rate, Rational::new(24, 1));
    assert_eq!(out.colour, tags, "the output carries the input's colour tags");
    // The video holds the rendered frames: ProRes is lossy, but decoding with the output's tags
    // gives back the frames without a colour shift (a wrong matrix shifts channels by several
    // levels).
    let frames: Vec<_> = freesynth_media::VideoDecoder::open(&ff, &dir.join("out.mov"), &out, &out.decode_colour())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(frames.len(), FRAMES);
    let (mut abs, mut bias, mut n) = (0.0, [0.0f64; 3], 0usize);
    for (t, f) in frames.iter().enumerate() {
        let png = io::load_rgba(dir.join(format!("png/{t:04}.png"))).unwrap();
        for (a, b) in f.data().as_chunks::<4>().0.iter().zip(png.data().as_chunks::<4>().0) {
            for c in 0..3 {
                let d = f64::from(a[c]) - f64::from(b[c]);
                abs += d.abs();
                bias[c] += d;
            }
            n += 1;
        }
    }
    let abs = abs / (3 * n) as f64;
    let bias = bias.map(|b| b / n as f64);
    eprintln!("decoded output vs frames: mean abs {abs:.2}, bias {bias:?}");
    assert!(abs < 4.0, "mean difference {abs}");
    assert!(bias.iter().all(|b| b.abs() < 0.75), "colour shift {bias:?}");

    // The decode ran alongside the render and completed the frame cache.
    let decoded = decoded_dirs(&project);
    assert_eq!(decoded.len(), 1);
    assert!(decoded[0].join("index.json").is_file());
    assert_eq!(std::fs::read_dir(&decoded[0]).unwrap().count(), FRAMES + 1);

    // Nothing changed: nothing renders, nothing is decoded again and the video is kept.
    let statuses = std::sync::Mutex::new(Vec::new());
    let record = |e: &Event| {
        if let Event::Status(s) = e {
            statuses.lock().unwrap().push(s.clone());
        }
    };
    let again = session.render(&project, Some(&record), &CancelToken::new()).unwrap();
    assert_eq!((again.rendered, again.video_reused), (0, true));
    let statuses = statuses.into_inner().unwrap();
    assert!(!statuses.iter().any(|s| s.starts_with("decoding")), "{statuses:?}");

    // Repainting a keyframe re-renders its reach and re-encodes the video.
    bump_red(&dir.join("scene/keys/00012.png"), 50);
    let edit = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((edit.rendered, edit.runs.clone(), edit.video_reused), (6, vec![(6, 12)], false));
    assert_eq!(probe(&ff, &dir.join("out.mov")).unwrap().frame_count, FRAMES);

    // The decoded input is cached and reused.
    assert_eq!(decoded_dirs(&project), decoded);
    std::fs::remove_dir_all(dir).unwrap();
}

/// The frame cache directories of a project's decoded video inputs.
fn decoded_dirs(project: &Project) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(project.default_cache_dir().join("decoded"))
        .map(|entries| entries.map(|e| e.unwrap().path()).collect())
        .unwrap_or_default();
    dirs.sort();
    dirs
}

#[test]
fn a_cancelled_first_render_leaves_no_partial_decode() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("cancel-decode");
    scene_with_video(&dir, &ff, &ColourTags::bt709());
    let project = video_project(&dir);
    let session = Session::new(Backend::Cpu).unwrap();
    let cancel = CancelToken::new();
    cancel.cancel();
    let err = session.render(&project, None, &cancel).unwrap_err();
    assert!(err.is_cancelled(), "{err}");
    assert!(decoded_dirs(&project).is_empty(), "the unfinished frame cache is removed");
    // The next render decodes and renders everything.
    let stats = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!(stats.rendered, FRAMES);
    assert!(decoded_dirs(&project)[0].join("index.json").is_file());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn video_only_output_in_16_bits() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("deep");
    scene_with_video(&dir, &ff, &ColourTags::default());
    let mut project = video_project(&dir);
    project.output.frames.clear();
    project.output.video = Some("deep.mkv".into());
    project.output.bit_depth = 16;
    let session = Session::new(Backend::Cpu).unwrap();
    session.render(&project, None, &CancelToken::new()).unwrap();
    let out = probe(&ff, &dir.join("deep.mkv")).unwrap();
    assert_eq!(out.codec, "ffv1");
    assert!(out.pix_fmt.contains("16"), "{}", out.pix_fmt);
    // Untagged 64x48 input is decoded as BT.601 and the output says so.
    assert_eq!(out.colour.primaries.as_deref(), Some("smpte170m"));
    // Without an output sequence the frames live in the cache, at 16 bits.
    let cached = project.default_cache_dir().join("frames").join("00003.png");
    let (_, _, data) = freesynth_media::load_rgba16(&cached).unwrap();
    assert!(data.iter().any(|v| v % 257 == 0 && *v != 0));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn variable_frame_rate_input_keeps_its_timing() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("vfr");
    let params = SceneParams { width: W, height: H, frames: 6, seed: 5, keyframes: vec![0], ..SceneParams::default() };
    write_scene(&params, "vfr", &dir.join("scene")).unwrap();
    let durations = [0.04, 0.08, 0.04, 0.12, 0.04, 0.04];
    let files: Vec<(PathBuf, f64)> =
        durations.iter().enumerate().map(|(t, &d)| (dir.join(format!("scene/frames/{t:05}.png")), d)).collect();
    let settings = EncodeSettings { codec: Codec::Png, ..EncodeSettings::default() };
    encode_files(&ff, &files, &dir.join("vfr.mov"), W, H, &settings, &dir).unwrap();
    let mut project = Project::new("vfr.mov", "out.mov");
    project.add_keyframe(0, "scene/keys/00000.png");
    project.engine.backend = Backend::Cpu;
    project.set_base_dir(&dir);
    let session = Session::new(Backend::Cpu).unwrap();
    session.render(&project, None, &CancelToken::new()).unwrap();
    let out = probe(&ff, &dir.join("out.mov")).unwrap();
    assert_eq!(out.frame_count, 6);
    for (i, d) in durations.iter().enumerate().take(5) {
        assert!((out.frame_duration(i) - d).abs() < 2e-3, "frame {i}: {} s, expected {d}", out.frame_duration(i));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// Writes an `.ebs` file for the scene: patterns relative to `dir`, keyframes laid out the way
/// EbSynth does (each covers up to its neighbours).
fn write_ebs(path: &Path, keys: &[usize], last: usize) {
    let mut out = b"EBS05\0".to_vec();
    let s = |out: &mut Vec<u8>, t: &str| {
        out.extend((t.len() as i32).to_le_bytes());
        out.extend(t.as_bytes());
    };
    s(&mut out, "scene\\frames\\[#####].png");
    s(&mut out, "scene\\mask\\[#####].png");
    s(&mut out, "scene\\keys\\[#####].png");
    out.push(0);
    for w in [1.0f32, 4.0, 1.0, 10.0, 1.0, 3500.0] {
        out.extend(w.to_le_bytes());
    }
    out.extend((keys.len() as i32).to_le_bytes());
    for (i, &k) in keys.iter().enumerate() {
        let back = if i == 0 { (k > 0).then_some(0) } else { Some(keys[i - 1]) };
        let fwd = if i + 1 == keys.len() { (k < last).then_some(last) } else { Some(keys[i + 1]) };
        out.extend((k as i32).to_le_bytes());
        out.push(u8::from(back.is_some()));
        out.push(u8::from(fwd.is_some()));
        out.extend((back.unwrap_or(1) as i32).to_le_bytes());
        out.extend((fwd.unwrap_or(1) as i32).to_le_bytes());
        s(&mut out, &format!("out_{k:05}\\[#####].png"));
    }
    out.extend(b"V02\0");
    out.extend(2i32.to_le_bytes());
    out.push(0);
    out.extend(24f32.to_le_bytes());
    out.extend(704i32.to_le_bytes());
    std::fs::write(path, out).unwrap();
}

#[test]
fn imported_ebs_project_renders() {
    let dir = scratch("ebs");
    let params = SceneParams {
        width: W,
        height: H,
        frames: FRAMES,
        seed: 4,
        keyframes: KEYS.to_vec(),
        ..SceneParams::default()
    };
    write_scene(&params, "ebs", &dir.join("scene")).unwrap();
    write_ebs(&dir.join("shot.ebs"), &KEYS, FRAMES - 1);
    let import = import_ebs(&dir.join("shot.ebs"), &dir.join("shot.fsyn")).unwrap();
    assert!(import.output_warnings.iter().any(|w| w.contains("output folders")), "{:?}", import.output_warnings);
    let project = import.project;
    assert_eq!(project.media.frames, "scene/frames/[#####].png");
    assert_eq!(project.keyframes.len(), 3);
    assert_eq!((project.media.first, project.media.last, project.media.fps), (Some(0), Some(12), Some(24.0)));
    assert_eq!(project.engine.backend, Backend::Cpu, "the file said: no GPU");
    project.save(dir.join("shot.fsyn")).unwrap();
    let project = Project::load(dir.join("shot.fsyn")).unwrap();
    let session = Session::new(Backend::Cpu).unwrap();
    let stats = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!(stats.rendered, FRAMES);
    assert!(dir.join("shot-out/00012.png").is_file());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn eight_bit_png_frames_from_a_video() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("png8");
    scene_with_video(&dir, &ff, &ColourTags::bt709());
    let mut project = video_project(&dir);
    project.output.video = None;
    let session = Session::new(Backend::Cpu).unwrap();
    session.render(&project, None, &CancelToken::new()).unwrap();
    let raw = std::fs::read(dir.join("png/0005.png")).unwrap();
    assert_eq!(raw[24], 8, "8-bit PNG");
    std::fs::remove_dir_all(dir).unwrap();
}

/// Largest and mean channel difference of two RGBA images.
fn diff(a: &freesynth_core::Image, b: &freesynth_core::Image) -> (u8, f64) {
    let (a, b) = (freesynth_media::to_rgba8(a), freesynth_media::to_rgba8(b));
    let worst = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    let mean = a.iter().zip(&b).map(|(x, y)| f64::from(x.abs_diff(*y))).sum::<f64>() / a.len() as f64;
    (worst, mean)
}

/// With JPEG output frames, a lossless video is still encoded from lossless frames (PNG copies
/// in the cache), not from the JPEG files.
#[test]
fn jpeg_frames_with_a_lossless_video() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("jpeg-master");
    scene_with_video(&dir, &ff, &ColourTags::bt709());
    let mut project = video_project(&dir);
    project.output.frames = "jpg/[####].jpg".into();
    project.output.video = Some("master.mkv".into());
    let session = Session::new(Backend::Cpu).unwrap();
    session.render(&project, None, &CancelToken::new()).unwrap();
    let out = probe(&ff, &dir.join("master.mkv")).unwrap();
    assert_eq!((out.codec.as_str(), out.frame_count), ("ffv1", FRAMES));
    let decoded: Vec<_> = freesynth_media::VideoDecoder::open(&ff, &dir.join("master.mkv"), &out, &out.decode_colour())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let masters = project.default_cache_dir().join("frames");
    let (mut worst_master, mut mean_jpeg) = (0u8, 0.0);
    for (t, frame) in decoded.iter().enumerate() {
        let master = io::load_rgba(masters.join(format!("{t:05}.png"))).unwrap();
        let jpeg = io::load_rgba(dir.join(format!("jpg/{t:04}.jpg"))).unwrap();
        worst_master = worst_master.max(diff(frame, &master).0);
        mean_jpeg += diff(frame, &jpeg).1 / FRAMES as f64;
    }
    assert_eq!(worst_master, 0, "FFV1 holds the rendered frames exactly");
    assert!(mean_jpeg > 0.2, "the JPEG files differ from the video ({mean_jpeg})");

    // Up to date, until a lossless copy goes missing: then its frame is rendered again.
    let again = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((again.rendered, again.video_reused), (0, true));
    std::fs::remove_file(masters.join("00009.png")).unwrap();
    let fixed = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!((fixed.rendered, fixed.video_reused), (1, false));
    std::fs::remove_dir_all(dir).unwrap();
}

/// Only the frames of the project's range are decoded, and a wider range decodes again.
#[test]
fn decodes_only_the_frames_it_renders() {
    let Some(ff) = ffmpeg() else { return };
    let dir = scratch("decode-range");
    scene_with_video(&dir, &ff, &ColourTags::bt709());
    let mut project = video_project(&dir);
    project.keyframes.retain(|k| k.frame >= 6);
    (project.media.first, project.media.last) = (Some(7), Some(12));
    project.output.video = None;
    let session = Session::new(Backend::Cpu).unwrap();
    let stats = session.render(&project, None, &CancelToken::new());
    assert!(stats.unwrap_err().to_string().contains("keyframe 6 is outside the frame range 7-12"));
    project.media.first = Some(6);
    let stats = session.render(&project, None, &CancelToken::new()).unwrap();
    assert_eq!(stats.rendered, 7);
    assert!(stats.seconds >= stats.prepare_seconds + stats.render_seconds);
    let files = |d: &Path| -> Vec<String> {
        let mut names: Vec<String> =
            std::fs::read_dir(d).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        names
    };
    let decoded = decoded_dirs(&project);
    let expected: Vec<String> =
        (6..=12).map(|i| format!("{i:06}.png")).chain(std::iter::once("index.json".to_owned())).collect();
    assert_eq!(files(&decoded[0]), expected);
    // The rendered frames are the same as from a whole decode (frame numbers are exact).
    let narrow = io::load_rgba(dir.join("png/0009.png")).unwrap();
    // A narrower range reuses the cache; a wider one decodes again.
    project.media.first = Some(8);
    project.keyframes.retain(|k| k.frame == 12);
    let statuses = std::sync::Mutex::new(Vec::new());
    let record = |e: &Event| {
        if let Event::Status(s) = e {
            statuses.lock().unwrap().push(s.clone());
        }
    };
    session.render(&project, Some(&record), &CancelToken::new()).unwrap();
    assert!(!statuses.lock().unwrap().iter().any(|s| s.starts_with("decoding")));
    let mut whole = video_project(&dir);
    whole.output.video = None;
    whole.output.frames = "whole/[####].png".into();
    session.render(&whole, Some(&record), &CancelToken::new()).unwrap();
    assert!(statuses.lock().unwrap().iter().any(|s| s.starts_with("decoding")));
    assert_eq!(files(&decoded_dirs(&whole)[0]).len(), FRAMES + 1);
    // Frame 9 lies between keyframes 6 and 12 in both projects: the same render, from the
    // same input frames, whichever part of the video was decoded.
    let from_whole = io::load_rgba(dir.join("whole/0009.png")).unwrap();
    let (_, mean) = diff(&narrow, &from_whole);
    assert!(mean < 0.5, "mean difference {mean}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// An output video that cannot be written fails before anything is rendered.
#[test]
fn an_unwritable_video_folder_fails_before_rendering() {
    let dir = scratch("blocker");
    let params = SceneParams { width: W, height: H, frames: 3, seed: 5, keyframes: vec![0], ..SceneParams::default() };
    write_scene(&params, "blocker", &dir.join("scene")).unwrap();
    std::fs::write(dir.join("blocker"), b"a file").unwrap();
    let mut project = Project::new("scene/frames", "blocker/x.mkv");
    project.output.frames = "out".into();
    project.add_keyframe(0, "scene/keys/00000.png");
    project.engine.backend = Backend::Cpu;
    project.set_base_dir(&dir);
    if Ffmpeg::locate().is_err() {
        eprintln!("skipped: no ffmpeg");
        std::fs::remove_dir_all(dir).unwrap();
        return;
    }
    let session = Session::new(Backend::Cpu).unwrap();
    let err = session.render(&project, None, &CancelToken::new()).unwrap_err().to_string();
    assert!(err.contains("cannot create the folder of output.video"), "{err}");
    assert_eq!(std::fs::read_dir(dir.join("out")).map(|d| d.count()).unwrap_or(0), 0, "nothing was rendered");
    std::fs::remove_dir_all(dir).unwrap();
}
