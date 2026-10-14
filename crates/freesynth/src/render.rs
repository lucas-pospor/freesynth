//! Incremental project renders: plan against the cache, render the dirty segments, write the
//! frames and the video.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use freesynth_core::io;
use freesynth_media::BitDepth;
use freesynth_media::{EncodeSettings, Ffmpeg, Rational, is_video_path};
use freesynth_video::plan::Plan;
use freesynth_video::{
    CancelToken, FlowBackend, FrameSource, Keyframe, Keyframe16, Progress, Shot, VideoError, VideoParams,
};

use crate::cache::{CacheLock, FrameKind, Manifest, dirty_runs, frame_keys, keyframe_reach};
use crate::error::Error;
use crate::event::{Event, ProjectStats};
use crate::hashing::{FileStamp, Hash, Hasher};
use crate::input::{Input, Opener, describe, frame_range, image_sequence, probe_cached, with_decodes};
use crate::output::{OutputFrames, ProjectSink, Ready, VideoJob};
use crate::project::Project;
use crate::session::Session;

// Revision of how FreeSynth renders frames, part of every frame's cache key. It is owned by
// freesynth-video (so the bump lives next to the rendering code it tracks); the cache layout has
// its own version, MANIFEST_VERSION in the cache module.
use freesynth_video::OUTPUT_REVISION;

/// What to bring up to date in a project render. New fields may be added: build it from
/// [`RenderRequest::default`] and set the fields you need.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RenderRequest {
    /// Only frames in `first..=last` are rendered (the rest of the output is left as it is).
    /// `None`: the project's whole frame range.
    pub range: Option<(usize, usize)>,
    /// Render every frame again, ignoring the cache.
    pub force: bool,
    /// Render the reach of these keyframes again (their own frames and the frames up to the
    /// neighbouring keyframes), even when the cache says they are current.
    pub keyframes: Vec<usize>,
}

/// The inputs of a project, opened.
pub(crate) struct Prepared {
    pub frames: Input,
    pub mask: Option<Input>,
    /// Keyframe paintings sorted by frame, with their files.
    pub keyframes: Vec<(Keyframe, PathBuf)>,
    /// The paintings at 16 bits, for 16-bit output (the 8-bit ones are these rounded).
    pub keyframes16: Option<Vec<Keyframe16>>,
    /// The project's frame range.
    pub first: usize,
    pub last: usize,
    /// The frame size.
    pub size: (usize, usize),
}

impl Prepared {
    /// Every file the render reads: input frames (or video files), mask frames and paintings.
    fn input_files(&self) -> Vec<PathBuf> {
        let mut files = self.frames.files();
        if let Some(m) = &self.mask {
            files.extend(m.files());
        }
        files.extend(self.keyframes.iter().map(|(_, p)| p.clone()));
        files
    }
}

/// The project's frame range (`media.first` and `media.last`, by default the input's ends),
/// checked against the input's frames `input` and the keyframes.
fn check_range(
    project: &Project,
    input: (usize, usize),
    describe: &dyn Fn() -> String,
    keyframes: &[usize],
) -> Result<(usize, usize), Error> {
    let first = project.media.first.unwrap_or(input.0);
    let last = project.media.last.unwrap_or(input.1);
    if first < input.0 || last > input.1 {
        return Err(Error::Project(format!("frames {first}-{last} are not all in the input ({})", describe())));
    }
    if let Some(k) = keyframes.iter().find(|&&k| k < first || k > last) {
        return Err(Error::Project(format!("keyframe {k} is outside the frame range {first}-{last}")));
    }
    Ok((first, last))
}

/// Checks that the mask, holding frames `mask`, covers `first..=last`.
fn check_mask(mask: (usize, usize), first: usize, last: usize) -> Result<(), Error> {
    if mask.0 > first || mask.1 < last {
        return Err(Error::Project(format!(
            "the mask holds frames {}-{}, but the render needs frames {first}-{last}",
            mask.0, mask.1
        )));
    }
    Ok(())
}

/// Checks that every painting `(frame, file, size)` has the frame size.
fn check_keyframe_sizes(keyframes: &[(usize, &Path, (usize, usize))], size: (usize, usize)) -> Result<(), Error> {
    for &(frame, path, (w, h)) in keyframes {
        if (w, h) != size {
            return Err(Error::Project(format!(
                "keyframe {frame} ('{}') is {w}x{h}; it must have the frame size, {}x{}",
                path.display(),
                size.0,
                size.1
            )));
        }
    }
    Ok(())
}

/// The file `path` names, for comparing paths: symbolic links and `..` resolved where the
/// file (or its folder) exists.
fn file_identity(path: &Path) -> PathBuf {
    if let Ok(p) = std::fs::canonicalize(path) {
        return p;
    }
    match (path.parent(), path.file_name()) {
        (Some(dir), Some(name)) => std::fs::canonicalize(if dir.as_os_str().is_empty() { Path::new(".") } else { dir })
            .map(|d| d.join(name))
            .unwrap_or_else(|_| crate::paths::absolute(path)),
        _ => crate::paths::absolute(path),
    }
}

/// Refuses outputs that would overwrite an input: the output video must not be an input file,
/// and no output frame of `first..=last` may be one.
fn check_outputs(
    inputs: &[PathBuf],
    frames: Option<&OutputFrames>,
    video: Option<&Path>,
    first: usize,
    last: usize,
) -> Result<(), Error> {
    let inputs: HashSet<PathBuf> = inputs.iter().map(|p| file_identity(p)).collect();
    if let Some(v) = video
        && inputs.contains(&file_identity(v))
    {
        return Err(Error::Project(format!(
            "output.video '{}' is an input of the project; it would be overwritten (choose another file)",
            v.display()
        )));
    }
    if let Some(target) = frames {
        for t in first..=last {
            let path = target.path(t);
            if inputs.contains(&file_identity(&path)) {
                return Err(Error::Project(format!(
                    "output frame {t} would overwrite the input file '{}' (set output.frames to another folder)",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

/// Opens a project's inputs (decoding video inputs into the cache when needed).
pub(crate) fn prepare(project: &Project, opener: &mut Opener<'_>) -> Result<Prepared, Error> {
    let m = &project.media;
    let range = (m.first, m.last);
    let frames = opener.open(&project.resolve(&m.frames), "frames", m.start_number, m.colour.as_ref(), range)?;
    let mask =
        m.mask.as_ref().map(|p| opener.open(&project.resolve(p), "mask", m.start_number, None, range)).transpose()?;
    // 16-bit output: the final result of every synthesis is voted again from the paintings at
    // 16 bits, and the passes are blended at 16 bits.
    let deep = project.output.depth()? == BitDepth::Sixteen;
    let mut keyframes = Vec::new();
    let mut keyframes16 = Vec::new();
    for k in &project.keyframes {
        let path = project.resolve(&k.image);
        let load = |e| VideoError::Load { path: path.clone(), source: e };
        let style = if deep {
            let style16 = io::load_rgba16(&path).map_err(load)?;
            let style = style16.to_image();
            keyframes16.push(Keyframe16 { frame: k.frame, style: style16 });
            style
        } else {
            io::load_rgba(&path).map_err(load)?
        };
        keyframes.push((Keyframe { frame: k.frame, style }, path));
    }
    keyframes.sort_by_key(|(k, _)| k.frame);
    let keyframe_frames: Vec<usize> = keyframes.iter().map(|(k, _)| k.frame).collect();
    let (first, last) = check_range(project, frame_range(&frames), &|| describe(&frames), &keyframe_frames)?;
    if let Some(mask) = &mask {
        check_mask(frame_range(mask), first, last)?;
    }
    let size = match &frames.video {
        Some(info) => (info.width, info.height),
        None => frames.seq.load(first)?.size(),
    };
    let sizes: Vec<(usize, &Path, (usize, usize))> =
        keyframes.iter().map(|(k, p)| (k.frame, p.as_path(), k.style.size())).collect();
    check_keyframe_sizes(&sizes, size)?;
    let keyframes16 = deep.then_some(keyframes16);
    Ok(Prepared { frames, mask, keyframes, keyframes16, first, last, size })
}

/// An input found without reading its frames (for [`preflight`]).
struct Located {
    /// The frame numbers it holds.
    range: (usize, usize),
    /// Its image files, or its video file.
    files: Vec<PathBuf>,
    /// The frame size.
    size: (usize, usize),
}

impl Located {
    fn find(path: &Path, start_number: usize, ffmpeg: &dyn Fn() -> Result<Ffmpeg, Error>) -> Result<Self, Error> {
        if is_video_path(path) {
            let info = probe_cached(&ffmpeg()?, path)?;
            let range = (start_number, start_number + info.frame_count - 1);
            return Ok(Self { range, files: vec![path.to_owned()], size: (info.width, info.height) });
        }
        let seq = image_sequence(path)?;
        let files: Vec<PathBuf> =
            (seq.first()..=seq.last()).filter_map(|t| seq.path(t)).map(Path::to_path_buf).collect();
        Ok(Self { range: (seq.first(), seq.last()), files, size: seq.load(seq.first())?.size() })
    }

    fn describe(&self) -> String {
        format!("{} frames ({}-{})", self.range.1 - self.range.0 + 1, self.range.0, self.range.1)
    }
}

/// Checks a project's inputs and outputs without rendering, decoding or writing anything:
/// [`Project::check_inputs`].
pub(crate) fn preflight(project: &Project) -> Result<(), Error> {
    let project = &project.absolute();
    project.validate()?;
    let m = &project.media;
    let located = std::sync::OnceLock::new();
    let ffmpeg = || {
        located
            .get_or_init(|| Ffmpeg::locate().map_err(|e| e.to_string()))
            .clone()
            .map_err(|msg| Error::Media(freesynth_media::MediaError::FfmpegMissing(msg)))
    };
    let frames = Located::find(&project.resolve(&m.frames), m.start_number, &ffmpeg)?;
    let mask = m.mask.as_ref().map(|p| Located::find(&project.resolve(p), m.start_number, &ffmpeg)).transpose()?;
    let mut keyframes = Vec::new();
    for k in &project.keyframes {
        let path = project.resolve(&k.image);
        let style = io::load_rgba(&path).map_err(|source| VideoError::Load { path: path.clone(), source })?;
        keyframes.push((k.frame, path, style.size()));
    }
    let keyframe_frames: Vec<usize> = keyframes.iter().map(|k| k.0).collect();
    let (first, last) = check_range(project, frames.range, &|| frames.describe(), &keyframe_frames)?;
    if let Some(mask) = &mask {
        check_mask(mask.range, first, last)?;
    }
    let sizes: Vec<(usize, &Path, (usize, usize))> = keyframes.iter().map(|(f, p, s)| (*f, p.as_path(), *s)).collect();
    check_keyframe_sizes(&sizes, frames.size)?;
    let mut inputs = frames.files;
    inputs.extend(mask.into_iter().flat_map(|m| m.files));
    inputs.extend(keyframes.into_iter().map(|k| k.1));
    let o = &project.output;
    let target =
        if o.frames.is_empty() { None } else { Some(OutputFrames::parse(&project.resolve(&o.frames), o.depth()?)?) };
    let video = o.video.as_ref().map(|v| project.resolve(v));
    check_outputs(&inputs, target.as_ref(), video.as_deref(), first, last)
}

/// The output video's encode settings for `project`.
fn encode_settings(project: &Project, prepared: &Prepared) -> Result<(EncodeSettings, Option<Vec<f64>>), Error> {
    let o = &project.output;
    let alpha = prepared.keyframes.iter().any(|(k, _)| io::eval_num_channels(&k.style) == 4);
    let (frame_rate, colour, durations) = match &prepared.frames.video {
        Some(info) => {
            let start = prepared.frames.start_number;
            let durations = info.is_vfr().then(|| {
                (prepared.first..=prepared.last).map(|t| info.frame_duration(t - start)).collect::<Vec<f64>>()
            });
            (
                info.frame_rate,
                prepared.frames.colour.clone().unwrap_or_else(freesynth_media::ColourTags::bt709),
                durations,
            )
        }
        None => {
            let fps = project.media.fps.unwrap_or(25.0);
            let rate = Rational::approximate(fps)
                .ok_or_else(|| Error::Project(format!("media.fps must be positive, not {fps}")))?;
            let colour = project.media.colour.clone().unwrap_or_default();
            (rate, colour.completed(1920, 1080, "yuv420p"), None)
        }
    };
    let settings = EncodeSettings { codec: o.codec, quality: o.quality, frame_rate, colour, alpha, depth: o.depth()? };
    Ok((settings, durations))
}

/// Creates the folder of the output video `path` and checks that files can be written in it,
/// before a long render rather than after it.
fn check_video_folder(path: &Path) -> Result<(), Error> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)
        .map_err(|e| Error::Project(format!("cannot create the folder of output.video '{}': {e}", path.display())))?;
    let probe = dir.join(format!(".freesynth-write-test-{}", std::process::id()));
    std::fs::File::create(&probe)
        .map_err(|e| Error::Project(format!("cannot write in the folder of output.video '{}': {e}", path.display())))?;
    let _ = std::fs::remove_file(&probe);
    Ok(())
}

/// The part of every frame's cache key that covers how frames are made: the render revision,
/// the look parameters, the synthesis device class and where optical flow runs (GPU and CPU
/// results differ slightly, so a frame made one way is not current for the other).
fn look_key(params: &VideoParams, gpu: bool, flow: FlowBackend) -> Hash {
    let flow_on_gpu = gpu && flow == FlowBackend::Device;
    Hasher::new("look")
        .u64(u64::from(OUTPUT_REVISION))
        .str(&format!("{params:?}"))
        .str(if gpu { "gpu" } else { "cpu" })
        .str(if flow_on_gpu { "gpu-flow" } else { "cpu-flow" })
        .finish()
}

/// The cache key of an output frame file: the frame's key, the bit depth and the format.
fn file_key(key: &Hash, bits: u32, format: &str) -> Hash {
    Hasher::new("file").hash(key).u64(u64::from(bits)).str(format).finish()
}

/// Ends a render's helpers when dropped, also when the render unwinds: stops the cancel
/// forwarder and releases the encoder waiting for frames that will not come.
struct RenderEnd<'a> {
    finished: &'a AtomicBool,
    ready: &'a Ready,
}

impl Drop for RenderEnd<'_> {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Relaxed);
        self.ready.abort();
    }
}

/// Cancels `child` when `parent` is cancelled, until `done` is set.
fn forward_cancel(parent: &CancelToken, child: &CancelToken, done: &AtomicBool) {
    while !done.load(Ordering::Relaxed) {
        if parent.is_cancelled() {
            child.cancel();
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Renders `project` incrementally; see [`Session::render_with`].
pub(crate) fn render_project(
    session: &Session,
    project: &Project,
    request: &RenderRequest,
    report: &(dyn Fn(&Event) + Sync),
    cancel: &CancelToken,
) -> Result<ProjectStats, Error> {
    let start = Instant::now();
    // Absolute paths: the cache does not depend on the working directory.
    let project = &project.absolute();
    project.validate()?;
    let params = project.video_params()?;
    let cache_dir = session.project_cache_dir(project);
    let _lock = CacheLock::acquire(&cache_dir)?;
    let mut manifest = Manifest::load(&cache_dir, project.base_dir());
    report(&Event::Status("checking inputs".into()));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let ffmpeg = || session.ffmpeg();
    let mut opener =
        Opener { cache_dir: &cache_dir, files: &mut manifest.files, ffmpeg: &ffmpeg, report: &|e| report(e), threads };
    let prepared = prepare(project, &mut opener)?;
    let (first, last) = (prepared.first, prepared.last);

    // Where frames go. Nothing may overwrite an input.
    let depth = project.output.depth()?;
    let frames_spec = if project.output.frames.is_empty() {
        cache_dir.join("frames")
    } else {
        project.resolve(&project.output.frames)
    };
    let target = OutputFrames::parse(&frames_spec, depth)?;
    let video_path = project.output.video.as_ref().map(|v| project.resolve(v));
    check_outputs(&prepared.input_files(), Some(&target), video_path.as_deref(), first, last)?;
    target.create_dir("output.frames")?;
    // A video is encoded from lossless frames: from the output frames when they are PNG,
    // otherwise from PNG copies in the cache.
    let format = target.format();
    let master = match &video_path {
        Some(_) if format != "png" => {
            let m = OutputFrames::parse(&cache_dir.join("frames"), depth)?;
            m.create_dir("the render cache")?;
            Some(m)
        }
        _ => None,
    };
    if let (Some(v), Some(info)) = (&video_path, &prepared.frames.video)
        && info.has_audio
    {
        report(&Event::Warning(format!(
            "'{}' has an audio track; '{}' is written without audio",
            prepared.frames.source.display(),
            v.display()
        )));
    }

    // A video input decodes on its own threads while the render reads the frames it has.
    let inputs: Vec<&Input> = std::iter::once(&prepared.frames).chain(prepared.mask.as_ref()).collect();
    let mut stats = with_decodes(&inputs, report, cancel, || {
        report(&Event::Status(format!("input: {}", describe(&prepared.frames))));

        // Keys of every output frame.
        let keyframe_frames: Vec<usize> = prepared.keyframes.iter().map(|(k, _)| k.frame).collect();
        let plan = Plan::new(first, last, &keyframe_frames)?;
        let paintings: Vec<Hash> =
            prepared.keyframes.iter().map(|(_, path)| manifest.files.hash(path)).collect::<Result<_, _>>()?;
        let device = session.device();
        let look = look_key(&params, device.is_gpu(), session.options().flow_backend);
        let (frames_in, mask_in) = (&prepared.frames, prepared.mask.as_ref());
        let keys = frame_keys(&plan, &look, &paintings, &|t| frames_in.key(t), &|t| mask_in.map(|m| m.key(t)));

        // Which frames are current.
        let bits = depth.bits();
        let file_keys: BTreeMap<usize, Hash> = keys.iter().map(|(&t, k)| (t, file_key(k, bits, &format))).collect();
        let master_keys: BTreeMap<usize, Hash> = match &master {
            Some(_) => keys.iter().map(|(&t, k)| (t, file_key(k, bits, "png"))).collect(),
            None => BTreeMap::new(),
        };
        let current = |m: &Manifest, t: usize| {
            m.is_current(FrameKind::Output, t, &file_keys[&t], &target.path(t))
                && master.as_ref().is_none_or(|mf| m.is_current(FrameKind::Master, t, &master_keys[&t], &mf.path(t)))
        };
        let (lo, hi) = request.range.map_or((first, last), |(a, b)| (a.max(first), b.min(last)));
        let mut forced = BTreeSet::new();
        for &k in &request.keyframes {
            let (a, b) = keyframe_reach(&plan, k)
                .ok_or_else(|| Error::Project(format!("frame {k} has no keyframe (--keyframe)")))?;
            forced.extend(a..=b);
        }
        let dirty: BTreeSet<usize> =
            (lo..=hi).filter(|&t| request.force || forced.contains(&t) || !current(&manifest, t)).collect();
        let runs = dirty_runs(&plan, &dirty);
        let run_plans: Vec<Plan> = runs
            .iter()
            .map(|&(a, b)| {
                let inside: Vec<usize> = keyframe_frames.iter().copied().filter(|k| (a..=b).contains(k)).collect();
                Plan::new(a, b, &inside)
            })
            .collect::<Result<_, _>>()?;
        let syntheses: usize = run_plans.iter().map(Plan::num_syntheses).sum();
        let frames_total = last - first + 1;
        report(&Event::Planned {
            frames: frames_total,
            render: dirty.len(),
            reused: frames_total - dirty.len(),
            runs: runs.clone(),
            syntheses,
        });

        // The video: encoded again when any frame changed or the file is not the one we wrote.
        // Frames outside the requested range that are not current: the video would be incomplete.
        let stale_outside: Vec<usize> =
            (first..=last).filter(|t| !(lo..=hi).contains(t) && !current(&manifest, *t)).collect();
        let video_frames = master.as_ref().unwrap_or(&target);
        let video_keys = if master.is_some() { &master_keys } else { &file_keys };
        let video = match &video_path {
            Some(v) if !stale_outside.is_empty() => {
                report(&Event::Warning(format!(
                    "'{}' is not written: {} frames outside {lo}-{hi} are not rendered (first: {})",
                    v.display(),
                    stale_outside.len(),
                    stale_outside[0]
                )));
                None
            }
            Some(path) => {
                let (settings, durations) = encode_settings(project, &prepared)?;
                let mut h = Hasher::new("video");
                for t in first..=last {
                    h.hash(&video_keys[&t]);
                }
                h.str(&format!("{settings:?}")).str(&format!("{durations:?}"));
                let key = h.finish();
                let current = dirty.is_empty() && manifest.video_is_current(&key, path);
                let ff = session.ffmpeg()?;
                freesynth_media::check_encoder(&ff, path, &settings)?;
                if !current {
                    check_video_folder(path)?;
                }
                let job = VideoJob {
                    ff,
                    path: path.clone(),
                    frames: video_frames,
                    size: prepared.size,
                    first,
                    last,
                    settings,
                    durations,
                };
                Some((job, key, current))
            }
            None => None,
        };

        // Forget the frames about to be replaced before touching them.
        let keep = files_in_use(project, &prepared);
        manifest.files.retain(&keep);
        for &t in &dirty {
            manifest.forget(t);
        }
        manifest.save(&cache_dir)?;
        let prepare_seconds = start.elapsed().as_secs_f64();

        let ready = Ready::new(first, last);
        for t in first..=last {
            if !dirty.contains(&t) {
                ready.mark(t);
            }
        }
        let wanted: BTreeMap<usize, Hash> = dirty.iter().map(|t| (*t, file_keys[t])).collect();
        let master_wanted: BTreeMap<usize, Hash> =
            dirty.iter().filter_map(|t| master_keys.get(t).map(|k| (*t, *k))).collect();
        let manifest = Mutex::new(manifest);
        let written = AtomicUsize::new(0);
        let last_save = Mutex::new(Instant::now());
        let sink = ProjectSink {
            target: &target,
            wanted: &wanted,
            master: master.as_ref().map(|m| (m, &master_wanted)),
            manifest: &manifest,
            cache_dir: &cache_dir,
            ready: &ready,
            written: &written,
            report,
            last_save: &last_save,
        };
        let synthesized = AtomicUsize::new(0);
        let on_progress = |p: &Progress| {
            if let Progress::Synthesized { frame, keyframe, .. } = p {
                let done = synthesized.fetch_add(1, Ordering::Relaxed) + 1;
                report(&Event::Synthesized { frame: *frame, keyframe: *keyframe, done, total: syntheses });
            }
        };

        let mut stats = ProjectStats {
            device: device.name().to_owned(),
            frames: frames_total,
            rendered: dirty.len(),
            reused: frames_total - dirty.len(),
            runs: runs.clone(),
            syntheses,
            prepare_seconds,
            ..ProjectStats::default()
        };
        let streaming = video.as_ref().filter(|(job, _, current)| !current && job.durations.is_none());
        let render_start = Instant::now();
        let mut render_end = None;
        // The render stops when the caller cancels or when the encoder fails (a failed encode
        // would waste the rest of a long render).
        let run_cancel = CancelToken::new();
        let finished = AtomicBool::new(false);
        let (render_result, encode_result) = std::thread::scope(|s| {
            s.spawn(|| forward_cancel(cancel, &run_cancel, &finished));
            let encoder = streaming.map(|(job, _, _)| {
                s.spawn(|| {
                    let result = job.stream(&ready, report);
                    if result.as_ref().is_err_and(|e| !e.is_cancelled()) {
                        run_cancel.cancel();
                    }
                    result
                })
            });
            let end = RenderEnd { finished: &finished, ready: &ready };
            let result = (|| {
                let keyframes_all: Vec<Keyframe> = prepared.keyframes.iter().map(|(k, _)| k.clone()).collect();
                for &(a, b) in &runs {
                    report(&Event::Status(format!("rendering frames {a}-{b}")));
                    let shot = Shot {
                        frames: &prepared.frames.seq,
                        keyframes: keyframes_all.iter().filter(|k| (a..=b).contains(&k.frame)).cloned().collect(),
                        mask: prepared.mask.as_ref().map(|m| &m.seq as &dyn FrameSource),
                        range: Some((a, b)),
                    };
                    let run_stats = session.run_shot(
                        &shot,
                        params.clone(),
                        prepared.keyframes16.as_deref(),
                        &cache_dir,
                        &sink,
                        Some(&on_progress),
                        &run_cancel,
                    )?;
                    stats.run_stats.push(run_stats);
                }
                Ok::<(), Error>(())
            })();
            render_end = Some(Instant::now());
            // Frames still missing now never come: the encoder must not wait for them.
            drop(end);
            let encoded =
                encoder.map(|h| h.join().unwrap_or_else(|_| Err(Error::Project("the encoder panicked".into()))));
            (result, encoded)
        });
        let render_end = render_end.unwrap_or_else(Instant::now);
        stats.render_seconds = (render_end - render_start).as_secs_f64();
        stats.syntheses = synthesized.load(Ordering::Relaxed);
        let mut manifest = manifest.into_inner().unwrap_or_else(|e| e.into_inner());
        let saved = manifest.save(&cache_dir);
        if let Err(e) = render_result {
            // A render stopped by a failed encode reports the encoder's error.
            if e.is_cancelled()
                && !cancel.is_cancelled()
                && let Some(Err(encode_error)) = encode_result
            {
                return Err(encode_error);
            }
            return Err(e);
        }
        saved?;

        if let Some((job, key, current)) = &video {
            if *current {
                stats.video_reused = true;
            } else {
                report(&Event::Status(format!("writing '{}'", job.path.display())));
                let notes = match (encode_result, &job.durations) {
                    (Some(result), _) => result?,
                    (None, Some(durations)) => job.encode_files(durations, &cache_dir)?,
                    (None, None) => unreachable!("a stale video without timestamps is streamed"),
                };
                for n in notes {
                    report(&Event::Warning(n));
                }
                let stamp = FileStamp::of(&job.path)?;
                manifest.record_video(*key, &job.path, stamp);
                manifest.save(&cache_dir)?;
            }
            stats.video = Some(job.path.clone());
        }
        stats.encode_seconds = render_end.elapsed().as_secs_f64();
        Ok(stats)
    })?;
    // After the decodes have finished too.
    stats.seconds = start.elapsed().as_secs_f64();
    Ok(stats)
}

/// The input files the project uses now (the remembered hashes of all others are dropped).
fn files_in_use(project: &Project, prepared: &Prepared) -> HashSet<PathBuf> {
    let mut keep: HashSet<PathBuf> = prepared.keyframes.iter().map(|(_, p)| p.clone()).collect();
    for input in std::iter::once(&prepared.frames).chain(prepared.mask.as_ref()) {
        if input.video.is_none() {
            keep.extend(input.files());
        }
    }
    let m = &project.media;
    for spec in std::iter::once(&m.frames).chain(m.mask.as_ref()) {
        if is_video_path(spec) {
            keep.insert(project.resolve(spec));
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flow_device_is_part_of_the_look() {
        let params = VideoParams::classic();
        let key = |gpu, flow| look_key(&params, gpu, flow);
        // With GPU synthesis, moving flow onto the GPU makes every frame stale.
        assert_ne!(key(true, FlowBackend::Device), key(true, FlowBackend::Cpu));
        // On the CPU both settings run the same flow.
        assert_eq!(key(false, FlowBackend::Device), key(false, FlowBackend::Cpu));
        assert_ne!(key(true, FlowBackend::Cpu), key(false, FlowBackend::Cpu));
    }

    #[test]
    fn outputs_must_not_overwrite_inputs() {
        let dir = crate::test_util::scratch("overwrite");
        std::fs::create_dir_all(dir.join("seq")).unwrap();
        let inputs: Vec<PathBuf> = (0..3).map(|t| dir.join(format!("seq/{t:05}.png"))).collect();
        for p in &inputs {
            std::fs::write(p, b"x").unwrap();
        }
        let video = dir.join("clip.mov");
        std::fs::write(&video, b"x").unwrap();
        let mut all = inputs.clone();
        all.push(video.clone());
        let same = OutputFrames::parse(&dir.join("seq"), freesynth_media::BitDepth::Eight).unwrap();
        let err = check_outputs(&all, Some(&same), None, 0, 2).unwrap_err().to_string();
        assert!(err.contains("output frame 0") && err.contains("00000.png"), "{err}");
        // The same file through another spelling.
        let spelled = dir.join("seq/../clip.mov");
        let err = check_outputs(&all, None, Some(&spelled), 0, 2).unwrap_err().to_string();
        assert!(err.contains("output.video"), "{err}");
        let other = OutputFrames::parse(&dir.join("out"), freesynth_media::BitDepth::Eight).unwrap();
        check_outputs(&all, Some(&other), Some(&dir.join("out.mov")), 0, 2).unwrap();
        // Frames outside the rendered range are not written.
        let shifted = OutputFrames::parse(&dir.join("seq/[#####].png"), freesynth_media::BitDepth::Eight).unwrap();
        check_outputs(&all, Some(&shifted), None, 3, 5).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
