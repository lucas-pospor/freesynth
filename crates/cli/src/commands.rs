//! The subcommands: keyframe video projects (`render`, `video`, `new`, `import-ebs`) and
//! `devices`.
//!
//! Exit codes: 0 success, 1 usage error, 2 bad input (project, files, media), 3 no usable GPU,
//! 4 render failure, 130 cancelled.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};
use freesynth::devices::{cpu_threads, gpu_adapters, gpu_support};
use freesynth::ebs::import_ebs;
use freesynth::paths::for_project;
use freesynth::project::{EXTENSION, find_keyframes};
use freesynth::{
    Backend, BlendMode, CancelToken, Codec, Error, Event, FlowBackend, Project, ProjectStats, RenderRequest, Session,
    SessionOptions, VideoError, WatchEvent, WatchOptions,
};
use freesynth_core::Preset;
use freesynth_media::{Ffmpeg, is_video_path};
use serde_json::{Value, json};

const AFTER_HELP: &str = "\
Single images: run `freesynth` with ebsynth-style options instead of a command, for example
  freesynth -style style.png -guide source.png target.png -output out.png
Run `freesynth` without arguments to list those options.";

/// Example-based stylization of images and video.
#[derive(Parser)]
#[command(name = "freesynth", version, after_help = AFTER_HELP)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Render a project file (.fsyn): bring its output up to date.
    ///
    /// Only frames whose inputs or settings changed since the last render are rendered again:
    /// after repainting one keyframe, the frames between its neighbouring keyframes. With
    /// --watch, keeps watching the keyframe images, video inputs and the project file, and
    /// renders again on every change.
    Render(RenderArgs),
    /// Stylize a video in one go, from frames or a video file and painted keyframes, without a
    /// project file.
    Video(VideoArgs),
    /// Create a project file for a video (or image sequence) and its keyframe paintings.
    New(NewArgs),
    /// Convert an EbSynth project (.ebs) into a FreeSynth project (.fsyn).
    ///
    /// Patterns, weights and keyframes carry over; the per-keyframe output folders become one
    /// blended output sequence. Whatever does not carry over exactly is listed.
    ImportEbs(ImportArgs),
    /// List GPU adapters, the CPU and the ffmpeg that video files use.
    Devices(DevicesArgs),
}

/// How progress is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ProgressMode {
    /// Progress and messages on stderr.
    Human,
    /// One JSON object per line on stdout (for scripts and other programs).
    Json,
    /// Warnings and errors only.
    None,
}

#[derive(Args)]
struct RunArgs {
    /// Device: auto (GPU when usable), gpu or cpu. Default: the project's setting.
    #[arg(long, value_name = "auto|gpu|cpu")]
    backend: Option<Backend>,
    /// How to show progress.
    #[arg(long, value_enum, default_value = "human")]
    progress: ProgressMode,
    /// Folder for render caches, which several projects may share (each gets its own
    /// directory in it). Default: .freesynth-cache/<project name> next to the project.
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
    /// Keyframe chains rendered at the same time (default: as many as fit in memory, at most 4).
    #[arg(long, value_name = "N")]
    parallel: Option<usize>,
    /// Memory budget of a render in MiB (default: half the memory, at most 4096).
    #[arg(long, value_name = "MB")]
    memory_mb: Option<usize>,
    /// Compute optical flow on the GPU too (by default it runs on the CPU next to GPU synthesis).
    #[arg(long)]
    flow_on_device: bool,
}

#[derive(Args)]
struct RenderArgs {
    /// The project file.
    project: PathBuf,
    /// Only bring frames A to B up to date (a single frame N is N-N).
    #[arg(long, value_name = "A-B", value_parser = parse_range)]
    frames: Option<(usize, usize)>,
    /// Render this keyframe's frames again even if they are up to date (repeatable).
    #[arg(long = "keyframe", value_name = "FRAME")]
    keyframes: Vec<usize>,
    /// Render every frame again, ignoring the cache.
    #[arg(long)]
    force: bool,
    /// After rendering, re-render whenever a keyframe image, video input or the project file
    /// changes. Stop with Ctrl-C.
    #[arg(long)]
    watch: bool,
    #[command(flatten)]
    run: RunArgs,
}

/// Look and output settings shared by `video` and `new`.
#[derive(Args)]
struct LookArgs {
    /// Mask sequence or video (white = subject).
    #[arg(long, value_name = "PATH")]
    mask: Option<String>,
    /// First frame to render.
    #[arg(long, value_name = "N")]
    first: Option<usize>,
    /// Last frame to render.
    #[arg(long, value_name = "N")]
    last: Option<usize>,
    /// Frame number of a video file's first frame (to match keyframe numbers).
    #[arg(long, value_name = "N")]
    start_number: Option<usize>,
    /// Frame rate of an image-sequence input, for video output.
    #[arg(long, value_name = "FPS")]
    fps: Option<f64>,
    /// Synthesis preset: default, fast, high or compat.
    #[arg(long, value_name = "NAME")]
    preset: Option<Preset>,
    /// Output bit depth: 8 or 16 (16 needs PNG frames).
    #[arg(long, value_name = "BITS")]
    bit_depth: Option<u32>,
    /// Video codec: auto, prores, prores4444, h264, h265, ffv1, png or vp9.
    #[arg(long, value_name = "CODEC")]
    codec: Option<Codec>,
    /// Quality factor (CRF) for h264, h265 and vp9; lower is better.
    #[arg(long, value_name = "CRF")]
    quality: Option<u32>,
    /// Weight of the mapping (positional) guide.
    #[arg(long, value_name = "W")]
    mapping: Option<f32>,
    /// Weight of the temporal (de-flicker) guide.
    #[arg(long, value_name = "W")]
    temporal: Option<f32>,
    /// Weight of the video (appearance) guide.
    #[arg(long = "video-weight", value_name = "W")]
    video_weight: Option<f32>,
    /// Weight of the uniformity term (how evenly the keyframe is used).
    #[arg(long, value_name = "W")]
    uniformity: Option<f32>,
    /// Random seed.
    #[arg(long, value_name = "N")]
    seed: Option<u32>,
    /// How the two passes between keyframes are combined: adaptive (default; keeps detail) or
    /// linear (crossfade).
    #[arg(long, value_name = "adaptive|linear")]
    blend: Option<BlendMode>,
    /// Also weight the temporal guide by flow consistency (helps with occlusions; slower).
    #[arg(long)]
    gate_temporal: bool,
    /// Start each frame from the previous one: about twice as fast per frame, a little less
    /// faithful to the keyframes (for previews).
    #[arg(long)]
    warm_start: bool,
}

#[derive(Args)]
struct VideoArgs {
    /// Input: a video file, a frame pattern (frames/[####].png) or a directory of numbered images.
    #[arg(short, long, value_name = "PATH")]
    input: String,
    /// Keyframe painting for frame N (repeatable).
    #[arg(long = "key", value_name = "N=IMAGE", value_parser = parse_key)]
    keys: Vec<(usize, String)>,
    /// Keyframe paintings: a directory or pattern of images numbered by frame.
    #[arg(long = "keys", value_name = "PATH")]
    keys_from: Option<String>,
    /// Output: a video file (.mov, .mp4, .mkv, .webm) or a frame pattern or directory. Give it
    /// twice for both.
    #[arg(short, long, value_name = "PATH", required = true)]
    output: Vec<String>,
    /// Also write the equivalent project file.
    #[arg(long, value_name = "FILE")]
    save_project: Option<PathBuf>,
    #[command(flatten)]
    look: LookArgs,
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Args)]
struct NewArgs {
    /// The project file to create (.fsyn).
    project: PathBuf,
    /// Input: a video file, a frame pattern or a directory of numbered images.
    #[arg(short, long, value_name = "PATH")]
    input: String,
    /// Keyframe paintings: a directory or pattern of images numbered by frame.
    #[arg(long = "keys", value_name = "PATH")]
    keys_from: Option<String>,
    /// Keyframe painting for frame N (repeatable).
    #[arg(long = "key", value_name = "N=IMAGE", value_parser = parse_key)]
    keys: Vec<(usize, String)>,
    /// Output: a video file or a frame pattern or directory (repeatable). Default:
    /// <name>-out/[#####].png next to the project.
    #[arg(short, long, value_name = "PATH")]
    output: Vec<String>,
    /// Overwrite an existing project file.
    #[arg(long)]
    force: bool,
    #[command(flatten)]
    look: LookArgs,
}

#[derive(Args)]
struct ImportArgs {
    /// The EbSynth project file.
    ebs: PathBuf,
    /// The project file to write (default: next to the .ebs file, with the .fsyn extension).
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
    /// Use this video file (or frame pattern) as input instead of the project's frames.
    #[arg(long, value_name = "PATH")]
    input: Option<String>,
    /// Write the result here instead: a video file or a frame pattern or directory (repeatable).
    #[arg(long = "out", value_name = "PATH")]
    outputs: Vec<String>,
    /// Overwrite an existing project file.
    #[arg(long)]
    force: bool,
}

#[derive(Args)]
struct DevicesArgs {
    /// Print JSON.
    #[arg(long)]
    json: bool,
}

fn parse_range(s: &str) -> Result<(usize, usize), String> {
    let bad = || format!("expected a frame range such as 10-40, or a frame number, not '{s}'");
    let (a, b) = s.split_once('-').unwrap_or((s, s));
    let (a, b): (usize, usize) = (a.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?);
    if a > b {
        return Err(format!("the range {a}-{b} is empty"));
    }
    Ok((a, b))
}

fn parse_key(s: &str) -> Result<(usize, String), String> {
    let (n, image) = s.split_once('=').ok_or_else(|| format!("expected N=IMAGE, not '{s}'"))?;
    let n = n.trim().parse().map_err(|_| format!("'{n}' is not a frame number"))?;
    if image.is_empty() {
        return Err("the keyframe image is missing".into());
    }
    Ok((n, image.to_owned()))
}

/// A failed command: what to print and the exit code.
struct Failure {
    message: String,
    code: u8,
}

impl From<Error> for Failure {
    fn from(e: Error) -> Self {
        let code = match &e {
            Error::Render(VideoError::Cancelled) => 130,
            Error::Render(VideoError::Device(_)) => 3,
            Error::Render(VideoError::Synthesis(_) | VideoError::Flow(_) | VideoError::Panicked(_)) => 4,
            _ => 2,
        };
        Failure { message: e.to_string(), code }
    }
}

/// A usage mistake (exit code 1).
impl From<String> for Failure {
    fn from(message: String) -> Self {
        Failure { message, code: 1 }
    }
}

impl Failure {
    /// Bad input that is not a library error (exit code 2).
    fn input(message: String) -> Self {
        Failure { message, code: 2 }
    }
}

/// Runs a subcommand; `args[0]` is the program name.
pub fn run(args: &[String]) -> u8 {
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return if e.use_stderr() { 1 } else { 0 };
        }
    };
    let json = match &cli.command {
        Command::Render(a) => a.run.progress == ProgressMode::Json,
        Command::Video(a) => a.run.progress == ProgressMode::Json,
        _ => false,
    };
    let result = match cli.command {
        Command::Render(a) => render(a),
        Command::Video(a) => video(a),
        Command::New(a) => new_project(a),
        Command::ImportEbs(a) => import(a),
        Command::Devices(a) => devices(a),
    };
    match result {
        Ok(()) => 0,
        Err(f) => {
            if json {
                emit_json(json!({"event": "error", "message": f.message, "code": f.code}));
            } else {
                eprintln!("error: {}", f.message);
            }
            f.code
        }
    }
}

/// A token cancelled by Ctrl-C (SIGINT) or SIGTERM. A second Ctrl-C exits at once.
fn interrupt_token() -> CancelToken {
    let token = CancelToken::new();
    #[cfg(unix)]
    {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            let _ = signal_hook::flag::register_conditional_shutdown(signal, 130, Arc::clone(&flag));
            let _ = signal_hook::flag::register(signal, Arc::clone(&flag));
        }
        let t = token.clone();
        std::thread::spawn(move || {
            while !flag.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
            }
            t.cancel();
        });
    }
    token
}

fn emit_json(value: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

/// Shows progress events.
struct Reporter {
    mode: ProgressMode,
    terminal: bool,
    /// The last progress line's tenth (non-terminal output) and whether a `\r` line is open.
    state: Mutex<(usize, bool)>,
}

impl Reporter {
    fn new(mode: ProgressMode) -> Self {
        Self { mode, terminal: std::io::stderr().is_terminal(), state: Mutex::new((usize::MAX, false)) }
    }

    /// Prints a message line (ending an open progress line first).
    fn line(&self, text: &str) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.1 {
            eprintln!();
            state.1 = false;
        }
        eprintln!("{text}");
    }

    fn progress(&self, text: &str, done: usize, total: usize) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if self.terminal {
            eprint!("\r\x1b[2K{text}");
            let _ = std::io::stderr().flush();
            state.1 = true;
        } else {
            let tenth = (done * 10).checked_div(total).unwrap_or(10);
            if tenth != state.0 {
                state.0 = tenth;
                eprintln!("{text}");
            }
        }
    }

    fn event(&self, e: &Event) {
        match self.mode {
            ProgressMode::None => {
                if let Event::Warning(w) = e {
                    self.line(&format!("warning: {w}"));
                }
            }
            ProgressMode::Json => emit_json(event_json(e)),
            ProgressMode::Human => match e {
                Event::Status(s) => self.line(s),
                Event::Warning(w) => self.line(&format!("warning: {w}")),
                // The decode runs alongside the render: one line when it is complete.
                Event::Decoded { done, total } if done == total => self.line(&format!("decoded {total} frames")),
                Event::Planned { frames, render, reused, runs, syntheses } => {
                    if *render == 0 {
                        self.line(&format!("all {frames} frames are up to date"));
                    } else {
                        let ranges: Vec<String> = runs.iter().map(|(a, b)| format!("{a}-{b}")).collect();
                        self.line(&format!(
                            "rendering {render} of {frames} frames ({reused} up to date; segments {}; {syntheses} \
                             syntheses)",
                            ranges.join(", ")
                        ));
                    }
                    *self.state.lock().unwrap_or_else(|e| e.into_inner()) = (usize::MAX, false);
                }
                Event::Synthesized { done, total, .. } => {
                    self.progress(&format!("synthesized {done}/{total}"), *done, *total);
                }
                Event::Encoded { done, total } if done == total => {
                    self.progress(&format!("encoded {done}/{total} frames"), *done, *total);
                }
                // Written frames (and event kinds this version does not know) print nothing.
                _ => {}
            },
        }
    }

    fn finished(&self, stats: &ProjectStats) {
        match self.mode {
            ProgressMode::Json => emit_json(stats_json(stats)),
            ProgressMode::None => {}
            ProgressMode::Human => {
                let video = match (&stats.video, stats.video_reused) {
                    (Some(v), false) => format!("; wrote {}", v.display()),
                    (Some(v), true) => format!("; {} is up to date", v.display()),
                    (None, _) => String::new(),
                };
                self.line(&format!(
                    "done: {} frames rendered, {} kept, in {:.1} s on {}{video}",
                    stats.rendered, stats.reused, stats.seconds, stats.device
                ));
            }
        }
    }
}

fn event_json(e: &Event) -> Value {
    match e {
        Event::Status(s) => json!({"event": "status", "message": s}),
        Event::Warning(w) => json!({"event": "warning", "message": w}),
        Event::Decoded { done, total } => json!({"event": "decoded", "done": done, "total": total}),
        Event::Planned { frames, render, reused, runs, syntheses } => json!({
            "event": "planned", "frames": frames, "render": render, "reused": reused, "runs": runs,
            "syntheses": syntheses,
        }),
        Event::Synthesized { frame, keyframe, done, total } => json!({
            "event": "synthesized", "frame": frame, "keyframe": keyframe, "done": done, "total": total,
        }),
        Event::Written { frame, done, total } => {
            json!({"event": "written", "frame": frame, "done": done, "total": total})
        }
        Event::Encoded { done, total } => json!({"event": "encoded", "done": done, "total": total}),
        other => json!({"event": "other", "message": format!("{other:?}")}),
    }
}

fn stats_json(s: &ProjectStats) -> Value {
    json!({
        "event": "done",
        "device": s.device,
        "frames": s.frames,
        "rendered": s.rendered,
        "reused": s.reused,
        "runs": s.runs,
        "syntheses": s.syntheses,
        "seconds": s.seconds,
        "prepare_seconds": s.prepare_seconds,
        "render_seconds": s.render_seconds,
        "encode_seconds": s.encode_seconds,
        "video": s.video.as_ref().map(|v| v.display().to_string()),
        "video_reused": s.video_reused,
    })
}

/// Opens the session for `backend` with the run options.
fn open_session(backend: Backend, run: &RunArgs, reporter: &Reporter) -> Result<Session, Failure> {
    let mut session = Session::new(backend)?;
    let mut options = SessionOptions::default();
    options.cache_dir = run.cache_dir.clone();
    if let Some(n) = run.parallel {
        options.parallel_chains = Some(n.max(1));
    }
    if let Some(mb) = run.memory_mb {
        options.memory_budget = mb << 20;
    }
    if run.flow_on_device {
        options.flow_backend = FlowBackend::Device;
    }
    session.set_options(options);
    if let Some(reason) = session.device().fallback_reason() {
        reporter.event(&Event::Warning(format!("no usable GPU ({reason}); rendering on the CPU")));
    }
    if reporter.mode == ProgressMode::Json {
        emit_json(json!({"event": "device", "name": session.device().name(), "gpu": session.device().is_gpu()}));
    } else if reporter.mode == ProgressMode::Human {
        reporter.line(&format!("device: {}", session.device().name()));
    }
    Ok(session)
}

fn render(a: RenderArgs) -> Result<(), Failure> {
    let reporter = Reporter::new(a.run.progress);
    // `render shot` finds shot.fsyn, as `new shot` creates it.
    let project_file = if a.project.exists() { a.project.clone() } else { project_path(&a.project) };
    let mut project = Project::load(&project_file)?;
    if let Some(b) = a.run.backend {
        project.engine.backend = b;
    }
    // Mistakes in the project show up before the device opens (watch mode waits them out).
    if !a.watch {
        project.check_inputs()?;
    }
    let session = open_session(project.backend(), &a.run, &reporter)?;
    let mut request = RenderRequest::default();
    request.range = a.frames;
    request.force = a.force;
    request.keyframes = a.keyframes;
    let cancel = interrupt_token();
    if !a.watch {
        let stats = session.render_with(&project, &request, Some(&|e: &Event| reporter.event(e)), &cancel)?;
        reporter.finished(&stats);
        return Ok(());
    }
    let on_event = |w: WatchEvent<'_>| match w {
        WatchEvent::Rendering { reason } => match reporter.mode {
            ProgressMode::Json => emit_json(json!({"event": "watch_render", "reason": reason})),
            _ => reporter.line(&format!("render ({reason})")),
        },
        WatchEvent::Render(e) => reporter.event(e),
        WatchEvent::Rendered(stats) => reporter.finished(stats),
        WatchEvent::Failed(e) => match reporter.mode {
            ProgressMode::Json => emit_json(json!({"event": "watch_error", "message": e.to_string()})),
            _ => reporter.line(&format!("error: {e}")),
        },
        WatchEvent::Restarting { changed } => {
            let names: Vec<String> = changed.iter().map(|p| p.display().to_string()).collect();
            match reporter.mode {
                ProgressMode::Json => emit_json(json!({"event": "watch_restart", "changed": names})),
                _ => reporter.line(&format!("changed during the render: {}; starting over", names.join(", "))),
            }
        }
        WatchEvent::Waiting { files } => match reporter.mode {
            ProgressMode::Json => emit_json(json!({"event": "watch_waiting", "files": files.len()})),
            _ => reporter.line(&format!("watching {} files for changes (Ctrl-C to stop)", files.len())),
        },
        _ => {}
    };
    // Watch mode: a forced render first when asked, then incremental renders on every change.
    if request.force || !request.keyframes.is_empty() {
        let stats = session.render_with(&project, &request, Some(&|e: &Event| reporter.event(e)), &cancel)?;
        reporter.finished(&stats);
    }
    let mut base = RenderRequest::default();
    base.range = request.range;
    session.watch(&project_file, &base, WatchOptions::default(), &on_event, &cancel)?;
    Ok(())
}

/// Applies the look options to a project.
fn apply_look(p: &mut Project, l: &LookArgs) {
    if let Some(m) = &l.mask {
        p.media.mask = Some(m.clone());
    }
    p.media.first = l.first.or(p.media.first);
    p.media.last = l.last.or(p.media.last);
    if let Some(n) = l.start_number {
        p.media.start_number = n;
    }
    p.media.fps = l.fps.or(p.media.fps);
    if let Some(preset) = l.preset {
        p.engine.preset = preset;
    }
    if let Some(b) = l.bit_depth {
        p.output.bit_depth = b;
    }
    if let Some(c) = l.codec {
        p.output.codec = c;
    }
    p.output.quality = l.quality.or(p.output.quality);
    if let Some(w) = l.mapping {
        p.guides.mapping = w;
    }
    if let Some(w) = l.temporal {
        p.guides.temporal = w;
    }
    if let Some(w) = l.video_weight {
        p.guides.video = w;
    }
    if let Some(u) = l.uniformity {
        p.engine.uniformity = u;
    }
    if let Some(s) = l.seed {
        p.engine.seed = s;
    }
    if let Some(b) = l.blend {
        p.output.blend = b;
    }
    p.guides.gate_temporal |= l.gate_temporal;
    p.engine.warm_start |= l.warm_start;
}

/// Sets the outputs of `p` from paths (video files and frame sequences); `rel` converts a
/// command-line path into a project path.
fn set_outputs(p: &mut Project, outputs: &[String], rel: &dyn Fn(&str) -> String) -> Result<(), Failure> {
    let (videos, frames): (Vec<&String>, Vec<&String>) = outputs.iter().partition(|o| is_video_path(o));
    if videos.len() > 1 || frames.len() > 1 {
        return Err("give at most one video file and one frame sequence as outputs".to_owned().into());
    }
    p.output.video = videos.first().map(|v| rel(v));
    p.output.frames = frames.first().map(|f| rel(f)).unwrap_or_default();
    Ok(())
}

/// Adds keyframes from `--key N=IMAGE` and a `--keys` directory or pattern.
fn add_keys(
    p: &mut Project,
    keys: &[(usize, String)],
    from: Option<&str>,
    rel: &dyn Fn(&str) -> String,
) -> Result<(), Failure> {
    if let Some(spec) = from {
        for (frame, path) in find_keyframes(Path::new(spec))? {
            p.add_keyframe(frame, rel(&path.to_string_lossy()));
        }
    }
    for (frame, image) in keys {
        p.keyframes.retain(|k| k.frame != *frame);
        p.add_keyframe(*frame, rel(image));
    }
    p.keyframes.sort_by_key(|k| k.frame);
    if p.keyframes.is_empty() {
        return Err("no keyframe: give --key N=IMAGE or a directory of numbered paintings".to_owned().into());
    }
    Ok(())
}

fn video(a: VideoArgs) -> Result<(), Failure> {
    let reporter = Reporter::new(a.run.progress);
    let mut project = Project::new(a.input.clone(), "");
    project.set_name("video");
    let same = |s: &str| s.to_owned();
    add_keys(&mut project, &a.keys, a.keys_from.as_deref(), &same)?;
    set_outputs(&mut project, &a.output, &same)?;
    apply_look(&mut project, &a.look);
    if let Some(b) = a.run.backend {
        project.engine.backend = b;
    }
    project.check_inputs()?;
    if let Some(path) = &a.save_project {
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let saved = rebased(&project, &dir);
        saved.save(path)?;
        reporter.line(&format!("wrote {}", path.display()));
    }
    let session = open_session(project.backend(), &a.run, &reporter)?;
    // A one-shot render keeps its cache only when asked to.
    let temporary = a.run.cache_dir.is_none();
    let mut session = session;
    if temporary {
        // On disk: the system temporary directory is often RAM-backed, and frames can be large.
        let base = freesynth::paths::user_cache_dir().unwrap_or_else(std::env::temp_dir);
        remove_stale_video_caches(&base);
        let dir = base.join(format!("{VIDEO_CACHE_PREFIX}{}", std::process::id()));
        let mut options = session.options().clone();
        options.cache_dir = Some(dir);
        session.set_options(options);
    }
    let cancel = interrupt_token();
    let result = session.render(&project, Some(&|e: &Event| reporter.event(e)), &cancel);
    if temporary && let Some(dir) = &session.options().cache_dir {
        let _ = std::fs::remove_dir_all(dir);
    }
    reporter.finished(&result?);
    Ok(())
}

/// Name prefix of the temporary render caches of `freesynth video` (followed by the process id).
const VIDEO_CACHE_PREFIX: &str = "freesynth-video-";

/// Removes the temporary caches in `base` left by `freesynth video` runs that were killed
/// (their process no longer runs). Only on Linux, where a process's liveness is visible in
/// `/proc`.
fn remove_stale_video_caches(base: &Path) {
    if !cfg!(target_os = "linux") {
        return;
    }
    let Ok(entries) = std::fs::read_dir(base) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|n| n.strip_prefix(VIDEO_CACHE_PREFIX)) else { continue };
        let Ok(pid) = pid.parse::<u32>() else { continue };
        if pid != std::process::id() && !Path::new(&format!("/proc/{pid}")).exists() {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// A copy of `p` (whose paths are relative to the current directory) with its paths relative to
/// `dir`.
fn rebased(p: &Project, dir: &Path) -> Project {
    let rel = |s: &str| for_project(Path::new(s), dir);
    let mut out = p.clone();
    out.media.frames = rel(&p.media.frames);
    out.media.mask = p.media.mask.as_deref().map(rel);
    for k in &mut out.keyframes {
        k.image = rel(&k.image);
    }
    if !p.output.frames.is_empty() {
        out.output.frames = rel(&p.output.frames);
    }
    out.output.video = p.output.video.as_deref().map(rel);
    out.set_base_dir(dir);
    out
}

/// The project path with the `.fsyn` extension added when it has none.
fn project_path(path: &Path) -> PathBuf {
    if path.extension().is_some() { path.to_owned() } else { path.with_extension(EXTENSION) }
}

fn check_new_file(path: &Path, force: bool) -> Result<(), Failure> {
    if path.exists() && !force {
        return Err(Failure::input(format!("'{}' exists; use --force to overwrite it", path.display())));
    }
    Ok(())
}

fn new_project(a: NewArgs) -> Result<(), Failure> {
    let path = project_path(&a.project);
    check_new_file(&path, a.force)?;
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let rel = |s: &str| for_project(Path::new(s), &dir);
    let mut project = Project::new(rel(&a.input), "");
    add_keys(&mut project, &a.keys, a.keys_from.as_deref(), &rel)?;
    if a.output.is_empty() {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        project.output.frames = format!("{stem}-out/[#####].png");
    } else {
        set_outputs(&mut project, &a.output, &rel)?;
    }
    let mut look = a.look;
    look.mask = look.mask.as_deref().map(rel);
    apply_look(&mut project, &look);
    project.set_base_dir(&dir);
    project.validate()?;
    project.save(&path)?;
    println!("wrote {} ({} keyframes: {})", path.display(), project.keyframes.len(), keyframe_list(&project));
    Ok(())
}

fn keyframe_list(p: &Project) -> String {
    let frames: Vec<String> = p.keyframes.iter().map(|k| k.frame.to_string()).collect();
    frames.join(", ")
}

fn import(a: ImportArgs) -> Result<(), Failure> {
    let path = project_path(&a.output.clone().unwrap_or_else(|| a.ebs.with_extension(EXTENSION)));
    check_new_file(&path, a.force)?;
    let imported = import_ebs(&a.ebs, &path)?;
    let mut project = imported.project;
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let rel = |s: &str| for_project(Path::new(s), &dir);
    let mut notes = imported.warnings;
    if a.outputs.is_empty() {
        notes.extend(imported.output_warnings);
    }
    if let Some(input) = &a.input {
        // Keep the frame numbering of the replaced sequence.
        if is_video_path(input) {
            let old = project.resolve(&project.media.frames);
            let start = freesynth::ImageSequence::from_pattern(&old)
                .ok()
                .map(|s| freesynth::FrameSource::first(&s))
                .unwrap_or(0);
            project.media.start_number = start;
            notes.push(format!("input replaced by {input}; its first frame is frame {start}"));
        }
        project.media.frames = rel(input);
    }
    if !a.outputs.is_empty() {
        set_outputs(&mut project, &a.outputs, &rel)?;
    }
    project.validate()?;
    project.save(&path)?;
    println!("wrote {} ({} keyframes: {})", path.display(), project.keyframes.len(), keyframe_list(&project));
    for n in notes {
        println!("note: {n}");
    }
    Ok(())
}

fn devices(a: DevicesArgs) -> Result<(), Failure> {
    let adapters = gpu_adapters();
    let ffmpeg = Ffmpeg::locate();
    if a.json {
        let gpus: Vec<Value> = adapters
            .iter()
            .map(|g| {
                json!({
                    "name": g.name, "type": g.device_type, "backend": g.backend, "driver": g.driver,
                    "usable": g.usable, "problem": g.problem, "default": g.default,
                })
            })
            .collect();
        let ff = match &ffmpeg {
            Ok(f) => json!({
                "path": f.ffmpeg_path().display().to_string(), "version": f.version(), "warning": f.version_warning(),
            }),
            Err(e) => json!({"error": e.to_string()}),
        };
        emit_json(json!({"gpu_support": gpu_support(), "gpus": gpus, "cpu_threads": cpu_threads(), "ffmpeg": ff}));
        return Ok(());
    }
    if !gpu_support() {
        println!("GPU: not built in (CPU only)");
    } else if adapters.is_empty() {
        println!("GPU: no adapter found (renders use the CPU)");
    } else {
        println!("GPU adapters:");
        for g in &adapters {
            let mark = if g.default && g.usable { "*" } else { " " };
            let status = match &g.problem {
                None => String::new(),
                Some(p) => format!(" (not usable: {p})"),
            };
            println!("  {mark} {} [{}, {}] {}{status}", g.name, g.backend, g.device_type, g.driver);
        }
        println!("  (* = used by --backend gpu and auto)");
    }
    println!("CPU: {} threads", cpu_threads());
    match ffmpeg {
        Ok(f) => {
            println!("ffmpeg: {} ({})", f.ffmpeg_path().display(), f.version());
            if let Some(w) = f.version_warning() {
                println!("  warning: {w}");
            }
        }
        Err(e) => println!("ffmpeg: not available: {e}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_and_keys() {
        assert_eq!(parse_range("10-40"), Ok((10, 40)));
        assert_eq!(parse_range("5"), Ok((5, 5)), "a single frame");
        assert!(parse_range("40-10").is_err());
        assert!(parse_range("x").is_err());
        assert!(parse_range("3-").is_err());
        assert_eq!(parse_key("12=keys/012.png"), Ok((12, "keys/012.png".into())));
        assert!(parse_key("12").is_err());
        assert!(parse_key("a=b").is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn stale_video_caches_are_removed() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/tmp")
            .join(format!("freesynth-stale-caches-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        // Process ids never reach 4e9 (the kernel's limit is about 4 million).
        let names = [
            format!("{VIDEO_CACHE_PREFIX}4000000000"),
            format!("{VIDEO_CACHE_PREFIX}{}", std::process::id()),
            format!("{VIDEO_CACHE_PREFIX}notapid"),
            "other".to_owned(),
        ];
        for n in &names {
            std::fs::create_dir_all(base.join(n).join("frames")).unwrap();
        }
        remove_stale_video_caches(&base);
        let left: Vec<bool> = names.iter().map(|n| base.join(n).exists()).collect();
        assert_eq!(left, [false, true, true, true]);
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn help_for_every_command() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        for name in ["render", "video", "new", "import-ebs", "devices"] {
            let err = Cli::try_parse_from(["freesynth", name, "--help"]).err().expect("help is an early exit");
            assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp, "{name}");
        }
    }
}
