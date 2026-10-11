//! Renders a keyframe video shot from an image sequence and painted keyframes.
//!
//! ```text
//! cargo run --release -p freesynth --example render_sequence -- \
//!     --frames shot/video --key 0=shot/keys/000.png --key 99=shot/keys/099.png --out shot/out
//! ```
//!
//! Options:
//!
//! | Option | Meaning |
//! |---|---|
//! | `--project FILE` | load a project file instead of (or before) the options below |
//! | `--frames PATH` | input frames: a directory of numbered images or a pattern like `v/[###].jpg` |
//! | `--key N=IMAGE` | keyframe painting for frame N (repeatable) |
//! | `--out PATH` | output directory or pattern |
//! | `--mask PATH` | optional mask sequence |
//! | `--first N`, `--last N` | frame range |
//! | `--backend auto\|gpu\|cpu` | device (default auto) |
//! | `--preset NAME` | synthesis preset: compat, default, fast, high |
//! | `--seed N` | random seed of the synthesis |
//! | `--mapping W`, `--temporal W`, `--video W` | guide weights |
//! | `--gate` | weight the mapping guide by tracking confidence |
//! | `--parallel N` | chains run at once (default: as many as fit in the memory budget, at most 4) |
//! | `--memory-mb N` | total memory budget (default: half the memory, at most 4096) |
//! | `--frame-cache-mb N`, `--buffer-mb N` | cache limits before spilling to disk |
//! | `--cache-dir DIR` | spill directory (default `.freesynth-cache` next to the project) |
//! | `--flow-on-device` | compute optical flow on the synthesis device (by default it runs on the CPU) |
//! | `--save-project FILE` | write the resulting project file |
//! | `--stats FILE` | write render statistics as JSON |
//! | `--quiet` | no progress output |

use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use freesynth::{CancelToken, FlowBackend, Progress, Project, RenderStats, Session, SessionOptions};

fn usage() -> ExitCode {
    eprintln!(
        "usage: render_sequence [--project FILE] --frames PATH --key N=IMAGE [--key ...] --out PATH\n\
         \x20      [--mask PATH] [--first N] [--last N] [--backend auto|gpu|cpu] [--preset NAME] [--seed N]\n\
         \x20      [--mapping W] [--temporal W] [--video W] [--gate] [--parallel N] [--memory-mb N]\n\
         \x20      [--frame-cache-mb N] [--buffer-mb N] [--cache-dir DIR] [--flow-on-device]\n\
         \x20      [--save-project FILE] [--stats FILE] [--quiet]"
    );
    ExitCode::from(1)
}

/// Peak resident memory of this process in bytes, where the OS reports it.
fn peak_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

fn stats_json(stats: &RenderStats, open_seconds: f64, peak: Option<u64>) -> String {
    let s = stats;
    format!(
        "{{\n  \"device\": {:?},\n  \"frames\": {},\n  \"syntheses\": {},\n  \"chains\": {},\n  \"parallel\": {},\n  \
         \"device_open_seconds\": {open_seconds:.4},\n  \"render_seconds\": {:.4},\n  \"synthesis_seconds\": {:.4},\n  \
         \"guide_seconds\": {:.4},\n  \"flows_computed\": {},\n  \"flows_reused\": {},\n  \"flow_seconds\": {:.4},\n  \
         \"frame_cache_peak_bytes\": {},\n  \"frame_cache_spilled\": {},\n  \"buffer_peak_bytes\": {},\n  \
         \"buffer_spilled\": {},\n  \"estimated_memory_bytes\": {},\n  \"peak_rss_bytes\": {}\n}}\n",
        s.device,
        s.frames,
        s.syntheses,
        s.chains,
        s.parallel,
        s.seconds,
        s.synthesis_seconds,
        s.guide_seconds,
        s.flow.computed,
        s.flow.reused,
        s.flow.seconds,
        s.frame_cache.peak_bytes,
        s.frame_cache.spilled,
        s.buffers.peak_bytes,
        s.buffers.spilled,
        s.resources.estimated_bytes,
        peak.map_or("null".to_string(), |p| p.to_string()),
    )
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut project = Project::new("", "");
    let mut options = SessionOptions::default();
    let (mut save, mut stats_file, mut quiet) = (None, None, false);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if arg == "--help" || arg == "-h" {
            usage();
            return ExitCode::SUCCESS;
        }
        let mut value = || it.next().cloned();
        let ok = match arg.as_str() {
            "--project" => value().and_then(|p| Project::load(&p).map_err(|e| eprintln!("{e}")).ok()).map(|p| project = p),
            "--frames" => value().map(|v| project.media.frames = v),
            "--out" => value().map(|v| project.output.frames = v),
            "--mask" => value().map(|v| project.media.mask = Some(v)),
            "--key" => value().and_then(|v| {
                let (n, image) = v.split_once('=')?;
                project.add_keyframe(n.parse().ok()?, image);
                Some(())
            }),
            "--first" => value().and_then(|v| v.parse().ok()).map(|n| project.media.first = Some(n)),
            "--last" => value().and_then(|v| v.parse().ok()).map(|n| project.media.last = Some(n)),
            "--backend" => value().and_then(|v| v.parse().map_err(|e| eprintln!("{e}")).ok()).map(|b| project.engine.backend = b),
            "--preset" => value().and_then(|v| v.parse().map_err(|e| eprintln!("{e}")).ok()).map(|p| project.engine.preset = p),
            "--seed" => value().and_then(|v| v.parse().ok()).map(|n| project.engine.seed = n),
            "--mapping" => value().and_then(|v| v.parse().ok()).map(|w| project.guides.mapping = w),
            "--temporal" => value().and_then(|v| v.parse().ok()).map(|w| project.guides.temporal = w),
            "--video" => value().and_then(|v| v.parse().ok()).map(|w| project.guides.video = w),
            "--gate" => {
                project.guides.gate_mapping = true;
                Some(())
            }
            "--parallel" => value().and_then(|v| v.parse().ok()).map(|n| options.parallel_chains = Some(n)),
            "--memory-mb" => value().and_then(|v| v.parse::<usize>().ok()).map(|n| options.memory_budget = n << 20),
            "--frame-cache-mb" => value().and_then(|v| v.parse::<usize>().ok()).map(|n| options.frame_cache_bytes = n << 20),
            "--buffer-mb" => value().and_then(|v| v.parse::<usize>().ok()).map(|n| options.buffer_bytes = n << 20),
            "--cache-dir" => value().map(|v| options.cache_dir = Some(v.into())),
            "--flow-on-device" => {
                options.flow_backend = FlowBackend::Device;
                Some(())
            }
            // The default; still accepted.
            "--flow-on-cpu" => {
                options.flow_backend = FlowBackend::Cpu;
                Some(())
            }
            "--save-project" => value().map(|v| save = Some(v)),
            "--stats" => value().map(|v| stats_file = Some(v)),
            "--quiet" => {
                quiet = true;
                Some(())
            }
            _ => None,
        };
        if ok.is_none() {
            eprintln!("bad or incomplete argument '{arg}'");
            return usage();
        }
    }
    if let Err(e) = project.validate() {
        eprintln!("{e}");
        return usage();
    }
    if let Some(path) = &save
        && let Err(e) = project.save(path)
    {
        eprintln!("{e}");
        return ExitCode::from(2);
    }

    let opened = Instant::now();
    let backend = project.backend();
    let mut session = match Session::new(backend) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(3);
        }
    };
    let open_seconds = opened.elapsed().as_secs_f64();
    session.set_options(options);
    if !quiet {
        eprintln!("device: {} (opened in {open_seconds:.2} s)", session.device().name());
        if let Some(reason) = session.device().fallback_reason() {
            eprintln!("no GPU: {reason}");
        }
    }

    let last_report = AtomicUsize::new(0);
    let report = |p: &Progress| match p {
        Progress::Started { frames, syntheses, chains, parallel } => {
            eprintln!("{frames} frames, {syntheses} syntheses, {chains} chains ({parallel} at once)");
        }
        Progress::Written { done, total, .. } if done * 10 / total > last_report.load(Ordering::Relaxed) => {
            last_report.store(done * 10 / total, Ordering::Relaxed);
            eprintln!("{done}/{total} frames written");
        }
        _ => {}
    };
    let progress: Option<&(dyn Fn(&Progress) + Sync)> = if quiet { None } else { Some(&report) };
    let stats = match session.render(&project, progress, &CancelToken::new()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("render failed: {e}");
            return ExitCode::from(if e.is_cancelled() { 130 } else { 2 });
        }
    };
    let peak = peak_rss();
    println!(
        "{} frames ({} syntheses, {} chains, {} at once) in {:.2} s on {}: synthesis {:.1} s, guides {:.1} s, \
         {} flows computed ({} reused) in {:.1} s; frame cache peak {} MB ({} spilled), pass buffers peak {} MB \
         ({} spilled){}",
        stats.frames,
        stats.syntheses,
        stats.chains,
        stats.parallel,
        stats.seconds,
        stats.device,
        stats.synthesis_seconds,
        stats.guide_seconds,
        stats.flow.computed,
        stats.flow.reused,
        stats.flow.seconds,
        stats.frame_cache.peak_bytes >> 20,
        stats.frame_cache.spilled,
        stats.buffers.peak_bytes >> 20,
        stats.buffers.spilled,
        peak.map_or(String::new(), |p| format!("; peak RSS {} MB", p >> 20)),
    );
    if let Some(path) = stats_file
        && let Err(e) = std::fs::write(&path, stats_json(&stats, open_seconds, peak))
    {
        eprintln!("cannot write '{path}': {e}");
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}
