//! `freesynth-eval`: quality metrics and paired comparisons for FreeSynth results.
//!
//! Run without arguments for usage. Every measuring command writes one JSON report (to standard
//! output, or to `--json FILE`); `compare` and `table` summarise reports.

mod args;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use freesynth_core::{GuideLayer, Image, PatchErrorEvaluator, SynthesisInput, io};
use freesynth_eval::bootstrap::BootstrapConfig;
use freesynth_eval::compare::{Level, compare_reports};
use freesynth_eval::energy::{nnf_energy, sample_pixels, sampled_energy_with};
use freesynth_eval::frame::RgbImage;
use freesynth_eval::occupancy::{Occupancy, occupancy};
use freesynth_eval::report::{format_value, markdown_table};
use freesynth_eval::stats::metric_stats;
use freesynth_eval::sharpness::{DEFAULT_SIGMA, MAX_SIGMA, Sharpness, valid_sigma};
use freesynth_eval::temporal::{FlowDirs, FrameDir, FrameSource, FrameValue, TemporalOptions, temporal_metrics};
use freesynth_eval::{PixelSet, Report};

use crate::args::{Arg, Args};

const USAGE: &str = "\
usage: freesynth-eval <command> [options]

commands:
  energy      exact patch energy and source occupancy of a synthesis, from its NNF
  sampled     exact coherence energy of any image on random pixels (exhaustive search)
  occupancy   source occupancy statistics of an NNF
  sharpness   band-pass and high-frequency energy of an image or sequence relative to a reference
  temporal    warping error, second difference and flicker of frame sequences
  compare     paired bootstrap comparison of two sets of reports
  table       Markdown table of the metrics of several reports
  stats       mean and spread of every metric over repeated runs (run-to-run noise)

synthesis problem (energy, sampled), as for the freesynth command:
  -style <style.png> [-weight <w>]
  -guide <source.png> <target.png> [-weight <w>]   (one or more)
  -patchsize <size>                                 (default 5)
  -extrapass3x3                                     (the NNF came from the 3x3 extra pass)

energy:     --image <out.png> --nnf <out.nnf> [--error-map <map.png>]
sampled:    --image <out.png> [--samples <n>] [--seed <s>] [--pixels <report.json>]
            [--nnf <out.nnf>]     (seeds the search and reports the NNF's energy on the pixels)
            [--no-prune]          (plain scan without lower bounds: same result, slower)
occupancy:  --nnf <out.nnf> (-style <style.png> | --source-size <w>x<h>) [-patchsize <size>]
sharpness:  (--image <img.png> | --frames <dir>) (--reference <ref.png> | --reference-frames <dir>)
            [--sigma <s>] [--first <t>] [--last <t>]
temporal:   --frames [<name>=]<dir> (one or more) (--scene <dir> | --flow-fw <dir> --flow-bw <dir>)
            [--gaps 1,5] [--direct <gap> <fw-dir> <bw-dir>] [--first <t>] [--last <t>]
            [--out-dir <dir>]   (required for more than one sequence)
compare:    <a.json> <b.json> | --a <a1.json>... --b <b1.json>...
            [--metric <name>]... [--level samples|runs] [--resamples <n>] [--seed <s>]
            [--confidence <c>]
            JSON goes to --json or standard output; Markdown to --markdown, else standard error
table:      <report.json>... [--metric <name>]...   (Markdown to --markdown or standard output)
stats:      <report.json>...                        (JSON as above; Markdown to --markdown,
                                                     else standard error)

common options (measuring commands; compare takes --json and --markdown, table --markdown,
stats --json, --markdown and --label):
  --json <file>       write the JSON result to a file instead of standard output
  --markdown <file>   also write a Markdown summary
  --label <text>      name of the measured item in reports and tables
  --threads <n>       worker threads (default: all cores)

Frames are NNNNN.png files (5 digits). Flows are Middlebury .flo files: flow-fw/NNNNN.flo is the
flow from frame N to N+1; flow-bw holds the flow back from N+1 to N, as NNNNN.flo or as
(N+1).flo (detected). --scene <dir> takes flow_fw/, flow_bw/ and, for a gap G, flow_fw_G/ and
flow_bw_G/ from a scene directory; gaps without direct flows chain the 1-step flows. The flow
must come from the input video and be independent of the method being measured. Options may be
written with one or two leading dashes.";

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || matches!(argv[0].as_str(), "-h" | "--help" | "help") {
        println!("{USAGE}");
        return if argv.is_empty() { ExitCode::from(2) } else { ExitCode::SUCCESS };
    }
    match run(&argv[0], Args::new(argv[1..].to_vec())) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

type Res<T> = Result<T, String>;

fn run(command: &str, args: Args) -> Res<()> {
    match command {
        "energy" => cmd_energy(args),
        "sampled" => cmd_sampled(args),
        "occupancy" => cmd_occupancy(args),
        "sharpness" => cmd_sharpness(args),
        "temporal" => cmd_temporal(args),
        "compare" => cmd_compare(args),
        "table" => cmd_table(args),
        "stats" => cmd_stats(args),
        _ => Err(format!("unknown command '{command}' (run without arguments for usage)")),
    }
}

/// Options shared by all commands.
#[derive(Default)]
struct Common {
    json: Option<PathBuf>,
    markdown: Option<PathBuf>,
    label: Option<String>,
}

/// The common options a command accepts.
const ALL_COMMON: &[&str] = &["json", "markdown", "label", "threads"];

impl Common {
    /// Consumes a common option; returns whether `name` was one.
    fn parse(&mut self, name: &str, args: &mut Args) -> Res<bool> {
        self.parse_only(name, args, "", ALL_COMMON)
    }

    /// Like [`parse`](Self::parse), for a command that accepts only the common options in
    /// `allowed`: the others are rejected instead of silently ignored.
    fn parse_only(&mut self, name: &str, args: &mut Args, command: &str, allowed: &[&str]) -> Res<bool> {
        if ALL_COMMON.contains(&name) && !allowed.contains(&name) {
            return Err(format!("{command} does not take --{name}"));
        }
        match name {
            "json" => self.json = Some(args.value(name)?.into()),
            "markdown" => self.markdown = Some(args.value(name)?.into()),
            "label" => self.label = Some(args.value(name)?),
            "threads" => {
                let n: usize = args.parse(name)?;
                if n == 0 {
                    return Err("--threads must be at least 1".into());
                }
                rayon::ThreadPoolBuilder::new().num_threads(n).build_global().map_err(|e| format!("cannot set up threads: {e}"))?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Writes `report` as JSON (to `--json` or standard output) and Markdown (to `--markdown`).
    fn emit(&self, mut report: Report) -> Res<()> {
        if report.label.is_none() {
            report.label = self.label.clone();
        }
        match &self.json {
            Some(p) => report.save(p).map_err(|e| format!("cannot write '{}': {e}", p.display()))?,
            None => println!("{}", report.to_json()),
        }
        if let Some(p) = &self.markdown {
            write_text(p, &report.to_markdown())?;
        }
        Ok(())
    }
}

fn write_text(path: &Path, text: &str) -> Res<()> {
    std::fs::write(path, text).map_err(|e| format!("cannot write '{}': {e}", path.display()))
}

fn load_rgba(path: &str) -> Res<Image> {
    io::load_rgba(path).map_err(|e| format!("cannot load '{path}': {e}"))
}

/// The synthesis problem given by freesynth-style layer options.
#[derive(Default)]
struct Problem {
    style: Option<String>,
    style_weight: Option<f32>,
    guides: Vec<(String, String, Option<f32>)>,
    patch_size: Option<usize>,
    extra_pass_3x3: bool,
    /// The layer a following `-weight` applies to.
    weight_target: Option<Layer>,
}

/// A layer option, for `-weight`.
#[derive(Clone, Copy)]
enum Layer {
    Style,
    Guide,
}

impl Problem {
    fn parse(&mut self, name: &str, args: &mut Args) -> Res<bool> {
        match name {
            "style" => {
                self.style = Some(args.value(name)?);
                self.weight_target = Some(Layer::Style);
            }
            "guide" => {
                let source = args.value(name)?;
                let target = args.value(name)?;
                self.guides.push((source, target, None));
                self.weight_target = Some(Layer::Guide);
            }
            "weight" => {
                let w: f32 = args.parse(name)?;
                if !(w >= 0.0 && w.is_finite()) {
                    return Err("weights must be non-negative".into());
                }
                match self.weight_target {
                    Some(Layer::Style) => self.style_weight = Some(w),
                    Some(Layer::Guide) => self.guides.last_mut().expect("a guide was given").2 = Some(w),
                    None => return Err("-weight must follow -style or -guide".into()),
                }
            }
            "patchsize" => {
                let p: usize = args.parse(name)?;
                if p < 3 || p.is_multiple_of(2) {
                    return Err("patch size must be an odd number >= 3".into());
                }
                self.patch_size = Some(p);
            }
            "extrapass3x3" => self.extra_pass_3x3 = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The patch size of the field being evaluated.
    fn patch(&self) -> usize {
        if self.extra_pass_3x3 { 3 } else { self.patch_size.unwrap_or(5) }
    }

    fn build(&self) -> Res<SynthesisInput> {
        let style = load_rgba(self.style.as_deref().ok_or("-style is required")?)?;
        if self.guides.is_empty() {
            return Err("at least one -guide is required".into());
        }
        let mut layers = Vec::with_capacity(self.guides.len());
        for (s, t, w) in &self.guides {
            let mut layer = GuideLayer::new(load_rgba(s)?, load_rgba(t)?);
            if let Some(w) = w {
                layer = layer.with_weight(*w);
            }
            layers.push(layer);
        }
        SynthesisInput::from_rgba_layers(&style, self.style_weight, &layers).map_err(|e| e.to_string())
    }

    fn record(&self, report: &mut Report) {
        if let Some(s) = &self.style {
            report.input("style", s.as_str());
        }
        if let Some(w) = self.style_weight {
            report.input("style_weight", w);
        }
        let guides: Vec<serde_json::Value> = self
            .guides
            .iter()
            .map(|(s, t, w)| serde_json::json!({ "source": s, "target": t, "weight": w }))
            .collect();
        report.input("guides", guides);
        report.input("patch_size", self.patch());
    }
}

/// Loads an output image with the style's channel selection.
fn load_target_style(path: &str, input: &SynthesisInput) -> Res<Image> {
    let img = load_rgba(path)?;
    Ok(io::select_channels(&img, input.source_style.channels()))
}

fn load_nnf(path: &str) -> Res<freesynth_core::Nnf> {
    io::load_nnf(path).map_err(|e| format!("cannot load '{path}': {e}"))
}

fn add_occupancy(report: &mut Report, o: &Occupancy) {
    report
        .metric("unused_frac", o.unused_frac)
        .metric("occupancy_cv", o.cv)
        .metric("norm_entropy", o.norm_entropy)
        .metric("max_over_mean", o.max_over_mean)
        .metric("p99_over_mean", o.p99_over_mean)
        .metric("unique_centres_frac", o.unique_centres_frac)
        .metric("same_offset_frac", o.same_offset_frac);
}

fn cmd_energy(mut args: Args) -> Res<()> {
    let (mut common, mut problem) = (Common::default(), Problem::default());
    let (mut image, mut nnf_path, mut error_map) = (None, None, None);
    while let Some(arg) = args.next() {
        let Arg::Opt(name) = arg else { return Err(format!("unexpected argument '{}'", arg.text())) };
        if common.parse(&name, &mut args)? || problem.parse(&name, &mut args)? {
            continue;
        }
        match name.as_str() {
            "image" => image = Some(args.value(&name)?),
            "nnf" => nnf_path = Some(args.value(&name)?),
            "error-map" => error_map = Some(args.value(&name)?),
            _ => return Err(format!("unknown option '{name}' for energy")),
        }
    }
    let image = image.ok_or("--image is required")?;
    let nnf_path = nnf_path.ok_or("--nnf is required")?;
    let input = problem.build()?;
    let target = load_target_style(&image, &input)?;
    let nnf = load_nnf(&nnf_path)?;
    let evaluator = PatchErrorEvaluator::new(&input, &target, problem.patch()).map_err(|e| e.to_string())?;
    let e = nnf_energy(&evaluator, &nnf).map_err(|e| e.to_string())?;
    let occ = occupancy(&nnf, input.source_style.size(), problem.patch()).map_err(|e| e.to_string())?;

    let mut report = Report::new("energy");
    problem.record(&mut report);
    report.input("image", image.as_str()).input("nnf", nnf_path.as_str());
    report
        .metric("style", e.style)
        .metric("guide", e.guide)
        .metric("total", e.total)
        .metric("total_p50", e.total_p50)
        .metric("total_p90", e.total_p90)
        .metric("total_p99", e.total_p99);
    add_occupancy(&mut report, &occ);
    if let Some(path) = error_map {
        // sqrt of the per-pixel energy, 4 grey levels per unit (255 = energy 4064).
        let (w, h) = evaluator.target_size();
        let data = e.per_pixel.iter().map(|v| (v.sqrt() * 4.0).round().min(255.0) as u8).collect();
        io::save_png(&Image::from_raw(w, h, 1, data), &path).map_err(|e| format!("cannot write '{path}': {e}"))?;
        report.input("error_map", path.as_str());
    }
    common.emit(report)
}

fn cmd_sampled(mut args: Args) -> Res<()> {
    let (mut common, mut problem) = (Common::default(), Problem::default());
    let (mut image, mut nnf_path, mut pixels_path) = (None, None, None);
    let (mut samples, mut seed, mut prune) = (20_000usize, 1u64, true);
    while let Some(arg) = args.next() {
        let Arg::Opt(name) = arg else { return Err(format!("unexpected argument '{}'", arg.text())) };
        if common.parse(&name, &mut args)? || problem.parse(&name, &mut args)? {
            continue;
        }
        match name.as_str() {
            "image" => image = Some(args.value(&name)?),
            "nnf" => nnf_path = Some(args.value(&name)?),
            "pixels" => pixels_path = Some(args.value(&name)?),
            "samples" => samples = args.parse(&name)?,
            "seed" => seed = args.parse(&name)?,
            "no-prune" => prune = false,
            _ => return Err(format!("unknown option '{name}' for sampled")),
        }
    }
    let image = image.ok_or("--image is required")?;
    let input = problem.build()?;
    let target = load_target_style(&image, &input)?;
    let (tw, th) = target.size();
    let pixels: PixelSet = match &pixels_path {
        Some(p) => Report::load(p).map_err(|e| e.to_string())?.pixels.ok_or(format!("'{p}' holds no pixel set"))?,
        None => {
            if samples == 0 {
                return Err("--samples must be at least 1".into());
            }
            sample_pixels(tw, th, samples, seed)
        }
    };
    let hint = nnf_path.as_deref().map(load_nnf).transpose()?;
    let evaluator = PatchErrorEvaluator::new(&input, &target, problem.patch()).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let s = sampled_energy_with(&evaluator, &pixels, hint.as_ref(), prune).map_err(|e| e.to_string())?;
    let elapsed = start.elapsed().as_secs_f64();

    let mut report = Report::new("sampled");
    problem.record(&mut report);
    report.input("image", image.as_str()).input("pixel_set", pixels.key.as_str()).input("samples", pixels.coords.len());
    if let Some(p) = &nnf_path {
        report.input("nnf", p.as_str());
    }
    report
        .metric("total", s.total)
        .metric("style", s.style)
        .metric("guide", s.guide)
        .metric("total_se", s.total_se)
        .metric("style_se", s.style_se)
        .metric("time_s", elapsed);
    if let (Some(h), Some(f)) = (s.hint_total, s.hint_suboptimal_frac) {
        report.metric("nnf_total", h).metric("nnf_suboptimal_frac", f);
    }
    let key = pixels.key.clone();
    let totals: Vec<f64> = s.matches.iter().map(|m| m.total as f64).collect();
    let styles: Vec<f64> = s.matches.iter().map(|m| m.style as f64).collect();
    let guides: Vec<f64> = totals.iter().zip(&styles).map(|(t, s)| t - s).collect();
    report.samples("total", key.as_str(), totals).samples("style", key.as_str(), styles).samples("guide", key.as_str(), guides);
    report.pixels = Some(pixels);
    common.emit(report)
}

fn parse_size(s: &str) -> Res<(usize, usize)> {
    let (w, h) = s.split_once(['x', 'X']).ok_or(format!("bad size '{s}' (expected <w>x<h>)"))?;
    let p = |v: &str| v.trim().parse::<usize>().map_err(|_| format!("bad size '{s}'"));
    Ok((p(w)?, p(h)?))
}

fn cmd_occupancy(mut args: Args) -> Res<()> {
    let mut common = Common::default();
    let (mut nnf_path, mut style, mut size, mut patch) = (None, None, None, 5usize);
    while let Some(arg) = args.next() {
        let Arg::Opt(name) = arg else { return Err(format!("unexpected argument '{}'", arg.text())) };
        if common.parse(&name, &mut args)? {
            continue;
        }
        match name.as_str() {
            "nnf" => nnf_path = Some(args.value(&name)?),
            "style" => style = Some(args.value(&name)?),
            "source-size" => size = Some(parse_size(&args.value(&name)?)?),
            "patchsize" => patch = args.parse(&name)?,
            "extrapass3x3" => patch = 3,
            _ => return Err(format!("unknown option '{name}' for occupancy")),
        }
    }
    let nnf_path = nnf_path.ok_or("--nnf is required")?;
    let size = match (size, &style) {
        (Some(s), _) => s,
        (None, Some(p)) => load_rgba(p)?.size(),
        (None, None) => return Err("give -style or --source-size".into()),
    };
    let nnf = load_nnf(&nnf_path)?;
    let occ = occupancy(&nnf, size, patch).map_err(|e| e.to_string())?;
    let mut report = Report::new("occupancy");
    report.input("nnf", nnf_path.as_str()).input("source_size", vec![size.0, size.1]).input("patch_size", patch);
    add_occupancy(&mut report, &occ);
    common.emit(report)
}

/// Frames `first..=last` of a directory, clipped to its contiguous range.
fn frame_range(dir: &FrameDir, first: Option<usize>, last: Option<usize>) -> Res<(usize, usize)> {
    let (lo, hi) = dir.range();
    let (first, last) = (first.unwrap_or(lo), last.unwrap_or(hi));
    if first < lo || last > hi || first > last {
        return Err(format!("frames {first}-{last} are not all present (found {lo}-{hi})"));
    }
    Ok((first, last))
}

fn cmd_sharpness(mut args: Args) -> Res<()> {
    let mut common = Common::default();
    let (mut image, mut frames, mut reference, mut reference_frames) = (None, None, None, None);
    let (mut sigma, mut first, mut last) = (DEFAULT_SIGMA, None, None);
    while let Some(arg) = args.next() {
        let Arg::Opt(name) = arg else { return Err(format!("unexpected argument '{}'", arg.text())) };
        if common.parse(&name, &mut args)? {
            continue;
        }
        match name.as_str() {
            "image" => image = Some(args.value(&name)?),
            "frames" => frames = Some(args.value(&name)?),
            "reference" => reference = Some(args.value(&name)?),
            "reference-frames" => reference_frames = Some(args.value(&name)?),
            "sigma" => sigma = args.parse(&name)?,
            "first" => first = Some(args.parse(&name)?),
            "last" => last = Some(args.parse(&name)?),
            _ => return Err(format!("unknown option '{name}' for sharpness")),
        }
    }
    if !valid_sigma(sigma) {
        return Err(format!("--sigma must be a number in (0, {MAX_SIGMA}]"));
    }
    let load = |p: &str| RgbImage::load(p).map_err(|e| e.to_string());
    let measure = |img: &RgbImage| Sharpness::of(img, sigma);
    // Ratios are undefined against a reference without any detail.
    let ratio = |out: &Sharpness, refs: &Sharpness, what: &str| out.ratio(refs).ok_or(format!("the reference {what} is flat (no detail to compare with)"));
    let mut report = Report::new("sharpness");
    report.input("sigma", sigma);
    match (&image, &frames) {
        (Some(path), None) => {
            let out = measure(&load(path)?);
            let refp = reference.as_deref().ok_or("--reference is required with --image")?;
            let refs = measure(&load(refp)?);
            let r = ratio(&out, &refs, "image")?;
            report.input("image", path.as_str()).input("reference", refp);
            report
                .metric("band_ratio", r.band)
                .metric("hf_ratio", r.hf)
                .metric("gradient_ratio", r.gradient)
                .metric("band", out.band)
                .metric("hf", out.hf)
                .metric("gradient", out.gradient)
                .metric("band_reference", refs.band)
                .metric("hf_reference", refs.hf)
                .metric("gradient_reference", refs.gradient);
        }
        (None, Some(dir)) => {
            let seq = FrameDir::open(dir).map_err(|e| e.to_string())?;
            let (first, last) = frame_range(&seq, first, last)?;
            let fixed = reference.as_deref().map(|p| load(p).map(|i| measure(&i))).transpose()?;
            let ref_dir = reference_frames.as_deref().map(FrameDir::open).transpose().map_err(|e| e.to_string())?;
            if fixed.is_some() == ref_dir.is_some() {
                return Err("give exactly one of --reference and --reference-frames".into());
            }
            let (mut band, mut hf, mut grad) = (Vec::new(), Vec::new(), Vec::new());
            for t in first..=last {
                let out = measure(&seq.frame(t).map_err(|e| e.to_string())?);
                let refs = match (&fixed, &ref_dir) {
                    (Some(s), _) => *s,
                    (None, Some(d)) => measure(&d.frame(t).map_err(|e| e.to_string())?),
                    (None, None) => unreachable!("checked above"),
                };
                let r = ratio(&out, &refs, &format!("for frame {t}"))?;
                band.push(r.band);
                hf.push(r.hf);
                grad.push(r.gradient);
            }
            report.input("frames", dir.as_str()).input("first", first).input("last", last);
            match (&reference, &reference_frames) {
                (Some(p), _) => report.input("reference", p.as_str()),
                (_, Some(d)) => report.input("reference_frames", d.as_str()),
                _ => unreachable!("checked above"),
            };
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            report
                .metric("band_ratio", mean(&band))
                .metric("band_ratio_first", band[0])
                .metric("band_ratio_last", *band.last().expect("at least one frame"))
                .metric("hf_ratio", mean(&hf))
                .metric("hf_ratio_first", hf[0])
                .metric("hf_ratio_last", *hf.last().expect("at least one frame"))
                .metric("gradient_ratio", mean(&grad))
                .metric("gradient_ratio_last", *grad.last().expect("at least one frame"));
            let key = format!("frames/{first}-{last}");
            report
                .samples("band_ratio", key.as_str(), band)
                .samples("hf_ratio", key.as_str(), hf)
                .samples("gradient_ratio", key.as_str(), grad);
        }
        _ => return Err("give exactly one of --image and --frames".into()),
    }
    common.emit(report)
}

/// Mean and per-unit values of per-frame results, with a key that names the measured units.
fn frame_samples(values: &[FrameValue], key: &str) -> (Option<f64>, String, Vec<f64>) {
    let kept: Vec<f64> = values.iter().filter_map(|v| v.value).collect();
    let skipped: Vec<String> = values.iter().filter(|v| v.value.is_none()).map(|v| v.t.to_string()).collect();
    let key = if skipped.is_empty() { key.to_string() } else { format!("{key}/without:{}", skipped.join(",")) };
    let mean = (!kept.is_empty()).then(|| kept.iter().sum::<f64>() / kept.len() as f64);
    (mean, key, kept)
}

/// Whether `label` is usable as a sequence name and report file name.
fn valid_label(label: &str) -> bool {
    !label.is_empty() && !label.chars().all(|c| c == '.') && label.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Parses a `--frames [NAME=]DIR` value into (label, directory).
///
/// Without a name, the label is the directory's name, or its parent's name for the usual
/// `<run>/frames` layout, with characters other than letters, digits, `.`, `_` and `-` replaced
/// by `_`.
fn sequence_arg(v: &str) -> Res<(String, String)> {
    if let Some((label, dir)) = v.split_once('=')
        && !label.contains(['/', '\\'])
    {
        if !valid_label(label) {
            return Err(format!("bad sequence name '{label}' (use letters, digits, '.', '_' and '-')"));
        }
        return Ok((label.to_string(), dir.to_string()));
    }
    let path = Path::new(v);
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
    let base = match name(path) {
        Some(n) if n == "frames" => path.parent().and_then(name).unwrap_or(n),
        other => other.unwrap_or_default(),
    };
    let label: String = base.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' }).collect();
    if !valid_label(&label) {
        return Err(format!("cannot name the sequence in '{v}'; name it with --frames NAME=DIR"));
    }
    Ok((label, v.to_string()))
}

fn cmd_temporal(mut args: Args) -> Res<()> {
    let mut common = Common::default();
    let mut sequences: Vec<(String, String)> = Vec::new();
    let (mut fw, mut bw, mut out_dir, mut scene) = (None, None, None, None);
    let mut direct: Vec<(usize, String, String)> = Vec::new();
    let (mut gaps, mut first, mut last) = (vec![1usize, 5], None, None);
    while let Some(arg) = args.next() {
        let Arg::Opt(name) = arg else { return Err(format!("unexpected argument '{}'", arg.text())) };
        if common.parse(&name, &mut args)? {
            continue;
        }
        match name.as_str() {
            "frames" => sequences.push(sequence_arg(&args.value(&name)?)?),
            "scene" => scene = Some(PathBuf::from(args.value(&name)?)),
            "flow-fw" => fw = Some(args.value(&name)?),
            "flow-bw" => bw = Some(args.value(&name)?),
            "direct" => {
                let gap: usize = args.parse(&name)?;
                direct.push((gap, args.value(&name)?, args.value(&name)?));
            }
            "gaps" => {
                gaps = args
                    .value(&name)?
                    .split(',')
                    .map(|g| g.trim().parse::<usize>().ok().filter(|&g| g > 0).ok_or(format!("bad gap '{g}'")))
                    .collect::<Res<_>>()?;
            }
            "first" => first = Some(args.parse(&name)?),
            "last" => last = Some(args.parse(&name)?),
            "out-dir" => out_dir = Some(PathBuf::from(args.value(&name)?)),
            _ => return Err(format!("unknown option '{name}' for temporal")),
        }
    }
    if sequences.is_empty() {
        return Err("at least one --frames is required".into());
    }
    if sequences.len() > 1 && out_dir.is_none() {
        return Err("--out-dir is required for more than one sequence".into());
    }
    for (i, (label, _)) in sequences.iter().enumerate() {
        if sequences[..i].iter().any(|(l, _)| l == label) {
            return Err(format!("two sequences are labelled '{label}'; name them with --frames NAME=DIR"));
        }
    }
    if let Some(dir) = &scene {
        // A scene directory: flow_fw/ and flow_bw/, plus flow_fw_G/ and flow_bw_G/ for gap G.
        let sub = |name: &str| dir.join(name).to_string_lossy().into_owned();
        fw.get_or_insert_with(|| sub("flow_fw"));
        bw.get_or_insert_with(|| sub("flow_bw"));
        for &g in gaps.iter().filter(|&&g| g > 1) {
            let (f, b) = (dir.join(format!("flow_fw_{g}")), dir.join(format!("flow_bw_{g}")));
            if f.is_dir() && b.is_dir() && !direct.iter().any(|d| d.0 == g) {
                direct.push((g, f.to_string_lossy().into_owned(), b.to_string_lossy().into_owned()));
            }
        }
    }
    let (fw, bw) = (fw.ok_or("--flow-fw is required")?, bw.ok_or("--flow-bw is required")?);
    let mut flows = FlowDirs::open(&fw, &bw).map_err(|e| e.to_string())?;
    for (gap, f, b) in &direct {
        flows = flows.with_direct(*gap, f, b).map_err(|e| e.to_string())?;
    }
    let dirs: Vec<FrameDir> = sequences.iter().map(|(_, d)| FrameDir::open(d)).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
    // Measure the frames that every sequence has.
    let found = || sequences.iter().zip(&dirs).map(|((l, _), d)| format!("{l}: {}-{}", d.range().0, d.range().1)).collect::<Vec<_>>().join(", ");
    let (lo, hi) = dirs.iter().map(FrameDir::range).fold((0, usize::MAX), |(lo, hi), (a, b)| (lo.max(a), hi.min(b)));
    if lo > hi {
        return Err(format!("the sequences have no frames in common (found {})", found()));
    }
    let range = (first.unwrap_or(lo), last.unwrap_or(hi));
    if range.0 < lo || range.1 > hi || range.0 > range.1 {
        return Err(format!("frames {}-{} are not present in every sequence (common frames {lo}-{hi}; found {})", range.0, range.1, found()));
    }
    let opts = TemporalOptions { first: range.0, last: range.1, gaps: gaps.clone() };
    let sources: Vec<&dyn FrameSource> = dirs.iter().map(|d| d as &dyn FrameSource).collect();
    let result = temporal_metrics(&sources, &flows, &opts).map_err(|e| e.to_string())?;

    let mut reports = Vec::new();
    for ((label, dir), seq) in sequences.iter().zip(&result.sequences) {
        let mut report = Report::new("temporal");
        report.label = Some(common.label.clone().filter(|_| sequences.len() == 1).unwrap_or(label.clone()));
        report
            .input("frames", dir.as_str())
            .input("flow_fw", fw.as_str())
            .input("flow_bw", bw.as_str())
            .input("flow_bw_convention", flows.backward_convention())
            .input("direct_flow_gaps", direct.iter().map(|d| d.0).collect::<Vec<_>>())
            .input("first", opts.first)
            .input("last", opts.last)
            .input("gaps", opts.gaps.clone());
        for (g, values) in &seq.warp {
            let (mean, key, kept) = frame_samples(values, &format!("pairs/gap{g}/{}-{}/flow={fw}", opts.first, opts.last));
            if let Some(m) = mean {
                report.metric(&format!("warp{g}"), m);
            }
            report.samples(&format!("warp{g}"), key, kept);
        }
        for (g, fr) in &result.mask_frac {
            report.metric(&format!("mask_frac{g}"), fr.iter().sum::<f64>() / fr.len() as f64);
        }
        let (mean, key, kept) = frame_samples(&seq.second_diff, &format!("frames/{}-{}/flow={fw}", opts.first + 1, opts.last - 1));
        if let Some(m) = mean {
            report.metric("second_diff", m);
        }
        if !kept.is_empty() {
            report.samples("second_diff", key, kept);
        }
        if let Some(f) = seq.global_flicker() {
            report.metric("global_flicker", f);
        }
        reports.push(report);
    }
    match &out_dir {
        None => common.emit(reports.pop().expect("one sequence")),
        Some(dir) => {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create '{}': {e}", dir.display()))?;
            let mut named = Vec::new();
            for ((name, _), r) in sequences.iter().zip(reports) {
                // Named by the checked sequence label, not by a free-form --label.
                let path = dir.join(format!("{name}.json"));
                r.save(&path).map_err(|e| format!("cannot write '{}': {e}", path.display()))?;
                named.push((name.clone(), r));
            }
            let table = markdown_table(&named, &[]);
            println!("{table}");
            if let Some(p) = &common.markdown {
                write_text(p, &table)?;
            }
            Ok(())
        }
    }
}

fn load_reports(paths: &[String]) -> Res<Vec<(String, Report)>> {
    paths.iter().map(|p| Report::load(p).map(|r| (p.clone(), r)).map_err(|e| e.to_string())).collect()
}

fn cmd_compare(mut args: Args) -> Res<()> {
    let mut common = Common::default();
    let (mut positional, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
    let mut metrics = Vec::new();
    let mut level = None;
    let mut cfg = BootstrapConfig::default();
    // Which list bare file names go to after --a / --b.
    let mut side: Option<bool> = None;
    while let Some(arg) = args.next() {
        let name = match arg {
            Arg::Value(v) => {
                match side {
                    Some(true) => a.push(v),
                    Some(false) => b.push(v),
                    None => positional.push(v),
                }
                continue;
            }
            Arg::Opt(name) => name,
        };
        if common.parse_only(&name, &mut args, "compare", &["json", "markdown"])? {
            side = None;
            continue;
        }
        match name.as_str() {
            "a" => side = Some(true),
            "b" => side = Some(false),
            "metric" => metrics.push(args.value(&name)?),
            "level" => {
                level = Some(match args.value(&name)?.as_str() {
                    "samples" => Level::Samples,
                    "runs" => Level::Runs,
                    other => return Err(format!("bad level '{other}' (samples or runs)")),
                })
            }
            "resamples" => cfg.resamples = args.parse(&name)?,
            "seed" => cfg.seed = args.parse(&name)?,
            "confidence" => cfg.confidence = args.parse(&name)?,
            _ => return Err(format!("unknown option '{name}' for compare")),
        }
        if !matches!(name.as_str(), "a" | "b") {
            side = None;
        }
    }
    if !positional.is_empty() {
        if positional.len() != 2 || !a.is_empty() || !b.is_empty() {
            return Err("give two reports, or --a and --b lists".into());
        }
        b.push(positional.pop().expect("two"));
        a.push(positional.pop().expect("two"));
    }
    let comparison = compare_reports(&load_reports(&a)?, &load_reports(&b)?, &metrics, level, &cfg).map_err(|e| e.to_string())?;
    match &common.json {
        Some(p) => write_text(p, &(comparison.to_json() + "\n"))?,
        None => println!("{}", comparison.to_json()),
    }
    let md = comparison.to_markdown();
    match &common.markdown {
        Some(p) => write_text(p, &md)?,
        None => eprintln!("{md}"),
    }
    Ok(())
}

fn cmd_table(mut args: Args) -> Res<()> {
    let mut common = Common::default();
    let (mut files, mut metrics) = (Vec::new(), Vec::new());
    while let Some(arg) = args.next() {
        match arg {
            Arg::Value(v) => files.push(v),
            Arg::Opt(name) => {
                if common.parse_only(&name, &mut args, "table", &["markdown"])? {
                    continue;
                }
                match name.as_str() {
                    "metric" => metrics.push(args.value(&name)?),
                    _ => return Err(format!("unknown option '{name}' for table")),
                }
            }
        }
    }
    if files.is_empty() {
        return Err("give at least one report".into());
    }
    let table = markdown_table(&load_reports(&files)?, &metrics);
    match &common.markdown {
        Some(p) => write_text(p, &table),
        None => {
            println!("{table}");
            Ok(())
        }
    }
}

fn cmd_stats(mut args: Args) -> Res<()> {
    let mut common = Common::default();
    let mut files = Vec::new();
    while let Some(arg) = args.next() {
        match arg {
            Arg::Value(v) => files.push(v),
            Arg::Opt(name) => {
                if !common.parse_only(&name, &mut args, "stats", &["json", "markdown", "label"])? {
                    return Err(format!("unknown option '{name}' for stats"));
                }
            }
        }
    }
    if files.len() < 2 {
        return Err("give at least two reports".into());
    }
    let reports = load_reports(&files)?;
    let kind = reports[0].1.kind.clone();
    if let Some((name, r)) = reports.iter().find(|(_, r)| r.kind != kind) {
        return Err(format!("'{name}' is a {} report, expected {kind}", r.kind));
    }
    let stats = metric_stats(&reports.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>());
    let mut report = Report::new("stats");
    report.input("of", kind.as_str()).input("runs", files.clone());
    let mut md = format!("Spread over {} {kind} reports.\n\n| metric | n | mean | sd | cv | min | max |\n|---|---:|---:|---:|---:|---:|---:|\n", files.len());
    for m in &stats {
        report
            .metric(&format!("{}_mean", m.metric), m.mean)
            .metric(&format!("{}_sd", m.metric), m.sd)
            .metric(&format!("{}_cv", m.metric), m.cv);
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.2}% | {} | {} |\n",
            m.metric,
            m.n,
            format_value(m.mean),
            format_value(m.sd),
            100.0 * m.cv,
            format_value(m.min),
            format_value(m.max)
        ));
    }
    if report.label.is_none() {
        report.label = common.label.clone();
    }
    match &common.json {
        Some(p) => report.save(p).map_err(|e| format!("cannot write '{}': {e}", p.display()))?,
        None => println!("{}", report.to_json()),
    }
    match &common.markdown {
        Some(p) => write_text(p, &md),
        None => {
            eprintln!("{md}");
            Ok(())
        }
    }
}
