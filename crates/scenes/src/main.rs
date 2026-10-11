//! `freesynth-scenes`: generates procedural test scenes and checks scene directories.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use freesynth_scenes::output::{flow_path, frame_path, occlusion_path, truth_path};
use freesynth_scenes::{Direction, FlowField, Manifest, SceneInfo, SceneParams, StyleFamily, Tier, write_scene};
use image::RgbImage;

const USAGE: &str = "\
usage:
  freesynth-scenes generate <out-dir> [options]
      --family <rigid|painterly|hatching>   style family (default rigid)
      --seed <n>                            random seed (default 1)
      --size <W>x<H>                        frame size (default 480x270)
      --frames <n>                          number of frames (default 30)
      --keys <a,b,...>                      keyframe indices (default first and last)
      --objects <n>                         moving objects (default 3)
      --motion <x>                          speed multiplier (default 1)
      --gaps <a,b,...>                      extra flow gaps, e.g. 5 (gap 1 is always written)
      --name <name>                         scene name (default: the directory name)
  freesynth-scenes corpus <manifest.toml> <out-dir> [--tier ci|full|all] [--only <name>]...
      generates manifest scenes into <out-dir>/<name>/ (default tier: ci)
  freesynth-scenes check <scene-dir>
      warps every frame pair with its flow and reports the error on unoccluded pixels
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("generate") => generate(&args[1..]),
        Some("corpus") => corpus(&args[1..]),
        Some("check") => check(&args[1..]),
        Some("-h" | "--help" | "help") => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        _ => Err(format!("missing or unknown command\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Positional arguments and `--flag value` pairs.
type SplitArgs<'a> = (Vec<&'a str>, Vec<(&'a str, &'a str)>);

/// Splits arguments into positionals and `--flag value` pairs.
fn split_args(args: &[String]) -> Result<SplitArgs<'_>, String> {
    let (mut positional, mut flags) = (Vec::new(), Vec::new());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(flag) = a.strip_prefix("--") {
            let value = it.next().ok_or_else(|| format!("--{flag} needs a value"))?;
            flags.push((flag, value.as_str()));
        } else {
            positional.push(a.as_str());
        }
    }
    Ok((positional, flags))
}

fn parse<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("invalid value '{value}' for --{flag}"))
}

fn parse_list(flag: &str, value: &str) -> Result<Vec<usize>, String> {
    value.split(',').filter(|s| !s.is_empty()).map(|s| parse(flag, s.trim())).collect()
}

fn generate(args: &[String]) -> Result<(), String> {
    let (positional, flags) = split_args(args)?;
    let [out] = positional[..] else {
        return Err(format!("generate takes one output directory\n\n{USAGE}"));
    };
    let mut params = SceneParams::default();
    let mut name = None;
    for (flag, value) in flags {
        match flag {
            "family" => params.family = value.parse::<StyleFamily>()?,
            "seed" => params.seed = parse(flag, value)?,
            "size" => {
                let (w, h) = value.split_once(['x', 'X']).ok_or_else(|| format!("--size expects WxH, got '{value}'"))?;
                params.width = parse(flag, w)?;
                params.height = parse(flag, h)?;
            }
            "frames" => params.frames = parse(flag, value)?,
            "keys" => params.keyframes = parse_list(flag, value)?,
            "objects" => params.objects = parse(flag, value)?,
            "motion" => params.motion = parse(flag, value)?,
            "gaps" => params.flow_gaps = parse_list(flag, value)?,
            "name" => name = Some(value.to_owned()),
            _ => return Err(format!("unknown option --{flag}\n\n{USAGE}")),
        }
    }
    let out = PathBuf::from(out);
    let name = match name {
        Some(n) => n,
        // `std::path::absolute` keeps `..` but drops a trailing `.`, so `generate .` is named
        // after the current directory.
        None => std::path::absolute(&out)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .filter(|n| !n.is_empty())
            .ok_or_else(|| format!("cannot name the scene after '{}'; give --name", out.display()))?,
    };
    generate_one(&params, &name, &out)
}

fn generate_one(params: &SceneParams, name: &str, out: &Path) -> Result<(), String> {
    let start = Instant::now();
    let info = write_scene(params, name, out).map_err(|e| format!("{}: {e}", out.display()))?;
    println!(
        "{}: {}x{}, {} frames, {} family, keys {:?}, flow gaps {:?} ({:.2} s)",
        out.display(),
        info.width,
        info.height,
        info.frames,
        info.family,
        info.keyframes,
        info.flow_gaps,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn corpus(args: &[String]) -> Result<(), String> {
    let (positional, flags) = split_args(args)?;
    let [manifest, out] = positional[..] else {
        return Err(format!("corpus takes a manifest and an output directory\n\n{USAGE}"));
    };
    let mut tiers = vec![Tier::Ci];
    let mut only = Vec::new();
    for (flag, value) in flags {
        match (flag, value) {
            ("tier", "ci") => tiers = vec![Tier::Ci],
            ("tier", "full") => tiers = vec![Tier::Full],
            ("tier", "all") => tiers = vec![Tier::Ci, Tier::Full],
            ("only", name) => only.push(name.to_owned()),
            _ => return Err(format!("invalid option --{flag} {value}\n\n{USAGE}")),
        }
    }
    let manifest = Manifest::load(Path::new(manifest)).map_err(|e| e.to_string())?;
    if let Some(missing) = only.iter().find(|n| manifest.scene(n).is_none()) {
        return Err(format!("no scene named '{missing}' in the manifest"));
    }
    for entry in &manifest.scenes {
        let selected = if only.is_empty() { tiers.contains(&entry.tier) } else { only.contains(&entry.name) };
        if selected {
            generate_one(&entry.params, &entry.name, &Path::new(out).join(&entry.name))?;
        }
    }
    Ok(())
}

fn load_rgb(path: &Path) -> Result<RgbImage, String> {
    image::open(path).map(|i| i.to_rgb8()).map_err(|e| format!("{}: {e}", path.display()))
}

/// Mean absolute error (8-bit levels, averaged over channels) between `a` and `b` warped back
/// with `flow`, over the pixels where `mask` is 0 (or all pixels without a mask).
fn warp_error(a: &RgbImage, b: &RgbImage, flow: &FlowField, mask: Option<&image::GrayImage>) -> (f64, usize) {
    let (mut sum, mut count) = (0.0, 0usize);
    for y in 0..a.height() {
        for x in 0..a.width() {
            if mask.is_some_and(|m| m.get_pixel(x, y)[0] != 0) {
                continue;
            }
            let [u, v] = flow.get(x as usize, y as usize);
            let w = bilinear(b, x as f32 + u, y as f32 + v);
            let p = a.get_pixel(x, y);
            sum += (0..3).map(|c| (p[c] as f64 - w[c] as f64).abs()).sum::<f64>() / 3.0;
            count += 1;
        }
    }
    (if count > 0 { sum / count as f64 } else { 0.0 }, count)
}

fn bilinear(img: &RgbImage, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (x, y) = (x.clamp(0.0, w - 1.0), y.clamp(0.0, h - 1.0));
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(img.width() - 1), (y0 + 1).min(img.height() - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let px = |x, y| img.get_pixel(x, y).0.map(f32::from);
    let lerp = |a: [f32; 3], b: [f32; 3], t: f32| std::array::from_fn(|c| a[c] + (b[c] - a[c]) * t);
    lerp(lerp(px(x0, y0), px(x1, y0), fx), lerp(px(x0, y1), px(x1, y1), fx), fy)
}

fn check(args: &[String]) -> Result<(), String> {
    let [dir] = args else {
        return Err(format!("check takes one scene directory\n\n{USAGE}"));
    };
    let dir = Path::new(dir);
    let info = SceneInfo::load(dir).map_err(|e| e.to_string())?;
    println!(
        "{}: {}x{}, {} frames, flow {:?}, gaps {:?}",
        info.name, info.width, info.height, info.frames, info.flow, info.flow_gaps
    );
    for &gap in &info.flow_gaps {
        for direction in [Direction::Forward, Direction::Backward] {
            let (mut frame_err, mut truth_err, mut occluded, mut pairs) = (0.0, 0.0, 0.0, 0usize);
            for first in 0..info.frames.saturating_sub(gap) {
                let (src, dst) = match direction {
                    Direction::Forward => (first, first + gap),
                    Direction::Backward => (first + gap, first),
                };
                let flow = FlowField::load(flow_path(dir, direction, gap, first)).map_err(|e| e.to_string())?;
                let mask = if info.occlusion_masks {
                    let p = occlusion_path(dir, direction, gap, first);
                    Some(image::open(&p).map_err(|e| format!("{}: {e}", p.display()))?.to_luma8())
                } else {
                    None
                };
                let (a, b) = (load_rgb(&frame_path(dir, src))?, load_rgb(&frame_path(dir, dst))?);
                let (err, count) = warp_error(&a, &b, &flow, mask.as_ref());
                frame_err += err;
                occluded += 1.0 - count as f64 / (info.width * info.height) as f64;
                if info.ground_truth {
                    let (a, b) = (load_rgb(&truth_path(dir, src))?, load_rgb(&truth_path(dir, dst))?);
                    truth_err += warp_error(&a, &b, &flow, mask.as_ref()).0;
                }
                pairs += 1;
            }
            if pairs == 0 {
                continue;
            }
            let n = pairs as f64;
            let truth = if info.ground_truth { format!(", truth MAE {:.3}", truth_err / n) } else { String::new() };
            let masks = if info.occlusion_masks { format!("occluded {:.1}%", 100.0 * occluded / n) } else { "no occlusion masks, all pixels".to_owned() };
            println!("  gap {gap} {direction:?}: {pairs} pairs, {masks}, frame MAE {:.3}{truth}", frame_err / n);
        }
    }
    Ok(())
}
