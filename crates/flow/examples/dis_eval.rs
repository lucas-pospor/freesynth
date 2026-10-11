//! Runs DIS on a list of frame pairs and reports the end-point error against reference flow.
//!
//! ```text
//! cargo run --release -p freesynth-flow --example dis_eval -- [options] LIST...
//! ```
//!
//! Each line of a list file is `GROUP FRAME0 FRAME1 REFERENCE.flo|- OCCLUSION.png|- ...`
//! (further columns are ignored). The flow from FRAME0 to FRAME1 is computed; when a reference
//! flow is given, the mean end-point error over the pixels where the occlusion mask is 0 (all
//! pixels without a mask) is reported, averaged per group. Frames may be grey or colour
//! (converted to Rec. 601 luma).
//!
//! Options: `--preset ultrafast|fast|medium|high` and overrides of single parameters: `--finest L`,
//! `--stride N`, `--iters N`, `--tol T`, `--prop N`, `--prop-iters N`, `--margin M`, `--vr-finest L`,
//! `--vr-levels N`, `--vr N`, `--inner N`, `--alpha A`, `--delta D`, `--gamma G`, `--zeta Z`,
//! `--epsilon E`, `--omega W`, `--no-mean-norm`. `--out DIR` writes every flow as
//! `DIR/NNNNN.flo` (line number).
//!
//! The GPU estimator runs by default; `--cpu` runs the CPU one instead, and `--compare` runs
//! both and also reports how far apart their flows are (mean and 99th percentile of the
//! per-pixel distance, and the largest distance).

use std::collections::BTreeMap;
use std::path::PathBuf;

use freesynth_flow::cpu::CpuDis;
use freesynth_flow::gpu::{GpuContext, GpuDis};
use freesynth_flow::{DisParams, FlowEstimator, FlowField, GrayImage};

/// Which estimators run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Gpu,
    Cpu,
    Compare,
}

/// Distances between two flows: mean, 99th percentile and maximum.
fn distance_stats(a: &FlowField, b: &FlowField) -> [f64; 3] {
    let mut d: Vec<f32> = a.data().iter().zip(b.data()).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).collect();
    let mean = d.iter().map(|&v| f64::from(v)).sum::<f64>() / d.len() as f64;
    d.sort_by(f32::total_cmp);
    [mean, f64::from(d[(d.len() - 1) * 99 / 100]), f64::from(d[d.len() - 1])]
}

fn load_gray(path: &str) -> Result<GrayImage, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok(match img {
        image::DynamicImage::ImageLuma8(g) => GrayImage::from_luma8(w, h, g.as_raw())?,
        other => GrayImage::from_rgb8(w, h, other.to_rgb8().as_raw())?,
    })
}

fn parse_args() -> Result<(DisParams, Vec<String>, Option<PathBuf>, Mode), String> {
    let mut args = std::env::args().skip(1);
    let mut params = DisParams::default();
    let mut lists = Vec::new();
    let mut out = None;
    let mut mode = Mode::Gpu;
    let mut overrides: Vec<(String, String)> = Vec::new();
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--preset" => params = DisParams::preset(value()?.parse().map_err(|e| format!("{e}"))?),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--cpu" => mode = Mode::Cpu,
            "--compare" => mode = Mode::Compare,
            "--no-mean-norm" => overrides.push((a.clone(), String::new())),
            s if s.starts_with("--") => overrides.push((a.clone(), value()?)),
            _ => lists.push(a),
        }
    }
    // Overrides apply on top of the preset, whatever the order on the command line.
    for (k, v) in overrides {
        let int = || v.parse::<usize>().map_err(|e| format!("{k}: {e}"));
        let float = || v.parse::<f32>().map_err(|e| format!("{k}: {e}"));
        match k.as_str() {
            "--finest" => params.finest_level = int()?,
            "--stride" => params.patch_stride = int()?,
            "--iters" => params.gd_iterations = int()?,
            "--tol" => params.gd_tolerance = float()?,
            "--prop" => params.propagation_passes = int()?,
            "--prop-iters" => params.propagation_iterations = int()?,
            "--margin" => params.propagation_margin = float()?,
            "--vr-finest" => params.vr_finest_level = int()?,
            "--vr-levels" => params.vr_levels = int()?,
            "--vr" => params.vr_iterations = int()?,
            "--inner" => params.vr_inner_iterations = int()?,
            "--alpha" => params.vr_alpha = float()?,
            "--delta" => params.vr_delta = float()?,
            "--gamma" => params.vr_gamma = float()?,
            "--zeta" => params.vr_zeta = float()?,
            "--epsilon" => params.vr_epsilon = float()?,
            "--omega" => params.vr_omega = float()?,
            "--no-mean-norm" => params.mean_normalization = false,
            _ => return Err(format!("unknown option {k}")),
        }
    }
    if lists.is_empty() {
        return Err("usage: dis_eval [options] LIST...".into());
    }
    Ok((params, lists, out, mode))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (params, lists, out, mode) = parse_args()?;
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir)?;
    }
    let ctx = if mode == Mode::Cpu { None } else { Some(GpuContext::new()?) };
    let mut gpu: Option<GpuDis> = None;
    let mut cpu: Option<CpuDis> = None;
    let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut cpu_groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut apart: BTreeMap<String, Vec<[f64; 3]>> = BTreeMap::new();
    let mut index = 0usize;
    for list in &lists {
        for line in std::fs::read_to_string(list)?.lines().filter(|l| !l.trim().is_empty()) {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 3 {
                return Err(format!("bad line in {list}: {line}").into());
            }
            let (a, b) = (load_gray(cols[1])?, load_gray(cols[2])?);
            let gpu_flow = match &ctx {
                Some(ctx) => {
                    if gpu.as_ref().is_none_or(|d| d.size() != a.size()) {
                        gpu = Some(GpuDis::new(ctx, a.width(), a.height(), &params)?);
                    }
                    Some(gpu.as_mut().expect("created above").compute(&a, &b)?)
                }
                None => None,
            };
            let cpu_flow = if mode == Mode::Gpu {
                None
            } else {
                if cpu.as_ref().is_none_or(|d| d.size() != a.size()) {
                    cpu = Some(CpuDis::new(a.width(), a.height(), &params)?);
                }
                Some(cpu.as_mut().expect("created above").compute(&a, &b)?)
            };
            if let (Some(g), Some(c)) = (&gpu_flow, &cpu_flow) {
                apart.entry(cols[0].to_owned()).or_default().push(distance_stats(g, c));
            }
            let flow = gpu_flow.as_ref().or(cpu_flow.as_ref()).expect("one estimator ran");
            if let Some(dir) = &out {
                flow.save(dir.join(format!("{index:05}.flo")))?;
            }
            index += 1;
            let Some(&reference) = cols.get(3).filter(|r| **r != "-") else { continue };
            let truth = FlowField::load(reference)?;
            let mask = match cols.get(4).filter(|m| **m != "-") {
                Some(m) => Some(image::open(m)?.to_luma8().as_raw().iter().map(|&v| v == 0).collect::<Vec<bool>>()),
                None => None,
            };
            let epe = flow.mean_epe(&truth, mask.as_deref()).ok_or("no unoccluded pixels")?;
            groups.entry(cols[0].to_owned()).or_default().push(epe);
            if let (Some(_), Some(c)) = (&gpu_flow, &cpu_flow) {
                cpu_groups
                    .entry(cols[0].to_owned())
                    .or_default()
                    .push(c.mean_epe(&truth, mask.as_deref()).ok_or("no unoccluded pixels")?);
            }
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    for (group, v) in &groups {
        let label = if mode == Mode::Cpu { "CPU" } else { "GPU" };
        print!("{group:24} {label} EPE {:.4} ({} pairs)", mean(v), v.len());
        if let Some(c) = cpu_groups.get(group) {
            print!("  CPU EPE {:.4}", mean(c));
        }
        println!();
    }
    for (group, v) in &apart {
        let col = |i: usize| v.iter().map(|s| s[i]).collect::<Vec<_>>();
        let max = col(2).into_iter().fold(0.0, f64::max);
        println!(
            "{group:24} GPU-CPU distance: mean {:.5} px, p99 {:.4} px (means over pairs), max {max:.3} px",
            mean(&col(0)),
            mean(&col(1))
        );
    }
    Ok(())
}
