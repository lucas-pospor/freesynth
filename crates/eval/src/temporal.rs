//! Temporal stability of a frame sequence, measured with an externally supplied optical flow.
//!
//! **Warping error** (Lai et al., "Learning Blind Video Temporal Consistency", ECCV 2018): for a
//! frame gap `g`, the mean over start frames `t` of
//!
//! ```text
//! E_warp(t) = sum_x M(x) * |O_t(x) - O_{t+g}(x + F(x))|^2 / sum_x M(x)
//! ```
//!
//! with RGB in `[0, 1]` (so the per-pixel error is at most 3), `F` the flow from frame `t` to
//! `t + g` of the *input* video, bilinear sampling with the border pixels repeated, and `M` an
//! occlusion mask from forward-backward consistency (Sundaram, Brox and Keutzer, ECCV 2010):
//! a pixel is valid when `x + F(x)` lies inside the frame and
//! `|F(x) + B(x + F(x))|^2 <= 0.01 (|F(x)|^2 + |B(x + F(x))|^2) + 0.5`, where `B` is the flow back
//! from `t + g` to `t`.
//!
//! For gaps above 1, flows are chained from the 1-step flows (`F = f_t + f_{t+1}(x + f_t) + ...`,
//! valid where every step is valid) unless direct flows for that gap are supplied.
//!
//! Also measured:
//!
//! - **Second difference** at frame `t`: the masked mean of
//!   `|O_t - (O_{t-1}(x + B_t) + O_{t+1}(x + F_t)) / 2|^2`. It isolates flicker (oscillation)
//!   from smooth drift.
//! - **Global flicker**: mean absolute change of the mean luma between consecutive frames, in
//!   8-bit levels.
//!
//! The flow must come from the input video and must not be the flow the method under test used:
//! a method built on a given flow estimator scores unfairly well when measured with the same
//! estimator. Use exact flow for synthetic scenes and an independent estimator for real footage.
//! On real footage the flow's own error adds a floor to every sequence, so the metric flags gross
//! flicker but may not separate good methods; read it next to the fidelity metrics (energy,
//! sharpness), because plain advection of the keyframe scores very well here while smearing.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{Error, invalid};
use crate::flo::{FlowField, load_flo};
use crate::frame::RgbImage;

/// Gives frame `t` of a sequence.
pub trait FrameSource: Sync {
    fn frame(&self, t: usize) -> Result<RgbImage, Error>;
}

impl<F: Fn(usize) -> Result<RgbImage, Error> + Sync> FrameSource for F {
    fn frame(&self, t: usize) -> Result<RgbImage, Error> {
        self(t)
    }
}

/// Gives the input video's optical flow.
pub trait FlowSource {
    /// The flow from frame `t` to `t + 1` (on frame `t`'s grid) and back from `t + 1` to `t`
    /// (on frame `t + 1`'s grid).
    fn pair(&self, t: usize) -> Result<(FlowField, FlowField), Error>;

    /// Direct flows between frames `t` and `t + gap`, forward and backward, if available.
    fn direct(&self, _t: usize, _gap: usize) -> Result<Option<(FlowField, FlowField)>, Error> {
        Ok(None)
    }
}

/// Which frames and gaps to measure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalOptions {
    /// First and last frame index (inclusive).
    pub first: usize,
    pub last: usize,
    /// Frame gaps for the warping error, for example `[1, 5]`.
    pub gaps: Vec<usize>,
}

/// A per-frame value; `None` where the occlusion mask is empty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameValue {
    /// The start frame (warping error) or middle frame (second difference).
    pub t: usize,
    pub value: Option<f64>,
}

/// Results for one sequence.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SequenceTemporal {
    /// Warping error per gap, per start frame.
    pub warp: BTreeMap<usize, Vec<FrameValue>>,
    /// Second difference per middle frame.
    pub second_diff: Vec<FrameValue>,
    /// Mean luma of every frame, in `[0, 1]`.
    pub mean_luma: Vec<f64>,
}

impl SequenceTemporal {
    /// Mean warping error over the frames with a non-empty mask.
    pub fn mean_warp(&self, gap: usize) -> Option<f64> {
        self.warp.get(&gap).and_then(|v| mean_of(v))
    }

    /// Mean second difference over the frames with a non-empty mask.
    pub fn mean_second_diff(&self) -> Option<f64> {
        mean_of(&self.second_diff)
    }

    /// Mean absolute change of the mean luma between consecutive frames, in 8-bit levels.
    pub fn global_flicker(&self) -> Option<f64> {
        (self.mean_luma.len() >= 2).then(|| {
            self.mean_luma.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (self.mean_luma.len() - 1) as f64 * 255.0
        })
    }
}

fn mean_of(v: &[FrameValue]) -> Option<f64> {
    let vals: Vec<f64> = v.iter().filter_map(|f| f.value).collect();
    (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
}

/// Results for all sequences measured with one flow.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalResult {
    /// Fraction of valid (non-occluded) pixels per gap, per start frame.
    pub mask_frac: BTreeMap<usize, Vec<f64>>,
    /// One entry per input sequence, in order.
    pub sequences: Vec<SequenceTemporal>,
}

/// Bilinear taps for position `(x, y)` with coordinates clamped to the image (border pixels
/// repeated): indices and weights of the four neighbours.
#[inline]
fn taps(x: f32, y: f32, w: usize, h: usize) -> ([usize; 4], [f32; 4]) {
    let x = if x.is_finite() { x.clamp(0.0, (w - 1) as f32) } else { 0.0 };
    let y = if y.is_finite() { y.clamp(0.0, (h - 1) as f32) } else { 0.0 };
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    (
        [y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1],
        [(1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy],
    )
}

#[inline]
fn sample<const N: usize>(data: &[[f32; N]], idx: [usize; 4], wt: [f32; 4]) -> [f32; N] {
    let mut out = [0.0; N];
    for (&i, &k) in idx.iter().zip(&wt) {
        for (o, &v) in out.iter_mut().zip(&data[i]) {
            *o += k * v;
        }
    }
    out
}

/// Whether `(x, y)` lies inside a `w x h` frame (pixel centres span `[0, w - 1]`).
#[inline]
fn inside(x: f32, y: f32, w: usize, h: usize) -> bool {
    x >= 0.0 && y >= 0.0 && x <= (w - 1) as f32 && y <= (h - 1) as f32
}

/// `img` warped by `flow`: `out(x) = img(x + flow(x))`, bilinear, border pixels repeated.
///
/// # Panics
/// If the sizes differ.
pub fn warp(img: &RgbImage, flow: &FlowField) -> RgbImage {
    let (w, h) = img.size();
    assert_eq!(flow.size(), (w, h), "flow and image sizes differ");
    let data = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let [u, v] = flow.data()[i];
            let (idx, wt) = taps((i % w) as f32 + u, (i / w) as f32 + v, w, h);
            sample(img.data(), idx, wt)
        })
        .collect();
    RgbImage::from_raw(w, h, data)
}

/// Forward-backward consistency mask on `fw`'s grid (see the module documentation).
///
/// # Panics
/// If the sizes differ.
pub fn consistency_mask(fw: &FlowField, bw: &FlowField) -> Vec<bool> {
    let (w, h) = fw.size();
    assert_eq!(bw.size(), (w, h), "flow sizes differ");
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let [u, v] = fw.data()[i];
            let (x, y) = ((i % w) as f32 + u, (i / w) as f32 + v);
            if !inside(x, y, w, h) {
                return false;
            }
            let (idx, wt) = taps(x, y, w, h);
            let [bu, bv] = sample(bw.data(), idx, wt);
            let lhs = (u + bu).powi(2) + (v + bv).powi(2);
            let rhs = 0.01 * (u * u + v * v + bu * bu + bv * bv) + 0.5;
            lhs <= rhs
        })
        .collect()
}

/// Chains 1-step flows `steps[k]` (frame `t + k` to `t + k + 1`) with their masks into a flow
/// from `t` to `t + steps.len()`. A pixel stays valid while every step lands inside the frame on
/// a valid neighbourhood.
fn chain(steps: &[(FlowField, Vec<bool>)]) -> (FlowField, Vec<bool>) {
    let (first, rest) = steps.split_first().expect("at least one step");
    let (w, h) = first.0.size();
    let pairs: Vec<([f32; 2], bool)> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (px, py) = ((i % w) as f32, (i / w) as f32);
            let mut f = first.0.data()[i];
            let mut valid = first.1[i];
            for (flow, mask) in rest {
                let (x, y) = (px + f[0], py + f[1]);
                if !valid || !inside(x, y, w, h) {
                    valid = false;
                    break;
                }
                let (idx, wt) = taps(x, y, w, h);
                let m: f32 = idx.iter().zip(&wt).map(|(&j, &k)| if mask[j] { k } else { 0.0 }).sum();
                valid = m > 0.99;
                let [du, dv] = sample(flow.data(), idx, wt);
                f = [f[0] + du, f[1] + dv];
            }
            (f, valid)
        })
        .collect();
    let (flow, mask) = pairs.into_iter().unzip();
    (FlowField::from_raw(w, h, flow), mask)
}

/// Masked mean of `|a - b|^2` (summed over RGB); `None` if the mask is empty.
///
/// Rows are summed in parallel and then in order, so the result does not depend on scheduling.
fn masked_error(a: &RgbImage, b: &RgbImage, mask: &[bool]) -> Option<f64> {
    let w = a.width();
    let rows: Vec<(f64, usize)> = a
        .data()
        .par_chunks(w)
        .zip(b.data().par_chunks(w))
        .zip(mask.par_chunks(w))
        .map(|((ra, rb), rm)| {
            let mut acc = (0.0, 0);
            for ((p, q), &m) in ra.iter().zip(rb).zip(rm) {
                if m {
                    acc.0 += p.iter().zip(q).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>();
                    acc.1 += 1;
                }
            }
            acc
        })
        .collect();
    let (sum, n) = rows.iter().fold((0.0, 0), |acc, r| (acc.0 + r.0, acc.1 + r.1));
    (n > 0).then(|| sum / n as f64)
}

fn frac(mask: &[bool]) -> f64 {
    mask.iter().filter(|&&m| m).count() as f64 / mask.len() as f64
}

/// A sliding cache of decoded frames for one sequence.
struct FrameCache<'a> {
    source: &'a dyn FrameSource,
    frames: VecDeque<(usize, RgbImage)>,
    size: Option<(usize, usize)>,
}

impl FrameCache<'_> {
    /// Decodes frame `t` unless it is cached.
    fn load(&mut self, t: usize) -> Result<(), Error> {
        if self.frames.iter().any(|(i, _)| *i == t) {
            return Ok(());
        }
        let img = self.source.frame(t)?;
        match self.size {
            Some(s) if s != img.size() => return invalid(format!("frame {t} is {}x{}, earlier frames are {}x{}", img.width(), img.height(), s.0, s.1)),
            _ => self.size = Some(img.size()),
        }
        self.frames.push_back((t, img));
        Ok(())
    }

    /// A frame loaded by [`load`](Self::load) and not yet dropped.
    fn get(&self, t: usize) -> &RgbImage {
        &self.frames.iter().find(|(i, _)| *i == t).expect("frame was loaded").1
    }

    fn keep_from(&mut self, t: usize) {
        self.frames.retain(|(i, _)| *i >= t);
    }
}

/// Measures every sequence in `sequences` over the frames and gaps in `opts`, with flows from
/// `flows`. Frames and flows are read one pair at a time; at most `max gap + 1` frames per
/// sequence are held in memory.
pub fn temporal_metrics(sequences: &[&dyn FrameSource], flows: &dyn FlowSource, opts: &TemporalOptions) -> Result<TemporalResult, Error> {
    if opts.last <= opts.first {
        return invalid("need at least two frames");
    }
    if opts.gaps.is_empty() || opts.gaps.contains(&0) {
        return invalid("frame gaps must be positive");
    }
    let gaps: BTreeSet<usize> = opts.gaps.iter().copied().collect();
    let max_gap = *gaps.last().expect("non-empty");
    let mut caches: Vec<FrameCache> = sequences.iter().map(|&s| FrameCache { source: s, frames: VecDeque::new(), size: None }).collect();
    let mut results = vec![SequenceTemporal::default(); sequences.len()];
    let mut mask_frac: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    // The last `max_gap` forward flows with their masks, oldest first.
    let mut window: VecDeque<(FlowField, Vec<bool>)> = VecDeque::new();
    // The backward flow of the previous pair (frame t to t - 1) and its mask.
    let mut prev_back: Option<(FlowField, Vec<bool>)> = None;

    for t in opts.first..opts.last {
        let (fw, bw) = flows.pair(t)?;
        if fw.size() != bw.size() {
            return invalid(format!("forward and backward flows of pair {t} differ in size"));
        }
        let m_fw = consistency_mask(&fw, &bw);
        let m_bw = consistency_mask(&bw, &fw);

        for (cache, res) in caches.iter_mut().zip(&mut results) {
            cache.load(t)?;
            cache.load(t + 1)?;
            let (o0, o1) = (cache.get(t), cache.get(t + 1));
            if o0.size() != fw.size() {
                return invalid(format!("frame {t} is {}x{}, the flow is {}x{}", o0.width(), o0.height(), fw.width(), fw.height()));
            }
            if t == opts.first {
                res.mean_luma.push(o0.mean_luma());
            }
            res.mean_luma.push(o1.mean_luma());
            let next = warp(o1, &fw);
            if gaps.contains(&1) {
                res.warp.entry(1).or_default().push(FrameValue { t, value: masked_error(o0, &next, &m_fw) });
            }
            if let Some((pb, pm)) = &prev_back {
                let prev = warp(cache.get(t - 1), pb);
                let pred = RgbImage::from_raw(
                    prev.width(),
                    prev.height(),
                    prev.data().iter().zip(next.data()).map(|(p, q)| [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])]).collect(),
                );
                let mask: Vec<bool> = pm.iter().zip(&m_fw).map(|(a, b)| *a && *b).collect();
                res.second_diff.push(FrameValue { t, value: masked_error(o0, &pred, &mask) });
            }
        }
        if gaps.contains(&1) {
            mask_frac.entry(1).or_default().push(frac(&m_fw));
        }

        window.push_back((fw, m_fw));
        if window.len() > max_gap {
            window.pop_front();
        }
        for &g in gaps.iter().filter(|&&g| g > 1 && g <= window.len()) {
            let s = t + 1 - g;
            let (flow, mask) = match flows.direct(s, g)? {
                Some((f, b)) => {
                    if f.size() != b.size() || f.size() != window[0].0.size() {
                        return invalid(format!("direct flows for frames {s}-{} have the wrong size", s + g));
                    }
                    let m = consistency_mask(&f, &b);
                    (f, m)
                }
                None => chain(window.range(window.len() - g..).cloned().collect::<Vec<_>>().as_slice()),
            };
            mask_frac.entry(g).or_default().push(frac(&mask));
            for (cache, res) in caches.iter_mut().zip(&mut results) {
                let later = warp(cache.get(s + g), &flow);
                let e = masked_error(cache.get(s), &later, &mask);
                res.warp.entry(g).or_default().push(FrameValue { t: s, value: e });
            }
        }
        prev_back = Some((bw, m_bw));
        // Next iteration needs frame t (second difference) and t + 2 - max_gap (largest gap).
        let keep = (t + 2).saturating_sub(max_gap).min(t);
        for cache in &mut caches {
            cache.keep_from(keep);
        }
    }
    Ok(TemporalResult { mask_frac, sequences: results })
}

/// Lists `NNNNN.<ext>` files (5 or more digits) in `dir` by index, for the given extensions.
fn indexed_files(dir: &Path, extensions: &[&str]) -> Result<BTreeMap<usize, PathBuf>, Error> {
    let entries = std::fs::read_dir(dir).map_err(|e| Error::Invalid(format!("cannot read directory '{}': {e}", dir.display())))?;
    let mut files = BTreeMap::new();
    for entry in entries {
        let path = entry?.path();
        let (Some(stem), Some(ext)) = (path.file_stem().and_then(|s| s.to_str()), path.extension().and_then(|s| s.to_str())) else {
            continue;
        };
        if stem.len() < 5 || !stem.bytes().all(|b| b.is_ascii_digit()) || !extensions.contains(&ext.to_ascii_lowercase().as_str()) {
            continue;
        }
        let Ok(t) = stem.parse::<usize>() else { continue };
        if let Some(old) = files.insert(t, path.clone()) {
            return invalid(format!("'{}' and '{}' have the same frame index", old.display(), path.display()));
        }
    }
    Ok(files)
}

/// A frame sequence stored as `NNNNN.png` (or `.jpg`, `.jpeg`, `.bmp`, `.tga`) files in a
/// directory, numbered with 5 digits.
#[derive(Clone, Debug)]
pub struct FrameDir {
    files: BTreeMap<usize, PathBuf>,
}

impl FrameDir {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let files = indexed_files(dir, &["png", "jpg", "jpeg", "bmp", "tga"])?;
        if files.is_empty() {
            return invalid(format!("no NNNNN.png frames in '{}'", dir.display()));
        }
        Ok(Self { files })
    }

    /// The first and last frame index of the leading contiguous run of frames.
    pub fn range(&self) -> (usize, usize) {
        let first = *self.files.keys().next().expect("non-empty");
        let mut last = first;
        while self.files.contains_key(&(last + 1)) {
            last += 1;
        }
        (first, last)
    }
}

impl FrameSource for FrameDir {
    fn frame(&self, t: usize) -> Result<RgbImage, Error> {
        match self.files.get(&t) {
            Some(p) => RgbImage::load(p),
            None => invalid(format!("frame {t:05} is missing")),
        }
    }
}

/// Forward and backward flows stored as `.flo` files.
///
/// `fw/NNNNN.flo` is the flow from frame `N` to `N + gap`. The backward flow from `N + gap` to `N`
/// may be stored under either index: `bw/NNNNN.flo` (indexed like the forward flow) or
/// `bw/MMMMM.flo` with `M = N + gap` (indexed by the frame it starts from). The convention is
/// detected from which files exist.
#[derive(Clone, Debug)]
struct FlowPairDir {
    fw: BTreeMap<usize, PathBuf>,
    bw: BTreeMap<usize, PathBuf>,
    /// Added to a forward index to get the backward file index.
    bw_offset: usize,
}

impl FlowPairDir {
    fn open(fw: &Path, bw: &Path, gap: usize) -> Result<Self, Error> {
        let fw_files = indexed_files(fw, &["flo"])?;
        let bw_files = indexed_files(bw, &["flo"])?;
        let Some(&first) = fw_files.keys().next() else {
            return invalid(format!("no NNNNN.flo files in '{}'", fw.display()));
        };
        let bw_offset = if bw_files.contains_key(&first) {
            0
        } else if bw_files.contains_key(&(first + gap)) {
            gap
        } else {
            return invalid(format!(
                "'{}' has no backward flow for the pair starting at frame {first} (expected {first:05}.flo or {:05}.flo)",
                bw.display(),
                first + gap
            ));
        };
        Ok(Self { fw: fw_files, bw: bw_files, bw_offset })
    }

    fn get(&self, t: usize) -> Result<Option<(FlowField, FlowField)>, Error> {
        match (self.fw.get(&t), self.bw.get(&(t + self.bw_offset))) {
            (Some(f), Some(b)) => Ok(Some((load_flo(f)?, load_flo(b)?))),
            _ => Ok(None),
        }
    }
}

/// Optical flow of the input video stored as `.flo` files (see [`FlowSource`]).
///
/// The 1-step flows are required. Direct flows for larger gaps are optional; without them the
/// 1-step flows are chained.
#[derive(Clone, Debug)]
pub struct FlowDirs {
    step: FlowPairDir,
    direct: BTreeMap<usize, FlowPairDir>,
}

impl FlowDirs {
    /// 1-step flows from `fw` (frame `t` to `t + 1`) and `bw` (frame `t + 1` to `t`).
    pub fn open(fw: impl AsRef<Path>, bw: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self { step: FlowPairDir::open(fw.as_ref(), bw.as_ref(), 1)?, direct: BTreeMap::new() })
    }

    /// Adds direct flows for frame gap `gap` (frame `t` to `t + gap` and back).
    pub fn with_direct(mut self, gap: usize, fw: impl AsRef<Path>, bw: impl AsRef<Path>) -> Result<Self, Error> {
        self.direct.insert(gap, FlowPairDir::open(fw.as_ref(), bw.as_ref(), gap)?);
        Ok(self)
    }

    /// The backward-file convention detected for the 1-step flows: `"pair"` when `bw/N.flo` is
    /// the flow from `N + 1` to `N`, `"source"` when it is the flow from `N` to `N - 1`.
    pub fn backward_convention(&self) -> &'static str {
        if self.step.bw_offset == 0 { "pair" } else { "source" }
    }
}

impl FlowSource for FlowDirs {
    fn pair(&self, t: usize) -> Result<(FlowField, FlowField), Error> {
        self.step.get(t)?.ok_or_else(|| Error::Invalid(format!("flow files for frames {t}-{} are missing", t + 1)))
    }

    fn direct(&self, t: usize, gap: usize) -> Result<Option<(FlowField, FlowField)>, Error> {
        match self.direct.get(&gap) {
            Some(d) => d.get(t)?.map(Some).ok_or_else(|| Error::Invalid(format!("direct flow files for frames {t}-{} are missing", t + gap))),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory on disk under the build directory (the system temp directory may be
    /// RAM-backed).
    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"));
        target.join("tmp").join(format!("{name}-{}", std::process::id()))
    }

    use crate::flo::save_flo;

    /// A smooth pattern sampled at a sub-pixel offset: frame t of a scene moving by `(du, dv)`
    /// pixels per frame.
    fn moving(w: usize, h: usize, t: usize, du: f32, dv: f32) -> RgbImage {
        let f = |x: f32, y: f32| {
            [
                0.5 + 0.4 * (x * 0.21).sin() * (y * 0.17).cos(),
                0.5 + 0.4 * (x * 0.07 + y * 0.13).sin(),
                0.5 + 0.3 * ((x - y) * 0.11).cos(),
            ]
        };
        let (ox, oy) = (du * t as f32, dv * t as f32);
        RgbImage::from_raw(w, h, (0..w * h).map(|i| f((i % w) as f32 - ox, (i / w) as f32 - oy)).collect())
    }

    struct ConstFlow {
        w: usize,
        h: usize,
        uv: [f32; 2],
    }

    impl FlowSource for ConstFlow {
        fn pair(&self, _t: usize) -> Result<(FlowField, FlowField), Error> {
            Ok((FlowField::constant(self.w, self.h, self.uv), FlowField::constant(self.w, self.h, [-self.uv[0], -self.uv[1]])))
        }
    }

    #[test]
    fn integer_translation_warps_exactly() {
        let img = moving(20, 10, 0, 0.0, 0.0);
        let shifted = warp(&img, &FlowField::constant(20, 10, [2.0, 1.0]));
        assert_eq!(shifted.data()[3 * 20 + 4], img.data()[4 * 20 + 6]);
        // Border pixels repeat.
        assert_eq!(shifted.data()[9 * 20 + 19], img.data()[9 * 20 + 19]);
    }

    #[test]
    fn consistency_masks_out_leaving_pixels_and_inconsistent_flow() {
        let (w, h) = (10, 8);
        let fw = FlowField::constant(w, h, [2.0, 0.0]);
        let m = consistency_mask(&fw, &FlowField::constant(w, h, [-2.0, 0.0]));
        for y in 0..h {
            for x in 0..w {
                assert_eq!(m[y * w + x], x + 2 < w, "({x}, {y})");
            }
        }
        // A backward flow that does not undo the forward flow invalidates everything.
        let bad = consistency_mask(&fw, &FlowField::constant(w, h, [1.0, 1.0]));
        assert!(bad.iter().all(|&v| !v));
    }

    #[test]
    fn steady_motion_has_tiny_error_and_flicker_is_detected() {
        let (w, h, n) = (48, 32, 8);
        let (du, dv) = (1.0, 0.5);
        let seq = move |t: usize| Ok(moving(w, h, t, du, dv));
        // The same scene with frame-alternating brightness.
        let flicker = move |t: usize| {
            let img = moving(w, h, t, du, dv);
            let k = if t.is_multiple_of(2) { 0.9 } else { 1.0 };
            Ok(RgbImage::from_raw(w, h, img.data().iter().map(|p| p.map(|v| v * k)).collect()))
        };
        let flows = ConstFlow { w, h, uv: [du, dv] };
        let opts = TemporalOptions { first: 0, last: n - 1, gaps: vec![1, 5] };
        let r = temporal_metrics(&[&seq, &flicker], &flows, &opts).unwrap();
        let (good, bad) = (&r.sequences[0], &r.sequences[1]);
        assert_eq!(good.warp[&1].len(), n - 1);
        assert_eq!(good.warp[&5].len(), n - 5);
        assert_eq!(good.second_diff.len(), n - 2);
        assert_eq!(good.mean_luma.len(), n);
        // Only bilinear interpolation error of a smooth pattern remains.
        assert!(good.mean_warp(1).unwrap() < 1e-4, "{:?}", good.mean_warp(1));
        assert!(good.mean_warp(5).unwrap() < 1e-4);
        assert!(good.mean_second_diff().unwrap() < 1e-4);
        assert!(bad.mean_warp(1).unwrap() > 50.0 * good.mean_warp(1).unwrap());
        assert!(bad.mean_second_diff().unwrap() > 50.0 * good.mean_second_diff().unwrap());
        // The pattern drifting through a small frame changes its mean luma a little too.
        let (fb, fg) = (bad.global_flicker().unwrap(), good.global_flicker().unwrap());
        assert!(fb > 5.0 && fg < fb / 3.0, "{fb} {fg}");
        // Chained 5-step flow masks out the 5 pixels that leave on the right, 3 at the bottom.
        let expected = ((w - 5) * (h - 3)) as f64 / (w * h) as f64;
        assert!((r.mask_frac[&5][0] - expected).abs() < 1e-9, "{} vs {expected}", r.mask_frac[&5][0]);
    }

    #[test]
    fn chained_flow_matches_direct_flow() {
        let (w, h) = (16, 12);
        let step = (FlowField::constant(w, h, [1.25, -0.5]), vec![true; w * h]);
        let (f, m) = chain(&[step.clone(), step.clone(), step]);
        assert!((f.get(4, 8)[0] - 3.75).abs() < 1e-5 && (f.get(4, 8)[1] + 1.5).abs() < 1e-5);
        assert!(m[8 * w + 4]);
        assert!(!m[w - 2], "leaves the frame");
    }

    #[test]
    fn reads_frame_and_flow_directories() {
        let dir = scratch_dir("freesynth-eval-temporal");
        let (frames, fw, bw) = (dir.join("frames"), dir.join("flow_fw"), dir.join("flow_bw"));
        for d in [&frames, &fw, &bw] {
            std::fs::create_dir_all(d).unwrap();
        }
        let (w, h, n) = (24, 16, 4);
        for t in 0..n {
            let img = moving(w, h, t, 1.0, 0.0);
            let bytes: Vec<u8> = img.data().iter().flat_map(|p| p.map(|v| (v * 255.0).round() as u8)).collect();
            freesynth_core::io::save_png(&freesynth_core::Image::from_raw(w, h, 3, bytes), frames.join(format!("{t:05}.png"))).unwrap();
        }
        // Backward flows indexed by the frame they start from.
        for t in 0..n - 1 {
            save_flo(&FlowField::constant(w, h, [1.0, 0.0]), fw.join(format!("{t:05}.flo"))).unwrap();
            save_flo(&FlowField::constant(w, h, [-1.0, 0.0]), bw.join(format!("{:05}.flo", t + 1))).unwrap();
        }
        std::fs::write(frames.join("notes.txt"), "ignored").unwrap();
        let seq = FrameDir::open(&frames).unwrap();
        assert_eq!(seq.range(), (0, n - 1));
        let flows = FlowDirs::open(&fw, &bw).unwrap();
        assert_eq!(flows.backward_convention(), "source");
        let r = temporal_metrics(&[&seq], &flows, &TemporalOptions { first: 0, last: n - 1, gaps: vec![1] }).unwrap();
        assert!(r.sequences[0].mean_warp(1).unwrap() < 1e-4);
        // Missing pairs are reported.
        assert!(temporal_metrics(&[&seq], &flows, &TemporalOptions { first: 0, last: n, gaps: vec![1] }).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
