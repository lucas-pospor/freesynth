//! Position fields: for every pixel of a frame, where its content was in a keyframe.
//!
//! A [`Tracker`] walks a sequence away from a keyframe (forward or backward in time, in the
//! order the frames are given) and maintains, for the latest frame `t`:
//!
//! - a [`PositionField`] `P_t`: the keyframe position of the content seen at each pixel;
//! - a [`ConfidenceMap`] `C_t` in `[0, 1]`: whether that position can be trusted.
//!
//! **Anchors.** Chaining one-frame flows resamples the field at every frame, which blurs it and
//! lets errors accumulate. Instead, the tracker keeps an *anchor* frame `a` with a known field
//! and computes the flow from each new frame straight to the anchor: `P_t(x) = P_a(x + f(x))`
//! with `f` the flow from frame `t` to frame `a`. A frame becomes the next anchor after
//! [`TrackParams::max_stride`] frames, so the field is resampled once per anchor rather than
//! once per frame, and flows never span more than `max_stride` frames.
//!
//! The stride adapts to the motion: when the median displacement to the anchor exceeds
//! [`TrackParams::max_anchor_motion`], or the share of pixels failing the consistency check
//! jumps by more than [`TrackParams::max_failure_jump`] from one frame to the next, the
//! *previous* frame (whose field is still good) becomes the anchor and the flows are recomputed
//! from it. The current frame is not made the anchor in that case, because an anchor passes its
//! failures on to every later frame. (The share of inconsistent pixels grows steadily with the
//! distance to the anchor, as disoccluded regions widen; only a sudden jump points to a flow
//! failure or a cut.)
//!
//! **Confidence.** For each frame the reverse flow (anchor to frame) is computed as well, and a
//! pixel passes the forward-backward check of Sundaram, Brox and Keutzer ("Dense point
//! trajectories by GPU-accelerated large displacement optical flow", ECCV 2010) when following
//! the flow there and back returns close to where it started ([`ForwardBackward`]). Pixels that
//! fail are mostly disocclusions (content that was hidden in the anchor) and flow errors. The
//! failure mask is grown by [`TrackParams::erode_radius`] pixels, and
//! `C_t(x) = C_a(x + f(x))` where the check passes, 0 where it fails: once lost, confidence does
//! not come back within a walk, since the content has no reliable keyframe position. Pixels
//! whose position is interpolated across an edge of the anchor's field (two surfaces that moved
//! apart) lose their confidence too ([`TrackParams::max_position_jump`]): the flow there can be
//! consistent while the interpolated position belongs to neither surface.
//!
//! **Precision.** Positions are kept as `f32`. For compact storage they convert to 16-bit fixed
//! point ([`FixedPositions`]) with the finest step that still covers the frame (1/64 pixel up to
//! 1024 pixels, 1/16 pixel up to 4096).

use rayon::prelude::*;

use crate::error::{FlowError, invalid};
use crate::flo::FlowField;
use crate::image::GrayImage;
use crate::FlowEstimator;

/// Bilinear sample of a two-channel field at `(x, y)`, clamped to the field.
#[inline]
fn sample2(data: &[[f32; 2]], w: usize, h: usize, x: f32, y: f32) -> [f32; 2] {
    let (x, y) = (x.clamp(0.0, (w - 1) as f32), y.clamp(0.0, (h - 1) as f32));
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let lerp = |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    lerp(lerp(data[y0 * w + x0], data[y0 * w + x1], fx), lerp(data[y1 * w + x0], data[y1 * w + x1], fx), fy)
}

/// Bilinear sample of a one-channel field at `(x, y)`, clamped to the field.
#[inline]
fn sample1(data: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (x, y) = (x.clamp(0.0, (w - 1) as f32), y.clamp(0.0, (h - 1) as f32));
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    lerp(lerp(data[y0 * w + x0], data[y0 * w + x1], fx), lerp(data[y1 * w + x0], data[y1 * w + x1], fx), fy)
}

/// For every pixel of a frame, a position in the keyframe (pixel centres at integer
/// coordinates), row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct PositionField {
    width: usize,
    height: usize,
    data: Vec<[f32; 2]>,
}

impl PositionField {
    /// The keyframe's own field: every pixel maps to itself.
    pub fn identity(width: usize, height: usize) -> Self {
        let data = (0..width * height).map(|i| [(i % width) as f32, (i / width) as f32]).collect();
        Self { width, height, data }
    }

    /// Wraps existing positions (`width * height` of them, row-major).
    ///
    /// # Panics
    /// If `data.len()` does not match the dimensions.
    pub fn from_vec(width: usize, height: usize, data: Vec<[f32; 2]>) -> Self {
        assert_eq!(data.len(), width * height, "position field length does not match {width}x{height}");
        Self { width, height, data }
    }

    /// The positions `x + flow(x)`: the field of a frame whose flow to the keyframe is `flow`.
    pub fn from_displacement(flow: &FlowField) -> Self {
        let (w, h) = flow.size();
        let data = flow.data().iter().enumerate().map(|(i, d)| [(i % w) as f32 + d[0], (i / w) as f32 + d[1]]).collect();
        Self { width: w, height: h, data }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// `(width, height)`.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Positions, row-major.
    pub fn data(&self) -> &[[f32; 2]] {
        &self.data
    }

    /// The keyframe position of pixel `(x, y)`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 2] {
        self.data[y * self.width + x]
    }

    /// Bilinear interpolation at a continuous position, clamped to the field.
    pub fn sample(&self, x: f32, y: f32) -> [f32; 2] {
        sample2(&self.data, self.width, self.height, x, y)
    }

    /// The displacements `P(x) - x`, i.e. the flow from this frame to the keyframe.
    pub fn displacement(&self) -> FlowField {
        let w = self.width;
        let data = self.data.iter().enumerate().map(|(i, p)| [p[0] - (i % w) as f32, p[1] - (i / w) as f32]).collect();
        FlowField::from_vec(self.width, self.height, data)
    }

    /// The field of a frame whose flow to this field's frame is `flow`:
    /// `Q(x) = P(x + flow(x))`, bilinear and clamped to the field.
    ///
    /// # Panics
    /// If the sizes differ.
    pub fn compose(&self, flow: &FlowField) -> PositionField {
        assert_eq!(self.size(), flow.size(), "flow and position field differ in size");
        let (w, h) = self.size();
        let mut data = vec![[0.0; 2]; w * h];
        data.par_chunks_mut(w).zip(flow.data().par_chunks(w)).enumerate().for_each(|(y, (out, f))| {
            for (x, (o, d)) in out.iter_mut().zip(f).enumerate() {
                *o = sample2(&self.data, w, h, x as f32 + d[0], y as f32 + d[1]);
            }
        });
        Self { width: w, height: h, data }
    }

    /// Median distance between horizontally or vertically adjacent positions, over a sample
    /// of every fourth pixel in each direction: about 1 unless the content was scaled since the
    /// keyframe (for example by a camera zoom).
    pub fn median_spacing(&self) -> f32 {
        let (w, h) = self.size();
        let mut d = Vec::new();
        for y in (0..h.saturating_sub(1)).step_by(4) {
            for x in (0..w.saturating_sub(1)).step_by(4) {
                let (p, r, b) = (self.get(x, y), self.get(x + 1, y), self.get(x, y + 1));
                d.push((r[0] - p[0]).hypot(r[1] - p[1]));
                d.push((b[0] - p[0]).hypot(b[1] - p[1]));
            }
        }
        if d.is_empty() {
            return 1.0;
        }
        let mid = d.len() / 2;
        *d.select_nth_unstable_by(mid, f32::total_cmp).1
    }

    /// For each pixel `x` of a frame whose flow to this field's frame is `flow`: true when the
    /// four positions that `P(x + flow(x))` interpolates lie within `max_jump` pixels of each
    /// other along the cell's edges. Neighbouring positions are normally about one pixel apart
    /// ([`median_spacing`](Self::median_spacing)); a much larger jump means that the sample
    /// blends two surfaces that moved differently (an occlusion edge in the field), and the
    /// blended position belongs to neither.
    pub fn continuous_at(&self, flow: &FlowField, max_jump: f32) -> Vec<bool> {
        assert_eq!(self.size(), flow.size(), "flow and position field differ in size");
        let (w, h) = self.size();
        let mut ok = vec![true; w * h];
        if !max_jump.is_finite() {
            return ok;
        }
        let limit = max_jump * max_jump;
        ok.par_chunks_mut(w).zip(flow.data().par_chunks(w)).enumerate().for_each(|(y, (out, f))| {
            for (x, (o, d)) in out.iter_mut().zip(f).enumerate() {
                let sx = (x as f32 + d[0]).clamp(0.0, (w - 1) as f32);
                let sy = (y as f32 + d[1]).clamp(0.0, (h - 1) as f32);
                let (x0, y0) = (sx as usize, sy as usize);
                let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                let p = |x: usize, y: usize| self.data[y * w + x];
                let far = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) > limit;
                let (p00, p10, p01, p11) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
                *o = !(far(p00, p10) || far(p01, p11) || far(p00, p01) || far(p10, p11));
            }
        });
        ok
    }

    /// The positions in 16-bit fixed point (see [`FixedPositions`]).
    pub fn to_fixed(&self) -> FixedPositions {
        FixedPositions::encode(self)
    }
}

/// Positions in 16-bit unsigned fixed point, for compact storage.
///
/// A value `v` stands for the position `v / 2^frac_bits`. The number of fraction bits is the
/// largest that still covers the frame: 16 minus the bits needed for the larger dimension. That
/// is a step of 1/64 pixel up to 1024 pixels, 1/32 up to 2048 and 1/16 up to 4096 (half floats,
/// in contrast, have 2 pixel steps between 2048 and 4096). Positions outside the representable
/// range are clamped to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPositions {
    width: usize,
    height: usize,
    frac_bits: u32,
    data: Vec<[u16; 2]>,
}

impl FixedPositions {
    /// Fraction bits for a `width` x `height` frame.
    pub fn frac_bits_for(width: usize, height: usize) -> u32 {
        let largest = width.max(height).max(1);
        let int_bits = usize::BITS - (largest - 1).leading_zeros();
        16u32.saturating_sub(int_bits)
    }

    fn encode(p: &PositionField) -> Self {
        let frac_bits = Self::frac_bits_for(p.width, p.height);
        let scale = (1u32 << frac_bits) as f32;
        let q = |v: f32| (v * scale).round().clamp(0.0, f32::from(u16::MAX)) as u16;
        let data = p.data.iter().map(|&[x, y]| [q(x), q(y)]).collect();
        Self { width: p.width, height: p.height, frac_bits, data }
    }

    /// Fraction bits of the encoding.
    pub fn frac_bits(&self) -> u32 {
        self.frac_bits
    }

    /// The step between representable positions, in pixels.
    pub fn step(&self) -> f32 {
        1.0 / (1u32 << self.frac_bits) as f32
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Encoded positions, row-major.
    pub fn data(&self) -> &[[u16; 2]] {
        &self.data
    }

    /// Decodes to floating-point positions.
    pub fn to_positions(&self) -> PositionField {
        let step = self.step();
        let data = self.data.iter().map(|&[x, y]| [f32::from(x) * step, f32::from(y) * step]).collect();
        PositionField { width: self.width, height: self.height, data }
    }
}

/// Per-pixel confidence in `[0, 1]`, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfidenceMap {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl ConfidenceMap {
    /// Full confidence everywhere (the keyframe).
    pub fn ones(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![1.0; width * height] }
    }

    /// Wraps existing values (`width * height` of them, row-major).
    ///
    /// # Panics
    /// If `data.len()` does not match the dimensions.
    pub fn from_vec(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height, "confidence map length does not match {width}x{height}");
        Self { width, height, data }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// `(width, height)`.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Values, row-major.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The confidence of pixel `(x, y)`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Pixels whose confidence exceeds `threshold` (0.5 separates the advected values well:
    /// they are 1 or 0 except near the edges of lost regions).
    pub fn confident(&self, threshold: f32) -> Vec<bool> {
        self.data.iter().map(|&c| c > threshold).collect()
    }

    /// The share of pixels whose confidence exceeds `threshold`.
    pub fn fraction_confident(&self, threshold: f32) -> f64 {
        self.data.iter().filter(|&&c| c > threshold).count() as f64 / self.data.len().max(1) as f64
    }

    /// The map of a frame whose flow to this map's frame is `flow`, multiplied by `keep` (pixels
    /// where `keep` is false get 0): `C'(x) = C(x + flow(x))` where kept, bilinear and clamped.
    fn advect(&self, flow: &FlowField, keep: &[bool]) -> ConfidenceMap {
        let (w, h) = self.size();
        let mut data = vec![0.0; w * h];
        data.par_chunks_mut(w).zip(flow.data().par_chunks(w)).zip(keep.par_chunks(w)).enumerate().for_each(|(y, ((out, f), k))| {
            for (x, ((o, d), &keep)) in out.iter_mut().zip(f).zip(k).enumerate() {
                if keep {
                    *o = sample1(&self.data, w, h, x as f32 + d[0], y as f32 + d[1]);
                }
            }
        });
        ConfidenceMap { width: w, height: h, data }
    }
}

/// The forward-backward consistency check: a pixel `x` of frame A with flow `f` to frame B and
/// reverse flow `g` (B to A) is consistent when `x + f(x)` is inside B and
/// `|f(x) + g(x + f(x))|^2 < alpha1 (|f(x)|^2 + |g(x + f(x))|^2) + alpha2`, with `g` sampled
/// bilinearly. The tolerance grows with the motion, since long flows are less precise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForwardBackward {
    /// Tolerance relative to the squared flow lengths.
    pub alpha1: f32,
    /// Absolute tolerance, in squared pixels.
    pub alpha2: f32,
}

impl Default for ForwardBackward {
    /// The constants of Sundaram, Brox and Keutzer (2010): 0.01 and 0.5.
    fn default() -> Self {
        Self { alpha1: 0.01, alpha2: 0.5 }
    }
}

impl ForwardBackward {
    /// The consistency of every pixel of frame A, given the flow `forward` from A to B and the
    /// flow `backward` from B to A.
    ///
    /// # Panics
    /// If the sizes differ.
    pub fn check(&self, forward: &FlowField, backward: &FlowField) -> Vec<bool> {
        assert_eq!(forward.size(), backward.size(), "flow fields differ in size");
        let (w, h) = forward.size();
        let mut ok = vec![false; w * h];
        ok.par_chunks_mut(w).zip(forward.data().par_chunks(w)).enumerate().for_each(|(y, (out, f))| {
            for (x, (o, f)) in out.iter_mut().zip(f).enumerate() {
                let (tx, ty) = (x as f32 + f[0], y as f32 + f[1]);
                let inside = tx >= 0.0 && ty >= 0.0 && tx <= (w - 1) as f32 && ty <= (h - 1) as f32;
                let g = sample2(backward.data(), w, h, tx, ty);
                let (sx, sy) = (f[0] + g[0], f[1] + g[1]);
                let lengths = f[0] * f[0] + f[1] * f[1] + g[0] * g[0] + g[1] * g[1];
                *o = inside && sx * sx + sy * sy < self.alpha1 * lengths + self.alpha2;
            }
        });
        ok
    }
}

/// Erodes a mask by a square of `2 radius + 1` pixels: a pixel stays set only if every pixel
/// within `radius` (in both directions, inside the image) is set.
pub fn erode(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    assert_eq!(mask.len(), width * height, "mask does not match {width}x{height}");
    if radius == 0 || mask.is_empty() {
        return mask.to_vec();
    }
    // Separable: rows, then columns. A pixel is kept when no unset pixel is within reach, which
    // a running distance to the last unset pixel answers in one pass each way.
    let run = |line: &[bool], out: &mut [bool]| {
        let n = line.len();
        let mut since = usize::MAX / 2; // distance to the last unset pixel on the left
        for i in 0..n {
            since = if line[i] { since + 1 } else { 0 };
            out[i] = since > radius;
        }
        since = usize::MAX / 2;
        for i in (0..n).rev() {
            since = if line[i] { since + 1 } else { 0 };
            out[i] &= since > radius;
        }
    };
    let mut rows = vec![false; mask.len()];
    rows.par_chunks_mut(width).zip(mask.par_chunks(width)).for_each(|(out, line)| run(line, out));
    let mut out = vec![false; mask.len()];
    // Columns, in vertical strips so that each task owns its output columns.
    let columns: Vec<Vec<bool>> = (0..width)
        .into_par_iter()
        .map(|x| {
            let line: Vec<bool> = (0..height).map(|y| rows[y * width + x]).collect();
            let mut col = vec![false; height];
            run(&line, &mut col);
            col
        })
        .collect();
    for (x, col) in columns.iter().enumerate() {
        for (y, &v) in col.iter().enumerate() {
            out[y * width + x] = v;
        }
    }
    out
}

/// Parameters of a [`Tracker`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackParams {
    /// A frame becomes the anchor this many frames after the previous anchor, at the latest.
    /// 1 chains one-frame flows.
    pub max_stride: usize,
    /// When the median displacement from a frame to the anchor exceeds this many pixels, the
    /// previous frame becomes the anchor instead. `f32::INFINITY` disables it.
    pub max_anchor_motion: f32,
    /// When the share of pixels failing the consistency check exceeds that of the previous
    /// frame (against the same anchor) by more than this, the previous frame becomes the anchor
    /// instead. `f32::INFINITY` disables it.
    pub max_failure_jump: f32,
    /// The consistency check.
    pub check: ForwardBackward,
    /// Inconsistent regions grow by this many pixels before they clear the confidence, which
    /// also catches their unreliable fringes.
    pub erode_radius: usize,
    /// A pixel also loses its confidence when the anchor positions it is interpolated from
    /// are further apart than this many times the anchor field's median spacing (see
    /// [`PositionField::continuous_at`]): its position would be a blend across an occlusion
    /// edge. `f32::INFINITY` disables it.
    pub max_position_jump: f32,
}

impl Default for TrackParams {
    /// Anchors every 10 frames at most, re-anchoring on 20 px of median motion or a failure
    /// jump of 0.1, the default consistency check, erosion by 3 pixels and position jumps of
    /// 1.5 times the median spacing. Smaller erosion and jump limits keep a few percent more
    /// pixels confident, but let more wrong positions through at the edges of moving objects.
    fn default() -> Self {
        Self {
            max_stride: 10,
            max_anchor_motion: 20.0,
            max_failure_jump: 0.1,
            check: ForwardBackward::default(),
            erode_radius: 3,
            max_position_jump: 1.5,
        }
    }
}

impl TrackParams {
    /// Anchors exactly every `stride` frames: no motion or failure rule.
    pub fn fixed_stride(stride: usize) -> Self {
        Self { max_stride: stride, max_anchor_motion: f32::INFINITY, max_failure_jump: f32::INFINITY, ..Self::default() }
    }

    /// Checks the parameters for consistency.
    pub fn validate(&self) -> Result<(), FlowError> {
        if self.max_stride == 0 {
            return invalid("the anchor stride must be at least 1");
        }
        if self.max_anchor_motion.is_nan() || self.max_anchor_motion <= 0.0 || self.max_failure_jump.is_nan() || self.max_failure_jump < 0.0 {
            return invalid("the re-anchoring thresholds must be positive");
        }
        let (a1, a2) = (self.check.alpha1, self.check.alpha2);
        if !(a1.is_finite() && a2.is_finite() && a1 >= 0.0 && a2 >= 0.0) {
            return invalid("the consistency tolerances must be finite and non-negative");
        }
        Ok(())
    }
}

/// What happened in one [`Tracker::step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepInfo {
    /// The frame's index in the walk (the keyframe is 0).
    pub index: usize,
    /// Index of the anchor the frame's field was computed from.
    pub anchor: usize,
    /// True if this frame became the anchor for the following frames.
    pub anchored: bool,
    /// True if the motion or failure rule moved the anchor to the previous frame.
    pub reanchored: bool,
    /// Median displacement from the frame to its anchor, in pixels.
    pub median_motion: f32,
    /// Share of pixels that failed the consistency check (before erosion).
    pub failure_fraction: f32,
    /// Flows computed in this step (2, or 4 after moving the anchor).
    pub flows: usize,
}

/// A frame with its position field and confidence.
#[derive(Clone, Debug)]
struct State {
    index: usize,
    image: GrayImage,
    positions: PositionField,
    confidence: ConfidenceMap,
    /// The field's median spacing, computed when the frame becomes the anchor.
    spacing: f32,
}

/// The flows between a frame and an anchor, with the check's verdict.
struct Link {
    to_anchor: FlowField,
    consistent: Vec<bool>,
    median_motion: f32,
    failure_fraction: f32,
}

/// Maintains the position field and confidence of a walk away from a keyframe (see the
/// [module documentation](self)), computing flows with any [`FlowEstimator`].
///
/// ```no_run
/// # fn main() -> Result<(), freesynth_flow::FlowError> {
/// use freesynth_flow::cpu::CpuDis;
/// use freesynth_flow::track::{TrackParams, Tracker};
/// use freesynth_flow::{DisParams, GrayImage};
///
/// let frames: Vec<GrayImage> = (0..30).map(|_| GrayImage::new(320, 180)).collect();
/// let dis = CpuDis::new(320, 180, &DisParams::default())?;
/// let mut tracker = Tracker::new(dis, frames[0].clone(), TrackParams::default())?;
/// for frame in &frames[1..] {
///     let info = tracker.step(frame)?;
///     let (positions, confidence) = (tracker.positions(), tracker.confidence());
///     println!("frame {}: anchor {}, {:.0}% confident", info.index, info.anchor, 100.0 * confidence.fraction_confident(0.5));
/// #   let _ = positions;
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Tracker<E> {
    estimator: E,
    params: TrackParams,
    index: usize,
    anchor: State,
    /// The latest frame, when it is not the anchor, with its share of inconsistent pixels.
    previous: Option<(State, f32)>,
}

impl<E> Tracker<E> {

    pub fn params(&self) -> &TrackParams {
        &self.params
    }

    /// Index of the latest frame (0 before the first step).
    pub fn index(&self) -> usize {
        self.index
    }

    /// Index of the current anchor.
    pub fn anchor_index(&self) -> usize {
        self.anchor.index
    }

    /// The estimator, e.g. to reuse it after the walk.
    pub fn into_estimator(self) -> E {
        self.estimator
    }

    fn latest(&self) -> &State {
        self.previous.as_ref().map_or(&self.anchor, |(s, _)| s)
    }

    /// The latest frame's position field.
    pub fn positions(&self) -> &PositionField {
        &self.latest().positions
    }

    /// The latest frame's confidence.
    pub fn confidence(&self) -> &ConfidenceMap {
        &self.latest().confidence
    }
}

impl<E: FlowEstimator> Tracker<E> {
    /// Starts a walk at `keyframe` (identity positions, full confidence).
    pub fn new(estimator: E, keyframe: GrayImage, params: TrackParams) -> Result<Self, FlowError> {
        params.validate()?;
        let (w, h) = keyframe.size();
        let anchor =
            State { index: 0, positions: PositionField::identity(w, h), confidence: ConfidenceMap::ones(w, h), image: keyframe, spacing: 1.0 };
        Ok(Self { estimator, params, index: 0, anchor, previous: None })
    }

    /// Flows between `frame` and the anchor, and their consistency.
    fn link(&mut self, frame: &GrayImage) -> Result<Link, FlowError> {
        let (to_anchor, from_anchor) = self.estimator.compute_pair(frame, &self.anchor.image)?;
        let consistent = self.params.check.check(&to_anchor, &from_anchor);
        let failures = consistent.iter().filter(|&&c| !c).count();
        let failure_fraction = failures as f32 / consistent.len().max(1) as f32;
        let median_motion = median_length(&to_anchor);
        Ok(Link { to_anchor, consistent, median_motion, failure_fraction })
    }

    /// Adds the next frame of the walk. Its field and confidence are then available from
    /// [`positions`](Self::positions) and [`confidence`](Self::confidence).
    pub fn step(&mut self, frame: &GrayImage) -> Result<StepInfo, FlowError> {
        if frame.size() != self.anchor.image.size() {
            let (w, h) = self.anchor.image.size();
            return invalid(format!("frames must be {w}x{h}"));
        }
        let index = self.index + 1;
        let mut link = self.link(frame)?;
        let mut flows = 2;
        let mut reanchored = false;
        if let Some((previous, _)) = self.previous.take_if(|(_, previous_failure)| {
            let too_far = link.median_motion > self.params.max_anchor_motion;
            let jump = link.failure_fraction > *previous_failure + self.params.max_failure_jump;
            too_far || jump
        }) {
            self.anchor = State { spacing: previous.positions.median_spacing(), ..previous };
            link = self.link(frame)?;
            flows += 2;
            reanchored = true;
        }
        let (w, h) = frame.size();
        let mut keep = erode(&link.consistent, w, h, self.params.erode_radius);
        let jump = self.params.max_position_jump * self.anchor.spacing;
        let continuous = self.anchor.positions.continuous_at(&link.to_anchor, jump);
        keep.iter_mut().zip(&continuous).for_each(|(k, &c)| *k &= c);
        let state = State {
            index,
            image: frame.clone(),
            positions: self.anchor.positions.compose(&link.to_anchor),
            confidence: self.anchor.confidence.advect(&link.to_anchor, &keep),
            spacing: 1.0,
        };
        let anchor = self.anchor.index;
        let anchored = index - anchor >= self.params.max_stride;
        if anchored {
            self.anchor = State { spacing: state.positions.median_spacing(), ..state };
            self.previous = None;
        } else {
            self.previous = Some((state, link.failure_fraction));
        }
        self.index = index;
        Ok(StepInfo {
            index,
            anchor,
            anchored,
            reanchored,
            median_motion: link.median_motion,
            failure_fraction: link.failure_fraction,
            flows,
        })
    }
}

/// Median flow length over a sample of every fourth pixel in each direction.
fn median_length(flow: &FlowField) -> f32 {
    let (w, h) = flow.size();
    let mut lengths: Vec<f32> = (0..h).step_by(4).flat_map(|y| (0..w).step_by(4).map(move |x| (x, y))).map(|(x, y)| {
        let [u, v] = flow.get(x, y);
        u.hypot(v)
    }).collect();
    if lengths.is_empty() {
        return 0.0;
    }
    let mid = lengths.len() / 2;
    *lengths.select_nth_unstable_by(mid, f32::total_cmp).1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flow estimator that returns a fixed translation between frames whose first sample
    /// encodes the frame's offset (for testing the bookkeeping without real flow).
    struct Shift {
        calls: usize,
    }

    impl FlowEstimator for Shift {
        fn compute(&mut self, a: &GrayImage, b: &GrayImage) -> Result<FlowField, FlowError> {
            self.calls += 1;
            let d = b.data()[0] - a.data()[0];
            Ok(FlowField::constant(a.width(), a.height(), [d, 0.0]))
        }
    }

    fn frame(offset: f32) -> GrayImage {
        GrayImage::from_vec(32, 16, vec![offset; 32 * 16]).unwrap()
    }

    #[test]
    fn identity_composition_and_displacement() {
        let p = PositionField::identity(4, 3);
        assert_eq!(p.get(3, 2), [3.0, 2.0]);
        let flow = FlowField::constant(4, 3, [1.0, 0.5]);
        let q = p.compose(&flow);
        assert_eq!(q.get(1, 1), [2.0, 1.5]);
        // Clamped at the border.
        assert_eq!(q.get(3, 2), [3.0, 2.0]);
        assert_eq!(PositionField::from_displacement(&q.displacement()), q);
        assert_eq!(q.sample(0.5, 0.0), [1.5, 0.5]);
    }

    #[test]
    fn spacing_and_continuity() {
        let (w, h) = (12, 6);
        let identity = PositionField::identity(w, h);
        assert_eq!(identity.median_spacing(), 1.0);
        let zero = FlowField::new(w, h);
        assert!(identity.continuous_at(&zero, 1.8).iter().all(|&c| c));
        // A field whose right half moved 10 px: samples straddling the edge (between columns 5
        // and 6) blend the two halves.
        let data = identity.data().iter().map(|&[x, y]| if x >= 6.0 { [x + 10.0, y] } else { [x, y] }).collect();
        let torn = PositionField::from_vec(w, h, data);
        let ok = torn.continuous_at(&FlowField::constant(w, h, [0.5, 0.0]), 1.8);
        for (x, &c) in ok[..w].iter().enumerate() {
            assert_eq!(c, x != 5, "column {x}");
        }
        assert!(torn.continuous_at(&zero, f32::INFINITY).iter().all(|&c| c));
        // Twice the spacing everywhere (a zoom) is still continuous at twice the threshold.
        let zoomed = PositionField::from_vec(w, h, identity.data().iter().map(|&[x, y]| [2.0 * x, 2.0 * y]).collect());
        assert_eq!(zoomed.median_spacing(), 2.0);
        assert!(zoomed.continuous_at(&zero, 1.8 * zoomed.median_spacing()).iter().all(|&c| c));
    }

    #[test]
    fn fixed_point_round_trip() {
        assert_eq!(FixedPositions::frac_bits_for(960, 540), 6);
        assert_eq!(FixedPositions::frac_bits_for(1024, 10), 6);
        assert_eq!(FixedPositions::frac_bits_for(1025, 10), 5);
        assert_eq!(FixedPositions::frac_bits_for(3840, 2160), 4);
        let mut data = PositionField::identity(960, 1).data().to_vec();
        data[0] = [0.0, 959.0];
        data[1] = [12.34567, -3.0];
        let f = PositionField::from_vec(960, 1, data).to_fixed();
        assert_eq!(f.step(), 1.0 / 64.0);
        let q = f.to_positions();
        assert_eq!(q.get(0, 0), [0.0, 959.0]);
        assert!((q.get(1, 0)[0] - 12.34567).abs() <= 0.5 / 64.0);
        // Outside the range: clamped.
        assert_eq!(q.get(1, 0)[1], 0.0);
        assert_eq!(q.get(959, 0), [959.0, 0.0]);
    }

    #[test]
    fn forward_backward_check() {
        let (w, h) = (8, 4);
        let check = ForwardBackward::default();
        let f = FlowField::constant(w, h, [2.0, 0.0]);
        let ok = check.check(&f, &FlowField::constant(w, h, [-2.0, 0.0]));
        // Consistent where x + 2 stays inside the frame.
        assert!((0..h).all(|y| (0..w).all(|x| ok[y * w + x] == (x + 2 < w))));
        let bad = check.check(&f, &FlowField::constant(w, h, [1.0, 0.0]));
        assert!(bad.iter().all(|&c| !c));
        // The tolerance grows with the flow length.
        let long = FlowField::constant(w, h, [0.0, 0.0]);
        assert!(check.check(&long, &FlowField::constant(w, h, [0.6, 0.0])).iter().all(|&c| c));
    }

    #[test]
    fn erosion() {
        let (w, h) = (7, 5);
        let mut mask = vec![true; w * h];
        mask[2 * w + 3] = false;
        let e = erode(&mask, w, h, 1);
        for y in 0..h {
            for x in 0..w {
                let near = x.abs_diff(3) <= 1 && y.abs_diff(2) <= 1;
                assert_eq!(e[y * w + x], !near, "({x}, {y})");
            }
        }
        assert_eq!(erode(&mask, w, h, 0), mask);
        // The image border does not erode.
        assert!(erode(&vec![true; w * h], w, h, 2).iter().all(|&v| v));
    }

    #[test]
    fn anchors_every_stride() {
        let params = TrackParams::fixed_stride(3);
        let mut t = Tracker::new(Shift { calls: 0 }, frame(0.0), params).unwrap();
        let mut anchors = Vec::new();
        for i in 1..=7 {
            let info = t.step(&frame(-(i as f32) * 0.5)).unwrap();
            anchors.push((info.anchor, info.anchored));
            // The flow from frame i to frame a is 0.5 (i - a) px, so pixel x of frame i shows
            // keyframe position x + 0.5 i.
            let p = t.positions().get(10, 5);
            assert!((p[0] - (10.0 + 0.5 * i as f32)).abs() < 1e-5, "frame {i}: {p:?}");
            assert_eq!(info.flows, 2);
        }
        assert_eq!(anchors, [(0, false), (0, false), (0, true), (3, false), (3, false), (3, true), (6, false)]);
        assert_eq!(t.into_estimator().calls, 14);
    }

    #[test]
    fn moves_the_anchor_back_on_large_motion() {
        let params = TrackParams { max_anchor_motion: 1.2, max_failure_jump: f32::INFINITY, ..TrackParams::default() };
        let mut t = Tracker::new(Shift { calls: 0 }, frame(0.0), params).unwrap();
        let first = t.step(&frame(-0.5)).unwrap();
        assert!(!first.reanchored && first.anchor == 0);
        let second = t.step(&frame(-1.0)).unwrap();
        assert!(!second.reanchored);
        // 1.5 px from the keyframe: frame 2 becomes the anchor, frame 3 links to it.
        let third = t.step(&frame(-1.5)).unwrap();
        assert!(third.reanchored && third.anchor == 2 && third.flows == 4);
        assert!((t.positions().get(10, 5)[0] - 11.5).abs() < 1e-5);
        assert_eq!(t.anchor_index(), 2);
    }

    #[test]
    fn confidence_is_cleared_where_inconsistent() {
        // A flow of 3 px to the right: pixels whose destination leaves the frame fail the check.
        let mut t = Tracker::new(Shift { calls: 0 }, frame(0.0), TrackParams { erode_radius: 1, ..TrackParams::fixed_stride(10) }).unwrap();
        t.step(&frame(-3.0)).unwrap();
        let c = t.confidence();
        // x + 3 > 31 for x > 28; the eroded region reaches one pixel further.
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(27, 8), 1.0);
        assert_eq!(c.get(28, 8), 0.0);
        assert_eq!(c.get(31, 8), 0.0);
        assert!((c.fraction_confident(0.5) - 28.0 / 32.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_parameters_and_frames() {
        assert!(Tracker::new(Shift { calls: 0 }, frame(0.0), TrackParams { max_stride: 0, ..TrackParams::default() }).is_err());
        assert!(TrackParams { max_anchor_motion: 0.0, ..TrackParams::default() }.validate().is_err());
        assert!(TrackParams { check: ForwardBackward { alpha1: -1.0, alpha2: 0.5 }, ..TrackParams::default() }.validate().is_err());
        let mut t = Tracker::new(Shift { calls: 0 }, frame(0.0), TrackParams::default()).unwrap();
        assert!(t.step(&GrayImage::new(8, 8)).is_err());
    }
}
