//! Inverse search, propagation passes and densification on one pyramid level (Kroeger et al.
//! 2016, sections 2.1 and 2.2), as in `search.wgsl` and `densify.wgsl`.
//!
//! The GPU kernel gives each of a patch's eight rows to one thread, accumulates the row sums
//! column by column and adds the eight row sums in a fixed tree. Here the eight rows are the
//! eight lanes of small arrays, with the same accumulation order, which the compiler turns into
//! vector instructions. To make the eight rows of a column contiguous in memory, the search
//! reads transposed copies of the frames and of the Sobel gradients (`x * h + y`).
//!
//! Two shortcuts save work without changing the result: a propagation pass skips the patches
//! whose own and neighbours' displacements did not change in the previous pass (that pass made
//! its decision on the same inputs and kept the patch), and the matching cost of a patch's
//! displacement is carried over from the pass that computed it.

use rayon::prelude::*;

use super::lanes::F8;
use super::{Isa, clamp_index, dispatch, resample_flow, split_coordinate};
use crate::params::{DisParams, PATCH_SIZE};

const PS: usize = PATCH_SIZE;
const PATCH_PIXELS: f32 = (PS * PS) as f32;
/// A search that ends further than this from its start is abandoned.
const MAX_TRAVEL: f32 = PS as f32;

/// Buffers reused between flows.
#[derive(Debug, Default)]
pub(super) struct Scratch {
    /// Frame 0, its Sobel gradients and frame 1, transposed.
    t0: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
    t1: Vec<f32>,
    /// Patch displacements of the current and the previous pass.
    patches: [Vec<[f32; 2]>; 2],
    /// Matching cost of each patch's displacement (NaN if not computed), for the current and
    /// the previous pass.
    costs: [Vec<f32>; 2],
    /// Which patches changed in the current and the previous pass.
    changed: [Vec<bool>; 2],
}

/// The flow of the next coarser level.
#[derive(Clone, Copy, Debug)]
pub(super) struct Coarse<'a> {
    pub flow: &'a [[f32; 2]],
    pub w: usize,
    pub h: usize,
}

/// Everything one search pass reads.
struct Ctx<'a> {
    w: usize,
    h: usize,
    t0: &'a [f32],
    gx: &'a [f32],
    gy: &'a [f32],
    t1: &'a [f32],
    stride: usize,
    pw: usize,
    ph: usize,
    mean_norm: bool,
    margin: f32,
    iters0: usize,
    iters1: usize,
    /// Squared step length below which a search stops.
    tol2: f32,
    coarse: Option<Coarse<'a>>,
    /// The previous pass's displacements and their costs, in a propagation pass.
    previous: Option<&'a [[f32; 2]]>,
    previous_costs: &'a [f32],
    /// Which patches the previous pass changed, when it was a propagation pass.
    changed: Option<&'a [bool]>,
}

/// Fills the transposed planes for a `w` x `h` level.
fn transpose_planes(s: &mut Scratch, i0: &[f32], i1: &[f32], w: usize, h: usize) {
    for plane in [&mut s.t0, &mut s.gx, &mut s.gy, &mut s.t1] {
        plane.resize(w * h, 0.0);
    }
    let at0 = |x: isize, y: isize| i0[clamp_index(y, h) * w + clamp_index(x, w)];
    (s.t0.par_chunks_mut(h), s.gx.par_chunks_mut(h), s.gy.par_chunks_mut(h), s.t1.par_chunks_mut(h)).into_par_iter().enumerate().for_each(
        |(x, (t0, gx, gy, t1))| {
            let xi = x as isize;
            for y in 0..h {
                let yi = y as isize;
                let (a, b, c) = (at0(xi - 1, yi - 1), at0(xi, yi - 1), at0(xi + 1, yi - 1));
                let (d, f) = (at0(xi - 1, yi), at0(xi + 1, yi));
                let (g, hh, i) = (at0(xi - 1, yi + 1), at0(xi, yi + 1), at0(xi + 1, yi + 1));
                t0[y] = i0[y * w + x];
                gx[y] = 0.125 * ((c + 2.0 * f + i) - (a + 2.0 * d + g));
                gy[y] = 0.125 * ((g + 2.0 * hh + i) - (a + 2.0 * b + c));
                t1[y] = i1[y * w + x];
            }
        },
    );
}

/// Runs the first search pass and the propagation passes on a level with a `pw` x `ph` patch
/// grid, and returns the final patch displacements.
#[allow(clippy::too_many_arguments)]
pub(super) fn search_level<'s>(
    s: &'s mut Scratch,
    i0: &[f32],
    i1: &[f32],
    w: usize,
    h: usize,
    p: &DisParams,
    pw: usize,
    ph: usize,
    coarse: Option<Coarse<'_>>,
    isa: Isa,
) -> &'s [[f32; 2]] {
    transpose_planes(s, i0, i1, w, h);
    for buf in &mut s.patches {
        buf.resize(pw * ph, [0.0; 2]);
    }
    for buf in &mut s.changed {
        buf.resize(pw * ph, true);
    }
    for buf in &mut s.costs {
        buf.resize(pw * ph, f32::NAN);
    }
    let passes = p.propagation_passes + 1;
    for pass in 0..passes {
        let (a, b) = s.patches.split_at_mut(1);
        let (out, previous) = if pass % 2 == 0 { (&mut a[0], &b[0]) } else { (&mut b[0], &a[0]) };
        let (a, b) = s.changed.split_at_mut(1);
        let (changed_out, changed) = if pass % 2 == 0 { (&mut a[0], &b[0]) } else { (&mut b[0], &a[0]) };
        let (a, b) = s.costs.split_at_mut(1);
        let (costs_out, costs) = if pass % 2 == 0 { (&mut a[0], &b[0]) } else { (&mut b[0], &a[0]) };
        let ctx = Ctx {
            w,
            h,
            t0: &s.t0,
            gx: &s.gx,
            gy: &s.gy,
            t1: &s.t1,
            stride: p.patch_stride,
            pw,
            ph,
            mean_norm: p.mean_normalization,
            margin: p.propagation_margin,
            iters0: p.gd_iterations,
            iters1: p.propagation_iterations,
            tol2: p.gd_tolerance * p.gd_tolerance,
            coarse,
            previous: (pass > 0).then_some(previous.as_slice()),
            previous_costs: costs,
            changed: (pass > 1).then_some(changed.as_slice()),
        };
        (out.par_chunks_mut(pw), costs_out.par_chunks_mut(pw), changed_out.par_chunks_mut(pw)).into_par_iter().enumerate().for_each(
            |(pj, (row, cost_row, changed_row))| {
                let ctx = &ctx;
                dispatch!(isa, (ctx: &Ctx<'_>, pj: usize, row: &mut [[f32; 2]], cost_row: &mut [f32], changed_row: &mut [bool]) {
                    search_row(ctx, pj, row, cost_row, changed_row)
                });
            },
        );
    }
    &s.patches[(passes - 1) % 2]
}

/// One row of patches of a search pass; also records each result's cost and which patches
/// changed.
#[inline(always)]
fn search_row(ctx: &Ctx<'_>, pj: usize, row: &mut [[f32; 2]], cost_row: &mut [f32], changed_row: &mut [bool]) {
    for (pi, ((out, cost), changed)) in row.iter_mut().zip(cost_row.iter_mut()).zip(changed_row.iter_mut()).enumerate() {
        let p = pj * ctx.pw + pi;
        (*out, *cost) = match ctx.previous {
            Some(previous) => {
                if ctx.changed.is_some_and(|c| !neighbourhood_changed(ctx, c, p)) {
                    // Same inputs as in the previous pass, which kept this patch: so does
                    // this one.
                    (previous[p], ctx.previous_costs[p])
                } else {
                    let patch = Patch::new(ctx, pi, pj);
                    match patch.propagation_start(ctx, p, previous) {
                        (start, Some(start_cost)) => patch.descend(ctx, start, Some(start_cost), ctx.iters1),
                        (own, None) => (own, ctx.previous_costs[p]),
                    }
                }
            }
            None => {
                let patch = Patch::new(ctx, pi, pj);
                let start = match ctx.coarse {
                    Some(c) => resample_flow(c.flow, c.w, c.h, 2.0, patch.px as f32 + 3.5, patch.py as f32 + 3.5),
                    None => [0.0; 2],
                };
                patch.descend(ctx, start, None, ctx.iters0)
            }
        };
        *changed = ctx.previous.is_none_or(|previous| *out != previous[p]);
    }
}

/// `!(a <= b)`: true when `a > b` or either is NaN.
#[inline(always)]
fn not_at_most(a: f32, b: f32) -> bool {
    a.partial_cmp(&b).is_none_or(|o| o.is_gt())
}

/// True if patch `p` or one of its four grid neighbours changed in the previous pass.
#[inline(always)]
fn neighbourhood_changed(ctx: &Ctx<'_>, changed: &[bool], p: usize) -> bool {
    let (pi, pj) = (p % ctx.pw, p / ctx.pw);
    changed[p]
        || (pi > 0 && changed[p - 1])
        || (pi + 1 < ctx.pw && changed[p + 1])
        || (pj > 0 && changed[p - ctx.pw])
        || (pj + 1 < ctx.ph && changed[p + ctx.pw])
}

/// A patch of frame 0: its template, gradients and inverse Hessian, one lane per patch row and
/// one entry per column.
struct Patch {
    px: usize,
    py: usize,
    t: [F8; PS],
    gx: [F8; PS],
    gy: [F8; PS],
    /// Sums of the gradients over the patch.
    gs: [f32; 2],
    /// Inverse of the (regularised) Hessian: `(xx, xy, yy)`.
    inv: [f32; 3],
}

impl Patch {
    #[inline(always)]
    fn new(ctx: &Ctx<'_>, pi: usize, pj: usize) -> Self {
        let px = (pi * ctx.stride).min(ctx.w - PS);
        let py = (pj * ctx.stride).min(ctx.h - PS);
        let zero = F8::splat(0.0);
        let (mut t, mut gx, mut gy) = ([zero; PS], [zero; PS], [zero; PS]);
        for k in 0..PS {
            let at = (px + k) * ctx.h + py;
            t[k] = F8::load(&ctx.t0[at..]);
            gx[k] = F8::load(&ctx.gx[at..]);
            gy[k] = F8::load(&ctx.gy[at..]);
        }
        let (mut sxx, mut sxy, mut syy, mut sx, mut sy) = (zero, zero, zero, zero, zero);
        for k in 0..PS {
            sxx = gx[k].mul_add(gx[k], sxx);
            sxy = gx[k].mul_add(gy[k], sxy);
            syy = gy[k].mul_add(gy[k], syy);
            sx += gx[k];
            sy += gy[k];
        }
        let gs = [sx.tree_sum(), sy.tree_sum()];
        let (mut hxx, mut hxy, mut hyy) = (sxx.tree_sum(), sxy.tree_sum(), syy.tree_sum());
        if ctx.mean_norm {
            hxx -= gs[0] * gs[0] / PATCH_PIXELS;
            hxy -= gs[0] * gs[1] / PATCH_PIXELS;
            hyy -= gs[1] * gs[1] / PATCH_PIXELS;
        }
        hxx += 1e-2;
        hyy += 1e-2;
        let det = hxx * hyy - hxy * hxy;
        Self { px, py, t, gx, gy, gs, inv: [hyy / det, -hxy / det, hxx / det] }
    }

    /// Sums over the patch for displacement `u`: `(sum gx r, sum gy r, sum r, sum r^2)`, with
    /// `r` the difference between frame 1 at the displaced position (bilinear, clamped) and the
    /// template.
    #[inline(always)]
    fn residual_sums(&self, ctx: &Ctx<'_>, u: [f32; 2]) -> [f32; 4] {
        let (w, h, t1) = (ctx.w, ctx.h, ctx.t1);
        let (ix, ax) = split_coordinate(self.px as f32 + u[0]);
        let (iy, ay) = split_coordinate(self.py as f32 + u[1]);
        // Frame 1 interpolated along y, for the nine columns that the patch touches.
        let mut col = [F8::splat(0.0); PS + 1];
        if ix >= 0 && iy >= 0 && ix as usize + PS < w && iy as usize + PS < h {
            for (c, out) in col.iter_mut().enumerate() {
                let v = &t1[(ix as usize + c) * h + iy as usize..];
                *out = F8::mix(F8::load(v), F8::load(&v[1..]), ay);
            }
        } else {
            for (c, out) in col.iter_mut().enumerate() {
                let base = clamp_index(ix + c as isize, w) * h;
                let r0 = F8::from_fn(|j| t1[base + clamp_index(iy + j as isize, h)]);
                let r1 = F8::from_fn(|j| t1[base + clamp_index(iy + j as isize + 1, h)]);
                *out = F8::mix(r0, r1, ay);
            }
        }
        let zero = F8::splat(0.0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        for k in 0..PS {
            let r = F8::mix(col[k], col[k + 1], ax) - self.t[k];
            a0 = self.gx[k].mul_add(r, a0);
            a1 = self.gy[k].mul_add(r, a1);
            a2 += r;
            a3 = r.mul_add(r, a3);
        }
        [a0.tree_sum(), a1.tree_sum(), a2.tree_sum(), a3.tree_sum()]
    }

    /// Matching cost from the residual sums.
    #[inline(always)]
    fn cost(ctx: &Ctx<'_>, s: [f32; 4]) -> f32 {
        if ctx.mean_norm { s[3] - s[2] * s[2] / PATCH_PIXELS } else { s[3] }
    }

    /// `iters` Gauss-Newton iterations from `u0`. Falls back to `u0` when the search ends with
    /// a higher cost than it started with or travels further than a patch size. Returns the
    /// displacement and its cost (NaN if it was not evaluated). `u0_cost` is the cost at `u0`
    /// if known.
    #[inline(always)]
    fn descend(&self, ctx: &Ctx<'_>, u0: [f32; 2], u0_cost: Option<f32>, iters: usize) -> ([f32; 2], f32) {
        let mut u = u0;
        let mut start_cost = u0_cost.unwrap_or(f32::NAN);
        for it in 0..iters {
            let s = self.residual_sums(ctx, u);
            if it == 0 {
                start_cost = Self::cost(ctx, s);
            }
            let (mut bx, mut by) = (s[0], s[1]);
            if ctx.mean_norm {
                bx -= self.gs[0] * s[2] / PATCH_PIXELS;
                by -= self.gs[1] * s[2] / PATCH_PIXELS;
            }
            let inv = self.inv;
            let step = [inv[0] * bx + inv[1] * by, inv[1] * bx + inv[2] * by];
            u = [u[0] - step[0], u[1] - step[1]];
            let (dx, dy) = (u[0] - u0[0], u[1] - u0[1]);
            // Equivalent to comparing the distance with MAX_TRAVEL; NaN counts as too far.
            if not_at_most(dx * dx + dy * dy, MAX_TRAVEL * MAX_TRAVEL) {
                return (u0, start_cost);
            }
            if step[0] * step[0] + step[1] * step[1] < ctx.tol2 {
                break;
            }
        }
        if iters > 0 {
            let end_cost = Self::cost(ctx, self.residual_sums(ctx, u));
            if not_at_most(end_cost, start_cost) {
                return (u0, start_cost);
            }
            return (u, end_cost);
        }
        (u, start_cost)
    }

    /// The start of a propagation pass for patch `p`: the best of the four grid neighbours'
    /// displacements and its cost if it costs less than `margin` times the patch's own;
    /// otherwise the patch's previous result, which it keeps, and `None`.
    #[inline(always)]
    fn propagation_start(&self, ctx: &Ctx<'_>, p: usize, previous: &[[f32; 2]]) -> ([f32; 2], Option<f32>) {
        let (pi, pj) = (p % ctx.pw, p / ctx.pw);
        let own = previous[p];
        let own_cost = match ctx.previous_costs[p] {
            c if c.is_nan() => Self::cost(ctx, self.residual_sums(ctx, own)),
            c => c,
        };
        let mut best = own_cost;
        let mut start = None;
        let neighbours = [
            (pi > 0).then(|| p - 1),
            (pi + 1 < ctx.pw).then(|| p + 1),
            (pj > 0).then(|| p - ctx.pw),
            (pj + 1 < ctx.ph).then(|| p + ctx.pw),
        ];
        for n in neighbours.into_iter().flatten() {
            let c = previous[n];
            if c == own {
                continue;
            }
            let e = Self::cost(ctx, self.residual_sums(ctx, c));
            if e < best && e < ctx.margin * own_cost {
                best = e;
                start = Some(c);
            }
        }
        match start {
            Some(c) => (c, Some(best)),
            None => (own, None),
        }
    }
}

/// Densification: the flow at a pixel is the average of the displacements of all patches that
/// cover it, each weighted by `1 / max(1, |d|)`, where `d` is the difference between frame 1 at
/// the displaced position and frame 0 at the pixel.
///
/// Each output row adds up the eight pixels that every covering patch contributes to it, in
/// the order of the GPU kernel (patch rows, then patches, ascending).
#[allow(clippy::too_many_arguments)]
pub(super) fn densify(
    i0: &[f32],
    i1: &[f32],
    w: usize,
    h: usize,
    stride: usize,
    pw: usize,
    ph: usize,
    patches: &[[f32; 2]],
    out: &mut [[f32; 2]],
    isa: Isa,
) {
    let ctx = Densify { i0, i1, w, h, stride, pw, ph, patches };
    out.par_chunks_mut(w).enumerate().for_each_init(
        || vec![0.0f32; 3 * w],
        |acc, (y, row)| {
            let ctx = &ctx;
            let acc = &mut acc[..];
            dispatch!(isa, (ctx: &Densify<'_>, y: usize, row: &mut [[f32; 2]], acc: &mut [f32]) { densify_row(ctx, y, row, acc) });
        },
    );
}

/// Inputs of the densification.
struct Densify<'a> {
    i0: &'a [f32],
    i1: &'a [f32],
    w: usize,
    h: usize,
    stride: usize,
    pw: usize,
    ph: usize,
    patches: &'a [[f32; 2]],
}

/// Output row `y` of the densification; `acc` is scratch space for `3 * w` values.
#[inline(always)]
fn densify_row(ctx: &Densify<'_>, y: usize, row: &mut [[f32; 2]], acc: &mut [f32]) {
    let (w, h, s) = (ctx.w, ctx.h, ctx.stride);
    acc.fill(0.0);
    let (acc_u, rest) = acc.split_at_mut(w);
    let (acc_v, acc_w) = rest.split_at_mut(w);
    let t_row = &ctx.i0[y * w..(y + 1) * w];
    // Patch rows that cover y: origins min(j * s, h - 8) with origin <= y < origin + 8. All
    // but the last patch row have origin j * s; the last one ends at the border.
    let hi = (y / s).min(ctx.ph - 1);
    let lo = if y >= PS - 1 { (y - (PS - 1)).div_ceil(s) } else { 0 };
    let extra = (ctx.ph - 1 > hi && y + PS >= h).then_some(ctx.ph - 1);
    for j in (lo..=hi).chain(extra) {
        let oy = (j * s).min(h - PS);
        if y < oy || y >= oy + PS {
            continue;
        }
        for i in 0..ctx.pw {
            let ox = (i * s).min(w - PS);
            let u = ctx.patches[j * ctx.pw + i];
            let (iy, ay) = split_coordinate(y as f32 + u[1]);
            let (r0, r1) = (clamp_index(iy, h) * w, clamp_index(iy + 1, h) * w);
            // Lane k is pixel ox + k. All lanes share the integer part of the first lane's
            // position; the per-lane fraction is taken relative to it, so that the sample
            // stays continuous even when rounding moves a lane across an integer.
            let sx = F8::from_fn(|k| (ox + k) as f32) + F8::splat(u[0]);
            let (ix, _) = split_coordinate(sx.0[0]);
            let ax = sx - F8::from_fn(|k| (ix + k as isize) as f32);
            let (top, bottom) = if ix >= 0 && ix as usize + PS < w {
                let (a, b) = (&ctx.i1[r0 + ix as usize..], &ctx.i1[r1 + ix as usize..]);
                (F8::mix_lanes(F8::load(a), F8::load(&a[1..]), ax), F8::mix_lanes(F8::load(b), F8::load(&b[1..]), ax))
            } else {
                let at = |r: usize, k: usize| ctx.i1[r + clamp_index(ix + k as isize, w)];
                (
                    F8::mix_lanes(F8::from_fn(|k| at(r0, k)), F8::from_fn(|k| at(r0, k + 1)), ax),
                    F8::mix_lanes(F8::from_fn(|k| at(r1, k)), F8::from_fn(|k| at(r1, k + 1)), ax),
                )
            };
            let d = F8::mix(top, bottom, ay) - F8::load(&t_row[ox..]);
            let wt = F8::splat(1.0) / d.abs().max(F8::splat(1.0));
            wt.mul_add(F8::splat(u[0]), F8::load(&acc_u[ox..])).store(&mut acc_u[ox..]);
            wt.mul_add(F8::splat(u[1]), F8::load(&acc_v[ox..])).store(&mut acc_v[ox..]);
            (F8::load(&acc_w[ox..]) + wt).store(&mut acc_w[ox..]);
        }
    }
    // Every pixel is covered by at least one patch.
    for (x, o) in row.iter_mut().enumerate() {
        let norm = acc_w[x].max(1e-20);
        *o = [acc_u[x] / norm, acc_v[x] / norm];
    }
}
