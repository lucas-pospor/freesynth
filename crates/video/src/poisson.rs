//! A multigrid solver for the screened Poisson equation, and the box filters the blend uses.
//!
//! The adaptive blend ([`crate::BlendMode::Adaptive`]) finds the image `u` that minimises
//! `lambda |u - f|^2 + |grad u - g|^2` for a data term `f` and a target gradient field `g`. Its
//! normal equations are the screened Poisson equation `(lambda - Laplacian) u = lambda f - div g`
//! with Neumann (reflecting) boundaries, which [`solve_screened`] solves with multigrid V-cycles:
//! red-black Gauss-Seidel smoothing, cell-centred coarsening by 2x2 averaging and bilinear
//! interpolation back. With the weak data terms the blend uses (`lambda` around 0.05) the system
//! is well conditioned, and a handful of cycles reach a residual far below one grey level.

use rayon::prelude::*;

/// Grids at most this large in either dimension are solved by smoothing alone.
const COARSEST: usize = 4;

/// Smoothing sweeps before and after each coarse-grid correction.
const SWEEPS: usize = 2;

/// Rows of a `w`-wide grid that one parallel task handles at least: small grids (the coarse
/// levels) are not worth splitting into many tasks.
fn min_rows(w: usize) -> usize {
    (1 << 14) / w.max(1) + 1
}

/// One level of the hierarchy: the operator is `lambda u - scale * Laplacian(u)` on a `w` x `h`
/// grid (`scale` is `4^-level`: the grid spacing doubles at every level).
#[derive(Clone, Copy)]
struct Level {
    w: usize,
    h: usize,
    lambda: f32,
    scale: f32,
}

impl Level {
    fn coarser(&self) -> Level {
        Level { w: self.w.div_ceil(2), h: self.h.div_ceil(2), lambda: self.lambda, scale: self.scale * 0.25 }
    }

    /// Sum of the inside neighbours of `(x, y)` in `u`, and their number.
    #[inline]
    fn neighbours(&self, u: &[f32], x: usize, y: usize) -> (f32, f32) {
        let (w, h) = (self.w, self.h);
        let i = y * w + x;
        let (mut sum, mut n) = (0.0, 0.0);
        if x > 0 {
            sum += u[i - 1];
            n += 1.0;
        }
        if x + 1 < w {
            sum += u[i + 1];
            n += 1.0;
        }
        if y > 0 {
            sum += u[i - w];
            n += 1.0;
        }
        if y + 1 < h {
            sum += u[i + w];
            n += 1.0;
        }
        (sum, n)
    }

    /// Red-black Gauss-Seidel sweeps on `u` for right-hand side `b`.
    fn smooth(&self, u: &mut [f32], b: &[f32], sweeps: usize) {
        let w = self.w;
        let mut snapshot = vec![0.0f32; u.len()];
        for _ in 0..sweeps {
            for colour in 0..2 {
                // Cells of one colour only read cells of the other, which this half-sweep leaves alone.
                snapshot.copy_from_slice(u);
                u.par_chunks_mut(w).enumerate().with_min_len(min_rows(w)).for_each(|(y, row)| {
                    for x in ((y + colour) % 2..w).step_by(2) {
                        let (sum, n) = self.neighbours(&snapshot, x, y);
                        row[x] = (b[y * w + x] + self.scale * sum) / (self.lambda + self.scale * n);
                    }
                });
            }
        }
    }

    /// `b - A u`.
    fn residual(&self, u: &[f32], b: &[f32]) -> Vec<f32> {
        let w = self.w;
        let mut r = vec![0.0f32; u.len()];
        r.par_chunks_mut(w).enumerate().with_min_len(min_rows(w)).for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                let (sum, n) = self.neighbours(u, x, y);
                let i = y * w + x;
                *v = b[i] - ((self.lambda + self.scale * n) * u[i] - self.scale * sum);
            }
        });
        r
    }

    /// Averages 2x2 blocks of `r` onto the coarser grid (partial blocks at odd edges average the
    /// cells they have).
    fn restrict(&self, r: &[f32], coarse: &Level) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let mut out = vec![0.0f32; coarse.w * coarse.h];
        out.par_chunks_mut(coarse.w).enumerate().with_min_len(min_rows(coarse.w)).for_each(|(cy, row)| {
            let ys = 2 * cy..(2 * cy + 2).min(h);
            for (cx, v) in row.iter_mut().enumerate() {
                let xs = 2 * cx..(2 * cx + 2).min(w);
                let n = (ys.len() * xs.len()) as f32;
                *v = ys.clone().flat_map(|y| xs.clone().map(move |x| r[y * w + x])).sum::<f32>() / n;
            }
        });
        out
    }

    /// Adds the bilinear (cell-centred) interpolation of the coarse correction `e` to `u`.
    fn prolong_add(&self, u: &mut [f32], e: &[f32], coarse: &Level) {
        let (cw, ch) = (coarse.w, coarse.h);
        // Fine cell i lies at coarse coordinate i / 2 - 1/4: 3/4 on its own coarse cell, 1/4 on
        // the neighbour towards it (clamped at the edges).
        let taps = |i: usize, n: usize| {
            let k = i / 2;
            let other = if i.is_multiple_of(2) { k.saturating_sub(1) } else { (k + 1).min(n - 1) };
            (k, other)
        };
        u.par_chunks_mut(self.w).enumerate().with_min_len(min_rows(self.w)).for_each(|(y, row)| {
            let (y0, y1) = taps(y, ch);
            for (x, v) in row.iter_mut().enumerate() {
                let (x0, x1) = taps(x, cw);
                let at = |cx: usize, cy: usize| e[cy * cw + cx];
                *v += 0.5625 * at(x0, y0) + 0.1875 * (at(x1, y0) + at(x0, y1)) + 0.0625 * at(x1, y1);
            }
        });
    }

    fn v_cycle(&self, u: &mut [f32], b: &[f32]) {
        if self.w <= COARSEST || self.h <= COARSEST {
            self.smooth(u, b, 32);
            return;
        }
        self.smooth(u, b, SWEEPS);
        let coarse = self.coarser();
        let rc = self.restrict(&self.residual(u, b), &coarse);
        let mut ec = vec![0.0f32; rc.len()];
        coarse.v_cycle(&mut ec, &rc);
        self.prolong_add(u, &ec, &coarse);
        self.smooth(u, b, SWEEPS);
    }
}

/// Euclidean norm, summed row by row in a fixed order (the result does not depend on thread
/// scheduling).
fn norm(v: &[f32], w: usize) -> f64 {
    let rows: Vec<f64> = v
        .par_chunks(w)
        .with_min_len(min_rows(w))
        .map(|r| r.iter().map(|&x| f64::from(x) * f64::from(x)).sum())
        .collect();
    rows.iter().sum::<f64>().sqrt()
}

/// Solves `(lambda - Laplacian) u = b` on a `w` x `h` grid (row-major, Neumann boundaries; the
/// Laplacian sums `u(q) - u(p)` over the 4 neighbours `q` inside the grid), improving the initial
/// guess in `u` until the residual norm is below `tolerance` times the norm of `b`, or after
/// `max_cycles` V-cycles. Returns the number of cycles run.
///
/// # Panics
/// If the lengths do not match the grid or `lambda` is not positive.
pub(crate) fn solve_screened(
    u: &mut [f32],
    b: &[f32],
    (w, h): (usize, usize),
    lambda: f32,
    tolerance: f64,
    max_cycles: usize,
) -> usize {
    assert!(u.len() == w * h && b.len() == w * h, "grid size mismatch");
    assert!(lambda > 0.0, "the screened Poisson equation needs a positive lambda");
    if w == 0 || h == 0 {
        return 0;
    }
    let level = Level { w, h, lambda, scale: 1.0 };
    let target = tolerance * norm(b, w);
    for cycle in 0..max_cycles {
        if norm(&level.residual(u, b), w) <= target {
            return cycle;
        }
        level.v_cycle(u, b);
    }
    max_cycles
}

/// Box sizes of three successive box filters that approximate a Gaussian of standard deviation
/// `sigma` (Kovesi, "Fast almost-Gaussian filtering", 2010), as radii.
fn box_radii(sigma: f32) -> [usize; 3] {
    let ideal = (12.0 * sigma * sigma / 3.0 + 1.0).sqrt();
    let mut lower = ideal.floor() as i64;
    if lower % 2 == 0 {
        lower -= 1;
    }
    let lower = lower.max(1);
    let upper = lower + 2;
    let m = ((12.0 * sigma * sigma - 3.0 * (lower * lower) as f32 - 12.0 * lower as f32 - 9.0)
        / (-4.0 * lower as f32 - 4.0))
        .round() as i64;
    std::array::from_fn(|i| if (i as i64) < m { (lower / 2) as usize } else { (upper / 2) as usize })
}

/// Mean over a `2 r + 1` window along rows, with the window clamped to the row.
fn box_rows(src: &[f32], dst: &mut [f32], w: usize, r: usize) {
    dst.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(out, row)| {
        let mut prefix = Vec::with_capacity(w + 1);
        prefix.push(0.0f64);
        for &v in row {
            prefix.push(prefix[prefix.len() - 1] + f64::from(v));
        }
        for (x, o) in out.iter_mut().enumerate() {
            let (a, b) = (x.saturating_sub(r), (x + r + 1).min(w));
            *o = ((prefix[b] - prefix[a]) / (b - a) as f64) as f32;
        }
    });
}

fn transpose(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for (y, v) in col.iter_mut().enumerate() {
            *v = src[y * w + x];
        }
    });
    out
}

/// An approximately Gaussian blur of standard deviation `sigma` (three box filters per axis,
/// windows clamped at the edges). `sigma <= 0` returns the input unchanged.
pub(crate) fn gaussian_blur(data: &[f32], (w, h): (usize, usize), sigma: f32) -> Vec<f32> {
    assert_eq!(data.len(), w * h, "grid size mismatch");
    if sigma <= 0.0 || w == 0 || h == 0 {
        return data.to_vec();
    }
    let radii = box_radii(sigma);
    let mut a = data.to_vec();
    let mut b = vec![0.0f32; a.len()];
    for &r in &radii {
        box_rows(&a, &mut b, w, r);
        std::mem::swap(&mut a, &mut b);
    }
    let mut a = transpose(&a, w, h);
    for &r in &radii {
        box_rows(&a, &mut b, h, r);
        std::mem::swap(&mut a, &mut b);
    }
    transpose(&a, h, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Applies `lambda - Laplacian` (Neumann), for checking solutions.
    fn apply(u: &[f32], w: usize, h: usize, lambda: f32) -> Vec<f32> {
        let level = Level { w, h, lambda, scale: 1.0 };
        let zeros = vec![0.0; u.len()];
        level.residual(u, &zeros).iter().map(|r| -r).collect()
    }

    fn field(w: usize, h: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(7);
        (0..w * h)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((i % w) as f32 * 0.7).sin() * 40.0 + (s >> 24) as f32 * 0.3
            })
            .collect()
    }

    #[test]
    fn converges_on_odd_and_even_grids() {
        for (w, h) in [(64, 48), (97, 61), (5, 300), (1, 1), (3, 2)] {
            for lambda in [0.05f32, 1.0] {
                let truth = field(w, h, (w * h) as u32);
                let b = apply(&truth, w, h, lambda);
                let mut u = vec![0.0f32; w * h];
                let cycles = solve_screened(&mut u, &b, (w, h), lambda, 1e-6, 40);
                assert!(cycles < 40, "{w}x{h} lambda {lambda}: no convergence");
                let err = u.iter().zip(&truth).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(err < 0.05, "{w}x{h} lambda {lambda}: max error {err} after {cycles} cycles");
            }
        }
    }

    #[test]
    fn multigrid_needs_few_cycles() {
        let (w, h) = (480, 270);
        let truth = field(w, h, 3);
        let b = apply(&truth, w, h, 0.05);
        let mut u = vec![0.0f32; w * h];
        let cycles = solve_screened(&mut u, &b, (w, h), 0.05, 1e-5, 50);
        assert!(cycles <= 12, "{cycles} cycles");
    }

    #[test]
    fn constant_data_is_a_fixed_point() {
        // With b = lambda * c the solution is the constant c.
        let (w, h) = (33, 17);
        let b = vec![0.05 * 100.0; w * h];
        let mut u = vec![100.0f32; w * h];
        assert!(solve_screened(&mut u, &b, (w, h), 0.05, 1e-4, 10) <= 1);
        assert!(u.iter().all(|v| (v - 100.0).abs() < 1e-3));
    }

    #[test]
    fn blur_matches_a_gaussian() {
        let (w, h) = (101, 1);
        let mut impulse = vec![0.0f32; w];
        impulse[50] = 1.0;
        for sigma in [1.0f32, 3.0, 8.0] {
            let out = gaussian_blur(&impulse, (w, h), sigma);
            let sum: f32 = out.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "mass {sum}");
            let var: f32 = out.iter().enumerate().map(|(x, v)| v * (x as f32 - 50.0).powi(2)).sum();
            assert!((var.sqrt() - sigma).abs() < 0.15 * sigma + 0.3, "sigma {sigma}: got {}", var.sqrt());
        }
        // Both axes, and a constant stays constant.
        let flat = gaussian_blur(&vec![3.0; 40 * 30], (40, 30), 5.0);
        assert!(flat.iter().all(|v| (v - 3.0).abs() < 1e-5));
        assert_eq!(gaussian_blur(&[1.0, 2.0], (2, 1), 0.0), [1.0, 2.0]);
    }
}
