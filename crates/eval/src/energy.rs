//! Patch energy: how well every output patch matches a source patch under the guides.
//!
//! The energy of a target pixel is the engine's patch error (weighted SSD over style and guide
//! channels, see [`PatchErrorEvaluator`]) divided by the number of pixels in a patch, so it is in
//! weighted 8-bit² units per pixel. The style and guide parts are reported separately: the guide
//! part depends mostly on the inputs, so differences in output quality show first in the style
//! part.
//!
//! Two estimators are provided:
//!
//! - [`nnf_energy`]: the exact energy of a given nearest-neighbour field, normally the engine's
//!   final field. It needs the field, so it works only for FreeSynth outputs.
//! - [`sampled_energy`]: the coherence energy of any image, measured by exhaustive search over
//!   every source patch for a seeded random subset of target pixels. It is exact on those pixels
//!   (no randomised search, so no search bias) and an unbiased estimate of the full-image mean.
//!   Two images measured on the same [`PixelSet`] can be compared pixel by pixel. Candidates that
//!   a cheap lower bound (patch means and norms) proves worse than the best match so far are
//!   skipped, which makes the search several times faster without changing its result.
//!
//! The sampled estimate of an absolute mean has a standard error of about 2% at 20,000 pixels,
//! because per-pixel energies are heavy-tailed; a paired comparison of two images on the same
//! pixels is much more precise than that.
//!
//! Both measure the coherence (weighted SSD) part of FreeSynth's objective only. The engine also
//! minimises a uniformity penalty on how often each source patch is used; that term is not
//! included here, so a synthesis that reuses a few patches everywhere scores lower than one that
//! spreads its matches over the source. Read energy next to [`crate::occupancy`], and see the
//! crate documentation for what energy can and cannot show.

use freesynth_core::{ErrorPart, Nnf, PatchErrorEvaluator};
use rayon::prelude::*;

use crate::error::{Error, invalid};
use crate::report::PixelSet;
use crate::rng::SplitMix64;
use crate::stats::{mean, quantile_sorted, sorted, std_err};

/// Energy of a nearest-neighbour field.
#[derive(Clone, Debug, PartialEq)]
pub struct NnfEnergy {
    /// Mean per-pixel style energy.
    pub style: f64,
    /// Mean per-pixel guide energy.
    pub guide: f64,
    /// `style + guide`.
    pub total: f64,
    /// Median, 90th and 99th percentile of the per-pixel total energy.
    pub total_p50: f64,
    pub total_p90: f64,
    pub total_p99: f64,
    /// Per-pixel total energy in scan-line order.
    pub per_pixel: Vec<f32>,
}

/// Exact energy of `nnf` (see the module documentation).
pub fn nnf_energy(evaluator: &PatchErrorEvaluator, nnf: &Nnf) -> Result<NnfEnergy, Error> {
    let errors = evaluator.nnf_errors(nnf)?;
    let area = (evaluator.patch_size() * evaluator.patch_size()) as f32;
    let per_pixel: Vec<f32> = errors.style.iter().zip(&errors.guide).map(|(s, g)| (s + g) / area).collect();
    let n = per_pixel.len() as f64;
    let style = errors.style.iter().map(|&v| v as f64).sum::<f64>() / n / area as f64;
    let guide = errors.guide.iter().map(|&v| v as f64).sum::<f64>() / n / area as f64;
    let s = sorted(&per_pixel.iter().map(|&v| v as f64).collect::<Vec<_>>());
    Ok(NnfEnergy {
        style,
        guide,
        total: style + guide,
        total_p50: quantile_sorted(&s, 0.5),
        total_p90: quantile_sorted(&s, 0.9),
        total_p99: quantile_sorted(&s, 0.99),
        per_pixel,
    })
}

/// Key prefix of pixel sets drawn by [`sample_pixels`]. Change the version when the sampling
/// changes, so that sets drawn by different versions never count as equal.
const SAMPLER: &str = "uniform-v1";

/// Draws `n` distinct target pixels uniformly at random (all pixels when `n >= width * height`),
/// in scan-line order. The result depends only on the arguments.
pub fn sample_pixels(width: usize, height: usize, n: usize, seed: u64) -> PixelSet {
    let total = width * height;
    let key = format!("pixels/{SAMPLER}/{width}x{height}/n{}/seed{seed}", n.min(total));
    let mut idx: Vec<u32> = (0..total as u32).collect();
    if n < total {
        // Partial Fisher-Yates shuffle: the first n entries are a uniform random subset.
        let mut rng = SplitMix64::new(seed);
        for i in 0..n {
            let j = i + rng.below(total - i);
            idx.swap(i, j);
        }
        idx.truncate(n);
        idx.sort_unstable();
    }
    let coords = idx.iter().map(|&i| [i % width as u32, i / width as u32]).collect();
    PixelSet { key, width, height, coords }
}

/// The exact best match of one sampled pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledMatch {
    /// The target pixel.
    pub target: [u32; 2],
    /// Centre of the best source patch (the first in scan-line order among equals).
    pub source: [i32; 2],
    /// Per-pixel total energy of the best match.
    pub total: f32,
    /// Per-pixel style energy of the best match.
    pub style: f32,
    /// Per-pixel total energy of the hint field at this pixel, when a hint was given.
    pub hint_total: Option<f32>,
}

/// Summary of a [`sampled_energy`] run.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledEnergy {
    pub matches: Vec<SampledMatch>,
    /// Mean per-pixel total, style and guide energy over the sampled pixels.
    pub total: f64,
    pub style: f64,
    pub guide: f64,
    /// Standard errors of `total` and `style` as estimates of the full-image means.
    pub total_se: f64,
    pub style_se: f64,
    /// Mean energy of the hint field on the same pixels.
    pub hint_total: Option<f64>,
    /// Fraction of pixels where the hint field is worse than the exact best match.
    pub hint_suboptimal_frac: Option<f64>,
}

/// Per-source-centre patch statistics that give a cheap lower bound on the patch error.
///
/// For any two patches `a` and `b` of `n` pixels and channel weights `w`,
///
/// ```text
/// sum_p sum_k w_k (a_pk - b_pk)^2 >= sum_k w_k (sum_p a_pk - sum_p b_pk)^2 / n   (Cauchy-Schwarz)
/// sum_p sum_k w_k (a_pk - b_pk)^2 >= (|a|_w - |b|_w)^2                         (triangle inequality)
/// ```
///
/// where `|a|_w` is the weighted norm of the whole patch. Candidates whose bound already exceeds
/// the best error found so far cannot be the minimum and are skipped without evaluating the
/// patch distance, which does not change the result of the search.
///
/// The centres are kept sorted by norm. The norm bound then rules out everything outside one
/// contiguous window around the target patch's norm, so the search only visits that window.
struct PatchBounds {
    /// Channels per pixel.
    nc: usize,
    /// Width of the grid of valid source centres.
    cw: usize,
    /// Pixels per patch.
    n: f32,
    weights: Vec<f32>,
    /// Centre numbers (scan-line order over valid centres), sorted by weighted patch norm.
    order: Vec<u32>,
    /// Position of every centre number in `order`.
    rank: Vec<u32>,
    /// Weighted patch norms, in `order`.
    norms: Vec<f32>,
    /// `nc` channel sums per centre, in `order`.
    sums: Vec<f32>,
}

impl PatchBounds {
    /// `None` when the guide error is modulated per target pixel (the weights then vary).
    fn new(ev: &PatchErrorEvaluator) -> Option<Self> {
        if ev.has_modulation() {
            return None;
        }
        let weights = ev.weights(ErrorPart::Total).to_vec();
        let nc = weights.len();
        let planes = nc + 1;
        let p = ev.patch_size();
        let (sw, sh) = ev.source_size();
        let (cw, ch) = (sw + 1 - p, sh + 1 - p);
        let value = |px: &[u8], k: usize| -> f64 {
            if k < nc {
                px[k] as f64
            } else {
                px.iter().zip(&weights).map(|(&v, &w)| w as f64 * (v as f64).powi(2)).sum()
            }
        };
        // Horizontal box sums over every source row, then vertical sums over those.
        let rows: Vec<f64> = (0..sh)
            .into_par_iter()
            .flat_map_iter(|y| {
                let mut out = vec![0.0f64; cw * planes];
                for cx in 0..cw {
                    for x in cx..cx + p {
                        let px = ev.source_pixel(x, y);
                        for k in 0..planes {
                            out[cx * planes + k] += value(px, k);
                        }
                    }
                }
                out
            })
            .collect();
        let mut table = vec![0.0f32; cw * ch * planes];
        table.par_chunks_mut(cw * planes).enumerate().for_each(|(cy, out)| {
            for (i, o) in out.iter_mut().enumerate() {
                let s: f64 = (cy..cy + p).map(|y| rows[y * cw * planes + i]).sum();
                *o = if i % planes == nc { s.sqrt() as f32 } else { s as f32 };
            }
        });
        drop(rows);
        let count = cw * ch;
        let mut order: Vec<u32> = (0..count as u32).collect();
        // Stable, so equal norms keep scan-line order (the result does not depend on it).
        order.par_sort_by(|&a, &b| table[a as usize * planes + nc].total_cmp(&table[b as usize * planes + nc]));
        let mut rank = vec![0u32; count];
        for (i, &c) in order.iter().enumerate() {
            rank[c as usize] = i as u32;
        }
        let norms = order.iter().map(|&c| table[c as usize * planes + nc]).collect();
        let sums = order.iter().flat_map(|&c| table[c as usize * planes..c as usize * planes + nc].iter().copied()).collect();
        Some(Self { nc, cw, n: (p * p) as f32, weights, order, rank, norms, sums })
    }

    /// The statistics of the (border-clamped) target patch at `t`: `nc` channel sums, then the
    /// weighted norm.
    fn target_stats(&self, ev: &PatchErrorEvaluator, t: [u32; 2]) -> Vec<f32> {
        let r = (ev.patch_size() / 2) as i64;
        let (tw, th) = ev.target_size();
        let mut sums = vec![0.0f64; self.nc + 1];
        for dy in -r..=r {
            for dx in -r..=r {
                let x = (t[0] as i64 + dx).clamp(0, tw as i64 - 1) as usize;
                let y = (t[1] as i64 + dy).clamp(0, th as i64 - 1) as usize;
                let px = ev.target_pixel(x, y);
                for (k, &v) in px.iter().enumerate() {
                    sums[k] += v as f64;
                    sums[self.nc] += self.weights[k] as f64 * (v as f64).powi(2);
                }
            }
        }
        sums[self.nc] = sums[self.nc].sqrt();
        sums.into_iter().map(|v| v as f32).collect()
    }

    /// Whether the source patch at sorted position `i` is proven worse than `bound` for the
    /// target patch with statistics `ts`: first by the norm bound (one subtraction), then by the
    /// mean bound.
    #[inline]
    fn skip(&self, ts: &[f32], i: usize, bound: f32) -> bool {
        let dn = ts[self.nc] - self.norms[i];
        if prunable(dn * dn, bound) {
            return true;
        }
        let s = &self.sums[i * self.nc..(i + 1) * self.nc];
        let mut mean_bound = 0.0f32;
        for k in 0..self.nc {
            let d = ts[k] - s[k];
            mean_bound += self.weights[k] * d * d;
        }
        prunable(mean_bound / self.n, bound)
    }

    /// The sorted positions whose norm bound does not exceed `bound`: every other candidate
    /// would be skipped by [`skip`](Self::skip) anyway. The window is slightly wider than the
    /// exact one, to stay on the safe side of rounding.
    fn window(&self, ts: &[f32], bound: f32) -> std::ops::Range<usize> {
        let tn = ts[self.nc];
        let r = (bound * (1.0 + 1e-4) + 1e-2).sqrt() * (1.0 + 1e-5) + 1e-3 + 1e-6 * tn;
        self.norms.partition_point(|&v| v < tn - r)..self.norms.partition_point(|&v| v <= tn + r)
    }
}

/// Spacing of the coarse grid of centres that seeds the search when no hint is given.
const START_STRIDE: usize = 4;

/// Whether a candidate with lower bound `lb` can be skipped against the running minimum `best`.
/// The margin absorbs rounding in the f32 bound and in the patch distance.
#[inline]
fn prunable(lb: f32, best: f32) -> bool {
    lb > best * (1.0 + 1e-4) + 1e-2
}

/// Exhaustive nearest-patch search for the given target pixels (see the module documentation).
///
/// Every source patch inside the source is considered; pixels run in parallel on the current
/// rayon pool. Candidates that a cheap lower bound proves worse than the best match so far are
/// skipped, so the result is the same as a plain scan. `hint` (for example the engine's final
/// field) only seeds the running minimum; the result is the same with or without it. When given,
/// its energy on the same pixels is reported as well.
///
/// The bounds need fixed channel weights. When the input has a target modulation (per-pixel
/// guide weights), the search falls back to the plain scan, whose cost is pixels x source
/// centres x patch area: hours for 20,000 pixels at a source size of about 1200x900.
pub fn sampled_energy(evaluator: &PatchErrorEvaluator, pixels: &PixelSet, hint: Option<&Nnf>) -> Result<SampledEnergy, Error> {
    sampled_energy_with(evaluator, pixels, hint, true)
}

/// [`sampled_energy`], optionally without the lower-bound pruning. The result is the same either
/// way; the unpruned scan is much slower and exists to verify that.
pub fn sampled_energy_with(evaluator: &PatchErrorEvaluator, pixels: &PixelSet, hint: Option<&Nnf>, prune: bool) -> Result<SampledEnergy, Error> {
    let (tw, th) = evaluator.target_size();
    if (pixels.width, pixels.height) != (tw, th) {
        return invalid(format!("pixel set is for a {}x{} target, the target is {tw}x{th}", pixels.width, pixels.height));
    }
    if pixels.coords.iter().any(|&[x, y]| x as usize >= tw || y as usize >= th) {
        return invalid("pixel set has pixels outside the target");
    }
    if pixels.coords.is_empty() {
        return invalid("pixel set is empty");
    }
    let (xs, ys) = evaluator.source_centres();
    let hint = match hint {
        Some(h) if (h.width(), h.height()) != (tw, th) => {
            return invalid(format!("hint NNF is {}x{}, the target is {tw}x{th}", h.width(), h.height()));
        }
        Some(h) => {
            if h.data().iter().any(|s| !xs.contains(&s[0]) || !ys.contains(&s[1])) {
                return invalid("hint NNF points outside the valid source centres");
            }
            Some(h)
        }
        None => None,
    };
    let total = evaluator.distance(ErrorPart::Total);
    let style = evaluator.distance(ErrorPart::Style);
    let area = (evaluator.patch_size() * evaluator.patch_size()) as f32;
    let (sw, sh) = evaluator.source_size();
    let bounds = if prune { PatchBounds::new(evaluator) } else { None };

    let matches: Vec<SampledMatch> = pixels
        .coords
        .par_iter()
        .map(|&[x, y]| {
            let t = [x as i32, y as i32];
            let hint_err = hint.map(|h| total.error(t, h.get(x as usize, y as usize), f32::INFINITY));
            let (best, best_s) = match &bounds {
                Some(b) => {
                    let ts = b.target_stats(evaluator, [x, y]);
                    // Seed the running minimum with the hint, or with a search over a coarse grid.
                    let start = hint_err.unwrap_or_else(|| {
                        let mut best = f32::INFINITY;
                        for sy in ys.clone().step_by(START_STRIDE) {
                            for sx in xs.clone().step_by(START_STRIDE) {
                                let c = (sy - ys.start) as usize * b.cw + (sx - xs.start) as usize;
                                if !b.skip(&ts, b.rank[c] as usize, best) {
                                    best = best.min(total.error(t, [sx, sy], best));
                                }
                            }
                        }
                        best
                    });
                    // Visit the norm window in norm order. Ties resolve to the lowest centre
                    // number, the first match in scan-line order, as in the plain scan.
                    let (mut best, mut best_c) = (f32::INFINITY, usize::MAX);
                    for i in b.window(&ts, start) {
                        let bound = best.min(start);
                        if b.skip(&ts, i, bound) {
                            continue;
                        }
                        let c = b.order[i] as usize;
                        let e = total.error(t, [xs.start + (c % b.cw) as i32, ys.start + (c / b.cw) as i32], bound);
                        if e <= bound && (e < best || (e == best && c < best_c)) {
                            best = e;
                            best_c = c;
                        }
                    }
                    let best_s = if best_c == usize::MAX {
                        [xs.start, ys.start]
                    } else {
                        [xs.start + (best_c % b.cw) as i32, ys.start + (best_c / b.cw) as i32]
                    };
                    (best, best_s)
                }
                None => {
                    // The plain scan in scan-line order, seeded with the hint or with the
                    // proportional position in the source. The seed only tightens the early
                    // exit, so ties resolve to the first match in scan-line order.
                    let start = hint_err.unwrap_or_else(|| {
                        let sx = ((x as usize * sw / tw) as i32).clamp(xs.start, xs.end - 1);
                        let sy = ((y as usize * sh / th) as i32).clamp(ys.start, ys.end - 1);
                        total.error(t, [sx, sy], f32::INFINITY)
                    });
                    let (mut best, mut best_s) = (f32::INFINITY, [xs.start, ys.start]);
                    for sy in ys.clone() {
                        for sx in xs.clone() {
                            let bound = best.min(start);
                            let e = total.error(t, [sx, sy], bound);
                            if e < best && e <= bound {
                                best = e;
                                best_s = [sx, sy];
                            }
                        }
                    }
                    (best, best_s)
                }
            };
            SampledMatch {
                target: [x, y],
                source: best_s,
                total: best / area,
                style: style.error(t, best_s, f32::INFINITY) / area,
                hint_total: hint_err.map(|e| e / area),
            }
        })
        .collect();

    let totals: Vec<f64> = matches.iter().map(|m| m.total as f64).collect();
    let styles: Vec<f64> = matches.iter().map(|m| m.style as f64).collect();
    let (total, style) = (mean(&totals), mean(&styles));
    let (hint_total, hint_suboptimal_frac) = if hint.is_some() {
        let h: Vec<f64> = matches.iter().map(|m| m.hint_total.unwrap_or(f32::NAN) as f64).collect();
        let worse = matches.iter().filter(|m| m.hint_total.is_some_and(|h| h > m.total * (1.0 + 1e-6))).count();
        (Some(mean(&h)), Some(worse as f64 / matches.len() as f64))
    } else {
        (None, None)
    };
    Ok(SampledEnergy {
        total,
        style,
        guide: total - style,
        total_se: std_err(&totals),
        style_se: std_err(&styles),
        hint_total,
        hint_suboptimal_frac,
        matches,
    })
}

#[cfg(test)]
mod tests {
    use freesynth_core::{Image, SynthesisInput};

    use super::*;

    /// A random-texture RGB image.
    pub(crate) fn noise_image(w: usize, h: usize, seed: u64) -> Image {
        let mut rng = SplitMix64::new(seed);
        Image::from_raw(w, h, 3, (0..w * h * 3).map(|_| rng.below(256) as u8).collect())
    }

    fn crop(img: &Image, x0: usize, y0: usize, w: usize, h: usize) -> Image {
        let mut out = Image::new(w, h, img.channels());
        for y in 0..h {
            for x in 0..w {
                out.pixel_mut(x, y).copy_from_slice(img.pixel(x0 + x, y0 + y));
            }
        }
        out
    }

    fn input(src: &Image, tgt_guide: &Image) -> SynthesisInput {
        SynthesisInput {
            source_style: src.clone(),
            source_guide: src.clone(),
            target_guide: tgt_guide.clone(),
            target_modulation: None,
            style_weights: vec![1.0 / 3.0; 3],
            guide_weights: vec![1.0 / 3.0; 3],
        }
    }

    #[test]
    fn pixel_sets_are_reproducible_and_distinct() {
        let a = sample_pixels(30, 20, 100, 9);
        assert_eq!(a, sample_pixels(30, 20, 100, 9));
        assert_ne!(a.coords, sample_pixels(30, 20, 100, 10).coords);
        assert_eq!(a.coords.len(), 100);
        let mut idx: Vec<u32> = a.coords.iter().map(|&[x, y]| y * 30 + x).collect();
        assert!(idx.windows(2).all(|w| w[0] < w[1]), "sorted and distinct");
        idx.dedup();
        assert_eq!(idx.len(), 100);
        let all = sample_pixels(5, 4, 1000, 1);
        assert_eq!(all.coords.len(), 20);
        assert_eq!(all.key, "pixels/uniform-v1/5x4/n20/seed1");
    }

    #[test]
    fn identity_synthesis_has_zero_energy() {
        let img = noise_image(32, 24, 1);
        let inp = input(&img, &img);
        let ev = PatchErrorEvaluator::new(&inp, &img, 5).unwrap();
        let nnf = Nnf::from_raw(32, 24, (0..32 * 24).map(|i| [(i % 32).clamp(2, 29), (i / 32).clamp(2, 21)]).collect());
        let e = nnf_energy(&ev, &nnf).unwrap();
        // Only the border pixels (mapped to the nearest valid centre) have energy.
        let interior: f64 = (2..22).flat_map(|y| (2..30).map(move |x| (x, y))).map(|(x, y)| e.per_pixel[y * 32 + x] as f64).sum();
        assert_eq!(interior, 0.0);

        // The exhaustive search finds zero energy at every interior pixel too.
        let px = PixelSet {
            key: "test".into(),
            width: 32,
            height: 24,
            coords: (2..22).step_by(3).flat_map(|y| (2..30).step_by(5).map(move |x| [x, y])).collect(),
        };
        let s = sampled_energy(&ev, &px, None).unwrap();
        assert_eq!(s.total, 0.0);
        for m in &s.matches {
            assert_eq!(m.source, [m.target[0] as i32, m.target[1] as i32]);
        }
    }

    #[test]
    fn known_shift_is_found_exactly() {
        // Target = source shifted by (7, 4): the exact search must find that offset everywhere
        // a full patch fits, with zero energy; the hint does not change the answer.
        let src = noise_image(40, 30, 2);
        let tgt = crop(&src, 7, 4, 26, 20);
        let inp = input(&src, &tgt);
        let ev = PatchErrorEvaluator::new(&inp, &tgt, 5).unwrap();
        let px = sample_pixels(26, 20, 60, 3);
        let s = sampled_energy(&ev, &px, None).unwrap();
        for m in &s.matches {
            let [x, y] = m.target;
            if (2..24).contains(&x) && (2..18).contains(&y) {
                assert_eq!(m.source, [x as i32 + 7, y as i32 + 4]);
                assert_eq!(m.total, 0.0);
            }
        }
        // A deliberately bad hint: same result, and the hint is reported as suboptimal.
        let bad = Nnf::from_raw(26, 20, vec![[2, 2]; 26 * 20]);
        let h = sampled_energy(&ev, &px, Some(&bad)).unwrap();
        assert_eq!(h.matches.iter().map(|m| m.source).collect::<Vec<_>>(), s.matches.iter().map(|m| m.source).collect::<Vec<_>>());
        assert_eq!(h.total, s.total);
        assert!(h.hint_total.unwrap() > 1000.0);
        assert!(h.hint_suboptimal_frac.unwrap() > 0.9);

        // The exact field of the shift gives the same energies through nnf_energy.
        let shift = Nnf::from_raw(26, 20, (0..26 * 20).map(|i| [(i % 26 + 7).clamp(2, 37), (i / 26 + 4).clamp(2, 27)]).collect());
        let e = nnf_energy(&ev, &shift).unwrap();
        for m in &s.matches {
            let i = (m.target[1] * 26 + m.target[0]) as usize;
            assert!(e.per_pixel[i] >= m.total * (1.0 - 1e-5), "exact search must not be beaten: {m:?} vs {}", e.per_pixel[i]);
        }
    }

    #[test]
    fn energy_parts_and_units() {
        // Style differs by a constant 10 on every channel; the guide matches exactly.
        let mut src_clamped = noise_image(20, 20, 4);
        for v in src_clamped.data_mut() {
            *v = (*v).min(240);
        }
        let mut style = src_clamped.clone();
        for v in style.data_mut() {
            *v += 10;
        }
        let inp = input(&src_clamped, &src_clamped);
        let ev = PatchErrorEvaluator::new(&inp, &style, 3).unwrap();
        let nnf = Nnf::from_raw(20, 20, (0..400).map(|i| [(i % 20).clamp(1, 18), (i / 20).clamp(1, 18)]).collect());
        let e = nnf_energy(&ev, &nnf).unwrap();
        // Interior pixels: 3 channels * (1/3) * 10^2 = 100 per patch pixel.
        assert!((e.per_pixel[5 * 20 + 5] - 100.0).abs() < 1e-3);
        assert!(e.style > 99.0);
        assert!((e.total - e.style - e.guide).abs() < 1e-9);
    }

    #[test]
    fn pruning_does_not_change_the_result() {
        // A smooth source with mild noise and a different, smooth target guide: many candidates
        // are close, so the bounds are tested near their limit.
        let smooth = |w: usize, h: usize, seed: u64, f: f32| {
            let mut rng = SplitMix64::new(seed);
            let data = (0..w * h * 3)
                .map(|i| {
                    let (x, y, c) = (((i / 3) % w) as f32, ((i / 3) / w) as f32, (i % 3) as f32);
                    (128.0 + 90.0 * (x * f + c).sin() * (y * 0.8 * f).cos() + rng.below(9) as f32 - 4.0).clamp(0.0, 255.0) as u8
                })
                .collect();
            Image::from_raw(w, h, 3, data)
        };
        let src = smooth(45, 37, 1, 0.13);
        let tgt = smooth(31, 29, 2, 0.17);
        let out = smooth(31, 29, 3, 0.15);
        let mut inp = input(&src, &tgt);
        inp.style_weights = vec![0.5, 1.0, 2.0];
        for patch in [3, 5] {
            let ev = PatchErrorEvaluator::new(&inp, &out, patch).unwrap();
            let px = sample_pixels(31, 29, 120, 4);
            let plain = sampled_energy_with(&ev, &px, None, false).unwrap();
            let pruned = sampled_energy_with(&ev, &px, None, true).unwrap();
            assert_eq!(plain.matches, pruned.matches, "patch {patch}");
            let hint = Nnf::from_raw(31, 29, vec![[20, 20]; 31 * 29]);
            let hinted = sampled_energy_with(&ev, &px, Some(&hint), true).unwrap();
            for (a, b) in plain.matches.iter().zip(&hinted.matches) {
                assert_eq!((a.source, a.total, a.style), (b.source, b.total, b.style));
            }
        }
    }

    #[test]
    fn ties_resolve_to_the_first_centre_in_scan_order() {
        // A periodic source: every patch occurs many times with exactly the same error, so the
        // norm-ordered search must pick the same (first) centre as the plain scan.
        let tile = noise_image(6, 5, 8);
        let mut src = Image::new(36, 30, 3);
        for y in 0..30 {
            for x in 0..36 {
                src.pixel_mut(x, y).copy_from_slice(tile.pixel(x % 6, y % 5));
            }
        }
        let tgt = crop(&src, 9, 7, 20, 16);
        let ev = PatchErrorEvaluator::new(&input(&src, &tgt), &tgt, 3).unwrap();
        let px = sample_pixels(20, 16, 80, 2);
        let plain = sampled_energy_with(&ev, &px, None, false).unwrap();
        let pruned = sampled_energy_with(&ev, &px, None, true).unwrap();
        assert_eq!(plain.matches, pruned.matches);
        for m in &pruned.matches {
            let [x, y] = m.target;
            if (1..19).contains(&x) && (1..15).contains(&y) {
                assert_eq!(m.total, 0.0);
                // The first centre with this phase of the tile.
                let (fx, fy) = ((x as i32 + 9 - 1).rem_euclid(6) + 1, (y as i32 + 7 - 1).rem_euclid(5) + 1);
                assert_eq!(m.source, [fx, fy], "{m:?}");
            }
        }
    }

    #[test]
    fn rejects_mismatched_pixel_sets() {
        let img = noise_image(16, 16, 5);
        let ev = PatchErrorEvaluator::new(&input(&img, &img), &img, 3).unwrap();
        assert!(sampled_energy(&ev, &sample_pixels(15, 16, 10, 1), None).is_err());
        let mut px = sample_pixels(16, 16, 10, 1);
        px.coords.push([16, 0]);
        assert!(sampled_energy(&ev, &px, None).is_err());
        assert!(sampled_energy(&ev, &sample_pixels(16, 16, 10, 1), Some(&Nnf::new(16, 16))).is_err());
    }
}
