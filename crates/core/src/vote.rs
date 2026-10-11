//! Patch voting: every target pixel becomes the average of the source pixels that the
//! overlapping target patches map onto it.

use rayon::prelude::*;

use crate::image::{Image, Nnf};

/// How a vote's average is converted back to an 8-bit value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VoteRounding {
    /// Round to the nearest value. Unbiased.
    #[default]
    Nearest,
    /// Drop the fraction, as the 2019 engines do. Biases the output down by about 0.2-0.5
    /// levels on average; needed to reproduce 2019 results bit for bit.
    Truncate,
}

impl VoteRounding {
    /// Converts an average in `[0, 255]` to 8 bits.
    #[inline(always)]
    fn apply(self, v: f32) -> u8 {
        match self {
            VoteRounding::Nearest => (v + 0.5) as u8,
            VoteRounding::Truncate => v as u8,
        }
    }
}

/// Plain voting (`krnlVotePlain`, `EBSYNTH_VOTEMODE_PLAIN`): every overlapping patch has weight 1.
///
/// Patches that fall outside the target, or whose source pixel would fall outside the source,
/// are skipped (as in the CPU engine). Writes `target` (which must have the NNF's size and the
/// source's channel count), converting averages with `rounding`.
pub fn vote_plain(target: &mut Image, source: &Image, nnf: &Nnf, patch_size: usize, rounding: VoteRounding) {
    let (w, h) = (nnf.width(), nnf.height());
    assert_eq!(target.size(), (w, h), "vote target must match the NNF size");
    assert_eq!(target.channels(), source.channels());
    if w == 0 || h == 0 || source.channels() == 0 {
        return;
    }
    match source.channels() {
        1 => vote_plain_n::<1>(target, source, nnf, patch_size, rounding),
        2 => vote_plain_n::<2>(target, source, nnf, patch_size, rounding),
        3 => vote_plain_n::<3>(target, source, nnf, patch_size, rounding),
        4 => vote_plain_n::<4>(target, source, nnf, patch_size, rounding),
        5 => vote_plain_n::<5>(target, source, nnf, patch_size, rounding),
        6 => vote_plain_n::<6>(target, source, nnf, patch_size, rounding),
        7 => vote_plain_n::<7>(target, source, nnf, patch_size, rounding),
        8 => vote_plain_n::<8>(target, source, nnf, patch_size, rounding),
        _ => vote_plain_any(target, source, nnf, patch_size, rounding),
    }
}

/// [`vote_plain`] for `N` channels: the accumulator lives in registers and the patch window is
/// clipped to the target once per pixel. Accumulation order is the same as the original's.
fn vote_plain_n<const N: usize>(
    target: &mut Image,
    source: &Image,
    nnf: &Nnf,
    patch_size: usize,
    rounding: VoteRounding,
) {
    let (w, h) = (nnf.width(), nnf.height());
    let r = patch_size / 2;
    let (sw, sh) = (source.width() as i32, source.height() as i32);
    let src = source.data().as_chunks::<N>().0;
    let field = nnf.data();
    target.data_mut().par_chunks_mut(w * N).enumerate().for_each(|(y, row)| {
        let (ylo, yhi) = (y.saturating_sub(r), (y + r).min(h - 1));
        for (x, out) in row.as_chunks_mut::<N>().0.iter_mut().enumerate() {
            let (xlo, xhi) = (x.saturating_sub(r), (x + r).min(w - 1));
            let mut sum = [0.0f32; N];
            let mut count = 0.0f32;
            for yy in ylo..=yhi {
                let py = yy as i32 - y as i32;
                let frow = &field[yy * w + xlo..=yy * w + xhi];
                for (k, &[nx, ny]) in frow.iter().enumerate() {
                    let px = (xlo + k) as i32 - x as i32;
                    let (sx, sy) = (nx - px, ny - py);
                    if sx >= 0 && sx < sw && sy >= 0 && sy < sh {
                        let s = &src[sy as usize * sw as usize + sx as usize];
                        for (a, &v) in sum.iter_mut().zip(s) {
                            *a += v as f32;
                        }
                        count += 1.0;
                    }
                }
            }
            // The centre patch always lands inside the source, so count > 0.
            for (o, &a) in out.iter_mut().zip(&sum) {
                *o = rounding.apply(a / count);
            }
        }
    });
}

/// [`vote_plain`] for any channel count.
fn vote_plain_any(target: &mut Image, source: &Image, nnf: &Nnf, patch_size: usize, rounding: VoteRounding) {
    let (w, h) = (nnf.width() as i32, nnf.height() as i32);
    let nc = source.channels();
    let r = (patch_size / 2) as i32;
    let (sw, sh) = (source.width() as i32, source.height() as i32);
    let src = source.data();
    let field = nnf.data();
    target.data_mut().par_chunks_mut(w as usize * nc).enumerate().for_each(|(y, row)| {
        let y = y as i32;
        let mut sum = vec![0.0f32; nc];
        for (x, out) in row.chunks_exact_mut(nc).enumerate() {
            let x = x as i32;
            sum.fill(0.0);
            let mut count = 0.0f32;
            for py in -r..=r {
                for px in -r..=r {
                    let (xx, yy) = (x + px, y + py);
                    if xx < 0 || xx >= w || yy < 0 || yy >= h {
                        continue;
                    }
                    let [nx, ny] = field[(yy * w + xx) as usize];
                    let (sx, sy) = (nx - px, ny - py);
                    if sx >= 0 && sx < sw && sy >= 0 && sy < sh {
                        let p = (sy * sw + sx) as usize * nc;
                        for (a, &v) in sum.iter_mut().zip(&src[p..p + nc]) {
                            *a += v as f32;
                        }
                        count += 1.0;
                    }
                }
            }
            for (o, &a) in out.iter_mut().zip(sum.iter()) {
                *o = rounding.apply(a / count);
            }
        }
    });
}

/// Weighted voting (`krnlVoteWeighted`, `EBSYNTH_VOTEMODE_WEIGHTED`): each patch contributes with
/// weight `1 / (1 + e)`, where `e` is its matching error normalised by the number of patch
/// samples (`patch_size^2 * channels`). Gives sharper results than plain voting.
///
/// As in the CUDA engine (whose textures clamp their coordinates), every target pixel sums all
/// `patch_size^2` overlapping patches: neighbours outside the target are read at the nearest
/// target pixel, and source positions outside the source at the nearest source pixel. If every
/// weight is zero (errors of `+inf`) or the weights are not finite, the pixel falls back to equal
/// weights.
///
/// `errors` holds the per-target-pixel patch errors of `nnf`. Averages are converted with
/// `rounding`.
pub fn vote_weighted(
    target: &mut Image,
    source: &Image,
    nnf: &Nnf,
    errors: &[f32],
    patch_size: usize,
    rounding: VoteRounding,
) {
    let (w, h) = (nnf.width(), nnf.height());
    assert_eq!(target.size(), (w, h), "vote target must match the NNF size");
    assert_eq!(target.channels(), source.channels());
    assert_eq!(errors.len(), w * h);
    let nc = source.channels();
    if w == 0 || h == 0 || nc == 0 {
        return;
    }
    assert!(source.width() > 0 && source.height() > 0, "cannot vote from an empty source");
    let norm = (patch_size * patch_size * nc) as f32;
    // One weight per target pixel (its patch's), computed once instead of once per overlap.
    let weights: Vec<f32> = errors.par_iter().map(|&e| 1.0 / (1.0 + e / norm)).collect();
    match nc {
        1 => vote_weighted_n::<1>(target, source, nnf, &weights, patch_size, rounding),
        2 => vote_weighted_n::<2>(target, source, nnf, &weights, patch_size, rounding),
        3 => vote_weighted_n::<3>(target, source, nnf, &weights, patch_size, rounding),
        4 => vote_weighted_n::<4>(target, source, nnf, &weights, patch_size, rounding),
        5 => vote_weighted_n::<5>(target, source, nnf, &weights, patch_size, rounding),
        6 => vote_weighted_n::<6>(target, source, nnf, &weights, patch_size, rounding),
        7 => vote_weighted_n::<7>(target, source, nnf, &weights, patch_size, rounding),
        8 => vote_weighted_n::<8>(target, source, nnf, &weights, patch_size, rounding),
        _ => vote_weighted_any(target, source, nnf, &weights, patch_size, rounding),
    }
}

/// [`vote_weighted`] for `N` channels with precomputed per-pixel `weights` (accumulators in
/// registers). Accumulation order is the same as [`vote_weighted_any`]'s, so results are too.
///
/// Away from the target border, and when every field entry is a valid patch centre (as the
/// engine's always are), no coordinate can need clamping, so those pixels take a loop without
/// clamps.
fn vote_weighted_n<const N: usize>(
    target: &mut Image,
    source: &Image,
    nnf: &Nnf,
    weights: &[f32],
    patch_size: usize,
    rounding: VoteRounding,
) {
    let (w, h) = (nnf.width() as i32, nnf.height() as i32);
    let r = (patch_size / 2) as i32;
    let (sw, sh) = (source.width() as i32, source.height() as i32);
    let src = source.data().as_chunks::<N>().0;
    let field = nnf.data();
    let centres_valid = field.par_iter().all(|&[x, y]| x >= r && x < sw - r && y >= r && y < sh - r);
    target.data_mut().par_chunks_mut(w as usize * N).enumerate().for_each(|(y, row)| {
        let y = y as i32;
        for (x, out) in row.as_chunks_mut::<N>().0.iter_mut().enumerate() {
            let x = x as i32;
            let interior = centres_valid && x >= r && x < w - r && y >= r && y < h - r;
            let accumulate = |weighted: bool| {
                let mut sum = [0.0f32; N];
                let mut sum_w = 0.0f32;
                if interior && weighted {
                    for py in -r..=r {
                        let base = (y + py) * w + x;
                        for px in -r..=r {
                            let idx = (base + px) as usize;
                            let [nx, ny] = field[idx];
                            let wgt = weights[idx];
                            for (a, &v) in sum.iter_mut().zip(&src[((ny - py) * sw + nx - px) as usize]) {
                                *a += wgt * v as f32;
                            }
                            sum_w += wgt;
                        }
                    }
                    return (sum, sum_w);
                }
                for py in -r..=r {
                    let yy = (y + py).clamp(0, h - 1);
                    for px in -r..=r {
                        let xx = (x + px).clamp(0, w - 1);
                        let idx = (yy * w + xx) as usize;
                        let [nx, ny] = field[idx];
                        let sx = (nx - px).clamp(0, sw - 1);
                        let sy = (ny - py).clamp(0, sh - 1);
                        let wgt = if weighted { weights[idx] } else { 1.0 };
                        for (a, &v) in sum.iter_mut().zip(&src[(sy * sw + sx) as usize]) {
                            *a += wgt * v as f32;
                        }
                        sum_w += wgt;
                    }
                }
                (sum, sum_w)
            };
            let (mut sum, mut sum_w) = accumulate(true);
            if !(sum_w > 0.0 && sum_w.is_finite()) {
                (sum, sum_w) = accumulate(false);
            }
            for (o, &a) in out.iter_mut().zip(&sum) {
                *o = rounding.apply(a / sum_w);
            }
        }
    });
}

/// [`vote_weighted`] for any channel count, with precomputed per-pixel `weights`.
fn vote_weighted_any(
    target: &mut Image,
    source: &Image,
    nnf: &Nnf,
    weights: &[f32],
    patch_size: usize,
    rounding: VoteRounding,
) {
    let (w, h) = (nnf.width(), nnf.height());
    let nc = source.channels();
    let r = (patch_size / 2) as i32;
    let (sw, sh) = (source.width() as i32, source.height() as i32);
    let src = source.data();
    let field = nnf.data();
    let (wi, hi) = (w as i32, h as i32);
    target.data_mut().par_chunks_mut(w * nc).enumerate().for_each(|(y, row)| {
        let y = y as i32;
        let mut sum = vec![0.0f32; nc];
        for (x, out) in row.chunks_exact_mut(nc).enumerate() {
            let x = x as i32;
            let mut accumulate = |weighted: bool| {
                sum.fill(0.0);
                let mut sum_w = 0.0f32;
                for py in -r..=r {
                    let yy = (y + py).clamp(0, hi - 1);
                    for px in -r..=r {
                        let xx = (x + px).clamp(0, wi - 1);
                        let idx = (yy * wi + xx) as usize;
                        let [nx, ny] = field[idx];
                        let sx = (nx - px).clamp(0, sw - 1);
                        let sy = (ny - py).clamp(0, sh - 1);
                        let wgt = if weighted { weights[idx] } else { 1.0 };
                        let p = (sy * sw + sx) as usize * nc;
                        for (a, &v) in sum.iter_mut().zip(&src[p..p + nc]) {
                            *a += wgt * v as f32;
                        }
                        sum_w += wgt;
                    }
                }
                sum_w
            };
            let mut sum_w = accumulate(true);
            if !(sum_w > 0.0 && sum_w.is_finite()) {
                sum_w = accumulate(false);
            }
            for (o, &a) in out.iter_mut().zip(sum.iter()) {
                *o = rounding.apply(a / sum_w);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize, nc: usize) -> Image {
        let mut img = Image::new(w, h, nc);
        for (i, v) in img.data_mut().iter_mut().enumerate() {
            *v = (i * 37 % 251) as u8;
        }
        img
    }

    fn identity_nnf(w: usize, h: usize) -> Nnf {
        let mut nnf = Nnf::new(w, h);
        for y in 0..h {
            for x in 0..w {
                nnf.set(x, y, [x as i32, y as i32]);
            }
        }
        nnf
    }

    #[test]
    fn identity_field_reproduces_source() {
        let src = ramp(13, 9, 3);
        let nnf = identity_nnf(13, 9);
        let mut out = Image::new(13, 9, 3);
        vote_plain(&mut out, &src, &nnf, 5, VoteRounding::Truncate);
        assert_eq!(out, src);

        let errors = vec![0.0; 13 * 9];
        let mut out = Image::new(13, 9, 3);
        vote_weighted(&mut out, &src, &nnf, &errors, 5, VoteRounding::Truncate);
        // Clamped addressing blends border pixels with their neighbours (as in the CUDA
        // engine); the interior is reproduced exactly.
        for y in 2..7 {
            for x in 2..11 {
                assert_eq!(out.pixel(x, y), src.pixel(x, y));
            }
        }
    }

    #[test]
    fn translated_field_reproduces_shifted_source() {
        // Target pixel (x, y) maps to source (x + 4, y + 2): the vote is a pure translation.
        let src = ramp(20, 12, 2);
        let mut nnf = Nnf::new(10, 6);
        for y in 0..6 {
            for x in 0..10 {
                nnf.set(x, y, [x as i32 + 4, y as i32 + 2]);
            }
        }
        let mut out = Image::new(10, 6, 2);
        vote_plain(&mut out, &src, &nnf, 3, VoteRounding::Truncate);
        for y in 0..6 {
            for x in 0..10 {
                assert_eq!(out.pixel(x, y), src.pixel(x + 4, y + 2));
            }
        }
    }

    #[test]
    fn constant_field_averages_neighbourhood() {
        // Every target pixel maps to source (2, 2). Interior target pixels then average the 3x3
        // source neighbourhood around (2, 2) (each overlapping patch contributes a different
        // offset).
        let src = Image::from_raw(5, 5, 1, (0..25).map(|v| v * 10).collect());
        let nnf = Nnf::from_raw(3, 3, vec![[2, 2]; 9]);
        let mut out = Image::new(3, 3, 1);
        vote_plain(&mut out, &src, &nnf, 3, VoteRounding::Truncate);
        // Centre: mean of src(1..=3, 1..=3) = mean of {60,70,80,110,120,130,160,170,180} = 120.
        assert_eq!(out.pixel(1, 1), &[120]);
        // Corner (0,0): patches centred at (0..=1, 0..=1) -> source offsets (2-dx, 2-dy) for
        // dx, dy in {0, 1}: pixels (2,2),(1,2),(2,1),(1,1) = 120,110,70,60 -> 90.
        assert_eq!(out.pixel(0, 0), &[90]);
    }

    #[test]
    fn const_generic_vote_matches_generic() {
        for nc in 1..=9 {
            let src = ramp(17, 11, nc);
            let mut nnf = Nnf::new(13, 9);
            for y in 0..9 {
                for x in 0..13 {
                    nnf.set(x, y, [((x * 7 + y * 3) % 15 + 1) as i32, ((x + y * 5) % 9 + 1) as i32]);
                }
            }
            for patch in [3, 5, 7] {
                let mut a = Image::new(13, 9, nc);
                vote_plain(&mut a, &src, &nnf, patch, VoteRounding::Truncate);
                let mut b = Image::new(13, 9, nc);
                vote_plain_any(&mut b, &src, &nnf, patch, VoteRounding::Truncate);
                assert_eq!(a, b, "{nc} channels, patch {patch}");
            }
        }
    }

    #[test]
    fn const_generic_weighted_vote_matches_generic() {
        for nc in 1..=9 {
            let src = ramp(17, 11, nc);
            let mut nnf = Nnf::new(13, 9);
            let mut errors = Vec::new();
            for y in 0..9 {
                for x in 0..13 {
                    nnf.set(x, y, [((x * 7 + y * 3) % 15 + 1) as i32, ((x + y * 5) % 9 + 1) as i32]);
                    errors.push(((x * 31 + y * 17) % 23) as f32 * 40.0);
                }
            }
            let norm = (3 * 3 * nc) as f32;
            let weights: Vec<f32> = errors.iter().map(|&e| 1.0 / (1.0 + e / norm)).collect();
            for rounding in [VoteRounding::Truncate, VoteRounding::Nearest] {
                let mut a = Image::new(13, 9, nc);
                vote_weighted(&mut a, &src, &nnf, &errors, 3, rounding);
                let mut b = Image::new(13, 9, nc);
                vote_weighted_any(&mut b, &src, &nnf, &weights, 3, rounding);
                assert_eq!(a, b, "{nc} channels");
            }
            // A field with an entry outside the valid centres takes the clamped path everywhere.
            let mut bad = nnf.clone();
            bad.set(6, 4, [0, 0]);
            let mut a = Image::new(13, 9, nc);
            vote_weighted(&mut a, &src, &bad, &errors, 3, VoteRounding::Nearest);
            let mut b = Image::new(13, 9, nc);
            vote_weighted_any(&mut b, &src, &bad, &weights, 3, VoteRounding::Nearest);
            assert_eq!(a, b, "{nc} channels, invalid centre");
        }
    }

    #[test]
    fn weighted_vote_clamps_at_borders() {
        // A constant field: with clamped addressing every target pixel sums all 9 patches,
        // reading clamped source pixels, so the corner differs from the plain (skipping) vote.
        let src = Image::from_raw(5, 5, 1, (0..25).map(|v| v * 10).collect());
        let nnf = Nnf::from_raw(3, 3, vec![[2, 2]; 9]);
        let mut out = Image::new(3, 3, 1);
        vote_weighted(&mut out, &src, &nnf, &[0.0; 9], 3, VoteRounding::Truncate);
        assert_eq!(out.pixel(1, 1), &[120]);
        // Corner (0,0): offsets (px, py) in -1..=1 read source (2 - px, 2 - py), all in range:
        // the full 3x3 neighbourhood of (2, 2) -> 120 (the plain vote gives 90).
        assert_eq!(out.pixel(0, 0), &[120]);
    }

    #[test]
    fn weighted_vote_falls_back_to_equal_weights() {
        let src = ramp(6, 6, 2);
        let nnf = identity_nnf(6, 6);
        let mut inf = Image::new(6, 6, 2);
        vote_weighted(&mut inf, &src, &nnf, &[f32::INFINITY; 36], 3, VoteRounding::Truncate);
        let mut zero_err = Image::new(6, 6, 2);
        vote_weighted(&mut zero_err, &src, &nnf, &[0.0; 36], 3, VoteRounding::Truncate);
        assert_eq!(inf, zero_err);
    }

    #[test]
    fn rounding_modes() {
        // Target pixel 0 averages source values 10 and 13 (11.5): truncation gives 11, rounding
        // 12. A constant field at (1, 1) with 3x3 patches: corner (0, 0) of a 2x1 target sees
        // patches centred at (0, 0) and (1, 0), reading source (1, 1), (0, 1) -> 10 and 13.
        let src = Image::from_raw(3, 3, 1, vec![0, 0, 0, 13, 10, 0, 0, 0, 0]);
        let nnf = Nnf::from_raw(2, 1, vec![[1, 1], [1, 1]]);
        let mut t = Image::new(2, 1, 1);
        vote_plain(&mut t, &src, &nnf, 3, VoteRounding::Truncate);
        assert_eq!(t.pixel(0, 0), &[11]);
        vote_plain(&mut t, &src, &nnf, 3, VoteRounding::Nearest);
        assert_eq!(t.pixel(0, 0), &[12]);
        // The generic path rounds the same way.
        let mut g = Image::new(2, 1, 1);
        vote_plain_any(&mut g, &src, &nnf, 3, VoteRounding::Nearest);
        assert_eq!(g, t);
        // Weighted: equal errors give equal weights; values stay in range.
        let bright = Image::from_raw(3, 3, 1, vec![255; 9]);
        let mut w = Image::new(2, 1, 1);
        vote_weighted(&mut w, &bright, &nnf, &[0.5, 0.5], 3, VoteRounding::Nearest);
        assert_eq!(w.data(), &[255, 255]);
    }

    #[test]
    fn rounding_is_unbiased_on_average() {
        // Random fields over a random source: truncation is biased down by close to half a
        // level of the averages' fractional parts, rounding by close to zero.
        let src = ramp(31, 23, 3);
        let mut nnf = Nnf::new(29, 21);
        let mut state = 12345u32;
        for y in 0..21 {
            for x in 0..29 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                nnf.set(x, y, [(2 + (state >> 8) % 27) as i32, (2 + (state >> 20) % 19) as i32]);
            }
        }
        let mean = |r| {
            let mut t = Image::new(29, 21, 3);
            vote_plain(&mut t, &src, &nnf, 5, r);
            t.data().iter().map(|&v| v as f64).sum::<f64>() / t.data().len() as f64
        };
        let (trunc, round) = (mean(VoteRounding::Truncate), mean(VoteRounding::Nearest));
        assert!(round - trunc > 0.3 && round - trunc < 0.7, "truncate {trunc}, round {round}");
    }

    #[test]
    fn weighted_vote_prefers_low_error_patches() {
        // Two candidate colours; the low-error patch must dominate.
        let src =
            Image::from_raw(6, 3, 1, vec![0, 0, 0, 200, 200, 200, 0, 0, 0, 200, 200, 200, 0, 0, 0, 200, 200, 200]);
        // Target 2x1: pixel 0 maps into the dark half, pixel 1 into the bright half.
        let nnf = Nnf::from_raw(2, 1, vec![[1, 1], [4, 1]]);
        let mut plain = Image::new(2, 1, 1);
        vote_plain(&mut plain, &src, &nnf, 3, VoteRounding::Truncate);
        let mut weighted = Image::new(2, 1, 1);
        // Pixel 1's patch has a huge error, so pixel 0's (dark) patch should dominate both.
        vote_weighted(&mut weighted, &src, &nnf, &[0.0, 1.0e6], 3, VoteRounding::Truncate);
        assert!(weighted.pixel(1, 0)[0] < plain.pixel(1, 0)[0]);
        assert!(weighted.pixel(0, 0)[0] <= plain.pixel(0, 0)[0]);
    }
}
