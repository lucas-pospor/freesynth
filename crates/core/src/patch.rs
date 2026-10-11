//! Patch distance (`PatchSSD_Split` in the original).
//!
//! The error between a target patch centred at `(tx, ty)` and a source patch centred at
//! `(sx, sy)` is the weighted sum of squared differences over all patch pixels and all channels:
//!
//! ```text
//! E = sum_p [ sum_k styleWeight[k] * (Ts[p][k] - Ss[p][k])^2
//!           + sum_k guideWeight[k] * mod[p][k]/255 * (Tg[p][k] - Sg[p][k])^2 ]
//! ```
//!
//! (`mod` is 255 when no modulation is given). Style and guide channels are packed together per
//! pixel (style first), so one patch row is a single contiguous run of `patch_size * channels`
//! bytes in both images. The fast kernel accumulates such runs in 16 independent lanes (with
//! AVX2/AVX-512 versions selected at run time, all bit-identical to the portable code); the exact
//! kernel accumulates sequentially in the original order.

use rayon::prelude::*;

/// Number of independent accumulators in the fast kernel.
pub(crate) const LANES: usize = 16;

/// Rows are zero-weight padded to a multiple of this many elements.
pub(crate) const ROW_PAD: usize = 8;

/// Which patch-distance implementation to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// Lane-parallel accumulation (vectorised); differs from the original only in rounding.
    Fast = 0,
    /// Sequential accumulation in the original order; bit-identical to the 2019 build.
    Exact = 1,
    /// Per-pixel guide modulation (the CUDA engine's `PatchSSD_Split_Modulation`).
    Modulated = 2,
}

pub(crate) const FAST: u8 = Kernel::Fast as u8;
pub(crate) const EXACT: u8 = Kernel::Exact as u8;
pub(crate) const MODULATED: u8 = Kernel::Modulated as u8;

/// Packs style and guide images into one buffer of `ns + ng` channels per pixel, followed by
/// [`LANES`] bytes of padding so that padded row reads never leave the buffer.
pub(crate) fn pack(style: &[u8], ns: usize, guide: &[u8], ng: usize, num_pixels: usize) -> Vec<u8> {
    let nc = ns + ng;
    let mut out = vec![0u8; num_pixels * nc + LANES];
    out[..num_pixels * nc]
        .par_chunks_exact_mut(nc)
        .zip(style.par_chunks_exact(ns).zip(guide.par_chunks_exact(ng)))
        .for_each(|(px, (s, g))| {
            px[..ns].copy_from_slice(s);
            px[ns..].copy_from_slice(g);
        });
    out
}

/// Everything needed to evaluate patch errors at one pyramid level.
pub(crate) struct PatchSpace<'a> {
    pub tw: usize,
    pub th: usize,
    pub sw: usize,
    pub sh: usize,
    /// Channels per packed pixel (style + guide).
    pub nc: usize,
    /// Number of leading style channels.
    pub ns: usize,
    pub patch: usize,
    /// Packed target pixels (see [`pack`]).
    pub target: &'a [u8],
    /// Packed source pixels (see [`pack`]).
    pub source: &'a [u8],
    /// Per-channel weights (`nc`).
    pub weights: &'a [f32],
    /// `weights` tiled `patch` times, zero-padded to a multiple of [`ROW_PAD`].
    pub row_weights: Vec<f32>,
    /// Target modulation, `nc - ns` channels per target pixel.
    pub modulation: Option<&'a [u8]>,
    /// Present when the CPU supports the AVX2 row kernel.
    simd: Option<simd::Token>,
}

impl<'a> PatchSpace<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target_size: (usize, usize),
        source_size: (usize, usize),
        ns: usize,
        patch: usize,
        target: &'a [u8],
        source: &'a [u8],
        weights: &'a [f32],
        modulation: Option<&'a [u8]>,
    ) -> Self {
        let nc = weights.len();
        let (tw, th) = target_size;
        let (sw, sh) = source_size;
        assert!(target.len() >= tw * th * nc + LANES);
        assert!(source.len() >= sw * sh * nc + LANES);
        if let Some(m) = modulation {
            assert_eq!(m.len(), tw * th * (nc - ns));
        }
        let row_len = patch * nc;
        let mut row_weights = vec![0.0f32; row_len.div_ceil(ROW_PAD) * ROW_PAD];
        for (i, w) in row_weights[..row_len].iter_mut().enumerate() {
            *w = weights[i % nc];
        }
        Self {
            tw,
            th,
            sw,
            sh,
            nc,
            ns,
            patch,
            target,
            source,
            weights,
            row_weights,
            modulation,
            simd: simd::Token::detect(),
        }
    }

    /// The kernel to use for this level.
    pub fn kernel(&self, exact: bool) -> Kernel {
        if self.modulation.is_some() {
            Kernel::Modulated
        } else if exact {
            Kernel::Exact
        } else {
            Kernel::Fast
        }
    }

    /// Whether the target patch at `(tx, ty)` lies entirely inside the target.
    #[inline(always)]
    pub fn is_interior(&self, tx: i32, ty: i32) -> bool {
        let r = (self.patch / 2) as i32;
        tx - r >= 0 && tx + r < self.tw as i32 && ty - r >= 0 && ty + r < self.th as i32
    }

    /// Patch error between target `(tx, ty)` and source `(sx, sy)`.
    ///
    /// Evaluation may stop early, after a full patch row, once the partial error exceeds
    /// `ebest`; the returned value is then a lower bound that is larger than `ebest`. The source
    /// patch must lie inside the source image.
    #[inline(always)]
    pub fn error<const K: u8>(&self, tx: i32, ty: i32, sx: i32, sy: i32, ebest: f32) -> f32 {
        if K == MODULATED {
            return self.error_modulated(tx, ty, sx, sy, ebest);
        }
        if !self.is_interior(tx, ty) {
            return self.error_clamped(tx, ty, sx, sy);
        }
        let r = (self.patch / 2) as i32;
        let nc = self.nc;
        let mut t = ((ty - r) as usize * self.tw + (tx - r) as usize) * nc;
        let mut s = ((sy - r) as usize * self.sw + (sx - r) as usize) * nc;
        let t_stride = self.tw * nc;
        let s_stride = self.sw * nc;
        let mut err = 0.0f32;
        if K == EXACT {
            let len = self.patch * nc;
            let w = &self.row_weights[..len];
            for _ in 0..self.patch {
                let a = &self.target[t..t + len];
                let b = &self.source[s..s + len];
                for ((&x, &y), &w) in a.iter().zip(b).zip(w) {
                    let d = x as f32 - y as f32;
                    err += w * d * d;
                }
                if err > ebest {
                    break;
                }
                t += t_stride;
                s += s_stride;
            }
        } else {
            let len = self.row_weights.len();
            for _ in 0..self.patch {
                let (a, b) = (&self.target[t..t + len], &self.source[s..s + len]);
                err += match self.simd {
                    Some(token) => token.row_ssd(a, b, &self.row_weights),
                    None => row_ssd(a, b, &self.row_weights),
                };
                if err > ebest {
                    break;
                }
                t += t_stride;
                s += s_stride;
            }
        }
        err
    }

    /// Full error with clamped addressing, for target patches that touch the border. Same
    /// accumulation order as the original (no early termination).
    #[inline(always)]
    fn error_clamped(&self, tx: i32, ty: i32, sx: i32, sy: i32) -> f32 {
        let r = (self.patch / 2) as i32;
        let nc = self.nc;
        let (tw, th, sw, sh) = (self.tw as i32, self.th as i32, self.sw as i32, self.sh as i32);
        let mut err = 0.0f32;
        for py in -r..=r {
            let ty = (ty + py).clamp(0, th - 1);
            let sy = (sy + py).clamp(0, sh - 1);
            for px in -r..=r {
                let t = (ty * tw + (tx + px).clamp(0, tw - 1)) as usize * nc;
                let s = (sy * sw + (sx + px).clamp(0, sw - 1)) as usize * nc;
                for ((&a, &b), &w) in self.target[t..t + nc].iter().zip(&self.source[s..s + nc]).zip(self.weights) {
                    let d = a as f32 - b as f32;
                    err += w * d * d;
                }
            }
        }
        err
    }

    /// Error with per-pixel guide modulation (clamped addressing, early termination per row).
    #[inline(always)]
    fn error_modulated(&self, tx: i32, ty: i32, sx: i32, sy: i32, ebest: f32) -> f32 {
        let modulation = self.modulation.expect("modulated kernel without modulation");
        let r = (self.patch / 2) as i32;
        let (nc, ns, ng) = (self.nc, self.ns, self.nc - self.ns);
        let (tw, th, sw, sh) = (self.tw as i32, self.th as i32, self.sw as i32, self.sh as i32);
        let mut err = 0.0f32;
        for py in -r..=r {
            let ty = (ty + py).clamp(0, th - 1);
            let sy = (sy + py).clamp(0, sh - 1);
            for px in -r..=r {
                let tp = (ty * tw + (tx + px).clamp(0, tw - 1)) as usize;
                let t = tp * nc;
                let s = (sy * sw + (sx + px).clamp(0, sw - 1)) as usize * nc;
                let (ta, sa) = (&self.target[t..t + nc], &self.source[s..s + nc]);
                for k in 0..ns {
                    let d = ta[k] as f32 - sa[k] as f32;
                    err += self.weights[k] * d * d;
                }
                let m = &modulation[tp * ng..(tp + 1) * ng];
                for k in 0..ng {
                    let d = ta[ns + k] as f32 - sa[ns + k] as f32;
                    err += self.weights[ns + k] * (m[k] as f32 / 255.0) * d * d;
                }
            }
            if err > ebest {
                break;
            }
        }
        err
    }
}

/// Weighted SSD of two byte runs (portable version).
///
/// `w.len()` must be a multiple of [`ROW_PAD`]; `a` and `b` must be at least as long. Element `i`
/// is accumulated into lane `i % LANES` (a trailing half chunk goes to lanes `0..8`), then the
/// lanes are summed pairwise. The SIMD versions in [`simd`] perform exactly the same operations,
/// so all variants agree bit for bit.
#[inline(always)]
fn row_ssd(a: &[u8], b: &[u8], w: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let n = w.len();
    let full = n / LANES * LANES;
    let mut i = 0;
    // (A `while` loop over fixed-size array copies is the form LLVM vectorises reliably here.)
    while i < full {
        let ca: [u8; LANES] = a[i..i + LANES].try_into().unwrap();
        let cb: [u8; LANES] = b[i..i + LANES].try_into().unwrap();
        let cw: [f32; LANES] = w[i..i + LANES].try_into().unwrap();
        for j in 0..LANES {
            let d = ca[j] as f32 - cb[j] as f32;
            acc[j] += cw[j] * d * d;
        }
        i += LANES;
    }
    if i < n {
        for j in 0..ROW_PAD {
            let d = a[i + j] as f32 - b[i + j] as f32;
            acc[j] += w[i + j] * d * d;
        }
    }
    let mut half = [0.0f32; LANES / 2];
    for j in 0..LANES / 2 {
        half[j] = acc[j] + acc[j + LANES / 2];
    }
    let mut quarter = [0.0f32; LANES / 4];
    for j in 0..LANES / 4 {
        quarter[j] = half[j] + half[j + LANES / 4];
    }
    (quarter[0] + quarter[2]) + (quarter[1] + quarter[3])
}

/// Explicit SIMD versions of [`row_ssd`], selected at run time.
mod simd {
    /// Proof that the CPU supports a SIMD row kernel; only [`Token::detect`] creates one.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Token(Level);

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Level {
        #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
        Avx2,
        #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
        Avx512,
    }

    impl Token {
        pub fn detect() -> Option<Self> {
            #[cfg(target_arch = "x86_64")]
            {
                use std::is_x86_feature_detected as has;
                if has!("avx512f") && has!("avx2") && has!("fma") {
                    return Some(Self(Level::Avx512));
                }
                if has!("avx2") && has!("fma") {
                    return Some(Self(Level::Avx2));
                }
            }
            None
        }

        #[inline(always)]
        pub fn row_ssd(self, a: &[u8], b: &[u8], w: &[f32]) -> f32 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: a `Token` with a given level only exists if the CPU features that the
            // corresponding function enables were detected at run time.
            unsafe {
                match self.0 {
                    Level::Avx512 => x86::row_ssd_avx512(a, b, w),
                    Level::Avx2 => x86::row_ssd_avx2(a, b, w),
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = (a, b, w);
                unreachable!("no SIMD token on this architecture")
            }
        }
    }

    #[cfg(all(test, target_arch = "x86_64"))]
    mod tests {
        use super::*;

        /// Every SIMD level must reproduce the portable kernel bit for bit.
        #[test]
        fn simd_kernels_match_portable() {
            use std::is_x86_feature_detected as has;
            let mut levels = Vec::new();
            if has!("avx2") && has!("fma") {
                levels.push(Level::Avx2);
                if has!("avx512f") {
                    levels.push(Level::Avx512);
                }
            }
            let mut state = 12345u32;
            let mut next = || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            };
            for len in (8..=136).step_by(8) {
                let a: Vec<u8> = (0..len + 16).map(|_| next()).collect();
                let b: Vec<u8> = (0..len + 16).map(|_| next()).collect();
                let w: Vec<f32> = (0..len).map(|i| 0.1 + (i % 7) as f32 * 0.37).collect();
                let expected = super::super::row_ssd(&a, &b, &w);
                for &level in &levels {
                    let got = Token(level).row_ssd(&a, &b, &w);
                    assert_eq!(got.to_bits(), expected.to_bits(), "{level:?}, len {len}");
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod x86 {
        use std::arch::x86_64::*;

        use super::super::{LANES, ROW_PAD};

        /// Loads 8 bytes as 8 `f32` lanes.
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        fn bytes8_ps(v: __m128i) -> __m256 {
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v))
        }

        /// `(q0 + q2) + (q1 + q3)` of the four lanes of `quarter`.
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        fn reduce4(quarter: __m128) -> f32 {
            let pairs = _mm_add_ps(quarter, _mm_movehl_ps(quarter, quarter));
            _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<1>(pairs, pairs)))
        }

        /// Accumulates 8 elements at `i` into `acc`.
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        fn step8(acc: __m256, a: &[u8], b: &[u8], w: &[f32], i: usize) -> __m256 {
            assert!(i + 8 <= w.len() && i + 8 <= a.len() && i + 8 <= b.len());
            // SAFETY: the assertion above keeps all three loads in bounds.
            let (va, vb, vw) = unsafe {
                (
                    _mm_loadl_epi64(a.as_ptr().add(i).cast()),
                    _mm_loadl_epi64(b.as_ptr().add(i).cast()),
                    _mm256_loadu_ps(w.as_ptr().add(i)),
                )
            };
            let d = _mm256_sub_ps(bytes8_ps(va), bytes8_ps(vb));
            _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(vw, d), d))
        }

        /// AVX2 kernel: lanes 0-7 live in `acc0`, lanes 8-15 in `acc1`.
        #[target_feature(enable = "avx2,fma")]
        pub(super) fn row_ssd_avx2(a: &[u8], b: &[u8], w: &[f32]) -> f32 {
            let n = w.len();
            debug_assert!(n.is_multiple_of(ROW_PAD));
            let full = n / LANES * LANES;
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i < full {
                acc0 = step8(acc0, a, b, w, i);
                acc1 = step8(acc1, a, b, w, i + 8);
                i += LANES;
            }
            if i < n {
                acc0 = step8(acc0, a, b, w, i);
            }
            let half = _mm256_add_ps(acc0, acc1);
            reduce4(_mm_add_ps(_mm256_castps256_ps128(half), _mm256_extractf128_ps::<1>(half)))
        }

        /// AVX-512 kernel: all 16 lanes in one register.
        #[target_feature(enable = "avx512f,avx2,fma")]
        pub(super) fn row_ssd_avx512(a: &[u8], b: &[u8], w: &[f32]) -> f32 {
            let n = w.len();
            debug_assert!(n.is_multiple_of(ROW_PAD));
            let full = n / LANES * LANES;
            assert!(full <= a.len() && full <= b.len());
            let mut acc = _mm512_setzero_ps();
            let mut i = 0;
            while i < full {
                // SAFETY: i + 16 <= full <= len of `a`, `b` and `w`; unaligned loads.
                let (va, vb, vw) = unsafe {
                    (
                        _mm_loadu_si128(a.as_ptr().add(i).cast()),
                        _mm_loadu_si128(b.as_ptr().add(i).cast()),
                        _mm512_loadu_ps(w.as_ptr().add(i)),
                    )
                };
                let d = _mm512_sub_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(va)),
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vb)),
                );
                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_mul_ps(vw, d), d));
                i += LANES;
            }
            let mut lo = _mm512_castps512_ps256(acc);
            let hi = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(acc)));
            if i < n {
                lo = step8(lo, a, b, w, i);
            }
            let half = _mm256_add_ps(lo, hi);
            reduce4(_mm_add_ps(_mm256_castps256_ps128(half), _mm256_extractf128_ps::<1>(half)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space<'a>(t: &'a [u8], s: &'a [u8], w: &'a [f32], size: usize) -> PatchSpace<'a> {
        PatchSpace::new((size, size), (size, size), 1, 3, t, s, w, None)
    }

    #[test]
    fn fast_and_exact_agree_on_small_values() {
        let n = 9 * 9;
        let style: Vec<u8> = (0..n).map(|i| (i * 7 % 256) as u8).collect();
        let guide: Vec<u8> = (0..n * 2).map(|i| (i * 13 % 256) as u8).collect();
        let style2: Vec<u8> = (0..n).map(|i| (i * 11 % 256) as u8).collect();
        let t = pack(&style, 1, &guide, 2, n);
        let s = pack(&style2, 1, &guide, 2, n);
        let w = [0.5f32, 0.25, 2.0];
        let sp = space(&t, &s, &w, 9);
        for (tx, ty, sx, sy) in [(4, 4, 4, 4), (1, 1, 6, 3), (0, 0, 4, 4), (8, 3, 2, 2)] {
            let e0 = sp.error::<EXACT>(tx, ty, sx, sy, f32::MAX);
            let e1 = sp.error::<FAST>(tx, ty, sx, sy, f32::MAX);
            assert!((e0 - e1).abs() <= e0 * 1e-6, "{e0} vs {e1}");
            // Brute-force reference.
            let mut e = 0.0f64;
            for py in -1..=1i32 {
                for px in -1..=1i32 {
                    let tp = ((ty + py).clamp(0, 8) * 9 + (tx + px).clamp(0, 8)) as usize;
                    let spx = ((sy + py).clamp(0, 8) * 9 + (sx + px).clamp(0, 8)) as usize;
                    for k in 0..3 {
                        let d = t[tp * 3 + k] as f64 - s[spx * 3 + k] as f64;
                        e += w[k] as f64 * d * d;
                    }
                }
            }
            assert!((e0 as f64 - e).abs() <= e * 1e-5, "{e0} vs {e}");
        }
    }

    #[test]
    fn identical_patches_have_zero_error_and_early_exit_bounds() {
        let n = 7 * 7;
        let style: Vec<u8> = (0..n).map(|i| (i * 5 % 256) as u8).collect();
        let t = pack(&style, 1, &style, 1, n);
        let w = [1.0f32, 1.0];
        let sp = PatchSpace::new((7, 7), (7, 7), 1, 3, &t, &t, &w, None);
        assert_eq!(sp.error::<FAST>(3, 3, 3, 3, f32::MAX), 0.0);
        let full = sp.error::<FAST>(3, 3, 4, 4, f32::MAX);
        let partial = sp.error::<FAST>(3, 3, 4, 4, 1.0);
        assert!(partial > 1.0 && partial <= full);
    }

    #[test]
    fn modulation_scales_guide_term() {
        let n = 5 * 5;
        let style = vec![0u8; n];
        let gt = vec![100u8; n];
        let gs = vec![90u8; n];
        let t = pack(&style, 1, &gt, 1, n);
        let s = pack(&style, 1, &gs, 1, n);
        let w = [1.0f32, 1.0];
        let full = vec![255u8; n];
        let half = vec![51u8; n]; // 0.2
        let e_full =
            PatchSpace::new((5, 5), (5, 5), 1, 3, &t, &s, &w, Some(&full)).error::<MODULATED>(2, 2, 2, 2, f32::MAX);
        let e_half =
            PatchSpace::new((5, 5), (5, 5), 1, 3, &t, &s, &w, Some(&half)).error::<MODULATED>(2, 2, 2, 2, f32::MAX);
        assert_eq!(e_full, 900.0);
        assert!((e_half - 180.0).abs() < 1e-3);
    }
}
