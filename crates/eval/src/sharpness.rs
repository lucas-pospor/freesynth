//! Sharpness: high-frequency content of an image relative to a reference.
//!
//! Patch voting averages overlapping patches, and repeated resampling (for example by a temporal
//! guide that warps the previous output) blurs further, so a synthesis can lose detail that the
//! style exemplar has. The ratio of an output's high-frequency energy to that of the keyframe or
//! style catches this; energy alone does not, because a blurred image can still match blurred
//! source patches.
//!
//! Three measures on Rec. 601 luma in 8-bit units, for a scale `sigma` (default 2):
//!
//! - **High-frequency energy**: mean of `(Y - G_sigma * Y)^2`, where `G_sigma` is a Gaussian blur
//!   (border reflected without repeating the edge pixel).
//! - **Band-pass energy**: mean of `(G_{sigma/2} * Y - G_sigma * Y)^2`, a difference of Gaussians
//!   (sigma 1 to 2 by default). It leaves out the finest, pixel-level band.
//! - **Mean gradient magnitude**: mean of `sqrt(dx^2 + dy^2)` with central differences
//!   (one-sided at the border).
//!
//! Ratios are output / reference; 1 means as sharp as the reference.
//!
//! # Noise reads as sharpness
//!
//! The high-frequency energy and the gradient include the band down to single pixels, so pixel
//! noise, blocking and other artefacts raise them just as real detail does. A noisier output
//! reads as sharper and a cleaner one as blurrier: on a 1200x912 synthesis, adding Gaussian noise
//! of sd 16 levels moved its high-frequency ratio against the exemplar from 0.81 to 1.41
//! ("sharper than the exemplar"). A rise in these ratios means something only if style energy
//! does not rise with it. The band-pass ratio is much less sensitive to noise (0.84 to 1.00 on the
//! same image) and still drops sharply with blur (0.35 after a blur of sigma 1), so use it as the
//! guard against blur, and the other two as supporting evidence.
use rayon::prelude::*;

use crate::frame::RgbImage;

/// Default Gaussian sigma for [`high_frequency_energy`].
pub const DEFAULT_SIGMA: f64 = 2.0;

/// Largest accepted Gaussian sigma. Larger scales measure little but the image mean, and the
/// kernel cost grows with sigma.
pub const MAX_SIGMA: f64 = 64.0;

/// Whether `sigma` is a valid scale for [`Sharpness::of`]: finite, positive and at most
/// [`MAX_SIGMA`].
pub fn valid_sigma(sigma: f64) -> bool {
    sigma.is_finite() && sigma > 0.0 && sigma <= MAX_SIGMA
}

/// Sharpness measures of one image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sharpness {
    /// High-frequency energy (8-bit² units).
    pub hf: f64,
    /// Band-pass (difference of Gaussians) energy (8-bit² units).
    pub band: f64,
    /// Mean gradient magnitude (8-bit levels per pixel).
    pub gradient: f64,
}

impl Sharpness {
    /// Measures `img` at scale `sigma` (see the module documentation).
    ///
    /// # Panics
    /// If `sigma` is not [valid](valid_sigma).
    pub fn of(img: &RgbImage, sigma: f64) -> Self {
        assert!(valid_sigma(sigma), "sharpness sigma {sigma} is not in (0, {MAX_SIGMA}]");
        let y: Vec<f32> = img.luma().iter().map(|v| v * 255.0).collect();
        let (w, h) = img.size();
        let coarse = gaussian_blur(&y, w, h, sigma);
        let fine = gaussian_blur(&y, w, h, sigma / 2.0);
        Self { hf: mean_sq_diff(&y, &coarse), band: mean_sq_diff(&fine, &coarse), gradient: mean_gradient(&y, w, h) }
    }

    /// Element-wise ratio `self / reference`, or `None` if the reference has no detail at all
    /// (one of its measures is zero), where the ratios are undefined.
    pub fn ratio(&self, reference: &Sharpness) -> Option<Sharpness> {
        (reference.hf > 0.0 && reference.band > 0.0 && reference.gradient > 0.0).then(|| Sharpness {
            hf: self.hf / reference.hf,
            band: self.band / reference.band,
            gradient: self.gradient / reference.gradient,
        })
    }
}

/// Mean of `(a - b)^2`.
fn mean_sq_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / a.len() as f64
}

/// Reflects an out-of-range index into `0..n` without repeating the edge (`dcb|abcd|cba`).
#[inline]
fn reflect101(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut i = i.rem_euclid(period);
    if i >= n {
        i = period - i;
    }
    i as usize
}

/// Separable Gaussian blur of a `w x h` plane, kernel radius `ceil(4 sigma)` (at most the larger
/// image side). `sigma = 0` returns the plane unchanged.
///
/// # Panics
/// If `sigma` is negative or not finite, or the plane does not have `w * h` values.
pub fn gaussian_blur(plane: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
    assert_eq!(plane.len(), w * h);
    assert!(sigma.is_finite() && sigma >= 0.0, "blur sigma {sigma} must be finite and non-negative");
    if sigma == 0.0 || w == 0 || h == 0 {
        return plane.to_vec();
    }
    // Beyond the image size, a longer kernel only folds back onto the same pixels.
    let r = (4.0 * sigma).ceil().min(w.max(h) as f64) as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);

    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &plane[y * w..(y + 1) * w];
        for (x, out) in row.iter_mut().enumerate() {
            *out = k.iter().enumerate().map(|(j, &kv)| kv * src[reflect101(x as isize + j as isize - r, w)]).sum();
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o =
                k.iter().enumerate().map(|(j, &kv)| kv * tmp[reflect101(y as isize + j as isize - r, h) * w + x]).sum();
        }
    });
    out
}

/// Mean of `(plane - blur(plane))^2` (see the module documentation).
pub fn high_frequency_energy(plane: &[f32], w: usize, h: usize, sigma: f64) -> f64 {
    mean_sq_diff(plane, &gaussian_blur(plane, w, h, sigma))
}

/// Mean of `(blur(plane, sigma / 2) - blur(plane, sigma))^2` (see the module documentation).
pub fn band_pass_energy(plane: &[f32], w: usize, h: usize, sigma: f64) -> f64 {
    mean_sq_diff(&gaussian_blur(plane, w, h, sigma / 2.0), &gaussian_blur(plane, w, h, sigma))
}

/// Mean gradient magnitude with central differences (see the module documentation).
pub fn mean_gradient(plane: &[f32], w: usize, h: usize) -> f64 {
    assert_eq!(plane.len(), w * h);
    let diff = |a: f32, b: f32, span: usize| if span == 0 { 0.0 } else { (a - b) / span as f32 };
    // Per-row sums in parallel, then summed in order (independent of scheduling).
    let rows: Vec<f64> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut s = 0.0f64;
            for x in 0..w {
                let (x0, x1) = (x.saturating_sub(1), (x + 1).min(w - 1));
                let (y0, y1) = (y.saturating_sub(1), (y + 1).min(h - 1));
                let dx = diff(plane[y * w + x1], plane[y * w + x0], x1 - x0);
                let dy = diff(plane[y1 * w + x], plane[y0 * w + x], y1 - y0);
                s += ((dx * dx + dy * dy) as f64).sqrt();
            }
            s
        })
        .collect();
    rows.iter().sum::<f64>() / plane.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(w: usize, h: usize, cell: usize) -> RgbImage {
        RgbImage::from_raw(
            w,
            h,
            (0..w * h)
                .map(|i| if ((i % w) / cell + (i / w) / cell).is_multiple_of(2) { [1.0; 3] } else { [0.0; 3] })
                .collect(),
        )
    }

    fn blur_rgb(img: &RgbImage, sigma: f64) -> RgbImage {
        let (w, h) = img.size();
        let plane: Vec<f32> = img.data().iter().map(|p| p[0]).collect();
        let b = gaussian_blur(&plane, w, h, sigma);
        RgbImage::from_raw(w, h, b.into_iter().map(|v| [v; 3]).collect())
    }

    #[test]
    fn reflect_indices() {
        let got: Vec<usize> = (-3..7).map(|i| reflect101(i, 4)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
        assert_eq!(reflect101(-5, 1), 0);
    }

    #[test]
    fn blur_preserves_constants_and_mean() {
        let flat = vec![3.5f32; 12 * 7];
        for v in gaussian_blur(&flat, 12, 7, 2.0) {
            assert!((v - 3.5).abs() < 1e-5);
        }
        assert!(high_frequency_energy(&flat, 12, 7, 2.0) < 1e-9);
        assert_eq!(mean_gradient(&flat, 12, 7), 0.0);
    }

    #[test]
    fn ramp_gradient() {
        // A horizontal ramp of 2 levels per pixel has gradient 2 everywhere.
        let ramp: Vec<f32> = (0..10 * 4).map(|i| (i % 10) as f32 * 2.0).collect();
        assert!((mean_gradient(&ramp, 10, 4) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn blur_lowers_the_ratio() {
        let sharp = checker(64, 48, 4);
        let s = Sharpness::of(&sharp, DEFAULT_SIGMA);
        let same = s.ratio(&s).unwrap();
        assert!((same.hf - 1.0).abs() < 1e-12 && (same.band - 1.0).abs() < 1e-12);
        let soft = Sharpness::of(&blur_rgb(&sharp, 1.0), DEFAULT_SIGMA).ratio(&s).unwrap();
        let softer = Sharpness::of(&blur_rgb(&sharp, 2.0), DEFAULT_SIGMA).ratio(&s).unwrap();
        assert!(soft.hf < 0.8 && softer.hf < soft.hf, "{soft:?} {softer:?}");
        assert!(soft.band < 0.9 && softer.band < soft.band, "{soft:?} {softer:?}");
        assert!(soft.gradient < 1.0 && softer.gradient < soft.gradient);
        let band = band_pass_energy(&sharp.luma().iter().map(|v| v * 255.0).collect::<Vec<_>>(), 64, 48, DEFAULT_SIGMA);
        assert!((band - s.band).abs() < 1e-9 * band);
    }

    #[test]
    fn noise_raises_high_frequencies_more_than_the_band() {
        // A smooth pattern with detail at a few pixels' scale, then the same with pixel noise.
        let (w, h) = (96, 64);
        let base: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                [0.5 + 0.3 * (x * 0.7).sin() * (y * 0.5).cos(); 3]
            })
            .collect();
        let clean = RgbImage::from_raw(w, h, base.clone());
        let mut rng = crate::rng::SplitMix64::new(4);
        let noisy = RgbImage::from_raw(
            w,
            h,
            base.iter().map(|p| [(p[0] + 0.06 * rng.normal() as f32).clamp(0.0, 1.0); 3]).collect(),
        );
        let r = Sharpness::of(&noisy, DEFAULT_SIGMA).ratio(&Sharpness::of(&clean, DEFAULT_SIGMA)).unwrap();
        assert!(r.hf > 1.0 && r.band > 1.0, "{r:?}");
        assert!(r.band - 1.0 < 0.5 * (r.hf - 1.0), "{r:?}");
    }

    #[test]
    fn flat_reference_and_bad_sigma() {
        let flat = RgbImage::from_raw(8, 8, vec![[0.5; 3]; 64]);
        let s = Sharpness::of(&flat, DEFAULT_SIGMA);
        assert!(s.ratio(&s).is_none());
        assert!(valid_sigma(0.5) && valid_sigma(MAX_SIGMA));
        assert!(!valid_sigma(0.0) && !valid_sigma(-1.0) && !valid_sigma(f64::NAN) && !valid_sigma(1e7));
        // A huge sigma would be capped by the image size, not allocate a huge kernel.
        let b = gaussian_blur(&[1.0, 2.0, 3.0, 4.0], 2, 2, 1e9);
        assert!(b.iter().all(|v| v.is_finite()));
    }
}
