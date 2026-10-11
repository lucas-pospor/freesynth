//! Pyramid level sizes and the bilinear resampler used to build the pyramid.
//!
//! As in the original, every coarse level is resampled directly from the finest level (no
//! low-pass prefilter), with a single scale factor derived from the widths.

use rayon::prelude::*;

use crate::image::Image;

/// Size of `level` in a pyramid of `num_levels` levels (level 0 is the coarsest,
/// `num_levels - 1` is the full resolution `base`).
///
/// Equivalent to the original `V2i(V2f(size) * pow(2, -(numLevels-1-level)))`: scaling by a power
/// of two is exact in floating point and the conversion truncates, so this is a right shift.
pub fn pyramid_level_size(base: (usize, usize), num_levels: usize, level: usize) -> (usize, usize) {
    let shift = (num_levels - 1 - level) as u32;
    (shr(base.0, shift), shr(base.1, shift))
}

fn shr(v: usize, s: u32) -> usize {
    v.checked_shr(s).unwrap_or(0)
}

/// The largest number of pyramid levels for which the coarsest level of both the source and the
/// target is at least `2 * patch_size + 1` pixels in each dimension (at least 1).
///
/// This is the default `-pyramidlevels` of the 2019 command-line tool.
pub fn max_pyramid_levels(source: (usize, usize), target: (usize, usize), patch_size: usize) -> usize {
    let min_w = source.0.min(target.0);
    let min_h = source.1.min(target.1);
    for level in (0..=32u32).rev() {
        if shr(min_w, level).min(shr(min_h, level)) > 2 * patch_size {
            return level as usize + 1;
        }
    }
    0
}

/// Bilinearly resamples `input` to `width x height` (`resampleCPU` in the original).
///
/// Output pixel `(x, y)` samples the input at `(s*x, s*y)` with `s = in_width / out_width`, with
/// clamped addressing. The arithmetic (a product followed by three fused multiply-adds) follows
/// the scalar code of the 2019 reference build (gcc `-O3 -march=native`), so results are
/// bit-identical to it. (That build vectorises some channel counts, e.g. 7, and leaves the
/// multiply-adds of the trailing lanes unfused; those rare off-by-one differences only affect
/// coarse pyramid levels, and are why `-bitexact` does not reproduce every input exactly.)
///
/// # Panics
///
/// If `input` is empty while the requested output is not (there is nothing to sample).
pub fn resample(input: &Image, width: usize, height: usize) -> Image {
    let nc = input.channels();
    let mut out = Image::new(width, height, nc);
    if width == 0 || height == 0 || nc == 0 {
        return out;
    }
    assert!(input.width() > 0 && input.height() > 0, "cannot resample an empty image to {width}x{height}");
    let s = input.width() as f32 / width as f32;
    let row_len = width * nc;
    #[cfg(target_arch = "x86_64")]
    let fma = std::is_x86_feature_detected!("fma");
    out.data_mut().par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        #[cfg(target_arch = "x86_64")]
        if fma {
            // SAFETY: the FMA instruction set was detected at run time above.
            unsafe { resample_row_fma(input, s, y, row) };
            return;
        }
        resample_row(input, s, y, row);
    });
    out
}

/// [`resample_row`] with hardware fused multiply-adds (otherwise `f32::mul_add` is a call to the
/// C library's `fmaf`). Both compute the same, correctly rounded results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
fn resample_row_fma(input: &Image, s: f32, y: usize, row: &mut [u8]) {
    resample_row(input, s, y, row);
}

/// Output row `y` of [`resample`].
#[inline(always)]
fn resample_row(input: &Image, s: f32, y: usize, row: &mut [u8]) {
    let nc = input.channels();
    let (iw, ih) = (input.width() as i32, input.height() as i32);
    let src = input.data();
    let fy = s * y as f32;
    let iy = fy as i32;
    let t = fy - iy as f32;
    let y0 = iy.clamp(0, ih - 1) as usize;
    let y1 = (iy + 1).clamp(0, ih - 1) as usize;
    for (x, px) in row.chunks_exact_mut(nc).enumerate() {
        let fx = s * x as f32;
        let ix = fx as i32;
        let u = fx - ix as f32;
        let x0 = ix.clamp(0, iw - 1) as usize;
        let x1 = (ix + 1).clamp(0, iw - 1) as usize;
        let w00 = (1.0 - u) * (1.0 - t);
        let w10 = u * (1.0 - t);
        let w01 = (1.0 - u) * t;
        let w11 = u * t;
        let p00 = (y0 * iw as usize + x0) * nc;
        let p10 = (y0 * iw as usize + x1) * nc;
        let p01 = (y1 * iw as usize + x0) * nc;
        let p11 = (y1 * iw as usize + x1) * nc;
        for (c, o) in px.iter_mut().enumerate() {
            let mut v = w00 * src[p00 + c] as f32;
            v = w10.mul_add(src[p10 + c] as f32, v);
            v = w01.mul_add(src[p01 + c] as f32, v);
            v = w11.mul_add(src[p11 + c] as f32, v);
            *o = v as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_sizes_halve_and_truncate() {
        assert_eq!(pyramid_level_size((1200, 912), 7, 6), (1200, 912));
        assert_eq!(pyramid_level_size((1200, 912), 7, 5), (600, 456));
        assert_eq!(pyramid_level_size((1200, 912), 7, 0), (18, 14));
        assert_eq!(pyramid_level_size((321, 463), 3, 1), (160, 231));
        assert_eq!(pyramid_level_size((321, 463), 3, 0), (80, 115));
    }

    #[test]
    fn max_levels_matches_cli_rule() {
        // 320x462, patch 3 -> coarsest min side must be >= 7: 320>>5 = 10, 320>>6 = 5.
        assert_eq!(max_pyramid_levels((320, 462), (320, 462), 3), 6);
        // 1200x912, patch 5 -> >= 11: 912>>6 = 14, 912>>7 = 7.
        assert_eq!(max_pyramid_levels((1200, 912), (1200, 912), 5), 7);
        // The smaller of source and target decides.
        assert_eq!(max_pyramid_levels((1200, 912), (100, 100), 5), 4);
        assert_eq!(max_pyramid_levels((5, 5), (5, 5), 5), 0);
    }

    #[test]
    fn resample_identity_and_constant() {
        let mut img = Image::new(7, 5, 2);
        for (i, v) in img.data_mut().iter_mut().enumerate() {
            *v = (i * 7 % 256) as u8;
        }
        assert_eq!(resample(&img, 7, 5), img);

        let flat = Image::from_raw(8, 8, 1, vec![200; 64]);
        let half = resample(&flat, 4, 4);
        assert!(half.data().iter().all(|&v| v == 200));
    }

    #[test]
    fn resample_downscale_picks_even_samples() {
        // s = 2 lands exactly on even input pixels, so the weights are (1, 0, 0, 0).
        let img = Image::from_raw(4, 4, 1, (0..16).collect());
        let half = resample(&img, 2, 2);
        assert_eq!(half.data(), &[0, 2, 8, 10]);
    }

    #[test]
    fn resample_fma_paths_agree() {
        // Odd sizes exercise fractional weights, where fusing matters.
        let mut img = Image::new(37, 23, 3);
        for (i, v) in img.data_mut().iter_mut().enumerate() {
            *v = (i * 97 % 251) as u8;
        }
        let out = resample(&img, 16, 11);
        let s = 37.0f32 / 16.0;
        let mut row = vec![0u8; 16 * 3];
        for y in 0..11 {
            resample_row(&img, s, y, &mut row);
            assert_eq!(&out.data()[y * 48..(y + 1) * 48], &row[..]);
        }
    }

    #[test]
    fn resample_empty_output() {
        let empty = Image::new(0, 0, 3);
        assert_eq!(resample(&empty, 0, 0).size(), (0, 0));
    }
}
