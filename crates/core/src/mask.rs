//! Stop-threshold masking (from the 2019 CUDA engine).
//!
//! After a vote, pixels whose colour changed by less than the stop threshold (in every channel)
//! are considered converged. The mask is dilated by the patch radius, and the next PatchMatch
//! pass skips pixels whose whole neighbourhood has converged.

use rayon::prelude::*;

use crate::image::Image;

/// Mask value for pixels that are still being improved.
pub(crate) const ACTIVE: u8 = 255;

/// `krnlEvalMask`: 255 where the largest per-channel change between `a` and `b` is at least
/// `stop_threshold`, 0 elsewhere.
pub(crate) fn eval_mask(a: &Image, b: &Image, stop_threshold: u32, mask: &mut [u8]) {
    let nc = a.channels();
    assert_eq!(a.size(), b.size());
    assert_eq!(mask.len(), a.width() * a.height());
    mask.par_iter_mut().zip(a.data().par_chunks(nc).zip(b.data().par_chunks(nc))).for_each(|(m, (pa, pb))| {
        let max_diff = pa.iter().zip(pb).map(|(&x, &y)| x.abs_diff(y) as u32).max().unwrap_or(0);
        *m = if max_diff < stop_threshold { 0 } else { ACTIVE };
    });
}

/// `krnlDilateMask`: a pixel becomes active if any pixel within the patch window (clamped at the
/// borders) is active.
pub(crate) fn dilate_mask(mask: &[u8], width: usize, height: usize, patch_size: usize) -> Vec<u8> {
    let r = (patch_size / 2) as isize;
    let mut out = vec![0u8; mask.len()];
    if width == 0 {
        return out;
    }
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut active = false;
            'win: for py in -r..=r {
                let yy = (y as isize + py).clamp(0, height as isize - 1) as usize;
                for px in -r..=r {
                    let xx = (x as isize + px).clamp(0, width as isize - 1) as usize;
                    if mask[yy * width + xx] == ACTIVE {
                        active = true;
                        break 'win;
                    }
                }
            }
            *o = if active { ACTIVE } else { 0 };
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_uses_max_channel_difference() {
        let a = Image::from_raw(3, 1, 2, vec![10, 10, 10, 10, 10, 10]);
        let b = Image::from_raw(3, 1, 2, vec![14, 10, 10, 15, 0, 10]);
        let mut m = vec![0; 3];
        eval_mask(&a, &b, 5, &mut m);
        assert_eq!(m, [0, ACTIVE, ACTIVE]);
    }

    #[test]
    fn dilate_grows_by_radius() {
        let mut m = vec![0u8; 7 * 5];
        m[2 * 7 + 3] = ACTIVE;
        let d = dilate_mask(&m, 7, 5, 3);
        for y in 0..5 {
            for x in 0..7 {
                let expect = (2..=4).contains(&x) && (1..=3).contains(&y);
                assert_eq!(d[y * 7 + x] == ACTIVE, expect, "({x},{y})");
            }
        }
    }
}
