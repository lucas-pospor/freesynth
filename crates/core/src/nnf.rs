//! Nearest-neighbour field initialisation and upscaling.

use rayon::prelude::*;

use crate::image::Nnf;
use crate::rng::GlibcRand;

/// Random initial field: every target pixel gets a source patch centre drawn uniformly from
/// `[r, size - r)` with `r = patch_size / 2` (`nnfInitRandom`).
///
/// Draws two numbers per pixel from `rng` in scan-line order (y first, then x, matching the
/// argument evaluation order of the 2019 build), so the result is identical to the original for
/// the same generator state.
///
/// # Panics
///
/// If the source is not larger than `2 * (patch_size / 2)` in both dimensions (no patch fits).
pub fn nnf_init_random(target: (usize, usize), source: (usize, usize), patch_size: usize, rng: &mut GlibcRand) -> Nnf {
    let r = (patch_size / 2) as i32;
    let range_x = source.0 as i32 - 2 * r;
    let range_y = source.1 as i32 - 2 * r;
    assert!(range_x > 0 && range_y > 0, "source is too small for the patch size");
    let data = (0..target.0 * target.1)
        .map(|_| {
            let y = r + rng.next_i32() % range_y;
            let x = r + rng.next_i32() % range_x;
            [x, y]
        })
        .collect();
    Nnf::from_raw(target.0, target.1, data)
}

/// Upscales a field to the next (2x) pyramid level (`nnfUpscale`).
///
/// Each fine pixel inherits its coarse parent's match, doubled and offset by the pixel's position
/// within its 2x2 block, then clamped to `[patch_size, source - patch_size - 1]`.
///
/// # Panics
///
/// If `nnf` is empty while `target` is not (there is no coarse match to inherit).
pub fn nnf_upscale(nnf: &Nnf, patch_size: usize, target: (usize, usize), source: (usize, usize)) -> Nnf {
    let mut out = Nnf::new(target.0, target.1);
    if target.0 == 0 || target.1 == 0 {
        return out;
    }
    assert!(nnf.width() > 0 && nnf.height() > 0, "cannot upscale an empty field to {}x{}", target.0, target.1);
    let p = patch_size as i32;
    let (max_x, max_y) = (source.0 as i32 - p - 1, source.1 as i32 - p - 1);
    let (cw, ch) = (nnf.width() as i32, nnf.height() as i32);
    out.data_mut().par_chunks_mut(target.0).enumerate().for_each(|(y, row)| {
        let y = y as i32;
        for (x, v) in row.iter_mut().enumerate() {
            let x = x as i32;
            let [nx, ny] = nnf.get((x / 2).clamp(0, cw - 1) as usize, (y / 2).clamp(0, ch - 1) as usize);
            // `clamp(v, lo, hi)` of the original is `min(max(v, lo), hi)`, which (unlike
            // `i32::clamp`) does not panic when lo > hi.
            *v = [(nx * 2 + x % 2).max(p).min(max_x), (ny * 2 + y % 2).max(p).min(max_y)];
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_init_is_in_range() {
        let mut rng = GlibcRand::default();
        let nnf = nnf_init_random((30, 20), (17, 13), 5, &mut rng);
        for &[x, y] in nnf.data() {
            assert!((2..15).contains(&x), "x = {x}");
            assert!((2..11).contains(&y), "y = {y}");
        }
    }

    #[test]
    fn upscale_doubles_and_offsets() {
        // A 2x2 coarse field upscaled to 4x4 with a large source: no clamping.
        let coarse = Nnf::from_raw(2, 2, vec![[10, 20], [11, 20], [10, 21], [30, 40]]);
        let fine = nnf_upscale(&coarse, 5, (4, 4), (100, 100));
        assert_eq!(fine.get(0, 0), [20, 40]);
        assert_eq!(fine.get(1, 0), [21, 40]);
        assert_eq!(fine.get(0, 1), [20, 41]);
        assert_eq!(fine.get(1, 1), [21, 41]);
        assert_eq!(fine.get(2, 0), [22, 40]);
        assert_eq!(fine.get(3, 3), [61, 81]);
    }

    #[test]
    fn upscale_handles_odd_sizes_and_clamps() {
        let coarse = Nnf::from_raw(2, 1, vec![[2, 2], [40, 40]]);
        // Fine level is 5x3: column 4 and row 2 reuse the last coarse column/row.
        let fine = nnf_upscale(&coarse, 5, (5, 3), (50, 50));
        assert_eq!(fine.get(0, 0), [5, 5]); // 4 -> clamped up to patch_size
        assert_eq!(fine.get(4, 2), [44, 44]); // 80 -> clamped down to 50 - 5 - 1
        assert_eq!(fine.get(2, 2), [44, 44]);
    }
}
