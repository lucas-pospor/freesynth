//! Source occupancy: how evenly a nearest-neighbour field uses the source.
//!
//! Every target pixel's source patch is splatted onto a per-source-pixel counter. A synthesis
//! that copies a few well-matching source regions everywhere ("wash-out") gets low energy but
//! leaves much of the source unused; these statistics catch that. They depend on which field they
//! are read from (an engine's field uses the source more evenly than a pure nearest-patch field),
//! so compare them only between fields of the same kind.

use freesynth_core::Nnf;

use crate::error::{Error, invalid};

/// Occupancy statistics of a field.
#[derive(Clone, Debug, PartialEq)]
pub struct Occupancy {
    /// Fraction of source pixels covered by no matched patch.
    pub unused_frac: f64,
    /// Coefficient of variation (std / mean) of the per-pixel counts.
    pub cv: f64,
    /// Entropy of the normalised counts divided by the log of the number of source pixels
    /// (1 = perfectly even).
    pub norm_entropy: f64,
    /// Largest count divided by the mean count.
    pub max_over_mean: f64,
    /// 99th percentile count divided by the mean count.
    pub p99_over_mean: f64,
    /// Number of distinct source centres used, divided by the number of target pixels.
    pub unique_centres_frac: f64,
    /// Fraction of horizontally or vertically adjacent target pixels with the same offset
    /// (match minus position): a measure of how large the copied chunks are.
    pub same_offset_frac: f64,
}

/// Occupancy of `nnf` over a source of `source_size` with patches of `patch_size`.
///
/// Fails if a field entry is not a valid source centre (its patch must fit inside the source).
pub fn occupancy(nnf: &Nnf, source_size: (usize, usize), patch_size: usize) -> Result<Occupancy, Error> {
    let (sw, sh) = source_size;
    let r = (patch_size / 2) as i32;
    if patch_size.is_multiple_of(2) || patch_size == 0 {
        return invalid("patch size must be odd");
    }
    if nnf.data().is_empty() || sw == 0 || sh == 0 {
        return invalid("empty field or source");
    }
    if nnf.data().iter().any(|&[x, y]| x < r || y < r || x >= sw as i32 - r || y >= sh as i32 - r) {
        return invalid(format!("field points outside the valid source centres for patch size {patch_size} in a {sw}x{sh} source"));
    }
    let mut counts = vec![0u32; sw * sh];
    let mut used = vec![false; sw * sh];
    let mut unique = 0usize;
    for &[x, y] in nnf.data() {
        let c = y as usize * sw + x as usize;
        if !used[c] {
            used[c] = true;
            unique += 1;
        }
        for dy in -r..=r {
            let row = (y + dy) as usize * sw;
            for cell in &mut counts[row + (x - r) as usize..=row + (x + r) as usize] {
                *cell += 1;
            }
        }
    }
    let n = counts.len() as f64;
    let total: f64 = counts.iter().map(|&c| c as f64).sum();
    let mean = total / n;
    let var = counts.iter().map(|&c| (c as f64 - mean).powi(2)).sum::<f64>() / n;
    let entropy: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum();
    let mut sorted = counts.clone();
    sorted.sort_unstable();
    let p99 = sorted[((sorted.len() - 1) as f64 * 0.99) as usize] as f64;
    let max = *sorted.last().expect("non-empty") as f64;

    let (w, h) = (nnf.width(), nnf.height());
    let offset = |x: usize, y: usize| {
        let [sx, sy] = nnf.get(x, y);
        [sx - x as i32, sy - y as i32]
    };
    let (mut same, mut pairs) = (0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if x + 1 < w {
                pairs += 1;
                same += usize::from(offset(x, y) == offset(x + 1, y));
            }
            if y + 1 < h {
                pairs += 1;
                same += usize::from(offset(x, y) == offset(x, y + 1));
            }
        }
    }

    Ok(Occupancy {
        unused_frac: counts.iter().filter(|&&c| c == 0).count() as f64 / n,
        cv: var.sqrt() / mean,
        norm_entropy: if n > 1.0 { entropy / n.ln() } else { 1.0 },
        max_over_mean: max / mean,
        p99_over_mean: p99 / mean,
        unique_centres_frac: unique as f64 / nnf.data().len() as f64,
        same_offset_frac: if pairs > 0 { same as f64 / pairs as f64 } else { 1.0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_centre_uses_one_patch() {
        // Every target pixel uses the patch at (2, 2) of a 10x10 source: 9 of 100 pixels used.
        let nnf = Nnf::from_raw(4, 4, vec![[2, 2]; 16]);
        let o = occupancy(&nnf, (10, 10), 3).unwrap();
        assert!((o.unused_frac - 0.91).abs() < 1e-12);
        assert!((o.max_over_mean - 100.0 / 9.0).abs() < 1e-9);
        assert!((o.unique_centres_frac - 1.0 / 16.0).abs() < 1e-12);
        assert!(o.same_offset_frac < 1e-12); // offsets differ everywhere (identity offset varies)
        assert!((o.norm_entropy - 9f64.ln() / 100f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn identity_field_covers_the_source() {
        // A 1:1 field over the interior of the source with patch 3.
        let (w, h) = (8, 6);
        let nnf = Nnf::from_raw(w - 2, h - 2, (0..(w - 2) * (h - 2)).map(|i| [(i % (w - 2)) as i32 + 1, (i / (w - 2)) as i32 + 1]).collect());
        let o = occupancy(&nnf, (w, h), 3).unwrap();
        assert_eq!(o.unused_frac, 0.0);
        assert_eq!(o.unique_centres_frac, 1.0);
        assert_eq!(o.same_offset_frac, 1.0);
        assert!(o.cv > 0.0); // corners are covered once, the middle nine times
    }

    #[test]
    fn rejects_out_of_range_centres() {
        assert!(occupancy(&Nnf::from_raw(1, 1, vec![[0, 1]]), (5, 5), 3).is_err());
        assert!(occupancy(&Nnf::from_raw(1, 1, vec![[4, 2]]), (5, 5), 3).is_err());
        assert!(occupancy(&Nnf::from_raw(1, 1, vec![[3, 3]]), (5, 5), 3).is_ok());
    }
}
