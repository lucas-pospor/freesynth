//! Optional inputs of a synthesis beyond the images and weights ([`SynthesisExtras`]): a warm
//! start from a known field and a high-precision style.

use rayon::prelude::*;

use crate::image::{Image16, Nnf};

/// Optional inputs of [`synthesize_with`](crate::synthesize_with). The default (all `None`)
/// runs the plain synthesis of [`synthesize`](crate::synthesize).
#[derive(Clone, Debug, Default)]
pub struct SynthesisExtras {
    /// Start from a known field and run only the finest pyramid levels.
    pub warm_start: Option<WarmStart>,
    /// The style at 16 bits per channel: the source style's size and channels (the search still
    /// runs on the 8-bit [`source_style`](crate::SynthesisInput::source_style)). The final field
    /// is then voted from it as well, with the weights of the 8-bit result's final vote, and the
    /// result is returned in [`SynthesisOutput::image16`](crate::SynthesisOutput::image16),
    /// rounded to the nearest 16-bit value. Votes are averages, so the 16-bit result carries
    /// fractions that the 8-bit one rounds away, even from an 8-bit style.
    pub style16: Option<Image16>,
}

impl SynthesisExtras {
    /// Whether every extra is absent.
    pub fn is_empty(&self) -> bool {
        self.warm_start.is_none() && self.style16.is_none()
    }
}

/// A known nearest-neighbour field to start from, for example the previous video frame's final
/// field moved along the optical flow. Only the finest [`levels`](Self::levels) pyramid levels
/// run: the coarse levels exist to find the large-scale layout, which the field already has.
#[derive(Clone, Debug, PartialEq)]
pub struct WarmStart {
    /// A field of the full-resolution target size whose entries are full-resolution source
    /// patch centres. Entries outside the valid centres are clamped.
    pub nnf: Nnf,
    /// Number of finest pyramid levels to run, at least 1 (more than the pyramid has runs them
    /// all). The field is scaled down to start the first of them.
    pub levels: usize,
}

impl WarmStart {
    /// The field scaled down to a pyramid level whose target is `target_level` and whose source
    /// is `source_level` (of a `source_full` source): each level pixel takes the entry at its
    /// sample position (as the pyramid samples images), divided by the source scale and clamped
    /// to the level's valid patch centres for `patch_size`.
    pub fn field_at_level(
        &self,
        target_level: (usize, usize),
        source_full: (usize, usize),
        source_level: (usize, usize),
        patch_size: usize,
    ) -> Nnf {
        warm_field(&self.nnf, target_level, source_full, source_level, patch_size)
    }
}

/// `nnf` (full-resolution target and source) scaled down to a pyramid level: each level pixel
/// takes the entry at its sample position (as the pyramid samples images), divided by the source
/// scale and clamped to the valid patch centres of the level's source.
pub(crate) fn warm_field(
    nnf: &Nnf,
    target_level: (usize, usize),
    source_full: (usize, usize),
    source_level: (usize, usize),
    patch_size: usize,
) -> Nnf {
    let (fw, fh) = (nnf.width(), nnf.height());
    let (tw, th) = target_level;
    let s_t = fw as f32 / tw.max(1) as f32;
    let s_s = source_full.0 as f32 / source_level.0.max(1) as f32;
    let r = (patch_size / 2) as i32;
    let (max_x, max_y) = (source_level.0 as i32 - r - 1, source_level.1 as i32 - r - 1);
    let mut out = Nnf::new(tw, th);
    out.data_mut().par_chunks_mut(tw.max(1)).enumerate().for_each(|(y, row)| {
        let fy = ((s_t * y as f32) as usize).min(fh - 1);
        for (x, v) in row.iter_mut().enumerate() {
            let fx = ((s_t * x as f32) as usize).min(fw - 1);
            let [nx, ny] = nnf.get(fx, fy);
            let lx = (nx as f32 / s_s).round() as i32;
            let ly = (ny as f32 / s_s).round() as i32;
            *v = [lx.max(r).min(max_x), ly.max(r).min(max_y)];
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_field_scales_and_clamps() {
        let nnf = Nnf::from_raw(4, 2, vec![[0, 0], [10, 2], [20, 4], [39, 19], [8, 8], [9, 9], [10, 10], [11, 11]]);
        let level = warm_field(&nnf, (2, 1), (40, 20), (20, 10), 5);
        // Entries halve and clamp to [2, 17] x [2, 7].
        assert_eq!(level.data(), &[[2, 2], [10, 2]]);
        let full = warm_field(&nnf, (4, 2), (40, 20), (40, 20), 5);
        assert_eq!(full.get(1, 0), [10, 2]);
        assert_eq!(full.get(3, 0), [37, 17]);
    }
}
