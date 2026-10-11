//! Blending the two passes of a keyframe interval.

use freesynth_core::Image;
use rayon::prelude::*;

use crate::guides::to_rgba;

/// How the two passes of an interval are combined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    /// Linear crossfade: `(1 - a) A + a B` with `a` going from 0 at the interval's first
    /// keyframe to 1 at its last. This is how keyframe passes have traditionally been combined
    /// (as an opacity crossfade in a compositor). It is temporally smooth, but where the passes
    /// differ it shows both (ghosting) and softens detail in the middle of the interval.
    #[default]
    Linear,
}

impl BlendMode {
    /// The mode's lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Linear => "linear",
        }
    }
}

impl std::str::FromStr for BlendMode {
    type Err = crate::VideoError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "linear" | "crossfade" => Ok(BlendMode::Linear),
            _ => Err(crate::VideoError::InvalidInput(format!("unknown blend mode '{s}' (expected linear)"))),
        }
    }
}

/// Blends pass `a` and pass `b` of one frame with weight `alpha` on `b` (rounded to the nearest
/// level). Images with different channel counts are blended as RGBA.
///
/// # Panics
/// If the sizes differ.
pub fn blend(mode: BlendMode, a: &Image, b: &Image, alpha: f32) -> Image {
    assert_eq!(a.size(), b.size(), "passes differ in size");
    if a.channels() != b.channels() {
        return blend(mode, &to_rgba(a), &to_rgba(b), alpha);
    }
    match mode {
        BlendMode::Linear => linear(a, b, alpha),
    }
}

fn linear(a: &Image, b: &Image, alpha: f32) -> Image {
    let alpha = alpha.clamp(0.0, 1.0);
    let mut data = vec![0u8; a.data().len()];
    data.par_chunks_mut(1 << 14).zip(a.data().par_chunks(1 << 14).zip(b.data().par_chunks(1 << 14))).for_each(
        |(out, (a, b))| {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                let (x, y) = (f32::from(x), f32::from(y));
                *o = (x + (y - x) * alpha).round() as u8;
            }
        },
    );
    Image::from_raw(a.width(), a.height(), a.channels(), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_crossfade() {
        let a = Image::from_raw(2, 1, 3, vec![0, 100, 255, 10, 10, 10]);
        let b = Image::from_raw(2, 1, 3, vec![255, 0, 255, 11, 10, 9]);
        assert_eq!(blend(BlendMode::Linear, &a, &b, 0.0), a);
        assert_eq!(blend(BlendMode::Linear, &a, &b, 1.0), b);
        assert_eq!(blend(BlendMode::Linear, &a, &b, 0.5).data(), &[128, 50, 255, 11, 10, 10]);
        let rgba = Image::from_raw(2, 1, 4, vec![255, 0, 255, 0, 11, 10, 9, 255]);
        let mixed = blend(BlendMode::Linear, &a, &rgba, 0.5);
        assert_eq!(mixed.channels(), 4);
        assert_eq!(mixed.data(), &[128, 50, 255, 128, 11, 10, 10, 255]);
        assert_eq!("crossfade".parse::<BlendMode>().unwrap(), BlendMode::Linear);
    }
}
