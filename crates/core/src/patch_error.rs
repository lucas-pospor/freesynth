//! Patch errors of a finished synthesis, measured with the engine's own patch distance.
//!
//! [`PatchErrorEvaluator`] packs a target image (the synthesised style plus the target guides) and
//! the source exactly as [`synthesize`](crate::synthesize) does at the finest pyramid level, so
//! the errors it reports are the ones the engine minimises: the weighted sum of squared
//! differences over a `patch_size x patch_size` patch, with clamped addressing for target patches
//! that touch the border and per-pixel guide modulation when the input has it. The uniformity term
//! is not included.
//!
//! This is meant for evaluation: the error of the engine's final nearest-neighbour field
//! ([`PatchErrorEvaluator::nnf_errors`], [`nnf_patch_errors`]), or of any other assignment, for
//! example the result of an exhaustive search built on [`PatchDistance::error`].

use std::ops::Range;

use rayon::prelude::*;

use crate::error::{Error, invalid};
use crate::image::{Image, Nnf};
use crate::patch::{EXACT, FAST, Kernel, MODULATED, PatchSpace, pack};
use crate::synth::{Params, SynthesisInput};

/// Which channels a [`PatchDistance`] sums over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorPart {
    /// Style and guide channels: the full patch error the engine minimises.
    Total,
    /// Style channels only.
    Style,
    /// Guide channels only (modulated when the input has a target modulation).
    Guide,
}

/// The packed target and source of one synthesis problem, ready to evaluate patch errors.
#[derive(Clone, Debug)]
pub struct PatchErrorEvaluator {
    target_size: (usize, usize),
    source_size: (usize, usize),
    ns: usize,
    patch: usize,
    target: Vec<u8>,
    source: Vec<u8>,
    modulation: Option<Vec<u8>>,
    weights_total: Vec<f32>,
    weights_style: Vec<f32>,
    weights_guide: Vec<f32>,
}

impl PatchErrorEvaluator {
    /// Packs `input` with `target_style` (for example a synthesis result) as the target's style
    /// channels.
    ///
    /// `target_style` must have the target guide's size and the source style's channel count.
    /// The source must be larger than `patch_size - 1` in both dimensions, so that at least one
    /// source patch fits inside it.
    pub fn new(input: &SynthesisInput, target_style: &Image, patch_size: usize) -> Result<Self, Error> {
        input.validate()?;
        if patch_size < 3 || patch_size.is_multiple_of(2) {
            return invalid("patch size must be an odd number >= 3");
        }
        let ns = input.source_style.channels();
        let ng = input.source_guide.channels();
        let target_size = input.target_guide.size();
        let source_size = input.source_style.size();
        if target_style.size() != target_size {
            return invalid(format!(
                "target image is {}x{}, but the target guide is {}x{}",
                target_style.width(),
                target_style.height(),
                target_size.0,
                target_size.1
            ));
        }
        if target_style.channels() != ns {
            return invalid(format!("target image has {} channels, the style has {ns}", target_style.channels()));
        }
        if source_size.0 < patch_size || source_size.1 < patch_size {
            return invalid(format!("source is too small for patch size {patch_size}"));
        }
        let target = pack(target_style.data(), ns, input.target_guide.data(), ng, target_size.0 * target_size.1);
        let source = pack(input.source_style.data(), ns, input.source_guide.data(), ng, source_size.0 * source_size.1);
        let weights_total: Vec<f32> = input.style_weights.iter().chain(&input.guide_weights).copied().collect();
        let mut weights_style = weights_total.clone();
        weights_style[ns..].fill(0.0);
        let mut weights_guide = weights_total.clone();
        weights_guide[..ns].fill(0.0);
        Ok(Self {
            target_size,
            source_size,
            ns,
            patch: patch_size,
            target,
            source,
            modulation: input.target_modulation.as_ref().map(|m| m.data().to_vec()),
            weights_total,
            weights_style,
            weights_guide,
        })
    }

    pub fn patch_size(&self) -> usize {
        self.patch
    }

    /// `(width, height)` of the target.
    pub fn target_size(&self) -> (usize, usize) {
        self.target_size
    }

    /// `(width, height)` of the source.
    pub fn source_size(&self) -> (usize, usize) {
        self.source_size
    }

    /// The source patch centres whose patch lies inside the source, as `(x range, y range)`.
    /// The engine only ever matches these.
    pub fn source_centres(&self) -> (Range<i32>, Range<i32>) {
        let r = (self.patch / 2) as i32;
        (r..self.source_size.0 as i32 - r, r..self.source_size.1 as i32 - r)
    }

    /// Number of style channels; the packed pixels hold these first, then the guide channels.
    pub fn style_channels(&self) -> usize {
        self.ns
    }

    /// The per-channel weights (style then guide) that a distance over `part` applies.
    pub fn weights(&self, part: ErrorPart) -> &[f32] {
        match part {
            ErrorPart::Total => &self.weights_total,
            ErrorPart::Style => &self.weights_style,
            ErrorPart::Guide => &self.weights_guide,
        }
    }

    /// Whether the guide error is modulated per target pixel.
    pub fn has_modulation(&self) -> bool {
        self.modulation.is_some()
    }

    /// The packed channels (style then guide) of source pixel `(x, y)`.
    #[inline]
    pub fn source_pixel(&self, x: usize, y: usize) -> &[u8] {
        let nc = self.weights_total.len();
        assert!(x < self.source_size.0 && y < self.source_size.1, "source pixel ({x}, {y}) out of range");
        let i = (y * self.source_size.0 + x) * nc;
        &self.source[i..i + nc]
    }

    /// The packed channels (style then guide) of target pixel `(x, y)`.
    #[inline]
    pub fn target_pixel(&self, x: usize, y: usize) -> &[u8] {
        let nc = self.weights_total.len();
        assert!(x < self.target_size.0 && y < self.target_size.1, "target pixel ({x}, {y}) out of range");
        let i = (y * self.target_size.0 + x) * nc;
        &self.target[i..i + nc]
    }

    /// A patch distance over the given channels.
    pub fn distance(&self, part: ErrorPart) -> PatchDistance<'_> {
        let weights = self.weights(part);
        let space = PatchSpace::new(
            self.target_size,
            self.source_size,
            self.ns,
            self.patch,
            &self.target,
            &self.source,
            weights,
            self.modulation.as_deref(),
        );
        let kernel = space.kernel(false);
        PatchDistance { space, kernel }
    }

    /// The style and guide parts of the patch error of every match in `nnf`.
    ///
    /// Fails if `nnf` does not have the target's size or points at a source patch that is not
    /// inside the source (see [`source_centres`](Self::source_centres)).
    pub fn nnf_errors(&self, nnf: &Nnf) -> Result<NnfErrors, Error> {
        let (tw, th) = self.target_size;
        if (nnf.width(), nnf.height()) != (tw, th) {
            return invalid(format!("NNF is {}x{}, but the target is {tw}x{th}", nnf.width(), nnf.height()));
        }
        let (xs, ys) = self.source_centres();
        if let Some((i, s)) = nnf.data().iter().enumerate().find(|(_, s)| !xs.contains(&s[0]) || !ys.contains(&s[1])) {
            return invalid(format!(
                "NNF entry at ({}, {}) points at ({}, {}), outside the valid source centres for patch size {}",
                i % tw,
                i / tw,
                s[0],
                s[1],
                self.patch
            ));
        }
        let eval = |part| {
            let d = self.distance(part);
            nnf.data()
                .par_iter()
                .enumerate()
                .map(|(i, &s)| d.error([(i % tw) as i32, (i / tw) as i32], s, f32::INFINITY))
                .collect::<Vec<f32>>()
        };
        Ok(NnfErrors { width: tw, height: th, style: eval(ErrorPart::Style), guide: eval(ErrorPart::Guide) })
    }
}

/// The engine's patch distance over one [`ErrorPart`] of a [`PatchErrorEvaluator`].
pub struct PatchDistance<'a> {
    space: PatchSpace<'a>,
    kernel: Kernel,
}

impl PatchDistance<'_> {
    /// Error between the target patch centred at `target` and the source patch centred at
    /// `source`.
    ///
    /// Evaluation may stop early, after a full patch row, once the partial error exceeds
    /// `bound`; the returned value is then larger than `bound` but smaller than the full error.
    /// Pass `f32::INFINITY` for the full error. (Target patches touching the border are always
    /// evaluated in full.)
    ///
    /// # Panics
    ///
    /// If `target` is outside the target or `source` is not a valid source centre (see
    /// [`PatchErrorEvaluator::source_centres`]).
    #[inline]
    pub fn error(&self, target: [i32; 2], source: [i32; 2], bound: f32) -> f32 {
        let s = &self.space;
        let r = (s.patch / 2) as i32;
        assert!(
            (0..s.tw as i32).contains(&target[0]) && (0..s.th as i32).contains(&target[1]),
            "target pixel {target:?} is outside the target"
        );
        assert!(
            (r..s.sw as i32 - r).contains(&source[0]) && (r..s.sh as i32 - r).contains(&source[1]),
            "source centre {source:?} is not inside the source"
        );
        let [tx, ty] = target;
        let [sx, sy] = source;
        match self.kernel {
            Kernel::Fast => s.error::<FAST>(tx, ty, sx, sy, bound),
            Kernel::Exact => s.error::<EXACT>(tx, ty, sx, sy, bound),
            Kernel::Modulated => s.error::<MODULATED>(tx, ty, sx, sy, bound),
        }
    }
}

/// Per-pixel patch errors of a nearest-neighbour field (sums over the patch, not averages).
#[derive(Clone, Debug, PartialEq)]
pub struct NnfErrors {
    pub width: usize,
    pub height: usize,
    /// Style part of every target pixel's patch error, in scan-line order.
    pub style: Vec<f32>,
    /// Guide part of every target pixel's patch error, in scan-line order.
    pub guide: Vec<f32>,
}

/// Patch errors of `nnf` for the synthesis problem `input` solved with `params`, with `image`
/// (normally the synthesis result) as the target style.
///
/// The patch size is the one the final field was computed with: 3 when
/// [`Params::extra_pass_3x3`] is set, otherwise [`Params::patch_size`]. Errors are evaluated with
/// the engine's fast kernel; with [`Params::bit_exact`] the engine's own errors can differ from
/// these in the last bits.
pub fn nnf_patch_errors(input: &SynthesisInput, params: &Params, image: &Image, nnf: &Nnf) -> Result<NnfErrors, Error> {
    let patch = if params.extra_pass_3x3 { 3 } else { params.patch_size };
    PatchErrorEvaluator::new(input, image, patch)?.nnf_errors(nnf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::synthesize;

    fn test_image(w: usize, h: usize, seed: u32) -> Image {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut img = Image::new(w, h, 3);
        for y in 0..h {
            for x in 0..w {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let p = img.pixel_mut(x, y);
                p[0] = ((x * 7 + y * 3) % 256) as u8;
                p[1] = (state >> 24) as u8;
                p[2] = ((x * y) % 251) as u8;
            }
        }
        img
    }

    fn input(src: &Image, tgt: &Image) -> SynthesisInput {
        SynthesisInput {
            source_style: src.clone(),
            source_guide: src.clone(),
            target_guide: tgt.clone(),
            target_modulation: None,
            style_weights: vec![1.0 / 3.0; 3],
            guide_weights: vec![2.0 / 3.0; 3],
            modulated_range: None,
        }
    }

    /// Brute-force reference: clamped addressing on both sides, f64 accumulation.
    fn reference(inp: &SynthesisInput, style: &Image, patch: i32, t: [i32; 2], s: [i32; 2]) -> (f64, f64) {
        let r = patch / 2;
        let (tw, th) = (style.width() as i32, style.height() as i32);
        let (sw, sh) = (inp.source_style.width() as i32, inp.source_style.height() as i32);
        let (mut es, mut eg) = (0.0, 0.0);
        for dy in -r..=r {
            for dx in -r..=r {
                let (tx, ty) = ((t[0] + dx).clamp(0, tw - 1) as usize, (t[1] + dy).clamp(0, th - 1) as usize);
                let (sx, sy) = ((s[0] + dx).clamp(0, sw - 1) as usize, (s[1] + dy).clamp(0, sh - 1) as usize);
                for (k, (&a, &b)) in style.pixel(tx, ty).iter().zip(inp.source_style.pixel(sx, sy)).enumerate() {
                    es += inp.style_weights[k] as f64 * (a as f64 - b as f64).powi(2);
                }
                for (k, (&a, &b)) in
                    inp.target_guide.pixel(tx, ty).iter().zip(inp.source_guide.pixel(sx, sy)).enumerate()
                {
                    eg += inp.guide_weights[k] as f64 * (a as f64 - b as f64).powi(2);
                }
            }
        }
        (es, eg)
    }

    #[test]
    fn matches_brute_force() {
        let src = test_image(23, 19, 1);
        let tgt = test_image(17, 13, 2);
        let style = test_image(17, 13, 3);
        let inp = input(&src, &tgt);
        let ev = PatchErrorEvaluator::new(&inp, &style, 5).unwrap();
        let (total, st, gd) =
            (ev.distance(ErrorPart::Total), ev.distance(ErrorPart::Style), ev.distance(ErrorPart::Guide));
        for (t, s) in [([8, 6], [10, 9]), ([0, 0], [2, 2]), ([16, 12], [20, 16]), ([3, 12], [11, 5])] {
            let (es, eg) = reference(&inp, &style, 5, t, s);
            let close = |a: f32, b: f64| (a as f64 - b).abs() <= 1e-4 * b.max(1.0);
            assert!(close(st.error(t, s, f32::INFINITY), es));
            assert!(close(gd.error(t, s, f32::INFINITY), eg));
            assert!(close(total.error(t, s, f32::INFINITY), es + eg));
        }
    }

    #[test]
    fn identity_field_has_zero_error() {
        let img = test_image(20, 16, 4);
        let inp = input(&img, &img);
        let ev = PatchErrorEvaluator::new(&inp, &img, 3).unwrap();
        // Interior pixels map to themselves; border pixels to the nearest valid centre.
        let nnf = Nnf::from_raw(20, 16, (0..20 * 16).map(|i| [(i % 20).clamp(1, 18), (i / 20).clamp(1, 14)]).collect());
        let e = ev.nnf_errors(&nnf).unwrap();
        for y in 1..15 {
            for x in 1..19 {
                assert_eq!(e.style[y * 20 + x], 0.0);
                assert_eq!(e.guide[y * 20 + x], 0.0);
            }
        }
    }

    #[test]
    fn rejects_bad_fields_and_sizes() {
        let img = test_image(12, 10, 5);
        let inp = input(&img, &img);
        let ev = PatchErrorEvaluator::new(&inp, &img, 5).unwrap();
        assert_eq!(ev.source_centres(), (2..10, 2..8));
        assert_eq!(ev.style_channels(), 3);
        assert_eq!(ev.source_pixel(3, 4), &[img.pixel(3, 4), img.pixel(3, 4)].concat()[..]);
        assert_eq!(ev.target_pixel(11, 9).len(), 6);
        assert_eq!(ev.weights(ErrorPart::Style)[3..], [0.0; 3]);
        assert!(!ev.has_modulation());
        assert!(ev.nnf_errors(&Nnf::new(11, 10)).is_err());
        assert!(ev.nnf_errors(&Nnf::new(12, 10)).is_err()); // [0, 0] is not a valid centre
        assert!(PatchErrorEvaluator::new(&inp, &test_image(11, 10, 5), 5).is_err());
        assert!(PatchErrorEvaluator::new(&inp, &img, 4).is_err());
        assert!(PatchErrorEvaluator::new(&inp, &img, 13).is_err());
    }

    #[test]
    fn final_field_errors_follow_the_engine() {
        let src = test_image(48, 40, 6);
        let mut inp = input(&src, &test_image(40, 36, 7));
        inp.target_guide = crate::resample(&src, 40, 36);
        let params = Params { num_threads: Some(2), ..Params::default() };
        let out = synthesize(&inp, &params).unwrap();
        let e = nnf_patch_errors(&inp, &params, &out.image, &out.nnf).unwrap();
        assert_eq!(e.style.len(), 40 * 36);
        // A random field must be much worse than the optimised one.
        let mut rng = crate::GlibcRand::new(3);
        let random = crate::nnf_init_random((40, 36), (48, 40), 5, &mut rng);
        let er = nnf_patch_errors(&inp, &params, &out.image, &random).unwrap();
        let sum = |e: &NnfErrors| e.style.iter().chain(&e.guide).map(|&v| v as f64).sum::<f64>();
        assert!(sum(&e) * 3.0 < sum(&er), "{} vs {}", sum(&e), sum(&er));
    }
}
