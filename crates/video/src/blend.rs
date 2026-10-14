//! Blending the two passes of a keyframe interval ([`BlendMode`], [`Blender`]).

use freesynth_core::{Image, Image16};
use rayon::prelude::*;

use crate::error::{VideoError, invalid};
use crate::poisson::{gaussian_blur, solve_screened};

/// How the two passes of an interval are combined.
///
/// Between two keyframes, pass A runs forward from the first and pass B backward from the last,
/// and every frame in between is synthesized twice. The passes agree in colour and layout but
/// not in their strokes: each copies its own keyframe's. Two ways to combine them are built in,
/// and a host can plug in its own ([`Blender`]).
///
/// **Linear** crossfades the passes with a weight that moves from one keyframe to the other. It
/// is smooth in time, but where the passes' strokes differ it shows both of them at half
/// strength: detail fades in the middle of the interval and structures appear doubled.
///
/// **Adaptive** (the default of [`VideoParams::default`](crate::VideoParams::default)) chooses
/// per pixel, softly, the pass that explains the frame better, judged by the synthesis engine's
/// own per-pixel patch error, and blends in the gradient domain:
///
/// 1. Each pass's patch error is smoothed spatially ([`BlendSettings::error_smoothing`]) and
///    turned into a soft weight `w_A = sigmoid(4 a (1 - a) (e_B - e_A) / s + gamma ln((1 - a) / a))`,
///    where `a` is the position in the interval. The second term is a temporal prior: it makes
///    each pass win completely at its own keyframe, and with equal errors (and `gamma` = 1) it
///    gives exactly the linear crossfade weight. The factor `4 a (1 - a)` fades the error term
///    out towards the keyframes, so the frames next to a keyframe stay as close to its pass as
///    in a crossfade, and the errors decide in the middle of the interval.
/// 2. The target gradient field mixes the passes' gradients with these weights. Mixing two
///    uncorrelated stroke textures halves their contrast where the weights are near 1/2, so the
///    mix can be partly normalised back to the passes' detail energy, using the local correlation
///    of the two gradient fields so that detail the passes share is not amplified
///    ([`BlendSettings::detail`]).
/// 3. The output is the image whose gradients best match the mixed field while staying close to
///    the linear crossfade in its low frequencies: a screened Poisson problem with a weak data
///    term ([`BlendSettings::data_weight`]), solved by multigrid.
///
/// Selection is never hard: hard per-pixel switching between passes makes seams that jump from
/// frame to frame. Alpha is blended with the selection weights directly (no gradient domain), and
/// colour is blended premultiplied, so mattes keep clean edges; its data term is then the linear
/// crossfade of the straight colour at the blended alpha, so that colour keeps its value where the
/// passes' mattes differ.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    /// Linear crossfade: `(1 - a) A + a B` with `a` going from 0 at the interval's first
    /// keyframe to 1 at its last. This is how keyframe passes have traditionally been combined
    /// (as an opacity crossfade in a compositor). It is temporally smooth, but where the passes
    /// differ it shows both (ghosting) and softens detail in the middle of the interval.
    Linear,
    /// Soft per-pixel selection of the pass with the lower synthesis error, blended in the
    /// gradient domain ([`BlendSettings`]; see [`BlendMode`]). Keeps the
    /// passes' detail without doubled structures. Needs the engine's per-pixel errors, which the
    /// renderer requests for the passes it blends. The default.
    #[default]
    Adaptive,
    /// Like [`Adaptive`](BlendMode::Adaptive), but only the fine detail is error-selected: each
    /// pass is split into a base (shading and shadows, Gaussian [`BlendSettings::shading_scale`])
    /// and a detail residual (the strokes). The base is crossfaded, so soft shadows stay soft
    /// instead of taking one pass's hard edge ("looking like an object"), while the detail is
    /// selected per pixel like the adaptive blend. No gradient-domain solve, so it is also cheaper.
    /// Also needs the engine's per-pixel errors.
    Layered,
}

impl BlendMode {
    /// The mode's lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Linear => "linear",
            BlendMode::Adaptive => "adaptive",
            BlendMode::Layered => "layered",
        }
    }
}

impl std::str::FromStr for BlendMode {
    type Err = VideoError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "linear" | "crossfade" => Ok(BlendMode::Linear),
            "adaptive" => Ok(BlendMode::Adaptive),
            "layered" => Ok(BlendMode::Layered),
            _ => invalid(format!("unknown blend mode '{s}' (expected linear, adaptive or layered)")),
        }
    }
}

/// Settings of the [adaptive](BlendMode::Adaptive) blend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendSettings {
    /// How soft the selection is, in grey levels of root-mean-square patch error: in the middle
    /// of an interval, a pass whose smoothed error is lower by this much gets `e` (2.7) times the
    /// odds of the other. Smaller
    /// values select more decisively (sharper, but less stable in time).
    pub softness: f32,
    /// Standard deviation of the spatial smoothing of the patch errors before the selection, as
    /// a fraction of the larger frame dimension. Smoothing makes the selection follow regions
    /// (the face that one pass tracks better) rather than single strokes, which keeps it stable
    /// from frame to frame.
    pub error_smoothing: f32,
    /// Strength of the temporal prior (`gamma`): how strongly the weight follows the position in
    /// the interval. 1 reproduces the linear crossfade weight when both passes are equally good;
    /// it is what makes each pass win completely at its own keyframe.
    pub temporal_prior: f32,
    /// Weight of the data term (`lambda`): how strongly the low frequencies stay at the linear
    /// crossfade. Smaller keeps more of the selected detail, larger is steadier in time.
    pub data_weight: f32,
    /// How much of the detail energy lost by mixing two uncorrelated stroke textures is
    /// restored, from 0 (plain weighted mix of the gradients) to 1 (full energy).
    pub detail: f32,
    /// [`Layered`](BlendMode::Layered) only: the scale that separates shading from detail, as a
    /// fraction of the larger frame side. Each pass is split into a base (Gaussian of this scale,
    /// carrying the shading and the shadows) and a detail residual; the base is crossfaded and only
    /// the detail is error-selected, so a soft shadow never gains a selected hard edge. About 0.006
    /// (6 to 12 px at SD/HD) softens the shading while keeping the strokes; larger moves more into
    /// the selected detail (sharper, closer to [`Adaptive`](BlendMode::Adaptive)), 0 is the crossfade.
    pub shading_scale: f32,
}

impl Default for BlendSettings {
    fn default() -> Self {
        Self {
            softness: 0.33,
            error_smoothing: 0.125,
            temporal_prior: 1.0,
            data_weight: 0.05,
            detail: 0.0,
            shading_scale: 0.006,
        }
    }
}

impl BlendSettings {
    /// Checks the settings.
    pub fn validate(&self) -> Result<(), VideoError> {
        let finite = [
            self.softness,
            self.error_smoothing,
            self.temporal_prior,
            self.data_weight,
            self.detail,
            self.shading_scale,
        ];
        if finite.iter().any(|v| !v.is_finite()) {
            return invalid("blend settings must be finite");
        }
        if self.softness <= 0.0 || self.data_weight <= 0.0 {
            return invalid("blend softness and data weight must be positive");
        }
        if self.error_smoothing < 0.0 || self.temporal_prior < 0.0 || !(0.0..=1.0).contains(&self.detail) {
            return invalid("blend error smoothing and temporal prior must be non-negative, detail in 0..=1");
        }
        if self.shading_scale < 0.0 {
            return invalid("blend shading scale must be non-negative");
        }
        Ok(())
    }
}

/// One pass's result for a frame, as handed to a [`Blender`]: an 8-bit [`Image`], or an
/// [`Image16`] when the render produces 16-bit output ([`Blender::blend16`]).
#[derive(Debug)]
pub struct PassFrame<'a, I = Image> {
    /// The synthesized frame (the style's channels: grey, grey + alpha, RGB or RGBA).
    pub image: &'a I,
    /// When the blender asked for them ([`Blender::needs_errors`]): the engine's final patch
    /// error of every pixel, as a root-mean-square difference in grey levels: the weighted
    /// squared differences over the patch divided by the total weight the patch actually had
    /// (summed over its pixels: the style weights plus each guide channel's weight times its
    /// per-pixel modulation, so a gated guide does not shrink the error), square-rooted.
    /// Row-major, one value per pixel.
    pub errors: Option<&'a [f32]>,
}

// Manual impls: the derives would require `I: Copy`.
impl<I> Clone for PassFrame<'_, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for PassFrame<'_, I> {}

/// Combines the two passes of an interval into one frame. Implement it to plug a custom blend
/// into a [`Renderer`](crate::Renderer::blender).
pub trait Blender: Sync {
    /// Whether [`PassFrame::errors`] should be filled (it costs an extra error evaluation per
    /// synthesis and memory while frames wait for their partner).
    fn needs_errors(&self) -> bool;

    /// Blends pass `a` (from the interval's first keyframe) and pass `b` (from its last) at
    /// position `alpha` in the interval: 0 at the first keyframe, 1 at the last. Both images
    /// have the same size; their channel counts may differ.
    fn blend(&self, a: PassFrame<'_>, b: PassFrame<'_>, alpha: f32) -> Result<Image, VideoError>;

    /// [`blend`](Self::blend) for 16-bit passes, used when the render produces 16-bit frames
    /// ([`Renderer::keyframes16`](crate::Renderer::keyframes16)). The default rounds the passes
    /// to 8 bits, blends them with [`blend`](Self::blend) and widens the result, so a blender
    /// that only implements `blend` still works but loses the extra precision.
    fn blend16(&self, a: PassFrame<'_, Image16>, b: PassFrame<'_, Image16>, alpha: f32) -> Result<Image16, VideoError> {
        let (a8, b8) = (a.image.to_image(), b.image.to_image());
        let out =
            self.blend(PassFrame { image: &a8, errors: a.errors }, PassFrame { image: &b8, errors: b.errors }, alpha)?;
        Ok(Image16::from_image(&out))
    }

    /// Peak working memory of one [`blend`](Self::blend) of `width` x `height` frames, in bytes,
    /// beyond its inputs and output. The renderer blends in its writer threads and reserves this
    /// much for each of them when it plans memory
    /// ([`RenderOptions::resources`](crate::RenderOptions::resources)). The default is 0 (a blend
    /// that works pixel by pixel).
    fn working_bytes(&self, width: usize, height: usize) -> usize {
        let _ = (width, height);
        0
    }
}

/// The built-in blends: a [`BlendMode`] with its settings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BuiltinBlender {
    pub mode: BlendMode,
    pub settings: BlendSettings,
}

impl Blender for BuiltinBlender {
    fn needs_errors(&self) -> bool {
        matches!(self.mode, BlendMode::Adaptive | BlendMode::Layered)
    }

    fn working_bytes(&self, width: usize, height: usize) -> usize {
        match self.mode {
            BlendMode::Linear => 0,
            BlendMode::Adaptive => adaptive_working_bytes(width * height),
            BlendMode::Layered => layered_working_bytes(width * height),
        }
    }

    fn blend(&self, a: PassFrame<'_>, b: PassFrame<'_>, alpha: f32) -> Result<Image, VideoError> {
        if a.image.size() != b.image.size() {
            return invalid("the passes of a frame differ in size");
        }
        Ok(blend_with(self.mode, &self.settings, a, b, alpha))
    }

    fn blend16(&self, a: PassFrame<'_, Image16>, b: PassFrame<'_, Image16>, alpha: f32) -> Result<Image16, VideoError> {
        if a.image.size() != b.image.size() {
            return invalid("the passes of a frame differ in size");
        }
        Ok(blend_with(self.mode, &self.settings, a, b, alpha))
    }
}

/// A sample of a blended frame: 8 or 16 bits.
trait Sample: Copy + Send + Sync + 'static {
    /// Units per 8-bit level (1 or 257): the blends compute on the 8-bit scale.
    const SCALE: f32;
    fn to_f32(self) -> f32;
    /// Rounds and clamps a value in native units.
    fn from_f32(v: f32) -> Self;
}

impl Sample for u8 {
    const SCALE: f32 = 1.0;

    fn to_f32(self) -> f32 {
        f32::from(self)
    }

    fn from_f32(v: f32) -> Self {
        v.round().clamp(0.0, 255.0) as u8
    }
}

impl Sample for u16 {
    const SCALE: f32 = 257.0;

    fn to_f32(self) -> f32 {
        f32::from(self)
    }

    fn from_f32(v: f32) -> Self {
        v.round().clamp(0.0, 65535.0) as u16
    }
}

/// The frame types the built-in blends work on ([`Image`] and [`Image16`]).
trait Frame: Clone + Send + Sync {
    type S: Sample;
    fn size(&self) -> (usize, usize);
    fn channels(&self) -> usize;
    fn samples(&self) -> &[Self::S];
    fn from_samples(width: usize, height: usize, channels: usize, data: Vec<Self::S>) -> Self;
}

impl Frame for Image {
    type S = u8;

    fn size(&self) -> (usize, usize) {
        Image::size(self)
    }

    fn channels(&self) -> usize {
        Image::channels(self)
    }

    fn samples(&self) -> &[u8] {
        self.data()
    }

    fn from_samples(width: usize, height: usize, channels: usize, data: Vec<u8>) -> Self {
        Image::from_raw(width, height, channels, data)
    }
}

impl Frame for Image16 {
    type S = u16;

    fn size(&self) -> (usize, usize) {
        Image16::size(self)
    }

    fn channels(&self) -> usize {
        Image16::channels(self)
    }

    fn samples(&self) -> &[u16] {
        self.data()
    }

    fn from_samples(width: usize, height: usize, channels: usize, data: Vec<u16>) -> Self {
        Image16::from_raw(width, height, channels, data)
    }
}

/// A 1- to 4-channel frame (the engine's layouts) expanded to RGBA.
fn to_rgba<F: Frame>(image: &F) -> F {
    let n = image.channels();
    if n == 4 {
        return image.clone();
    }
    let opaque = F::S::from_f32(255.0 * F::S::SCALE);
    let mut data = Vec::with_capacity(image.samples().len() / n * 4);
    for px in image.samples().chunks_exact(n) {
        data.extend_from_slice(&match *px {
            [g] => [g, g, g, opaque],
            [g, a] => [g, g, g, a],
            [r, g, b] => [r, g, b, opaque],
            _ => panic!("to_rgba expects 1 to 4 channels, got {n}"),
        });
    }
    let (w, h) = image.size();
    F::from_samples(w, h, 4, data)
}

/// Blends pass `a` and pass `b` of one frame with weight `alpha` on `b` (rounded to the nearest
/// level), with the default [`BlendSettings`] and without patch errors (the adaptive blend then
/// relies on its temporal prior alone). Images with different channel counts are blended as
/// RGBA.
///
/// # Panics
/// If the sizes differ.
pub fn blend(mode: BlendMode, a: &Image, b: &Image, alpha: f32) -> Image {
    assert_eq!(a.size(), b.size(), "passes differ in size");
    let pass = |image| PassFrame { image, errors: None };
    blend_with(mode, &BlendSettings::default(), pass(a), pass(b), alpha)
}

fn blend_with<F: Frame>(
    mode: BlendMode,
    settings: &BlendSettings,
    a: PassFrame<'_, F>,
    b: PassFrame<'_, F>,
    alpha: f32,
) -> F {
    if a.image.channels() != b.image.channels() {
        let (ra, rb) = (to_rgba(a.image), to_rgba(b.image));
        return blend_with(mode, settings, PassFrame { image: &ra, ..a }, PassFrame { image: &rb, ..b }, alpha);
    }
    let alpha = alpha.clamp(0.0, 1.0);
    match mode {
        BlendMode::Linear => linear(a.image, b.image, alpha),
        BlendMode::Adaptive => adaptive(settings, a, b, alpha),
        BlendMode::Layered => layered(settings, a, b, alpha),
    }
}

fn linear<F: Frame>(a: &F, b: &F, alpha: f32) -> F {
    let mut data = vec![F::S::from_f32(0.0); a.samples().len()];
    const CHUNK: usize = 1 << 14;
    data.par_chunks_mut(CHUNK).zip(a.samples().par_chunks(CHUNK).zip(b.samples().par_chunks(CHUNK))).for_each(
        |(out, (a, b))| {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                let (x, y) = (x.to_f32(), y.to_f32());
                *o = F::S::from_f32(x + (y - x) * alpha);
            }
        },
    );
    let (w, h) = a.size();
    F::from_samples(w, h, a.channels(), data)
}

/// Soft selection weight of pass A per pixel (see the module documentation).
fn selection_weights<F: Frame>(
    settings: &BlendSettings,
    a: PassFrame<'_, F>,
    b: PassFrame<'_, F>,
    alpha: f32,
) -> Vec<f32> {
    let (w, h) = a.image.size();
    let prior = settings.temporal_prior * ((1.0 - alpha) / alpha).ln();
    let (Some(ea), Some(eb)) = (a.errors, b.errors) else {
        return vec![sigmoid(prior); w * h];
    };
    let sigma = settings.error_smoothing * w.max(h) as f32;
    let (ea, eb) = rayon::join(|| gaussian_blur(ea, (w, h), sigma), || gaussian_blur(eb, (w, h), sigma));
    // The error term fades out towards the keyframes, so that next to a keyframe its own pass
    // wins as surely as it does in the linear crossfade (no jump between the keyframe's frame and
    // the next one), while the middle of the interval is decided by the errors.
    let strength = 4.0 * alpha * (1.0 - alpha) / settings.softness;
    ea.par_iter().zip(&eb).map(|(ea, eb)| sigmoid((eb - ea) * strength + prior)).collect()
}

fn sigmoid(z: f32) -> f32 {
    // Saturates cleanly at the keyframes, where the prior is infinite.
    1.0 / (1.0 + (-z.clamp(-60.0, 60.0)).exp())
}

/// Index of the alpha channel for the engine's channel layouts (grey + alpha, RGBA).
fn alpha_channel(channels: usize) -> Option<usize> {
    matches!(channels, 2 | 4).then_some(channels - 1)
}

/// Frames up to this many pixels solve their colour channels concurrently (see [`adaptive`]).
const CONCURRENT_SOLVE_PIXELS: usize = 1 << 20;

/// Peak working memory of the adaptive blend for frames of `pixels` pixels, in steady state
/// (a writer blends hundreds of frames). Measured peak resident memory beyond the inputs (glibc):
/// when the channels are solved concurrently (960x540, 1280x720), a single blend needs about
/// 110-240 bytes per pixel, but after some 30 blends the heap retains 385-395 (RGB) or 385-415
/// (RGBA) bytes per pixel (the allocator keeps the freed buffers of several threads' arenas);
/// solved one at a time (1920x1080, 3840x2160) it stays at 70-77 (RGB) or about 94 (RGBA).
/// Rounded up.
fn adaptive_working_bytes(pixels: usize) -> usize {
    pixels * if pixels <= CONCURRENT_SOLVE_PIXELS { 420 } else { 96 }
}

fn adaptive<F: Frame>(settings: &BlendSettings, a: PassFrame<'_, F>, b: PassFrame<'_, F>, alpha: f32) -> F {
    // The keyframe ends of the interval are the passes themselves.
    if alpha <= 0.0 {
        return a.image.clone();
    }
    if alpha >= 1.0 {
        return b.image.clone();
    }
    let (w, h) = a.image.size();
    let nc = a.image.channels();
    let n = w * h;
    let wa = selection_weights(settings, a, b, alpha);
    let alpha_ch = alpha_channel(nc);
    // Computed on the 8-bit scale whatever the sample type.
    let channel = |img: &F, c: usize| -> Vec<f32> {
        img.samples().par_chunks(nc).map(|px| px[c].to_f32() / F::S::SCALE).collect()
    };
    // Weights of the forward differences: the mean of their two pixels.
    let wx: Vec<f32> =
        (0..n).into_par_iter().map(|i| if i % w + 1 < w { 0.5 * (wa[i] + wa[i + 1]) } else { wa[i] }).collect();
    let wy: Vec<f32> =
        (0..n).into_par_iter().map(|i| if i + w < n { 0.5 * (wa[i] + wa[i + w]) } else { wa[i] }).collect();
    let matte = alpha_ch.map(|c| Matte::new(channel(a.image, c), channel(b.image, c), &wa, w));

    let mut out = vec![F::S::from_f32(0.0); a.image.samples().len()];
    let solve = |c: usize| -> Vec<f32> {
        if let Some(m) = matte.as_ref().filter(|_| Some(c) == alpha_ch) {
            return m.out.clone();
        }
        // Premultiplied colour, so that transparent pixels do not bleed into the gradients.
        let premultiply = |mut v: Vec<f32>, al: Option<&[f32]>| {
            if let Some(al) = al {
                v.iter_mut().zip(al).for_each(|(v, a)| *v *= a / 255.0);
            }
            v
        };
        let ca = premultiply(channel(a.image, c), matte.as_ref().map(|m| m.a.as_slice()));
        let cb = premultiply(channel(b.image, c), matte.as_ref().map(|m| m.b.as_slice()));
        integrate(settings, &ca, &cb, matte.as_ref(), (&wx, &wy), (w, h), alpha)
    };
    // The channels are independent. Up to about a megapixel they are solved at the same time,
    // which keeps the threads busy on the small coarse levels of each solve (about a third faster
    // at 960x540); larger frames are solved one channel at a time, which caps the working memory
    // (see `adaptive_working_bytes`).
    let solved: Vec<Vec<f32>> = if n <= CONCURRENT_SOLVE_PIXELS {
        (0..nc).into_par_iter().map(solve).collect()
    } else {
        (0..nc).map(solve).collect()
    };
    out.par_chunks_mut(nc).enumerate().for_each(|(i, px)| {
        for (c, o) in px.iter_mut().enumerate() {
            let v = solved[c][i];
            let v = match (matte.as_ref().map(|m| &m.out), Some(c) != alpha_ch) {
                // Back from premultiplied, clamped to the blended alpha.
                (Some(al), true) if al[i] > 0.0 => v.clamp(0.0, al[i]) * 255.0 / al[i],
                (Some(_), true) => 0.0,
                _ => v,
            };
            *o = F::S::from_f32(v * F::S::SCALE);
        }
    });
    F::from_samples(w, h, nc, out)
}

/// Peak working memory of the layered blend per pixel. It keeps a few f32 buffers per channel
/// (the premultiplied passes, their bases and the mixed detail) but no multigrid hierarchy, so it
/// is lighter than the adaptive blend. Same concurrency split as [`adaptive`]; measured generously.
fn layered_working_bytes(pixels: usize) -> usize {
    pixels * if pixels <= CONCURRENT_SOLVE_PIXELS { 200 } else { 56 }
}

/// The [layered](BlendMode::Layered) blend: split each pass into a base (shading, Gaussian of
/// `shading_scale`) and a detail residual, crossfade the base and error-select the detail. Mirrors
/// [`adaptive`]'s channel and alpha handling but combines in image space (no gradient-domain solve).
fn layered<F: Frame>(settings: &BlendSettings, a: PassFrame<'_, F>, b: PassFrame<'_, F>, alpha: f32) -> F {
    if alpha <= 0.0 {
        return a.image.clone();
    }
    if alpha >= 1.0 {
        return b.image.clone();
    }
    let (w, h) = a.image.size();
    let nc = a.image.channels();
    let n = w * h;
    let wa = selection_weights(settings, a, b, alpha);
    let alpha_ch = alpha_channel(nc);
    let sigma = settings.shading_scale * w.max(h) as f32;
    let channel = |img: &F, c: usize| -> Vec<f32> {
        img.samples().par_chunks(nc).map(|px| px[c].to_f32() / F::S::SCALE).collect()
    };
    // The output alpha follows the selection weights (as the adaptive matte does).
    let (aa, ab, out_alpha) = match alpha_ch {
        Some(c) => {
            let (aa, ab) = (channel(a.image, c), channel(b.image, c));
            let out: Vec<f32> = aa.par_iter().zip(&ab).zip(&wa).map(|((x, y), w)| w * x + (1.0 - w) * y).collect();
            (Some(aa), Some(ab), Some(out))
        }
        None => (None, None, None),
    };
    let solve = |c: usize| -> Vec<f32> {
        if Some(c) == alpha_ch {
            return out_alpha.clone().unwrap();
        }
        // Premultiplied colour, so transparent pixels do not bleed into the blurred base.
        let premul = |v: Vec<f32>, al: &Option<Vec<f32>>| -> Vec<f32> {
            match al {
                Some(al) => v.iter().zip(al).map(|(v, a)| v * a / 255.0).collect(),
                None => v,
            }
        };
        let pa = premul(channel(a.image, c), &aa);
        let pb = premul(channel(b.image, c), &ab);
        let base_a = if sigma > 0.0 { gaussian_blur(&pa, (w, h), sigma) } else { pa.clone() };
        let base_b = if sigma > 0.0 { gaussian_blur(&pb, (w, h), sigma) } else { pb.clone() };
        let res_a: Vec<f32> = pa.par_iter().zip(&base_a).map(|(v, bse)| v - bse).collect();
        let res_b: Vec<f32> = pb.par_iter().zip(&base_b).map(|(v, bse)| v - bse).collect();
        // Base crossfaded (scalar alpha), detail error-selected (variance-preserving if `detail`).
        let det = mix_gradients(&res_a, &res_b, &wa, (w, h), settings.detail);
        (0..n).into_par_iter().map(|i| (1.0 - alpha) * base_a[i] + alpha * base_b[i] + det[i]).collect()
    };
    let solved: Vec<Vec<f32>> = if n <= CONCURRENT_SOLVE_PIXELS {
        (0..nc).into_par_iter().map(solve).collect()
    } else {
        (0..nc).map(solve).collect()
    };
    let mut out = vec![F::S::from_f32(0.0); a.image.samples().len()];
    out.par_chunks_mut(nc).enumerate().for_each(|(i, px)| {
        for (c, o) in px.iter_mut().enumerate() {
            let v = solved[c][i];
            let v = match (&out_alpha, Some(c) != alpha_ch) {
                // Back from premultiplied, clamped to the blended alpha.
                (Some(al), true) if al[i] > 0.0 => v.clamp(0.0, al[i]) * 255.0 / al[i],
                (Some(_), true) => 0.0,
                _ => v,
            };
            *o = F::S::from_f32(v * F::S::SCALE);
        }
    });
    F::from_samples(w, h, nc, out)
}

/// Variance-preserving mix of two gradient components with weights `wt` on `ga`: the weighted
/// mix divided by `(wt^2 + (1 - wt)^2 + 2 wt (1 - wt) rho)^(detail / 2)`, with `rho` the local
/// correlation of the two fields (clamped to 0..1, Gaussian window of 3 pixels). For
/// uncorrelated fields and `detail` = 1 the mix keeps the passes' energy; identical fields are
/// never amplified.
fn mix_gradients(ga: &[f32], gb: &[f32], wt: &[f32], size: (usize, usize), detail: f32) -> Vec<f32> {
    let mixed = ga.par_iter().zip(gb).zip(wt).map(|((a, b), w)| w * a + (1.0 - w) * b);
    if detail <= 0.0 {
        return mixed.collect();
    }
    const WINDOW: f32 = 3.0;
    let prod = |x: &[f32], y: &[f32]| -> Vec<f32> { x.par_iter().zip(y).map(|(x, y)| x * y).collect() };
    let ab = gaussian_blur(&prod(ga, gb), size, WINDOW);
    let aa = gaussian_blur(&prod(ga, ga), size, WINDOW);
    let bb = gaussian_blur(&prod(gb, gb), size, WINDOW);
    mixed
        .zip(&ab)
        .zip(aa.par_iter().zip(&bb))
        .zip(wt)
        .map(|(((m, ab), (aa, bb)), w)| {
            let rho = (ab / (aa * bb).max(1e-6).sqrt()).clamp(0.0, 1.0);
            let energy = w * w + (1.0 - w) * (1.0 - w) + 2.0 * w * (1.0 - w) * rho;
            m * energy.powf(-0.5 * detail)
        })
        .collect()
}

/// The alpha channels of a blend with alpha, and what the colour channels need from them.
///
/// The output alpha follows the selection weights `w`: `out = w aA + (1 - w) aB`. Colour is
/// integrated premultiplied, and the premultiplied colour of the output should be the blended
/// straight colour times `out`. Two things make the colour channels agree with that alpha:
///
/// - the data term is the linear crossfade of the straight colour, premultiplied by `out` (the
///   crossfade of the premultiplied colour would darken it wherever the passes' alphas differ);
/// - the gradient field gets the part of `grad(out)` that the weighted mix of the passes'
///   gradients misses, `(aA - aB) grad w`, times the straight colour (the product rule; it
///   vanishes where both passes have the same alpha).
struct Matte {
    /// Alpha of pass A, pass B and the output (0..255).
    a: Vec<f32>,
    b: Vec<f32>,
    out: Vec<f32>,
    /// `(aA - aB) / 255 * D w` on the forward differences along x and y (midpoint means).
    dx: Vec<f32>,
    dy: Vec<f32>,
}

impl Matte {
    fn new(a: Vec<f32>, b: Vec<f32>, wa: &[f32], w: usize) -> Self {
        let n = a.len();
        let out = a.par_iter().zip(&b).zip(wa).map(|((x, y), w)| w * x + (1.0 - w) * y).collect();
        let term = |i: usize, j: usize| (a[i] + a[j] - b[i] - b[j]) / 510.0 * (wa[j] - wa[i]);
        let dx = (0..n).into_par_iter().map(|i| if i % w + 1 < w { term(i, i + 1) } else { 0.0 }).collect();
        let dy = (0..n).into_par_iter().map(|i| if i + w < n { term(i, i + w) } else { 0.0 }).collect();
        Self { a, b, out, dx, dy }
    }
}

/// Integrates the mixed gradients of one colour channel (`a`, `b`: the passes' values,
/// premultiplied when there is a [`Matte`]): argmin `lambda |u - f|^2 + |grad u - g|^2` with `f`
/// the linear crossfade (of the straight colour at the output alpha, with a matte).
fn integrate(
    settings: &BlendSettings,
    a: &[f32],
    b: &[f32],
    matte: Option<&Matte>,
    (wx, wy): (&[f32], &[f32]),
    (w, h): (usize, usize),
    alpha: f32,
) -> Vec<f32> {
    let lerp = |x: f32, y: f32| x + (y - x) * alpha;
    let n = w * h;
    // Forward differences, zero across the last column and row (Neumann).
    let gx = |v: &[f32]| -> Vec<f32> {
        (0..n).into_par_iter().map(|i| if i % w + 1 < w { v[i + 1] - v[i] } else { 0.0 }).collect()
    };
    let gy = |v: &[f32]| -> Vec<f32> {
        (0..n).into_par_iter().map(|i| if i + w < n { v[i + w] - v[i] } else { 0.0 }).collect()
    };
    let mut mx = mix_gradients(&gx(a), &gx(b), wx, (w, h), settings.detail);
    let mut my = mix_gradients(&gy(a), &gy(b), wy, (w, h), settings.detail);
    let f: Vec<f32> = match matte {
        None => a.par_iter().zip(b).map(|(&x, &y)| lerp(x, y)).collect(),
        Some(m) => {
            // The straight colour of the crossfade, alpha-weighted over a difference's two pixels
            // (transparent pixels have no colour).
            let straight = |i: usize, j: usize| {
                let al = lerp(m.a[i], m.b[i]) + lerp(m.a[j], m.b[j]);
                if al > 0.0 { 255.0 * (lerp(a[i], b[i]) + lerp(a[j], b[j])) / al } else { 0.0 }
            };
            mx.par_iter_mut().zip(&m.dx).enumerate().for_each(|(i, (g, d))| {
                if *d != 0.0 {
                    *g += straight(i, i + 1) * d;
                }
            });
            my.par_iter_mut().zip(&m.dy).enumerate().for_each(|(i, (g, d))| {
                if *d != 0.0 {
                    *g += straight(i, i + w) * d;
                }
            });
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let al = lerp(m.a[i], m.b[i]);
                    // Where both passes are transparent, so is the output.
                    if al > 0.0 { lerp(a[i], b[i]) * m.out[i] / al } else { 0.0 }
                })
                .collect()
        }
    };
    let lambda = settings.data_weight;
    // rhs = lambda f - div g, with div the negative adjoint of the forward differences.
    let rhs: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let mut div = mx[i] + my[i];
            if x > 0 {
                div -= mx[i - 1];
            }
            if y > 0 {
                div -= my[i - w];
            }
            lambda * f[i] - div
        })
        .collect();
    let mut u = f;
    // A residual of 1e-4 of the right-hand side is far below one grey level.
    solve_screened(&mut u, &rhs, (w, h), lambda, 1e-4, 20);
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(w: usize, h: usize, channels: usize, seed: u32) -> Image {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let data = (0..w * h * channels)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let base = ((i / channels) % w * 3 + (i / channels) / w * 2) as u32;
                (base.wrapping_add(s >> 26) % 200 + 20) as u8
            })
            .collect();
        Image::from_raw(w, h, channels, data)
    }

    fn hf_energy(img: &Image) -> f64 {
        let (w, h) = img.size();
        let nc = img.channels();
        let mut e = 0.0;
        for y in 0..h {
            for x in 0..w - 1 {
                let d = f64::from(img.pixel(x + 1, y)[0]) - f64::from(img.pixel(x, y)[0]);
                e += d * d;
            }
        }
        e / ((w - 1) * h * nc) as f64
    }

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
        assert_eq!("adaptive".parse::<BlendMode>().unwrap(), BlendMode::Adaptive);
        assert!("poisson".parse::<BlendMode>().is_err());
    }

    #[test]
    fn adaptive_returns_each_pass_at_its_keyframe() {
        let a = textured(40, 30, 3, 1);
        let b = textured(40, 30, 3, 2);
        let errors = vec![1.0f32; 40 * 30];
        let blender = BuiltinBlender { mode: BlendMode::Adaptive, settings: BlendSettings::default() };
        let pa = PassFrame { image: &a, errors: Some(&errors) };
        let pb = PassFrame { image: &b, errors: Some(&errors) };
        assert_eq!(blender.blend(pa, pb, 0.0).unwrap(), a);
        assert_eq!(blender.blend(pa, pb, 1.0).unwrap(), b);
        // Next to a keyframe the prior dominates: within a fraction of a level of the near pass.
        let near = blender.blend(pa, pb, 0.01).unwrap();
        let mad = near.data().iter().zip(a.data()).map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs()).sum::<f64>()
            / a.data().len() as f64;
        assert!(mad < 1.5, "mean difference {mad}");
    }

    #[test]
    fn adaptive_follows_the_lower_error() {
        // Pass A is better on the left half, pass B on the right half.
        let (w, h) = (64, 32);
        let a = textured(w, h, 3, 3);
        let b = textured(w, h, 3, 4);
        let ea: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 1.0 } else { 20.0 }).collect();
        let eb: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 20.0 } else { 1.0 }).collect();
        let settings = BlendSettings { error_smoothing: 0.0, ..BlendSettings::default() };
        let blender = BuiltinBlender { mode: BlendMode::Adaptive, settings };
        let out = blender
            .blend(PassFrame { image: &a, errors: Some(&ea) }, PassFrame { image: &b, errors: Some(&eb) }, 0.5)
            .unwrap();
        // Gradients follow the selected pass away from the seam.
        let grad = |img: &Image, x: usize, y: usize| f32::from(img.pixel(x + 1, y)[0]) - f32::from(img.pixel(x, y)[0]);
        let mut close = 0;
        let mut total = 0;
        for y in 2..h - 2 {
            for x in (2..w / 2 - 4).chain(w / 2 + 4..w - 3) {
                let want = if x < w / 2 { grad(&a, x, y) } else { grad(&b, x, y) };
                total += 1;
                close += usize::from((grad(&out, x, y) - want).abs() <= 2.0);
            }
        }
        assert!(close * 10 >= total * 9, "{close} of {total} gradients follow the better pass");
        // Detail is kept, unlike the crossfade.
        let lin = blend(BlendMode::Linear, &a, &b, 0.5);
        assert!(hf_energy(&out) > 1.5 * hf_energy(&lin), "{} vs {}", hf_energy(&out), hf_energy(&lin));
    }

    #[test]
    fn detail_restores_energy_of_uncorrelated_textures() {
        let (w, h) = (64, 48);
        let a = textured(w, h, 1, 5);
        let b = textured(w, h, 1, 6);
        let plain = blend(BlendMode::Adaptive, &a, &b, 0.5);
        let settings = BlendSettings { detail: 1.0, ..BlendSettings::default() };
        let pass = |image| PassFrame { image, errors: None };
        let full = blend_with(BlendMode::Adaptive, &settings, pass(&a), pass(&b), 0.5);
        let passes = 0.5 * (hf_energy(&a) + hf_energy(&b));
        assert!(hf_energy(&plain) < 0.7 * passes);
        assert!(hf_energy(&full) > 0.8 * passes, "{} vs {passes}", hf_energy(&full));
        // Identical passes are not amplified.
        let same = blend_with(BlendMode::Adaptive, &settings, pass(&a), pass(&a), 0.5);
        let diff = same.data().iter().zip(a.data()).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
        assert!(diff <= 1, "max difference {diff}");
    }

    #[test]
    fn adaptive_keeps_alpha_clean() {
        let (w, h) = (32, 24);
        let mut a = textured(w, h, 4, 7);
        let mut b = textured(w, h, 4, 8);
        // Left third transparent in both passes.
        for img in [&mut a, &mut b] {
            for y in 0..h {
                for x in 0..w {
                    img.pixel_mut(x, y)[3] = if x < w / 3 { 0 } else { 255 };
                }
            }
        }
        let out = blend(BlendMode::Adaptive, &a, &b, 0.4);
        assert_eq!(out.channels(), 4);
        for y in 0..h {
            for x in 0..w {
                let p = out.pixel(x, y);
                assert_eq!(p[3], if x < w / 3 { 0 } else { 255 }, "alpha at ({x}, {y})");
                if x < w / 3 {
                    assert_eq!(&p[..3], &[0, 0, 0]);
                }
            }
        }
    }

    #[test]
    fn adaptive_keeps_colour_where_alphas_differ() {
        // Same colour in both passes; on the right half A is opaque and B transparent, and the
        // errors favour A there. The output alpha follows the selection there, the colour must not.
        let (w, h) = (64, 32);
        let pass = |right_alpha: u8| {
            let data = (0..w * h).flat_map(|i| [200, 120, 60, if i % w >= w / 2 { right_alpha } else { 255 }]);
            Image::from_raw(w, h, 4, data.collect())
        };
        let (a, b) = (pass(255), pass(0));
        let ea = vec![1.0f32; w * h];
        let eb: Vec<f32> = (0..w * h).map(|i| if i % w >= w / 2 { 3.0 } else { 1.0 }).collect();
        for settings in [BlendSettings::default(), BlendSettings { error_smoothing: 0.0, ..BlendSettings::default() }] {
            let blender = BuiltinBlender { mode: BlendMode::Adaptive, settings };
            for alpha in [0.3, 0.5, 0.7] {
                let out = blender
                    .blend(
                        PassFrame { image: &a, errors: Some(&ea) },
                        PassFrame { image: &b, errors: Some(&eb) },
                        alpha,
                    )
                    .unwrap();
                for y in 0..h {
                    for x in 0..w {
                        let p = out.pixel(x, y);
                        if p[3] < 8 {
                            continue; // too transparent to carry a colour
                        }
                        for (c, want) in [200u8, 120, 60].into_iter().enumerate() {
                            assert!(p[c].abs_diff(want) <= 2, "alpha {alpha}, ({x}, {y}): {p:?}");
                        }
                    }
                }
                // The selection, not the crossfade, decides the alpha on the right.
                assert!(out.pixel(3 * w / 4, h / 2)[3] > 128, "alpha {alpha}: {:?}", out.pixel(3 * w / 4, h / 2));
            }
        }
    }

    #[test]
    fn layered_returns_each_pass_at_its_keyframe() {
        let a = textured(40, 30, 3, 1);
        let b = textured(40, 30, 3, 2);
        let errors = vec![1.0f32; 40 * 30];
        let blender = BuiltinBlender { mode: BlendMode::Layered, settings: BlendSettings::default() };
        let pa = PassFrame { image: &a, errors: Some(&errors) };
        let pb = PassFrame { image: &b, errors: Some(&errors) };
        assert_eq!(blender.blend(pa, pb, 0.0).unwrap(), a);
        assert_eq!(blender.blend(pa, pb, 1.0).unwrap(), b);
    }

    #[test]
    fn layered_crossfades_shading_even_when_errors_select_one_pass() {
        // Shared fine texture, plus a low-frequency shading step: A brighter on the left, B on the
        // right. Errors strongly favour A everywhere.
        let (w, h) = (80, 60);
        let base = textured(w, h, 3, 21);
        let (mut a, mut b) = (base.clone(), base.clone());
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let bump = |p: &mut u8| *p = (i32::from(*p) + 40).clamp(0, 255) as u8;
                    if x < w / 2 {
                        bump(&mut a.pixel_mut(x, y)[c]);
                    } else {
                        bump(&mut b.pixel_mut(x, y)[c]);
                    }
                }
            }
        }
        let ea = vec![1.0f32; w * h];
        let eb = vec![20.0f32; w * h];
        let settings = BlendSettings { shading_scale: 0.05, error_smoothing: 0.0, ..BlendSettings::default() };
        let blender = BuiltinBlender { mode: BlendMode::Layered, settings };
        let out = blender
            .blend(PassFrame { image: &a, errors: Some(&ea) }, PassFrame { image: &b, errors: Some(&eb) }, 0.5)
            .unwrap();
        let lin = blend(BlendMode::Linear, &a, &b, 0.5);
        // The shading (low-pass of the red channel) follows the crossfade, not the error-selected
        // pass A.
        let low = |img: &Image| -> Vec<f32> {
            let red: Vec<f32> = img.data().iter().step_by(3).map(|&v| f32::from(v)).collect();
            gaussian_blur(&red, (w, h), 0.05 * w as f32)
        };
        let (lo, ll, la) = (low(&out), low(&lin), low(&a));
        let dist = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(a, b)| (a - b).abs()).sum::<f32>() / x.len() as f32;
        assert!(dist(&lo, &ll) < 0.3 * dist(&lo, &la), "{} vs {}", dist(&lo, &ll), dist(&lo, &la));
    }

    #[test]
    fn layered_keeps_detail() {
        // Different textures, A better on the left, B on the right: the detail is selected and kept,
        // unlike the crossfade which halves it.
        let (w, h) = (64, 32);
        let a = textured(w, h, 3, 22);
        let b = textured(w, h, 3, 23);
        let ea: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 1.0 } else { 20.0 }).collect();
        let eb: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 20.0 } else { 1.0 }).collect();
        let settings = BlendSettings { shading_scale: 0.03, error_smoothing: 0.0, ..BlendSettings::default() };
        let blender = BuiltinBlender { mode: BlendMode::Layered, settings };
        let out = blender
            .blend(PassFrame { image: &a, errors: Some(&ea) }, PassFrame { image: &b, errors: Some(&eb) }, 0.5)
            .unwrap();
        let lin = blend(BlendMode::Linear, &a, &b, 0.5);
        assert!(hf_energy(&out) > 1.4 * hf_energy(&lin), "{} vs {}", hf_energy(&out), hf_energy(&lin));
    }

    #[test]
    fn layered_keeps_alpha_clean() {
        let (w, h) = (32, 24);
        let mut a = textured(w, h, 4, 7);
        let mut b = textured(w, h, 4, 8);
        for img in [&mut a, &mut b] {
            for y in 0..h {
                for x in 0..w {
                    img.pixel_mut(x, y)[3] = if x < w / 3 { 0 } else { 255 };
                }
            }
        }
        let errors = vec![1.0f32; w * h];
        let out = BuiltinBlender { mode: BlendMode::Layered, settings: BlendSettings::default() }
            .blend(PassFrame { image: &a, errors: Some(&errors) }, PassFrame { image: &b, errors: Some(&errors) }, 0.4)
            .unwrap();
        assert_eq!(out.channels(), 4);
        for y in 0..h {
            for x in 0..w {
                let p = out.pixel(x, y);
                assert_eq!(p[3], if x < w / 3 { 0 } else { 255 }, "alpha at ({x}, {y})");
                if x < w / 3 {
                    assert_eq!(&p[..3], &[0, 0, 0]);
                }
            }
        }
    }

    #[test]
    fn working_memory_is_declared() {
        let linear = BuiltinBlender { mode: BlendMode::Linear, settings: BlendSettings::default() };
        let adaptive = BuiltinBlender { mode: BlendMode::Adaptive, ..linear };
        assert_eq!(linear.working_bytes(3840, 2160), 0);
        // Small frames solve their channels concurrently, large ones one at a time: per pixel,
        // large frames need less.
        let (small, large) = (adaptive.working_bytes(960, 540), adaptive.working_bytes(3840, 2160));
        // Steady state after many blends (allocator retention included), as measured.
        assert!(small >= 960 * 540 * 415 && large >= 3840 * 2160 * 94, "{small} {large}");
        assert!(large / (3840 * 2160) < small / (960 * 540));
    }

    #[test]
    fn settings_are_checked() {
        BlendSettings::default().validate().unwrap();
        for bad in [
            BlendSettings { softness: 0.0, ..BlendSettings::default() },
            BlendSettings { data_weight: -1.0, ..BlendSettings::default() },
            BlendSettings { detail: 1.5, ..BlendSettings::default() },
            BlendSettings { error_smoothing: f32::NAN, ..BlendSettings::default() },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn sixteen_bit_blends_match_the_8_bit_ones() {
        let (w, h) = (48, 32);
        let a = textured(w, h, 4, 11);
        let b = textured(w, h, 4, 12);
        let ea: Vec<f32> = (0..w * h).map(|i| (i % 7) as f32).collect();
        let eb: Vec<f32> = (0..w * h).map(|i| (i % 5) as f32).collect();
        let (a16, b16) = (Image16::from_image(&a), Image16::from_image(&b));
        for mode in [BlendMode::Linear, BlendMode::Adaptive] {
            let blender = BuiltinBlender { mode, settings: BlendSettings::default() };
            let eight = blender
                .blend(PassFrame { image: &a, errors: Some(&ea) }, PassFrame { image: &b, errors: Some(&eb) }, 0.37)
                .unwrap();
            let pa = PassFrame { image: &a16, errors: Some(ea.as_slice()) };
            let pb = PassFrame { image: &b16, errors: Some(eb.as_slice()) };
            let sixteen = blender.blend16(pa, pb, 0.37).unwrap();
            // The same blend at more precision: within rounding of the 8-bit result.
            let max = sixteen.to_image().data().iter().zip(eight.data()).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
            assert!(max <= 1, "{mode:?}: {max}");
            assert!(sixteen.data().iter().any(|v| v % 257 != 0), "{mode:?}: no extra precision");
            assert_eq!(blender.blend16(pa, pb, 0.0).unwrap(), a16);
        }
        // The default implementation goes through the 8-bit blend.
        struct Eight;
        impl Blender for Eight {
            fn needs_errors(&self) -> bool {
                false
            }
            fn blend(&self, a: PassFrame<'_>, _: PassFrame<'_>, _: f32) -> Result<Image, VideoError> {
                Ok(a.image.clone())
            }
        }
        let pass = |image| PassFrame { image, errors: None };
        assert_eq!(Eight.blend16(pass(&a16), pass(&b16), 0.5).unwrap(), a16);
    }
}
