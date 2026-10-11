//! The synthesis driver: pyramid construction and the per-level search/vote schedule
//! (`ebsynthCpu` in the original).

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::error::{Error, invalid};
use crate::image::{Image, Nnf};
use crate::io::{eval_num_channels, select_channels};
use crate::mask::{ACTIVE, dilate_mask, eval_mask};
use crate::nnf::{nnf_init_random, nnf_upscale};
use crate::patch::{PatchSpace, pack};
use crate::patchmatch::{OmegaWorkspace, PatchMatchConfig, patchmatch};
use crate::pyramid::{max_pyramid_levels, pyramid_level_size, resample};
use crate::rng::GlibcRand;
use crate::vote::{vote_plain, vote_weighted};
use crate::{MAX_GUIDE_CHANNELS, MAX_STYLE_CHANNELS};

/// How patches are blended into the output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VoteMode {
    /// Every overlapping patch has weight 1 (`EBSYNTH_VOTEMODE_PLAIN`; used by the 2019 CLI).
    #[default]
    Plain,
    /// Weight `1 / (1 + error)`: sharper (`EBSYNTH_VOTEMODE_WEIGHTED`).
    Weighted,
}

/// How PatchMatch threads share the occupancy map used by the uniformity term.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OmegaSync {
    /// One map shared by all threads and updated immediately, as in the original. Results vary
    /// slightly between runs when more than one thread is used.
    #[default]
    Shared,
    /// Each band works on a private copy during an iteration; changes are merged between
    /// iterations. Fully deterministic for a given seed and band count (see
    /// [`Params::num_bands`]).
    PerTile,
}

/// A value given either once for all pyramid levels or per level (coarsest first).
#[derive(Clone, Debug, PartialEq)]
pub enum PerLevel<T> {
    Uniform(T),
    Levels(Vec<T>),
}

impl<T: Copy> PerLevel<T> {
    /// The value for `level` (0 = coarsest).
    pub fn get(&self, level: usize) -> T {
        match self {
            PerLevel::Uniform(v) => *v,
            PerLevel::Levels(v) => v[level],
        }
    }

    fn covers(&self, num_levels: usize) -> bool {
        match self {
            PerLevel::Uniform(_) => true,
            PerLevel::Levels(v) => v.len() >= num_levels,
        }
    }
}

/// Synthesis parameters. [`Params::default`] matches the defaults of the 2019 `ebsynth` CLI.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Weight of the uniformity term (`-uniformity`, default 3500; 500-15000 is reasonable).
    pub uniformity_weight: f32,
    /// Patch size, odd and at least 3 (`-patchsize`, default 5).
    pub patch_size: usize,
    /// Voting mode (the 2019 CLI always uses [`VoteMode::Plain`]).
    pub vote_mode: VoteMode,
    /// Number of pyramid levels (`-pyramidlevels`). `None` uses the maximum for which the
    /// coarsest level is at least `2 * patch_size + 1` pixels; larger values are clamped to it.
    pub num_pyramid_levels: Option<usize>,
    /// Search/vote iterations per level (`-searchvoteiters`, default 6).
    pub search_vote_iters: PerLevel<usize>,
    /// PatchMatch iterations per search/vote iteration (`-patchmatchiters`, default 4).
    pub patch_match_iters: PerLevel<usize>,
    /// Stop threshold per level (`-stopthreshold`, default 5). Only used when
    /// [`use_stop_threshold`](Self::use_stop_threshold) is set.
    pub stop_threshold: PerLevel<u32>,
    /// Skip PatchMatch for pixels whose neighbourhood changed by less than the stop threshold in
    /// the last vote. The 2019 CPU engine ignored the threshold and only its CUDA engine applied
    /// it, so this is off by default to reproduce the CPU engine.
    pub use_stop_threshold: bool,
    /// Finish with an extra pass using 3x3 patches and no uniformity term (`-extrapass3x3`).
    pub extra_pass_3x3: bool,
    /// Random seed, with the semantics of `srand()` (the original never seeds: 1).
    pub seed: u32,
    /// Number of threads. `None` uses the current rayon thread pool; otherwise a pool of this
    /// size is created on first use and reused by later calls.
    pub num_threads: Option<usize>,
    /// Number of horizontal bands the PatchMatch scan is split into. Results depend on it.
    /// `None` uses one band per thread, as the original does; set it to get the same
    /// [`OmegaSync::PerTile`] results on machines with different core counts.
    pub num_bands: Option<usize>,
    /// How threads share the uniformity occupancy map.
    pub omega_sync: OmegaSync,
    /// Accumulate patch errors sequentially in the original order so that, single-threaded,
    /// results match the 2019 build bit for bit. Slower; mostly useful for validation.
    pub bit_exact: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            uniformity_weight: 3500.0,
            patch_size: 5,
            vote_mode: VoteMode::Plain,
            num_pyramid_levels: None,
            search_vote_iters: PerLevel::Uniform(6),
            patch_match_iters: PerLevel::Uniform(4),
            stop_threshold: PerLevel::Uniform(5),
            use_stop_threshold: false,
            extra_pass_3x3: false,
            seed: 1,
            num_threads: None,
            num_bands: None,
            omega_sync: OmegaSync::Shared,
            bit_exact: false,
        }
    }
}

/// The images and weights to synthesise from (the data arguments of `ebsynthRun`).
///
/// Guide channels of all guide layers are concatenated into one multi-channel image per side.
#[derive(Clone, Debug)]
pub struct SynthesisInput {
    /// Source style (exemplar), 1 to 8 channels.
    pub source_style: Image,
    /// Source guides, 1 to 24 channels, same size as `source_style`.
    pub source_guide: Image,
    /// Target guides, same channels as `source_guide`; defines the output size.
    pub target_guide: Image,
    /// Optional per-pixel guide modulation, same shape as `target_guide`:
    /// `guideError *= modulation / 255`.
    pub target_modulation: Option<Image>,
    /// One weight per style channel.
    pub style_weights: Vec<f32>,
    /// One weight per guide channel.
    pub guide_weights: Vec<f32>,
}

/// A pair of guide images (RGBA) with an optional weight, as given by `-guide` / `-weight`.
#[derive(Clone, Debug)]
pub struct GuideLayer {
    pub source: Image,
    pub target: Image,
    /// Total weight of the layer; `None` means `1 / number_of_guides` (the CLI default).
    pub weight: Option<f32>,
}

impl GuideLayer {
    pub fn new(source: Image, target: Image) -> Self {
        Self { source, target, weight: None }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }
}

impl SynthesisInput {
    /// Builds the input from RGBA layers exactly like the 2019 CLI: each image keeps only its
    /// informative channels (see [`eval_num_channels`]), a guide pair uses the larger channel
    /// count of its two images, and each layer's weight is split evenly over its channels.
    /// The style weight defaults to 1.
    pub fn from_rgba_layers(style: &Image, style_weight: Option<f32>, guides: &[GuideLayer]) -> Result<Self, Error> {
        if style.channels() != 4 || guides.iter().any(|g| g.source.channels() != 4 || g.target.channels() != 4) {
            return invalid("from_rgba_layers expects RGBA images");
        }
        if guides.is_empty() {
            return invalid("at least one guide is required");
        }
        let ns = eval_num_channels(style);
        let target_size = guides[0].target.size();
        let mut counts = Vec::with_capacity(guides.len());
        for (i, g) in guides.iter().enumerate() {
            if g.source.size() != style.size() {
                return invalid(format!("source guide {i} doesn't match the resolution of the style"));
            }
            if g.target.size() != target_size {
                return invalid(format!("target guide {i} doesn't match the resolution of target guide 0"));
            }
            counts.push(eval_num_channels(&g.source).max(eval_num_channels(&g.target)));
        }
        let ng: usize = counts.iter().sum();
        if ns > MAX_STYLE_CHANNELS {
            return invalid(format!("too many style channels ({ns}), maximum number is {MAX_STYLE_CHANNELS}"));
        }
        if ng > MAX_GUIDE_CHANNELS {
            return invalid(format!("too many guide channels ({ng}), maximum number is {MAX_GUIDE_CHANNELS}"));
        }

        let concat = |side: fn(&GuideLayer) -> &Image, (w, h): (usize, usize)| {
            let parts: Vec<Image> = guides.iter().zip(&counts).map(|(g, &n)| select_channels(side(g), n)).collect();
            let mut data = Vec::with_capacity(w * h * ng);
            for i in 0..w * h {
                for (p, &n) in parts.iter().zip(&counts) {
                    data.extend_from_slice(&p.data()[i * n..(i + 1) * n]);
                }
            }
            Image::from_raw(w, h, ng, data)
        };
        let source_guide = concat(|g| &g.source, style.size());
        let target_guide = concat(|g| &g.target, target_size);

        let sw = style_weight.unwrap_or(1.0);
        let style_weights = vec![sw / ns as f32; ns];
        let mut guide_weights = Vec::with_capacity(ng);
        for (g, &n) in guides.iter().zip(&counts) {
            let w = g.weight.unwrap_or(1.0 / guides.len() as f32);
            guide_weights.extend(std::iter::repeat_n(w / n as f32, n));
        }
        Ok(Self {
            source_style: select_channels(style, ns),
            source_guide,
            target_guide,
            target_modulation: None,
            style_weights,
            guide_weights,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        let ns = self.source_style.channels();
        let ng = self.source_guide.channels();
        if !(1..=MAX_STYLE_CHANNELS).contains(&ns) {
            return invalid(format!("number of style channels must be 1..={MAX_STYLE_CHANNELS}, got {ns}"));
        }
        if !(1..=MAX_GUIDE_CHANNELS).contains(&ng) {
            return invalid(format!("number of guide channels must be 1..={MAX_GUIDE_CHANNELS}, got {ng}"));
        }
        if self.source_guide.size() != self.source_style.size() {
            return invalid("source guide and source style differ in size");
        }
        if self.target_guide.channels() != ng {
            return invalid("source and target guides differ in channel count");
        }
        if let Some(m) = &self.target_modulation
            && (m.size() != self.target_guide.size() || m.channels() != ng) {
                return invalid("target modulation must have the target guide's size and channels");
            }
        if self.style_weights.len() != ns || self.guide_weights.len() != ng {
            return invalid("need one weight per style channel and one per guide channel");
        }
        if self.style_weights.iter().chain(&self.guide_weights).any(|w| !w.is_finite() || *w < 0.0) {
            return invalid("weights must be finite and non-negative");
        }
        Ok(())
    }
}

/// Result of [`synthesize`].
#[derive(Clone, Debug)]
pub struct SynthesisOutput {
    /// The synthesised target style, with the source style's channel count.
    pub image: Image,
    /// The final nearest-neighbour field (source patch centre for every target pixel).
    pub nnf: Nnf,
}

/// Per-level images.
struct Level<'a> {
    source_size: (usize, usize),
    target_size: (usize, usize),
    source_style: Cow<'a, Image>,
    target_modulation: Option<Cow<'a, Image>>,
    /// Packed source style + guide.
    source_packed: Vec<u8>,
    /// Packed target style + guide (the style part follows the current target style).
    target_packed: Vec<u8>,
    /// Channels per packed pixel.
    nc: usize,
}

impl<'a> Level<'a> {
    fn build(input: &'a SynthesisInput, num_levels: usize, level: usize) -> Self {
        let source_size = pyramid_level_size(input.source_style.size(), num_levels, level);
        let target_size = pyramid_level_size(input.target_guide.size(), num_levels, level);
        let finest = level == num_levels - 1;
        let at = |img: &'a Image, (w, h): (usize, usize)| -> Cow<'a, Image> {
            if finest { Cow::Borrowed(img) } else { Cow::Owned(resample(img, w, h)) }
        };
        let source_style = at(&input.source_style, source_size);
        let source_guide = at(&input.source_guide, source_size);
        let target_guide = at(&input.target_guide, target_size);
        let target_modulation = input.target_modulation.as_ref().map(|m| at(m, target_size));
        let (ns, ng) = (source_style.channels(), source_guide.channels());
        let source_packed = pack(source_style.data(), ns, source_guide.data(), ng, source_size.0 * source_size.1);
        let zeros = vec![0u8; target_size.0 * target_size.1 * ns];
        let target_packed = pack(&zeros, ns, target_guide.data(), ng, target_size.0 * target_size.1);
        Self { source_size, target_size, source_style, target_modulation, source_packed, target_packed, nc: ns + ng }
    }

    /// Copies `style` into the style slots of the packed target.
    fn set_target_style(&mut self, style: &Image) {
        let ns = style.channels();
        let nc = self.nc;
        let n = self.target_size.0 * self.target_size.1;
        self.target_packed[..n * nc].par_chunks_mut(nc).zip(style.data().par_chunks(ns)).for_each(|(d, s)| {
            d[..ns].copy_from_slice(s);
        });
    }
}

/// Runs example-based synthesis (`ebsynthRun` with the CPU backend).
///
/// Builds a pyramid by resampling the inputs, initialises a random nearest-neighbour field at the
/// coarsest level and then, at every level, alternates PatchMatch searches and votes; the field is
/// upscaled between levels. Returns the synthesised image and the final field.
pub fn synthesize(input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, Error> {
    match params.num_threads {
        Some(n) if n.max(1) != rayon::current_num_threads() => {
            thread_pool(n.max(1))?.install(|| synthesize_here(input, params))
        }
        _ => synthesize_here(input, params),
    }
}

/// A rayon pool with `n` threads. The most recently used pool is kept, so repeated calls with
/// the same thread count (e.g. one synthesis per video frame) do not spawn threads each time.
fn thread_pool(n: usize) -> Result<Arc<rayon::ThreadPool>, Error> {
    static POOL: Mutex<Option<(usize, Arc<rayon::ThreadPool>)>> = Mutex::new(None);
    let mut cached = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((size, pool)) = cached.as_ref()
        && *size == n
    {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| Error::InvalidInput(format!("cannot create a thread pool: {e}")))?,
    );
    *cached = Some((n, Arc::clone(&pool)));
    Ok(pool)
}

/// [`synthesize`] on the current rayon thread pool.
fn synthesize_here(input: &SynthesisInput, params: &Params) -> Result<SynthesisOutput, Error> {
    input.validate()?;
    if params.patch_size < 3 || params.patch_size.is_multiple_of(2) {
        return invalid("patch size must be an odd number >= 3");
    }
    if !params.uniformity_weight.is_finite() {
        return invalid("uniformity weight must be finite");
    }
    let source_size = input.source_style.size();
    let target_size = input.target_guide.size();
    let max_levels = max_pyramid_levels(source_size, target_size, params.patch_size);
    if max_levels == 0 {
        return invalid(format!(
            "source and target must be at least {0}x{0} pixels for patch size {1}",
            2 * params.patch_size + 1,
            params.patch_size
        ));
    }
    let num_levels = match params.num_pyramid_levels {
        Some(0) => return invalid("number of pyramid levels must be at least 1"),
        Some(n) => n.min(max_levels),
        None => max_levels,
    };
    if !params.search_vote_iters.covers(num_levels) || !params.patch_match_iters.covers(num_levels) || !params.stop_threshold.covers(num_levels) {
        return invalid(format!("per-level settings must cover all {num_levels} pyramid levels"));
    }

    let bands = params.num_bands.unwrap_or_else(rayon::current_num_threads).max(1);
    let ns = input.source_style.channels();
    let weights: Vec<f32> = input.style_weights.iter().chain(&input.guide_weights).copied().collect();
    let mut rng = GlibcRand::new(params.seed);

    let mut patch_size = params.patch_size;
    let mut lambda = params.uniformity_weight;
    let mut in_extra_pass = false;

    let mut nnf = Nnf::new(0, 0);
    let mut errors: Vec<f32> = Vec::new();
    let mut style = Image::new(0, 0, ns);
    let mut style2 = Image::new(0, 0, ns);
    let mut omega_ws = OmegaWorkspace::default();
    let mut level_data: Option<Level<'_>> = None;

    let mut level = 0;
    loop {
        if !in_extra_pass {
            let lv = Level::build(input, num_levels, level);
            nnf = if level > 0 {
                nnf_upscale(&nnf, patch_size, lv.target_size, lv.source_size)
            } else {
                nnf_init_random(lv.target_size, lv.source_size, patch_size, &mut rng)
            };
            style = Image::new(lv.target_size.0, lv.target_size.1, ns);
            style2 = Image::new(lv.target_size.0, lv.target_size.1, ns);
            omega_ws = OmegaWorkspace::default();
            level_data = Some(lv);
        }
        let lv = level_data.as_mut().expect("level data");
        let (tw, th) = lv.target_size;

        vote_plain(&mut style, &lv.source_style, &nnf, patch_size);
        lv.set_target_style(&style);

        let mut mask = params.use_stop_threshold.then(|| vec![ACTIVE; tw * th]);
        let num_sv = params.search_vote_iters.get(level);
        for vote_iter in 0..num_sv {
            {
                let space = PatchSpace::new(
                    lv.target_size,
                    lv.source_size,
                    ns,
                    patch_size,
                    &lv.target_packed,
                    &lv.source_packed,
                    &weights,
                    lv.target_modulation.as_deref().map(|m| m.data()),
                );
                let cfg = PatchMatchConfig {
                    lambda,
                    num_iters: params.patch_match_iters.get(level),
                    num_bands: bands,
                    sync: params.omega_sync,
                    kernel: space.kernel(params.bit_exact),
                };
                patchmatch(&space, &mut nnf, &mut errors, &cfg, &mut rng, mask.as_deref(), &mut omega_ws);
            }
            match params.vote_mode {
                VoteMode::Plain => vote_plain(&mut style2, &lv.source_style, &nnf, patch_size),
                VoteMode::Weighted => vote_weighted(&mut style2, &lv.source_style, &nnf, &errors, patch_size),
            }
            std::mem::swap(&mut style, &mut style2);
            lv.set_target_style(&style);

            if let Some(m) = mask.as_mut()
                && vote_iter + 1 < num_sv {
                    eval_mask(&style, &style2, params.stop_threshold.get(level), m);
                    *m = dilate_mask(m, tw, th, patch_size);
                }
        }

        if level == num_levels - 1 {
            if params.extra_pass_3x3 && !in_extra_pass {
                // Repeat the finest level with 3x3 patches and no uniformity term, keeping the
                // current field and image.
                in_extra_pass = true;
                patch_size = 3;
                lambda = 0.0;
                continue;
            }
            return Ok(SynthesisOutput { image: style, nnf });
        }
        level += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth but non-repetitive colour image.
    fn test_image(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h, 3);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32, y as f32);
                let p = img.pixel_mut(x, y);
                p[0] = (128.0 + 100.0 * (fx * 0.11).sin() * (fy * 0.07).cos()) as u8;
                p[1] = (128.0 + 100.0 * (fx * 0.05 + fy * 0.09).sin()) as u8;
                p[2] = ((x * 3 + y * 5) % 256) as u8;
            }
        }
        img
    }

    fn psnr(a: &Image, b: &Image) -> f64 {
        let mse = a.data().iter().zip(b.data()).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.data().len() as f64;
        if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
    }

    fn identity_input(img: &Image) -> SynthesisInput {
        SynthesisInput {
            source_style: img.clone(),
            source_guide: img.clone(),
            target_guide: img.clone(),
            target_modulation: None,
            style_weights: vec![1.0 / 3.0; 3],
            guide_weights: vec![1.0 / 3.0; 3],
        }
    }

    #[test]
    fn identity_synthesis_reproduces_style() {
        let img = test_image(96, 80);
        let input = identity_input(&img);
        for sync in [OmegaSync::Shared, OmegaSync::PerTile] {
            let params = Params { omega_sync: sync, num_threads: Some(4), ..Params::default() };
            let out = synthesize(&input, &params).unwrap();
            assert_eq!(out.image.size(), (96, 80));
            let p = psnr(&out.image, &img);
            assert!(p > 35.0, "{sync:?}: PSNR {p:.2} dB");
        }
    }

    #[test]
    fn identity_synthesis_other_modes() {
        let img = test_image(64, 64);
        let mut input = identity_input(&img);
        let params = Params {
            vote_mode: VoteMode::Weighted,
            use_stop_threshold: true,
            extra_pass_3x3: true,
            ..Params::default()
        };
        let out = synthesize(&input, &params).unwrap();
        assert!(psnr(&out.image, &img) > 35.0);

        input.target_modulation = Some(Image::from_raw(64, 64, 3, vec![255; 64 * 64 * 3]));
        let out = synthesize(&input, &Params::default()).unwrap();
        assert!(psnr(&out.image, &img) > 35.0);
    }

    #[test]
    fn deterministic_with_per_tile_sync() {
        let img = test_image(70, 50);
        let mut input = identity_input(&img);
        // Make the target differ from the source so the result is non-trivial.
        input.target_guide = resample(&test_image(35, 25), 70, 50);
        let params = Params { omega_sync: OmegaSync::PerTile, num_threads: Some(3), seed: 7, ..Params::default() };
        let a = synthesize(&input, &params).unwrap();
        let b = synthesize(&input, &params).unwrap();
        assert_eq!(a.image, b.image);
        assert_eq!(a.nnf, b.nnf);
        let c = synthesize(&input, &Params { seed: 8, ..params }).unwrap();
        assert_ne!(a.nnf, c.nnf);
    }

    #[test]
    fn nnf_points_into_source() {
        let src = test_image(50, 40);
        let tgt = test_image(30, 60);
        let input = SynthesisInput {
            source_style: src.clone(),
            source_guide: src,
            target_guide: tgt,
            target_modulation: None,
            style_weights: vec![1.0; 3],
            guide_weights: vec![1.0; 3],
        };
        let out = synthesize(&input, &Params { patch_size: 3, ..Params::default() }).unwrap();
        assert_eq!(out.image.size(), (30, 60));
        assert_eq!((out.nnf.width(), out.nnf.height()), (30, 60));
        for &[x, y] in out.nnf.data() {
            assert!((1..49).contains(&x) && (1..39).contains(&y));
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let img = test_image(16, 16);
        let input = identity_input(&img);
        assert!(synthesize(&input, &Params { patch_size: 4, ..Params::default() }).is_err());
        assert!(synthesize(&input, &Params { patch_size: 9, ..Params::default() }).is_err()); // 16 < 19
        let mut bad = input.clone();
        bad.guide_weights.pop();
        assert!(synthesize(&bad, &Params::default()).is_err());
        let mut bad = input.clone();
        bad.style_weights[1] = -0.5;
        assert!(synthesize(&bad, &Params::default()).is_err());
    }

    #[test]
    fn band_count_not_thread_count_decides_per_tile_results() {
        let img = test_image(60, 44);
        let mut input = identity_input(&img);
        input.target_guide = resample(&test_image(30, 22), 60, 44);
        let base = Params { omega_sync: OmegaSync::PerTile, num_bands: Some(5), ..Params::default() };
        let a = synthesize(&input, &Params { num_threads: Some(2), ..base.clone() }).unwrap();
        let b = synthesize(&input, &Params { num_threads: Some(3), ..base.clone() }).unwrap();
        assert_eq!(a.nnf, b.nnf);
        assert_eq!(a.image, b.image);
    }

    #[test]
    fn layers_follow_cli_packing() {
        let w = 4;
        let rgba = |f: &dyn Fn(usize) -> [u8; 4]| Image::from_raw(w, 1, 4, (0..w).flat_map(f).collect());
        let style = rgba(&|i| [i as u8, 2, 3, 255]); // colour, opaque -> 3 channels
        let gray = rgba(&|i| [i as u8 * 9, i as u8 * 9, i as u8 * 9, 255]); // 1 channel
        let gray_alpha = rgba(&|i| [7, 7, 7, 100 + i as u8]); // 2 channels
        let input = SynthesisInput::from_rgba_layers(
            &style,
            Some(2.0),
            &[GuideLayer::new(gray.clone(), gray.clone()), GuideLayer::new(gray.clone(), gray_alpha).with_weight(3.0)],
        )
        .unwrap();
        assert_eq!(input.source_style.channels(), 3);
        assert_eq!(input.style_weights, vec![2.0 / 3.0; 3]);
        assert_eq!(input.source_guide.channels(), 3); // 1 + max(1, 2)
        assert_eq!(input.guide_weights, vec![0.5, 1.5, 1.5]);
        // Second guide uses R and A: source alpha is 255, target alpha varies.
        assert_eq!(input.source_guide.pixel(1, 0), &[9, 9, 255]);
        assert_eq!(input.target_guide.pixel(1, 0), &[9, 7, 101]);
    }
}
