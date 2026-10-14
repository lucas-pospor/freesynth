//! The synthesis driver: pyramid construction and the per-level search/vote schedule
//! (`ebsynthCpu` in the original).

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::error::{Error, invalid};
use crate::extras::{SynthesisExtras, warm_field};
use crate::image::{Image, Image16, Nnf};
use crate::io::{eval_num_channels, select_channels};
use crate::mask::{ACTIVE, dilate_mask, eval_mask};
use crate::nnf::{nnf_init_random, nnf_upscale};
use crate::patch::{PatchSpace, pack};
use crate::patchmatch::{OmegaWorkspace, PatchMatchConfig, nnf_errors, patchmatch};
use crate::pyramid::{pyramid_level_size, resample};
use crate::rng::GlibcRand;
use crate::schedule::{Engine, Preset, SearchMode};
use crate::vote::{VoteRounding, vote_plain, vote_plain16, vote_weighted, vote_weighted16};
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

    /// Whether a value is given for each of `num_levels` levels.
    pub(crate) fn covers(&self, num_levels: usize) -> bool {
        match self {
            PerLevel::Uniform(_) => true,
            PerLevel::Levels(v) => v.len() >= num_levels,
        }
    }
}

/// Synthesis parameters.
///
/// [`Params::default`] uses the [`Preset::Default`] preset; [`Params::compat`] reproduces the
/// defaults of the 2019 `ebsynth` CLI. The schedule settings that are `Option`s (iteration counts,
/// stop mask, voting) come from the preset unless they are set explicitly; see
/// [`Params::schedule`] for the resolved values.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Speed/quality preset that supplies every schedule setting left at `None`.
    pub preset: Preset,
    /// Weight of the uniformity term (`-uniformity`, default 3500; 500-15000 is reasonable).
    pub uniformity_weight: f32,
    /// Patch size, odd and at least 3 (`-patchsize`, default 5).
    pub patch_size: usize,
    /// Voting mode. `None`: weighted, except plain in [`Preset::Compat`] (the 2019 CLI always
    /// voted plain).
    pub vote_mode: Option<VoteMode>,
    /// How votes are converted to 8 bits. `None`: rounding, except truncation (as in 2019) in
    /// [`Preset::Compat`].
    pub vote_rounding: Option<VoteRounding>,
    /// Number of pyramid levels (`-pyramidlevels`). `None` uses the maximum for which the
    /// coarsest level is at least `2 * patch_size + 1` pixels; larger values are clamped to it.
    pub num_pyramid_levels: Option<usize>,
    /// Search/vote iterations per level (`-searchvoteiters`). `None`: 6, except 4 at the finest
    /// level in [`Preset::Fast`] on the GPU.
    pub search_vote_iters: Option<PerLevel<usize>>,
    /// PatchMatch iterations per search/vote iteration (`-patchmatchiters`). `None`: 4 in
    /// [`Preset::Compat`] and [`Preset::High`], 2 in the other presets (4 for 3x3 patches).
    pub patch_match_iters: Option<PerLevel<usize>>,
    /// PatchMatch variant of the GPU engine (`-search`; the CPU engine has one). `None`:
    /// [`SearchMode::Compat`] in [`Preset::Compat`], [`SearchMode::Improved`] in the other presets.
    pub search: Option<SearchMode>,
    /// Stop threshold per level (`-stopthreshold`, default 5). Only used with the stop mask.
    pub stop_threshold: PerLevel<u32>,
    /// Skip PatchMatch for pixels whose neighbourhood changed by less than the stop threshold in
    /// the last vote. `None`: only in [`Preset::Fast`]. (The 2019 CPU engine ignored the
    /// threshold; only its CUDA engine applied it.)
    pub use_stop_threshold: Option<bool>,
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
    /// Accumulate patch errors sequentially in the original order so that, single-threaded and
    /// with [`Preset::Compat`], results match the 2019 build bit for bit. Slower; mostly useful
    /// for validation. Only changes how the CPU engine sums errors: it does not change the
    /// schedule or the voting, so with any other preset it reproduces nothing.
    pub bit_exact: bool,
    /// Also return the final per-pixel patch errors ([`SynthesisOutput::errors`]). Costs one
    /// extra error evaluation at the finest level.
    pub return_errors: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            preset: Preset::Default,
            uniformity_weight: 3500.0,
            patch_size: 5,
            vote_mode: None,
            vote_rounding: None,
            num_pyramid_levels: None,
            search_vote_iters: None,
            patch_match_iters: None,
            search: None,
            stop_threshold: PerLevel::Uniform(5),
            use_stop_threshold: None,
            extra_pass_3x3: false,
            seed: 1,
            num_threads: None,
            num_bands: None,
            omega_sync: OmegaSync::Shared,
            bit_exact: false,
            return_errors: false,
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
    /// Optional per-pixel, per-channel guide weights, same shape as `target_guide`. At target
    /// pixel `p`, the squared difference of guide channel `k` is multiplied by
    /// `modulation[p][k] / 255` on top of `guide_weights[k]`: 255 keeps the channel's full
    /// weight, 0 switches it off at that pixel. The style channels are not affected. Use it to
    /// lower a guide's influence where the guide is unreliable (for example, where an advected
    /// position guide has lost track). [`GuideLayer::with_modulation`] builds it per layer.
    pub target_modulation: Option<Image>,
    /// One weight per style channel.
    pub style_weights: Vec<f32>,
    /// One weight per guide channel.
    pub guide_weights: Vec<f32>,
    /// The inclusive range of guide channels that carry a non-trivial modulation (the gated
    /// layers), as `(first, last)`. The channels outside it are 255 everywhere in
    /// [`target_modulation`], so modulating them is a no-op; backends may skip it. `None` means the
    /// range is unknown, so the whole guide is treated as modulated. [`from_rgba_layers`] fills it.
    pub modulated_range: Option<(u32, u32)>,
}

/// A pair of guide images (RGBA) with an optional weight, as given by `-guide` / `-weight`.
#[derive(Clone, Debug)]
pub struct GuideLayer {
    pub source: Image,
    pub target: Image,
    /// Total weight of the layer; `None` means `1 / number_of_guides` (the CLI default).
    pub weight: Option<f32>,
    /// Optional per-pixel weight map for this layer: one channel, the size of `target`. At each
    /// target pixel the layer's guide error is scaled by `value / 255` (see
    /// [`SynthesisInput::target_modulation`]). `None` means full weight everywhere.
    pub modulation: Option<Image>,
}

impl GuideLayer {
    pub fn new(source: Image, target: Image) -> Self {
        Self { source, target, weight: None, modulation: None }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Sets the layer's per-pixel weight map (one channel, the target's size).
    pub fn with_modulation(mut self, map: Image) -> Self {
        self.modulation = Some(map);
        self
    }
}

impl SynthesisInput {
    /// Builds the input from RGBA layers exactly like the 2019 CLI: each image keeps only its
    /// informative channels (see [`eval_num_channels`]), a guide pair uses the larger channel
    /// count of its two images, and each layer's weight is split evenly over its channels.
    /// The style weight defaults to 1. If any layer has a [modulation map](GuideLayer::modulation),
    /// the result gets a [`target_modulation`](Self::target_modulation) that repeats each map
    /// over its layer's channels (255 for layers without one).
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
            if let Some(m) = &g.modulation
                && (m.size() != target_size || m.channels() != 1)
            {
                return invalid(format!("modulation map of guide {i} must be one channel of the target's size"));
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

        let target_modulation = guides.iter().any(|g| g.modulation.is_some()).then(|| {
            let (w, h) = target_size;
            let mut data = Vec::with_capacity(w * h * ng);
            for i in 0..w * h {
                for (g, &n) in guides.iter().zip(&counts) {
                    let v = g.modulation.as_ref().map_or(255, |m| m.data()[i]);
                    data.extend(std::iter::repeat_n(v, n));
                }
            }
            Image::from_raw(w, h, ng, data)
        });
        // The channel span of the gated layers, so a backend modulates only those channels.
        let mut modulated_range: Option<(u32, u32)> = None;
        let mut offset = 0u32;
        for (g, &n) in guides.iter().zip(&counts) {
            if g.modulation.is_some() {
                let (lo, hi) = (offset, offset + n as u32 - 1);
                modulated_range = Some(modulated_range.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            }
            offset += n as u32;
        }

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
            target_modulation,
            style_weights,
            guide_weights,
            modulated_range,
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
            && (m.size() != self.target_guide.size() || m.channels() != ng)
        {
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
    /// With [`Params::return_errors`], the final patch error of every target pixel in scan-line
    /// order: the weighted SSD (style plus modulated guide channels, summed over the patch, no
    /// uniformity term) between the target patch of the output image and the source patch the
    /// final field points at. The patch size is 3 after [`Params::extra_pass_3x3`]. `None`
    /// when not requested. The map has the size of [`image`](Self::image).
    pub errors: Option<Vec<f32>>,
    /// With [`SynthesisExtras::style16`], the final field voted from the 16-bit style with the
    /// final vote's weights: the 16-bit counterpart of [`image`](Self::image). `None` otherwise.
    pub image16: Option<Image16>,
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

    /// The patch space of this level for the current target style.
    fn space<'s>(&'s self, ns: usize, patch_size: usize, weights: &'s [f32]) -> PatchSpace<'s> {
        PatchSpace::new(
            self.target_size,
            self.source_size,
            ns,
            patch_size,
            &self.target_packed,
            &self.source_packed,
            weights,
            self.target_modulation.as_deref().map(|m| m.data()),
        )
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
    synthesize_with(input, params, &SynthesisExtras::default())
}

/// [`synthesize`] with optional extra inputs: a warm start and a 16-bit style
/// ([`SynthesisExtras`]).
pub fn synthesize_with(
    input: &SynthesisInput,
    params: &Params,
    extras: &SynthesisExtras,
) -> Result<SynthesisOutput, Error> {
    match params.num_threads {
        Some(n) if n.max(1) != rayon::current_num_threads() => {
            thread_pool(n.max(1))?.install(|| synthesize_here(input, params, extras))
        }
        _ => synthesize_here(input, params, extras),
    }
}

impl SynthesisExtras {
    /// Checks the extras against the input they go with.
    pub fn validate(&self, input: &SynthesisInput) -> Result<(), Error> {
        let target = input.target_guide.size();
        if let Some(w) = &self.warm_start {
            if (w.nnf.width(), w.nnf.height()) != target {
                return invalid("the warm-start field must have the target's size");
            }
            if w.levels == 0 {
                return invalid("a warm start must run at least one pyramid level");
            }
        }
        if let Some(s) = &self.style16
            && (s.size() != input.source_style.size() || s.channels() != input.source_style.channels())
        {
            return invalid("the 16-bit style must have the source style's size and channels");
        }
        Ok(())
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
fn synthesize_here(
    input: &SynthesisInput,
    params: &Params,
    extras: &SynthesisExtras,
) -> Result<SynthesisOutput, Error> {
    let sched = params.schedule(input, Engine::Cpu)?;
    extras.validate(input)?;
    let num_levels = sched.num_levels;
    let first_level = extras.warm_start.as_ref().map_or(0, |w| num_levels.saturating_sub(w.levels));
    let rounding = sched.vote_rounding;

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

    // Whether the latest vote of the current level was weighted (with `errors`).
    let mut weighted_vote;
    let mut level = first_level;
    loop {
        if !in_extra_pass {
            let lv = Level::build(input, num_levels, level);
            nnf = match &extras.warm_start {
                Some(w) if level == first_level => {
                    warm_field(&w.nnf, lv.target_size, input.source_style.size(), lv.source_size, patch_size)
                }
                _ if level > first_level => nnf_upscale(&nnf, patch_size, lv.target_size, lv.source_size),
                _ => nnf_init_random(lv.target_size, lv.source_size, patch_size, &mut rng),
            };
            style = Image::new(lv.target_size.0, lv.target_size.1, ns);
            style2 = Image::new(lv.target_size.0, lv.target_size.1, ns);
            omega_ws = OmegaWorkspace::default();
            level_data = Some(lv);
        }
        let lv = level_data.as_mut().expect("level data");
        let (tw, th) = lv.target_size;

        vote_plain(&mut style, &lv.source_style, &nnf, patch_size, rounding);
        lv.set_target_style(&style);
        weighted_vote = false;

        let mut mask = sched.use_stop_threshold.then(|| vec![ACTIVE; tw * th]);
        let num_sv = sched.search_vote_iters[level];
        for vote_iter in 0..num_sv {
            {
                let space = lv.space(ns, patch_size, &weights);
                let cfg = PatchMatchConfig {
                    lambda,
                    num_iters: sched.patch_match_iters[level],
                    num_bands: bands,
                    sync: params.omega_sync,
                    kernel: space.kernel(params.bit_exact),
                };
                patchmatch(&space, &mut nnf, &mut errors, &cfg, &mut rng, mask.as_deref(), &mut omega_ws);
            }
            match sched.vote_mode {
                VoteMode::Plain => vote_plain(&mut style2, &lv.source_style, &nnf, patch_size, rounding),
                VoteMode::Weighted => vote_weighted(&mut style2, &lv.source_style, &nnf, &errors, patch_size, rounding),
            }
            weighted_vote = sched.vote_mode == VoteMode::Weighted;
            std::mem::swap(&mut style, &mut style2);
            lv.set_target_style(&style);

            if let Some(m) = mask.as_mut()
                && vote_iter + 1 < num_sv
            {
                eval_mask(&style, &style2, sched.stop_threshold[level], m);
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
            // The 16-bit result: the final field voted like the 8-bit one, from the 16-bit style.
            let image16 = extras.style16.as_ref().map(|s16| {
                let mut out = Image16::new(tw, th, ns);
                if weighted_vote {
                    vote_weighted16(&mut out, s16, &nnf, &errors, patch_size, VoteRounding::Nearest);
                } else {
                    vote_plain16(&mut out, s16, &nnf, patch_size, VoteRounding::Nearest);
                }
                out
            });
            let errors = params.return_errors.then(|| {
                // The target already holds the final image, so this is the error of the final
                // field against the output, with the kernel the search used.
                let space = lv.space(ns, patch_size, &weights);
                nnf_errors(&space, &nnf, &mut errors, space.kernel(params.bit_exact));
                errors
            });
            return Ok(SynthesisOutput { image: style, nnf, errors, image16 });
        }
        level += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extras::WarmStart;

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
        let mse = a.data().iter().zip(b.data()).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>()
            / a.data().len() as f64;
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
            modulated_range: None,
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
        for preset in Preset::ALL {
            let params = Params {
                vote_mode: Some(VoteMode::Weighted),
                use_stop_threshold: Some(true),
                extra_pass_3x3: true,
                ..Params::with_preset(preset)
            };
            let out = synthesize(&input, &params).unwrap();
            assert!(psnr(&out.image, &img) > 35.0, "{preset}");
            let out = synthesize(&input, &Params::with_preset(preset)).unwrap();
            assert!(psnr(&out.image, &img) > 35.0, "{preset}");
        }

        input.target_modulation = Some(Image::from_raw(64, 64, 3, vec![255; 64 * 64 * 3]));
        let out = synthesize(&input, &Params::default()).unwrap();
        assert!(psnr(&out.image, &img) > 35.0);
    }

    /// A target that differs from the source, so that the synthesis has real work to do.
    fn shifted_input() -> SynthesisInput {
        let mut input = identity_input(&test_image(72, 56));
        input.target_guide = resample(&test_image(36, 28), 72, 56);
        input
    }

    #[test]
    fn returned_errors_are_the_final_patch_errors() {
        let input = shifted_input();
        for extra_pass_3x3 in [false, true] {
            let params = Params {
                return_errors: true,
                extra_pass_3x3,
                omega_sync: OmegaSync::PerTile,
                num_threads: Some(2),
                ..Params::default()
            };
            let out = synthesize(&input, &params).unwrap();
            let e = crate::nnf_patch_errors(&input, &params, &out.image, &out.nnf).unwrap();
            let errors = out.errors.as_deref().expect("errors requested");
            assert_eq!(errors.len(), 72 * 56);
            for (i, &v) in errors.iter().enumerate() {
                let want = e.style[i] + e.guide[i];
                assert!((v - want).abs() <= 1e-3 * want.max(1.0), "pixel {i}: {v} vs {want}");
            }
            // The flag does not change the synthesis.
            let plain = synthesize(&input, &Params { return_errors: false, ..params }).unwrap();
            assert!(plain.errors.is_none());
            assert_eq!((plain.image, plain.nnf), (out.image, out.nnf));
        }
    }

    #[test]
    fn compat_preset_runs_the_2019_schedule() {
        // Compat must equal explicitly setting the 2019 schedule, voting and search on another
        // preset.
        let input = shifted_input();
        let base = Params { omega_sync: OmegaSync::PerTile, num_threads: Some(3), ..Params::compat() };
        let explicit = Params {
            search_vote_iters: Some(PerLevel::Uniform(6)),
            patch_match_iters: Some(PerLevel::Uniform(4)),
            use_stop_threshold: Some(false),
            vote_mode: Some(VoteMode::Plain),
            vote_rounding: Some(VoteRounding::Truncate),
            search: Some(SearchMode::Compat),
            ..Params { preset: Preset::Fast, ..base.clone() }
        };
        let (a, b) = (synthesize(&input, &base).unwrap(), synthesize(&input, &explicit).unwrap());
        assert_eq!((&a.image, &a.nnf), (&b.image, &b.nnf));
        // The CPU engine has one search, so the search mode does not change its result.
        let improved = synthesize(&input, &Params { search: Some(SearchMode::Improved), ..base.clone() }).unwrap();
        assert_eq!((&improved.image, &improved.nnf), (&a.image, &a.nnf));
        // And the default preset gives a different (faster) result.
        let d = synthesize(&input, &Params { preset: Preset::Default, ..base }).unwrap();
        assert_ne!(d.nnf, b.nnf);
    }

    #[test]
    fn layer_modulation_builds_target_modulation() {
        let w = 3;
        let rgba = |f: &dyn Fn(usize) -> [u8; 4]| Image::from_raw(w, 1, 4, (0..w).flat_map(f).collect());
        let style = rgba(&|i| [i as u8, 2, 3, 255]);
        let colour = rgba(&|i| [i as u8, 9, 1, 255]); // 3 channels
        let gray = rgba(&|i| [i as u8 * 9, i as u8 * 9, i as u8 * 9, 255]); // 1 channel
        let map = Image::from_raw(w, 1, 1, vec![0, 100, 255]);
        let plain =
            SynthesisInput::from_rgba_layers(&style, None, &[GuideLayer::new(colour.clone(), colour.clone())]).unwrap();
        assert!(plain.target_modulation.is_none());
        assert_eq!(plain.modulated_range, None);
        let input = SynthesisInput::from_rgba_layers(
            &style,
            None,
            &[
                GuideLayer::new(colour.clone(), colour.clone()),
                GuideLayer::new(gray.clone(), gray.clone()).with_modulation(map.clone()),
            ],
        )
        .unwrap();
        let m = input.target_modulation.as_ref().unwrap();
        assert_eq!((m.size(), m.channels()), ((3, 1), 4));
        // Only the second layer (the gray guide, channel 3) is gated.
        assert_eq!(input.modulated_range, Some((3, 3)));
        assert_eq!(m.pixel(0, 0), &[255, 255, 255, 0]);
        assert_eq!(m.pixel(1, 0), &[255, 255, 255, 100]);
        assert_eq!(m.pixel(2, 0), &[255, 255, 255, 255]);
        let bad = GuideLayer::new(gray.clone(), gray.clone()).with_modulation(Image::new(2, 1, 1));
        assert!(SynthesisInput::from_rgba_layers(&style, None, &[bad]).is_err());
        let bad = GuideLayer::new(gray.clone(), gray).with_modulation(Image::new(3, 1, 2));
        assert!(SynthesisInput::from_rgba_layers(&style, None, &[bad]).is_err());
    }

    #[test]
    fn zero_modulation_switches_a_guide_off() {
        // With the only guide modulated to zero everywhere, guide errors vanish: every final
        // patch error is a pure style error.
        let mut input = shifted_input();
        input.target_modulation = Some(Image::new(72, 56, 3));
        let params = Params { return_errors: true, num_threads: Some(2), ..Params::default() };
        let out = synthesize(&input, &params).unwrap();
        let e = crate::nnf_patch_errors(&input, &params, &out.image, &out.nnf).unwrap();
        assert!(e.guide.iter().all(|&g| g == 0.0));
        let full = crate::nnf_patch_errors(&shifted_input(), &params, &out.image, &out.nnf).unwrap();
        assert!(full.guide.iter().sum::<f32>() > 0.0);
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
            modulated_range: None,
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
        let base = Params {
            omega_sync: OmegaSync::PerTile,
            num_bands: Some(5),
            use_stop_threshold: Some(true),
            ..Params::default()
        };
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

    #[test]
    fn style16_reproduces_the_8_bit_result_at_full_precision() {
        let input = shifted_input();
        let wide = Image16::from_image(&input.source_style);
        let extras = SynthesisExtras { style16: Some(wide), ..SynthesisExtras::default() };
        for (vote, extra_pass_3x3) in
            [(VoteMode::Plain, false), (VoteMode::Weighted, false), (VoteMode::Weighted, true)]
        {
            let params = Params {
                vote_mode: Some(vote),
                extra_pass_3x3,
                omega_sync: OmegaSync::PerTile,
                num_threads: Some(2),
                ..Params::default()
            };
            let out = synthesize_with(&input, &params, &extras).unwrap();
            let plain = synthesize(&input, &params).unwrap();
            // The 16-bit style does not change the search.
            assert_eq!((&plain.image, &plain.nnf), (&out.image, &out.nnf));
            let image16 = out.image16.expect("16-bit result");
            let differ = image16.to_image().data().iter().zip(out.image.data()).filter(|(a, b)| a != b).count();
            assert!(differ <= 2, "{vote:?}, extra pass {extra_pass_3x3}: {differ} values differ");
            assert!(image16.data().iter().any(|&v| v % 257 != 0), "no fractions kept");
        }
        let bad = SynthesisExtras { style16: Some(Image16::new(3, 3, 3)), ..SynthesisExtras::default() };
        assert!(synthesize_with(&input, &Params::default(), &bad).is_err());
    }

    #[test]
    fn warm_start_from_a_good_field_stays_good() {
        let input = shifted_input();
        let params =
            Params { omega_sync: OmegaSync::PerTile, num_threads: Some(2), return_errors: true, ..Params::default() };
        let cold = synthesize(&input, &params).unwrap();
        let mean = |e: &[f32]| e.iter().map(|&v| f64::from(v)).sum::<f64>() / e.len() as f64;
        let cold_err = mean(cold.errors.as_deref().unwrap());
        for levels in [1, 2, 9] {
            let extras = SynthesisExtras {
                warm_start: Some(WarmStart { nnf: cold.nnf.clone(), levels }),
                ..SynthesisExtras::default()
            };
            let warm = synthesize_with(&input, &params, &extras).unwrap();
            let warm_err = mean(warm.errors.as_deref().unwrap());
            assert!(warm_err <= cold_err * 1.05, "{levels} levels: {warm_err} vs cold {cold_err}");
        }
        // A random field run at the finest level only is worse than the full pyramid.
        let mut rng = GlibcRand::new(3);
        let random = nnf_init_random((72, 56), (72, 56), 5, &mut rng);
        let extras =
            SynthesisExtras { warm_start: Some(WarmStart { nnf: random, levels: 1 }), ..SynthesisExtras::default() };
        let one = synthesize_with(
            &input,
            &Params { search_vote_iters: Some(PerLevel::Uniform(1)), ..params.clone() },
            &extras,
        )
        .unwrap();
        assert!(mean(one.errors.as_deref().unwrap()) > cold_err);
        let bad = SynthesisExtras {
            warm_start: Some(WarmStart { nnf: Nnf::new(3, 3), levels: 1 }),
            ..SynthesisExtras::default()
        };
        assert!(synthesize(&input, &params).is_ok() && synthesize_with(&input, &params, &bad).is_err());
    }
}
