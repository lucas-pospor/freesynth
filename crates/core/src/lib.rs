//! FreeSynth core: example-based image synthesis on the CPU.
//!
//! This crate is a Rust port of the public-domain 2019 ebsynth CPU engine
//! (`ebsynth_cpu.cpp`): a coarse-to-fine pyramid of PatchMatch nearest-neighbour searches with a
//! uniformity (source occupancy) term, alternated with patch voting. It also carries the optional
//! features that the 2019 code only implemented in its CUDA engine (weighted voting, guide
//! modulation, stop-threshold masking), so that every backend can share one parameter set.
//!
//! The entry point is [`synthesize`], which mirrors `ebsynthRun`:
//!
//! ```no_run
//! use freesynth_core::{io, GuideLayer, Params, SynthesisInput, synthesize};
//!
//! let style = io::load_rgba("style.png")?;
//! let guide = GuideLayer::new(io::load_rgba("src.png")?, io::load_rgba("tgt.png")?);
//! let input = SynthesisInput::from_rgba_layers(&style, None, &[guide])?;
//! let out = synthesize(&input, &Params::default())?;
//! io::save_png(&out.image, "out.png")?;
//! # Ok::<(), freesynth_core::Error>(())
//! ```
//!
//! # Presets
//!
//! [`Params::default`] runs the [`Preset::Default`] schedule: half the PatchMatch iterations of
//! the 2019 engine (which converge to about the same result), weighted voting, rounded votes and,
//! on the GPU, the improved search ([`SearchMode::Improved`]). [`Preset::Fast`] trades a few
//! percent of patch error for speed, [`Preset::High`] keeps the full 2019 schedule, and
//! [`Preset::Compat`] (or [`Params::compat`]) reproduces the 2019 tool. Any schedule setting given explicitly in
//! [`Params`] overrides the preset; [`Params::schedule`] shows the resolved values.
//!
//! # Outputs
//!
//! Besides the image, [`SynthesisOutput`] holds the final nearest-neighbour field and, with
//! [`Params::return_errors`], the final per-pixel patch errors (for example to find regions the
//! source cannot explain). [`SynthesisInput::target_modulation`] weights the guides per pixel.
//!
//! # Extras
//!
//! [`synthesize_with`] takes optional [`SynthesisExtras`]: a [`WarmStart`] from a known field
//! (for video, the previous frame's), which runs only the finest pyramid levels; and a 16-bit
//! style ([`Image16`]), from which the final field is voted again for a 16-bit result
//! ([`SynthesisOutput::image16`]).

mod error;
mod extras;
mod image;
pub mod io;
mod mask;
mod nnf;
mod patch;
mod patch_error;
mod patchmatch;
mod pyramid;
mod rng;
mod schedule;
mod synth;
mod vote;

pub use crate::error::Error;
pub use crate::extras::{SynthesisExtras, WarmStart};
pub use crate::image::{Image, Image16, Nnf};
pub use crate::nnf::{nnf_init_random, nnf_upscale};
pub use crate::patch_error::{ErrorPart, NnfErrors, PatchDistance, PatchErrorEvaluator, nnf_patch_errors};
pub use crate::pyramid::{max_pyramid_levels, pyramid_level_size, resample};
pub use crate::rng::GlibcRand;
pub use crate::schedule::{Engine, Preset, Schedule, SearchMode};
pub use crate::synth::{
    GuideLayer, OmegaSync, Params, PerLevel, SynthesisInput, SynthesisOutput, VoteMode, synthesize, synthesize_with,
};
pub use crate::vote::{VoteRounding, vote_plain, vote_plain16, vote_weighted, vote_weighted16};

/// Maximum number of style channels, as in the original (`EBSYNTH_MAX_STYLE_CHANNELS`).
pub const MAX_STYLE_CHANNELS: usize = 8;
/// Maximum number of guide channels, as in the original (`EBSYNTH_MAX_GUIDE_CHANNELS`).
pub const MAX_GUIDE_CHANNELS: usize = 24;
