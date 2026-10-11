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

mod error;
mod image;
pub mod io;
mod mask;
mod nnf;
mod patch;
mod patchmatch;
mod pyramid;
mod rng;
mod synth;
mod vote;

pub use crate::error::Error;
pub use crate::image::{Image, Nnf};
pub use crate::nnf::{nnf_init_random, nnf_upscale};
pub use crate::pyramid::{max_pyramid_levels, pyramid_level_size, resample};
pub use crate::rng::GlibcRand;
pub use crate::synth::{
    GuideLayer, OmegaSync, Params, PerLevel, SynthesisInput, SynthesisOutput, VoteMode,
    synthesize,
};
pub use crate::vote::{vote_plain, vote_weighted};

/// Maximum number of style channels, as in the original (`EBSYNTH_MAX_STYLE_CHANNELS`).
pub const MAX_STYLE_CHANNELS: usize = 8;
/// Maximum number of guide channels, as in the original (`EBSYNTH_MAX_GUIDE_CHANNELS`).
pub const MAX_GUIDE_CHANNELS: usize = 24;
