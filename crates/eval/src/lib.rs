//! FreeSynth evaluation: quality metrics for example-based synthesis and keyframe video
//! stylization, and paired comparisons between variants.
//!
//! No single number judges a stylization: patch energy rewards copying a few source patches
//! everywhere, warping error rewards smearing, and pixel metrics against a reference rank a
//! blurred copy above a valid re-synthesis. This crate therefore provides a small set of metrics
//! that are meant to be read together, each guarding against the failure modes of the others:
//!
//! | Module | Metric | Needs |
//! |---|---|---|
//! | [`energy`] | style and total patch energy, exact from an NNF | the engine's final NNF |
//! | [`energy`] | sampled exact coherence energy (exhaustive search on random pixels) | any output image |
//! | [`occupancy`] | unused-source fraction and other occupancy statistics | an NNF |
//! | [`sharpness`] | band-pass, high-frequency and gradient ratios against a reference (noise and artefacts raise the last two) | any image |
//! | [`temporal`] | warping error (1 and 5 frames), second difference, global flicker | a frame sequence and an independent flow |
//! | [`bootstrap`], [`compare`] | paired differences with bootstrap confidence intervals | two sets of reports |
//!
//! Every measurement produces a [`Report`] (JSON), and comparisons render as Markdown. The
//! `freesynth-eval` binary exposes all of this on the command line.
//!
//! # What energy can and cannot show
//!
//! Patch energy is the coherence part of FreeSynth's own objective: the weighted sum of squared
//! differences between output patches and source patches under the guides, with FreeSynth's
//! channel selection and weights. The engine minimises this plus a uniformity penalty that
//! discourages reusing the same source patches; energy leaves that penalty out. Turning
//! uniformity off therefore gives the lowest energy, by reusing a few good patches everywhere,
//! which is why energy must always be read next to [`occupancy`]. Energy is the right tool for
//! comparing two syntheses of the same problem with each other,
//! for example a faster search schedule against the current one. It is **not** an absolute measure
//! of quality:
//!
//! - A result produced by a different method, with different descriptors, guides or weights,
//!   is scored under an objective it did not optimise. Energy is biased toward FreeSynth in such
//!   a comparison, and it would equally penalise a future objective that is genuinely better.
//! - A correct answer that contains content the exemplar does not have (for example disoccluded
//!   or rescaled texture in a ground-truth video frame) scores worse than a synthesis, because
//!   synthesis minimises exactly this number.
//! - Low energy alone can mean wash-out (the uniformity penalty is not part of it): always read
//!   it next to [`occupancy`].
//!
//! Comparisons across objectives need ground-truth scenes, the temporal metrics and viewing.
//!
//! # Noise
//!
//! Syntheses with different seeds are equally valid and differ by about 1% in energy. Decide
//! about algorithm changes with paired runs over several images and seeds ([`compare`] at the
//! run level), not from a single run.
//!
//! # Quick check
//!
//! `cargo test -p freesynth-eval --test ci_quick` runs the whole chain in a few seconds: it
//! generates tiny scenes with `freesynth-scenes`, stylizes them, computes every metric and checks
//! the results against sanity bounds. It needs no external data and no GPU; the GPU engine is
//! checked against the CPU engine in `freesynth-gpu`'s tests, which skip themselves when no
//! adapter is available.

pub mod bootstrap;
pub mod compare;
pub mod energy;
mod error;
pub mod flo;
pub mod frame;
pub mod occupancy;
pub mod report;
pub mod rng;
pub mod sharpness;
pub mod stats;
pub mod temporal;

pub use crate::error::Error;
pub use crate::report::{PixelSet, Report, Samples};
