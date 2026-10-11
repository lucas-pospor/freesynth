//! Keyframe video stylization for FreeSynth.
//!
//! Given a video (a sequence of frames) and a few painted keyframes, this crate stylizes every
//! frame by example-based synthesis from the nearest keyframes. For each frame `t` of a chain
//! that walks away from keyframe `k`, the engine gets the keyframe painting as its style and
//! these guides:
//!
//! | Guide | Source (keyframe side) | Target (frame side) |
//! |---|---|---|
//! | appearance | video frame `k` | video frame `t` |
//! | temporal | the keyframe painting | the chain's previous output, warped to `t` by optical flow |
//! | mapping | keyframe coordinates | the keyframe coordinates of the content seen at `t` |
//! | mask (optional) | the painting's alpha | mask frame `t` |
//!
//! Positions come from `freesynth-flow`'s anchored tracker; they are encoded into 8-bit guide
//! channels ([`guides::encode_positions`]), and can be weighted per pixel by the tracker's
//! confidence ([`VideoParams::gate_mapping`]).
//!
//! # Scheduling
//!
//! Each keyframe's own frame is synthesized once. Between two neighbouring keyframes a forward
//! and a backward chain run, each only across that interval, and their results are blended
//! ([`BlendMode`]). Before the first and after the last keyframe a single chain runs (see
//! [`plan`]). All chains are independent, so they run at the same time on one device: a
//! [`ComputeDevice`] shares one wgpu device between the synthesis engine and the flow
//! estimator, or runs both on the CPU. Optical flow is computed once per frame pair and shared
//! by every chain that needs it ([`FlowBank`]).
//!
//! # Memory
//!
//! Input frames are decoded once into a [`FrameCache`]; synthesized frames wait in a buffer
//! until the other pass of their interval reaches them. Both hold a bounded number of bytes in
//! memory and spill the rest to disk ([`RenderOptions`]), which matters for 4K shots, where one
//! interval of frames is several gigabytes.
//!
//! ```no_run
//! use freesynth_core::io;
//! use freesynth_video::{Backend, ComputeDevice, ImageSequence, ImageSequenceWriter, Keyframe};
//! use freesynth_video::{RenderOptions, Renderer, Shot, VideoParams};
//!
//! let frames = ImageSequence::from_dir("shot/frames")?;
//! let keyframes = vec![
//!     Keyframe { frame: 0, style: io::load_rgba("shot/keys/000.png").unwrap() },
//!     Keyframe { frame: 99, style: io::load_rgba("shot/keys/099.png").unwrap() },
//! ];
//! let shot = Shot { frames: &frames, keyframes, mask: None, range: None };
//! let device = ComputeDevice::new(Backend::Auto)?;
//! let options = RenderOptions { spill_dir: Some("shot/cache".into()), ..RenderOptions::default() };
//! let stats = Renderer::new(&device, VideoParams::classic())
//!     .options(options)
//!     .render(&shot, &ImageSequenceWriter::into_dir("shot/out")?)?;
//! println!("{} frames in {:.1} s", stats.frames, stats.seconds);
//! # Ok::<(), freesynth_video::VideoError>(())
//! ```

mod blend;
mod device;
mod error;
mod flowbank;
mod frames;
pub mod guides;
mod params;
pub mod plan;
mod progress;
mod render;
pub mod resources;
mod store;

pub use crate::blend::{BlendMode, blend};
pub use crate::device::{Backend, ComputeDevice, FlowProvider, Synthesizer};
pub use crate::error::VideoError;
pub use crate::flowbank::{FlowBank, FlowStats};
pub use crate::frames::{
    FrameCache, FrameSink, FrameSource, ImageSequence, ImageSequenceWriter, MemoryFrames, MemorySink, pattern_path,
};
pub use crate::params::{FlowBackend, GuideWeights, KeyframeOutput, RenderOptions, VideoParams};
pub use crate::plan::Plan;
pub use crate::progress::{CancelToken, Progress};
pub use crate::render::{Keyframe, RenderStats, Renderer, Shot};
pub use crate::resources::{ResourcePlan, default_memory_budget};
pub use crate::store::{SpillKey, SpillStore, StoreStats};

#[cfg(test)]
pub(crate) mod test_util {
    use std::path::PathBuf;

    /// A fresh scratch directory path under the build directory (the system temporary directory
    /// may be RAM-backed). Not created.
    pub fn scratch_dir(name: &str) -> PathBuf {
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"));
        let dir = target.join("tmp").join(format!("freesynth-video-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }
}
