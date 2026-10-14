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
//! channels ([`guides::encode_positions`]). Guides can be weighted per pixel by how far they can
//! be trusted: the mapping guide by the tracker's confidence ([`VideoParams::gate_mapping`], on
//! by default) and the temporal guide by the forward-backward consistency of its flow
//! ([`VideoParams::gate_temporal`]).
//!
//! # Scheduling
//!
//! Each keyframe's own frame is synthesized once. Between two neighbouring keyframes a forward
//! and a backward chain run, each only across that interval, and their results are blended
//! ([`BlendMode`]): by default with the adaptive blend, which keeps per pixel the pass with the
//! lower synthesis error and works in the gradient domain, so the middle of an interval keeps
//! its detail; the linear crossfade remains available ([`VideoParams::classic`]), and hosts can
//! plug in their own [`Blender`]. Before the first and after the last keyframe a single chain runs (see
//! [`plan`]). All chains are independent, so they run at the same time on one device: a
//! [`ComputeDevice`] shares one wgpu device between the synthesis engine and the flow
//! estimator, or runs both on the CPU. Optical flow is computed once per frame pair and shared
//! by every chain that needs it ([`FlowBank`]).
//!
//! # Options
//!
//! - **Warm start** ([`VideoParams::warm_start`], off by default): each frame of a chain starts
//!   from the previous frame's final nearest-neighbour field, moved along the optical flow, and
//!   runs only the finest pyramid levels.
//! - **16-bit output** ([`Renderer::keyframes16`]): with 16-bit keyframe paintings, the search
//!   still runs at 8 bits, but every synthesis's final field is voted again from the 16-bit
//!   painting, passes are blended at 16 bits and frames reach [`FrameSink::write16`].
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
mod poisson;
mod progress;
mod render;
pub mod resources;
mod store;

pub use crate::blend::{BlendMode, BlendSettings, Blender, BuiltinBlender, PassFrame, blend};
pub use crate::device::{Backend, ComputeDevice, FlowProvider, Synthesizer};
pub use crate::error::VideoError;
pub use crate::flowbank::{FlowBank, FlowStats};
pub use crate::frames::{
    FrameCache, FrameSink, FrameSource, ImageSequence, ImageSequenceWriter, MemoryFrames, MemorySink, pattern_path,
};
pub use crate::params::{FlowBackend, GuideWeights, KeyframeOutput, RenderOptions, VideoParams, WarmStartSettings};
pub use crate::plan::Plan;
pub use crate::progress::{CancelToken, Progress};
pub use crate::render::{Keyframe, Keyframe16, RenderStats, Renderer, Shot};
pub use crate::resources::{ResourcePlan, ShotFormat, default_memory_budget};
pub use crate::store::{SpillKey, SpillStore, StoreStats};

/// Revision of how this crate renders frames. Bump it whenever a change to synthesis, flow, the
/// guides or the blend alters rendered frames without a visible change to [`VideoParams`], so that
/// a cache keyed on the parameters does not reuse frames from an earlier build. It lives with the
/// rendering code (rather than in the host) so the bump is next to the change that needs it.
pub const OUTPUT_REVISION: u32 = 1;

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
