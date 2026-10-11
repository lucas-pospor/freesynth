//! FreeSynth: example-based stylization of images and video.
//!
//! This crate is the library that front ends (the command line, a GUI, bindings) build on. It
//! holds the product model on top of the engine crates:
//!
//! - [`Project`]: a shot (input frames, optional masks), its painted keyframes, the look
//!   settings and the output, stored as a TOML project file;
//! - [`Session`]: a compute device kept open across renders (GPU with a CPU fallback), which
//!   renders projects ([`Session::render`]) or shots the host holds in memory
//!   ([`Session::render_shot`] with a [`Shot`] of [`MemoryFrames`] or any [`FrameSource`]);
//! - progress reports ([`Progress`]) and cancellation ([`CancelToken`]) for long renders.
//!
//! ```no_run
//! use freesynth::{CancelToken, Progress, Project, Session};
//!
//! let project = Project::load("shot.fsproj")?;
//! let session = Session::new(project.backend())?;
//! let report = |p: &Progress| {
//!     if let Progress::Written { done, total, .. } = p {
//!         eprintln!("{done}/{total}");
//!     }
//! };
//! let stats = session.render(&project, Some(&report), &CancelToken::new())?;
//! println!("{} frames in {:.1} s on {}", stats.frames, stats.seconds, stats.device);
//! # Ok::<(), freesynth::Error>(())
//! ```
//!
//! The engines are available directly too: `freesynth-core` (CPU synthesis), `freesynth-gpu`
//! (GPU synthesis), `freesynth-flow` (optical flow) and `freesynth-video` (the keyframe video
//! pipeline), whose main types are re-exported here.

mod error;
pub mod project;
mod session;

pub use crate::error::Error;
pub use crate::project::Project;
pub use crate::session::{Session, SessionOptions};
pub use freesynth_video::{
    Backend, BlendMode, CancelToken, ComputeDevice, FlowBackend, FrameSink, FrameSource, GuideWeights, ImageSequence,
    ImageSequenceWriter, Keyframe, KeyframeOutput, MemoryFrames, MemorySink, Progress, RenderOptions, RenderStats, Renderer,
    ResourcePlan, Shot, VideoError, VideoParams,
};
