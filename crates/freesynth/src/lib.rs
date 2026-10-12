//! FreeSynth: example-based stylization of images and video.
//!
//! This crate is the library that front ends (the command line, a GUI, bindings) build on. It
//! holds the product model on top of the engine crates:
//!
//! - [`Project`]: a shot (input frames or a video file, optional masks), its painted keyframes,
//!   the look settings and the output (an image sequence, 8 or 16 bits, and/or a video file),
//!   stored as a TOML project file (`.fsyn`). [`ebs`] imports EbSynth `.ebs` project files.
//! - [`Session`]: a compute device kept open across renders (GPU with a CPU fallback). It
//!   renders projects incrementally ([`Session::render`]): a render cache remembers which inputs
//!   every output frame came from, so after editing one keyframe only the frames between its
//!   neighbouring keyframes are rendered again. [`Session::watch`] re-renders whenever a
//!   keyframe file changes. Shots the host holds in memory render through
//!   [`Session::render_shot`].
//! - Video files are read and written through the system's ffmpeg (`freesynth-media`), with
//!   their colour tags; image sequences always work without it.
//! - Progress events ([`Event`]) and cancellation ([`CancelToken`]) for long renders, and
//!   [`devices`] to list GPU adapters.
//!
//! ```no_run
//! use freesynth::{CancelToken, Event, Project, Session};
//!
//! let project = Project::load("shot.fsyn")?;
//! let session = Session::new(project.backend())?;
//! let report = |e: &Event| {
//!     if let Event::Written { done, total, .. } = e {
//!         eprintln!("{done}/{total}");
//!     }
//! };
//! let stats = session.render(&project, Some(&report), &CancelToken::new())?;
//! println!("{} of {} frames rendered in {:.1} s", stats.rendered, stats.frames, stats.seconds);
//! # Ok::<(), freesynth::Error>(())
//! ```
//!
//! The engines are available directly too: `freesynth-core` (CPU synthesis), `freesynth-gpu`
//! (GPU synthesis), `freesynth-flow` (optical flow), `freesynth-video` (the keyframe video
//! pipeline) and `freesynth-media` (video I/O), whose main types are re-exported here.

mod cache;
pub mod devices;
pub mod ebs;
mod error;
mod event;
mod hashing;
mod input;
mod output;
pub mod paths;
pub mod project;
mod render;
mod session;
mod watch;

pub use crate::error::Error;
pub use crate::event::{Event, ProjectStats};
pub use crate::project::Project;
pub use crate::render::RenderRequest;
pub use crate::session::{Session, SessionOptions};
pub use crate::watch::{WatchEvent, WatchOptions};
pub use freesynth_media::{BitDepth, Codec, ColourTags, Ffmpeg, MediaError};
pub use freesynth_video::{
    Backend, BlendMode, BlendSettings, CancelToken, ComputeDevice, FlowBackend, FrameSink, FrameSource, GuideWeights,
    ImageSequence, ImageSequenceWriter, Keyframe, Keyframe16, KeyframeOutput, MemoryFrames, MemorySink, Progress,
    RenderOptions, RenderStats, Renderer, ResourcePlan, Shot, VideoError, VideoParams, WarmStartSettings,
};

#[cfg(test)]
pub(crate) mod test_util {
    use std::path::PathBuf;

    /// A fresh scratch directory path under the build directory (the system temporary directory
    /// may be RAM-backed). Not created.
    pub fn scratch(name: &str) -> PathBuf {
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"));
        let dir = target.join("tmp").join(format!("freesynth-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }
}
