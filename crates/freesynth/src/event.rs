//! Progress events of a project render.

use std::path::PathBuf;

use freesynth_video::RenderStats;

/// What a project render reports while it runs ([`Session::render`](crate::Session::render)).
/// Events come from several threads. New kinds of events may be added, so matches need a
/// catch-all arm.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Event {
    /// A phase started ("hashing inputs", "decoding shot.mov", ...).
    Status(String),
    /// Something the user should know that does not stop the render.
    Warning(String),
    /// Frames of a video input decoded into the frame cache so far.
    Decoded { done: usize, total: usize },
    /// The render was planned against the cache.
    Planned {
        /// Output frames of the shot.
        frames: usize,
        /// Frames that will be rendered.
        render: usize,
        /// Frames that are up to date and kept.
        reused: usize,
        /// The shot segments rendered, as frame ranges (each starts and ends at a keyframe or
        /// at the shot's ends).
        runs: Vec<(usize, usize)>,
        /// Syntheses to run in all segments.
        syntheses: usize,
    },
    /// One synthesis finished (`done` counts over all segments).
    Synthesized { frame: usize, keyframe: usize, done: usize, total: usize },
    /// One output frame was written (`done` counts the frames rendered by this render).
    Written { frame: usize, done: usize, total: usize },
    /// Frames handed to the video encoder so far.
    Encoded { done: usize, total: usize },
}

/// What a project render did. Fields may be added.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ProjectStats {
    /// Description of the device.
    pub device: String,
    /// Output frames of the shot.
    pub frames: usize,
    /// Frames rendered.
    pub rendered: usize,
    /// Frames that were up to date and kept.
    pub reused: usize,
    /// The shot segments rendered.
    pub runs: Vec<(usize, usize)>,
    /// Statistics of each segment's render.
    pub run_stats: Vec<RenderStats>,
    /// Syntheses run.
    pub syntheses: usize,
    /// Wall-clock time of the whole render, in seconds.
    pub seconds: f64,
    /// Time spent hashing inputs and decoding video input before rendering, in seconds.
    pub prepare_seconds: f64,
    /// Time spent rendering segments, in seconds.
    pub render_seconds: f64,
    /// Time spent finishing the output video after the last segment, in seconds (most of the
    /// encode overlaps the render).
    pub encode_seconds: f64,
    /// The output video, when one was written by this render.
    pub video: Option<PathBuf>,
    /// The output video was up to date and not encoded again.
    pub video_reused: bool,
}
