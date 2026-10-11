//! Render sessions: a device kept open across renders.

use std::path::{Path, PathBuf};

use freesynth_core::io;
use freesynth_video::{
    Backend, CancelToken, ComputeDevice, FlowBackend, FrameSink, FrameSource, ImageSequence, ImageSequenceWriter, Keyframe, Progress,
    RenderOptions, RenderStats, Renderer, Shot, VideoError, VideoParams,
};

use crate::error::Error;
use crate::project::Project;

/// Memory and parallelism settings of a [`Session`] (see [`RenderOptions`] for details).
#[derive(Clone, Debug, PartialEq)]
pub struct SessionOptions {
    /// Total memory a render should stay within, in bytes. Default: half the memory, at most
    /// 4 GiB.
    pub memory_budget: usize,
    /// Keyframe chains run at the same time. `None` runs as many as fit in the memory budget
    /// (at most four).
    pub parallel_chains: Option<usize>,
    /// The most bytes of decoded input frames kept in memory (per input sequence).
    pub frame_cache_bytes: usize,
    /// The most bytes of synthesized frames waiting for their blend partner kept in memory.
    pub buffer_bytes: usize,
    /// Bytes of optical flow fields kept for reuse.
    pub flow_cache_bytes: usize,
    /// Where frames beyond the memory budgets go. `None` uses `.freesynth-cache` in the
    /// project's directory, and the user's cache directory for [`Session::render_shot`].
    /// Several renders may share it at once.
    pub cache_dir: Option<PathBuf>,
    /// Where optical flow is computed.
    pub flow_backend: FlowBackend,
}

impl Default for SessionOptions {
    fn default() -> Self {
        let r = RenderOptions::default();
        Self {
            memory_budget: r.memory_budget,
            parallel_chains: r.parallel_chains,
            frame_cache_bytes: r.frame_cache_bytes,
            buffer_bytes: r.buffer_bytes,
            flow_cache_bytes: r.flow_cache_bytes,
            cache_dir: None,
            flow_backend: r.flow_backend,
        }
    }
}

impl SessionOptions {
    /// The render options for a render spilling into `spill_dir`.
    fn render_options(&self, spill_dir: Option<PathBuf>) -> RenderOptions {
        RenderOptions {
            memory_budget: self.memory_budget,
            parallel_chains: self.parallel_chains,
            frame_cache_bytes: self.frame_cache_bytes,
            buffer_bytes: self.buffer_bytes,
            flow_cache_bytes: self.flow_cache_bytes,
            flow_backend: self.flow_backend,
            spill_dir,
        }
    }
}

/// The user's cache directory for FreeSynth: `$XDG_CACHE_HOME/freesynth` or
/// `~/.cache/freesynth` on Unix, `%LOCALAPPDATA%\freesynth` on Windows.
fn user_cache_dir() -> Option<PathBuf> {
    let non_empty = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        non_empty("LOCALAPPDATA")?
    } else {
        non_empty("XDG_CACHE_HOME").or_else(|| non_empty("HOME").map(|h| h.join(".cache")))?
    };
    Some(base.join("freesynth"))
}

/// Owns a compute device and renders projects on it. Opening a GPU device and compiling the
/// kernels is slow, so a host (the command line, a GUI) keeps one session for many renders.
///
/// A session is `Send + Sync`: renders take `&self`, so several can run at once from different
/// threads (they then share the device).
pub struct Session {
    device: ComputeDevice,
    options: SessionOptions,
}

/// Loads a sequence given as a pattern (with a `#` run) or a directory.
fn sequence(path: &Path) -> Result<ImageSequence, VideoError> {
    let is_pattern = path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains('#'));
    if is_pattern { ImageSequence::from_pattern(path) } else { ImageSequence::from_dir(path) }
}

impl Session {
    /// Opens the device for `backend` (`Auto` falls back to the CPU without a GPU; the reason
    /// is in [`ComputeDevice::fallback_reason`]).
    pub fn new(backend: Backend) -> Result<Self, Error> {
        Ok(Self::with_device(ComputeDevice::new(backend)?))
    }

    /// Uses an existing device (for example one shared with a GUI).
    pub fn with_device(device: ComputeDevice) -> Self {
        Self { device, options: SessionOptions::default() }
    }

    /// Sets the memory and parallelism options.
    pub fn set_options(&mut self, options: SessionOptions) {
        self.options = options;
    }

    /// The memory and parallelism options.
    pub fn options(&self) -> &SessionOptions {
        &self.options
    }

    /// The device.
    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// Renders every frame of `project` to its output.
    pub fn render(
        &self,
        project: &Project,
        progress: Option<&(dyn Fn(&Progress) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<RenderStats, Error> {
        let out = project.resolve(&project.output.frames);
        let sink = if out.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains('#')) {
            ImageSequenceWriter::new(out)?
        } else {
            ImageSequenceWriter::into_dir(out)?
        };
        self.render_to(project, None, &sink, progress, cancel)
    }

    /// Renders frames `range` (all of the project's frames when `None`) of `project` into
    /// `sink`.
    pub fn render_to(
        &self,
        project: &Project,
        range: Option<(usize, usize)>,
        sink: &dyn FrameSink,
        progress: Option<&(dyn Fn(&Progress) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<RenderStats, Error> {
        project.validate()?;
        let params = project.video_params()?;
        let frames = sequence(&project.resolve(&project.media.frames))?;
        let mask = project.media.mask.as_ref().map(|m| sequence(&project.resolve(m))).transpose()?;
        let keyframes = project
            .keyframes
            .iter()
            .map(|k| {
                let path = project.resolve(&k.image);
                let style = io::load_rgba(&path).map_err(|source| VideoError::Load { path, source })?;
                Ok(Keyframe { frame: k.frame, style })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let range = range.unwrap_or((
            project.media.first.unwrap_or(frames.first()),
            project.media.last.unwrap_or(frames.last()),
        ));
        let shot = Shot { frames: &frames, keyframes, mask: mask.as_ref().map(|m| m as &dyn FrameSource), range: Some(range) };
        let o = &self.options;
        let project_cache = project.base_dir().join(".freesynth-cache");
        let spill_dir = o.cache_dir.clone().unwrap_or_else(|| project_cache.clone()).join("spill");
        let result = self.run(&shot, params, o.render_options(Some(spill_dir)), sink, progress, cancel);
        // Remove the cache directory if the render left it empty.
        if o.cache_dir.is_none() {
            let _ = std::fs::remove_dir(project_cache);
        }
        result
    }

    /// Renders a shot whose frames, masks and keyframes the host holds (in memory, or behind
    /// its own [`FrameSource`]) with the look `params`, into `sink`. Frames beyond the memory
    /// budgets spill to [`SessionOptions::cache_dir`] or, without one, to the user's cache
    /// directory.
    pub fn render_shot(
        &self,
        shot: &Shot<'_>,
        params: VideoParams,
        sink: &dyn FrameSink,
        progress: Option<&(dyn Fn(&Progress) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<RenderStats, Error> {
        let spill_dir = self.options.cache_dir.clone().or_else(user_cache_dir).map(|d| d.join("spill"));
        self.run(shot, params, self.options.render_options(spill_dir), sink, progress, cancel)
    }

    fn run(
        &self,
        shot: &Shot<'_>,
        params: VideoParams,
        options: RenderOptions,
        sink: &dyn FrameSink,
        progress: Option<&(dyn Fn(&Progress) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<RenderStats, Error> {
        let mut renderer = Renderer::new(&self.device, params).options(options).cancel_token(cancel.clone());
        if let Some(p) = progress {
            renderer = renderer.on_progress(p);
        }
        Ok(renderer.render(shot, sink)?)
    }
}

/// A host keeps a session in an `Arc` and renders from worker threads.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Session>();
};
