//! Render sessions: a device kept open across renders.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use freesynth_media::{Ffmpeg, MediaError};
use freesynth_video::{
    Backend, CancelToken, ComputeDevice, FlowBackend, FrameSink, FrameSource, Keyframe16, Progress, RenderOptions,
    RenderStats, Renderer, Shot, VideoParams,
};

use crate::error::Error;
use crate::event::{Event, ProjectStats};
use crate::input::{Input, Opener, with_decodes};
use crate::project::Project;
use crate::render::{RenderRequest, prepare, render_project};

/// Memory and parallelism settings of a [`Session`] (see [`RenderOptions`] for details). New
/// fields may be added: start from [`SessionOptions::default`] and set the fields you need.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
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
    /// A folder for render caches, which many projects may share: each project gets its own
    /// directory inside it ([`Project::cache_dir_in`]) holding its rendered frames (when the
    /// output is only a video), decoded video input and the record of what was rendered.
    /// [`Session::render_shot`] spills frames beyond the memory budgets into it. `None` uses
    /// [`Project::default_cache_dir`] for projects and the user's cache directory for
    /// [`Session::render_shot`].
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

/// Owns a compute device and renders projects on it. Opening a GPU device and compiling the
/// kernels is slow, so a host (the command line, a GUI) keeps one session for many renders.
///
/// A session is `Send + Sync`: renders take `&self`, so several can run at once from different
/// threads (they then share the device).
pub struct Session {
    device: ComputeDevice,
    options: SessionOptions,
    /// ffmpeg, found on first use.
    ffmpeg: OnceLock<Result<Ffmpeg, String>>,
}

impl Session {
    /// Opens the device for `backend` (`Auto` falls back to the CPU without a GPU; the reason
    /// is in [`ComputeDevice::fallback_reason`]).
    pub fn new(backend: Backend) -> Result<Self, Error> {
        Ok(Self::with_device(ComputeDevice::new(backend)?))
    }

    /// Uses an existing device (for example one shared with a GUI).
    pub fn with_device(device: ComputeDevice) -> Self {
        Self { device, options: SessionOptions::default(), ffmpeg: OnceLock::new() }
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

    /// The render cache directory of `project` (see [`SessionOptions::cache_dir`]).
    pub fn project_cache_dir(&self, project: &Project) -> PathBuf {
        match &self.options.cache_dir {
            Some(root) => project.cache_dir_in(root),
            None => project.default_cache_dir(),
        }
    }

    /// ffmpeg, located on first use (see [`Ffmpeg::locate`]).
    pub fn ffmpeg(&self) -> Result<Ffmpeg, Error> {
        self.ffmpeg
            .get_or_init(|| Ffmpeg::locate().map_err(|e| e.to_string()))
            .clone()
            .map_err(|msg| Error::Media(MediaError::FfmpegMissing(msg)))
    }

    /// Brings `project`'s output up to date: renders the frames whose inputs or settings
    /// changed since the last render, keeps the rest, and encodes the output video when any
    /// frame changed. The first render of a project renders everything.
    ///
    /// The render cache ([`Session::project_cache_dir`], by default
    /// [`Project::default_cache_dir`]) remembers what was rendered from which inputs. Editing
    /// one keyframe re-renders only the frames between its neighbouring keyframes.
    pub fn render(
        &self,
        project: &Project,
        progress: Option<&(dyn Fn(&Event) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<ProjectStats, Error> {
        self.render_with(project, &RenderRequest::default(), progress, cancel)
    }

    /// [`render`](Self::render) with a frame range, forced keyframes or a full re-render.
    pub fn render_with(
        &self,
        project: &Project,
        request: &RenderRequest,
        progress: Option<&(dyn Fn(&Event) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<ProjectStats, Error> {
        let report = |e: &Event| {
            if let Some(p) = progress {
                p(e);
            }
        };
        render_project(self, project, request, &report, cancel)
    }

    /// Renders frames `range` (all of the project's frames when `None`) of `project` into
    /// `sink`, without the render cache's bookkeeping. Video inputs are still decoded into the
    /// cache directory.
    pub fn render_to(
        &self,
        project: &Project,
        range: Option<(usize, usize)>,
        sink: &dyn FrameSink,
        progress: Option<&(dyn Fn(&Progress) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<RenderStats, Error> {
        let project = &project.absolute();
        project.validate()?;
        let params = project.video_params()?;
        let cache_dir = self.project_cache_dir(project);
        let mut files = crate::hashing::FileHashes::default();
        let ffmpeg = || self.ffmpeg();
        let mut opener = Opener {
            cache_dir: &cache_dir,
            files: &mut files,
            ffmpeg: &ffmpeg,
            report: &|_| {},
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        };
        let prepared = prepare(project, &mut opener)?;
        let shot = Shot {
            frames: &prepared.frames.seq,
            keyframes: prepared.keyframes.iter().map(|(k, _)| k.clone()).collect(),
            mask: prepared.mask.as_ref().map(|m| &m.seq as &dyn FrameSource),
            range: Some(range.unwrap_or((prepared.first, prepared.last))),
        };
        let inputs: Vec<&Input> = std::iter::once(&prepared.frames).chain(prepared.mask.as_ref()).collect();
        let keyframes16 = prepared.keyframes16.as_deref();
        let result = with_decodes(&inputs, &|_| {}, cancel, || {
            self.run_shot(&shot, params, keyframes16, &cache_dir, sink, progress, cancel)
        });
        // Remove the cache directory if the render left it empty.
        if self.options.cache_dir.is_none() {
            let _ = std::fs::remove_dir(&cache_dir);
            let _ = std::fs::remove_dir(project.base_dir().join(".freesynth-cache"));
        }
        result
    }

    /// Runs `shot` (at 16 bits with `keyframes16`), spilling into `cache_dir/spill`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_shot(
        &self,
        shot: &Shot<'_>,
        params: VideoParams,
        keyframes16: Option<&[Keyframe16]>,
        cache_dir: &Path,
        sink: &dyn FrameSink,
        progress: Option<&(dyn Fn(&Progress) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<RenderStats, Error> {
        let options = self.options.render_options(Some(cache_dir.join("spill")));
        self.run(shot, params, keyframes16, options, sink, progress, cancel)
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
        let spill_dir = self.options.cache_dir.clone().or_else(crate::paths::user_cache_dir).map(|d| d.join("spill"));
        self.run(shot, params, None, self.options.render_options(spill_dir), sink, progress, cancel)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        shot: &Shot<'_>,
        params: VideoParams,
        keyframes16: Option<&[Keyframe16]>,
        options: RenderOptions,
        sink: &dyn FrameSink,
        progress: Option<&(dyn Fn(&Progress) + Sync)>,
        cancel: &CancelToken,
    ) -> Result<RenderStats, Error> {
        let mut renderer = Renderer::new(&self.device, params).options(options).cancel_token(cancel.clone());
        if let Some(p) = progress {
            renderer = renderer.on_progress(p);
        }
        if let Some(k) = keyframes16 {
            renderer = renderer.keyframes16(k);
        }
        Ok(renderer.render(shot, sink)?)
    }
}

/// A host keeps a session in an `Arc` and renders from worker threads.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Session>();
};
