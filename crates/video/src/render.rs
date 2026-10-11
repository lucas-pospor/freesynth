//! The renderer: runs a shot's keyframe chains concurrently, blends their passes and writes the
//! output frames.

use std::collections::VecDeque;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use freesynth_core::{GuideLayer, Image, SynthesisInput, io};
use freesynth_flow::GrayImage;
use freesynth_flow::track::{PositionField, Tracker};

use crate::blend::blend;
use crate::device::{ComputeDevice, FlowProvider, Synthesizer};
use crate::error::{VideoError, invalid, panicked};
use crate::flowbank::{BankEstimator, FlowBank, FlowStats};
use crate::frames::{FrameCache, FrameSink, FrameSource};
use crate::guides::{alpha_as_gray, confidence_weights, encode_positions, to_rgba, warp_rgba};
use crate::params::{FlowBackend, KeyframeOutput, RenderOptions, VideoParams};
use crate::plan::{Chain, ChainOutput, Pass, Plan};
use crate::progress::{CancelToken, Progress};
use crate::resources::ResourcePlan;
use crate::store::{SpillKey, SpillStore, StoreStats};

/// A painted version of one frame of the shot.
#[derive(Clone, Debug)]
pub struct Keyframe {
    /// The frame it stylizes.
    pub frame: usize,
    /// The painting (RGBA, the frame's size). Transparent pixels make the output RGBA too.
    pub style: Image,
}

/// What to render: the video frames, the keyframes and an optional mask sequence.
pub struct Shot<'a> {
    /// The input video frames (the appearance guide).
    pub frames: &'a dyn FrameSource,
    /// The keyframes, in any order.
    pub keyframes: Vec<Keyframe>,
    /// Optional per-frame masks (white = subject). Adds the mask guide.
    pub mask: Option<&'a dyn FrameSource>,
    /// The frames to render, `first..=last`; `None` renders every frame of `frames`. Every
    /// keyframe must lie inside the range.
    pub range: Option<(usize, usize)>,
}

/// What a render did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderStats {
    /// Description of the device.
    pub device: String,
    /// Output frames written.
    pub frames: usize,
    /// Syntheses run.
    pub syntheses: usize,
    /// Keyframe chains.
    pub chains: usize,
    /// Chains run at the same time.
    pub parallel: usize,
    /// How the memory budget was split (parallelism, cache budgets, estimated peak).
    pub resources: ResourcePlan,
    /// Wall-clock time of the render, in seconds.
    pub seconds: f64,
    /// Time spent in synthesis calls, summed over threads, in seconds.
    pub synthesis_seconds: f64,
    /// Time spent tracking positions and building guides (flow included), summed over threads.
    pub guide_seconds: f64,
    /// Flow computation and reuse.
    pub flow: FlowStats,
    /// Decoded input frames: memory peak and spills.
    pub frame_cache: StoreStats,
    /// Synthesized frames waiting for their blend partner: memory peak and spills.
    pub buffers: StoreStats,
}

/// Key of a synthesized frame waiting for its blend partner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PassKey {
    interval: usize,
    pass: Pass,
    frame: usize,
}

impl SpillKey for PassKey {
    fn file_stem(&self) -> String {
        let pass = match self.pass {
            Pass::A => 'a',
            Pass::B => 'b',
        };
        format!("i{:04}_{pass}_{:06}", self.interval, self.frame)
    }
}

enum Task {
    Keyframe(usize),
    Chain(usize),
}

/// Numbers the renders of this process, to give each its own spill directory.
static NEXT_RENDER: AtomicUsize = AtomicUsize::new(0);

/// Parent spill directories that renders of this process found missing (and so will create),
/// with the number of live renders using each. The last render to finish removes the parent if it
/// is empty, so concurrent renders sharing a new parent never leave it behind.
static CREATED_PARENTS: Mutex<Vec<(PathBuf, usize)>> = Mutex::new(Vec::new());

/// The spill directory of one render: a fresh subdirectory of [`RenderOptions::spill_dir`],
/// removed on drop (after the stores in it removed their files), together with the parent
/// directory when a render of this process created it and it is empty.
struct SpillDirs {
    dir: Option<PathBuf>,
    /// The parent, when it is counted in [`CREATED_PARENTS`].
    created_parent: Option<PathBuf>,
}

impl SpillDirs {
    fn new(parent: Option<&PathBuf>) -> Self {
        let Some(parent) = parent else { return Self { dir: None, created_parent: None } };
        let id = NEXT_RENDER.fetch_add(1, Ordering::Relaxed);
        let mut created = CREATED_PARENTS.lock().unwrap_or_else(|e| e.into_inner());
        let created_parent = if let Some(entry) = created.iter_mut().find(|(p, _)| p == parent) {
            entry.1 += 1;
            Some(parent.clone())
        } else if !parent.exists() {
            created.push((parent.clone(), 1));
            Some(parent.clone())
        } else {
            None
        };
        Self { dir: Some(parent.join(format!("render-{}-{id}", std::process::id()))), created_parent }
    }

    /// The directory for one store.
    fn store(&self, name: &str) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join(name))
    }
}

impl Drop for SpillDirs {
    fn drop(&mut self) {
        // Only succeeds when empty: files of a store still alive are never removed here.
        if let Some(dir) = &self.dir {
            let _ = fs::remove_dir(dir);
        }
        if let Some(parent) = &self.created_parent {
            let mut created = CREATED_PARENTS.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = created.iter().position(|(p, _)| p == parent) {
                created[i].1 -= 1;
                if created[i].1 == 0 {
                    created.swap_remove(i);
                    let _ = fs::remove_dir(parent);
                }
            }
        }
    }
}

/// Sets a keyframe's result to "failed" if its task ends without a result, so that chains
/// waiting for it do not wait forever.
struct KeyframeGuard<'a>(&'a OnceLock<Option<Arc<Image>>>);

impl Drop for KeyframeGuard<'_> {
    fn drop(&mut self) {
        let _ = self.0.set(None);
    }
}

/// State shared by the threads of one render.
struct Job<'s> {
    plan: Plan,
    size: (usize, usize),
    frames: FrameCache<'s>,
    masks: Option<FrameCache<'s>>,
    /// Keyframe paintings (RGBA), in plan order.
    styles: Vec<Arc<Image>>,
    /// The mapping guide of the keyframe itself (identity positions).
    identity_map: Image,
    flows: FlowBank,
    buffers: SpillStore<PassKey>,
    keyframe_results: Vec<OnceLock<Option<Arc<Image>>>>,
    tasks: Mutex<VecDeque<Task>>,
    stop: AtomicBool,
    error: Mutex<Option<VideoError>>,
    synthesized: AtomicUsize,
    written: AtomicUsize,
    synthesis_nanos: AtomicU64,
    guide_nanos: AtomicU64,
}

impl Job<'_> {
    /// Records the first error and stops every thread.
    fn fail(&self, e: VideoError) {
        self.stop.store(true, Ordering::Relaxed);
        let mut slot = self.error.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() && !matches!(e, VideoError::Cancelled) {
            *slot = Some(e);
        }
    }

    fn next_task(&self) -> Option<Task> {
        self.tasks.lock().unwrap_or_else(|e| e.into_inner()).pop_front()
    }
}

/// Sends a finished frame to the writers.
fn emit(tx: &SyncSender<(usize, Arc<Image>)>, frame: usize, image: Arc<Image>) -> Result<(), VideoError> {
    tx.send((frame, image)).map_err(|_| VideoError::Cancelled)
}

/// Runs shots on a device; see the [crate documentation](crate).
pub struct Renderer<'a> {
    synthesizer: &'a dyn Synthesizer,
    flows: &'a dyn FlowProvider,
    device_name: String,
    params: VideoParams,
    options: RenderOptions,
    progress: Option<&'a (dyn Fn(&Progress) + Sync)>,
    cancel: CancelToken,
}

impl<'a> Renderer<'a> {
    /// A renderer on `device` with the given look.
    pub fn new(device: &'a ComputeDevice, params: VideoParams) -> Self {
        let mut r = Self::with_engines(device, device, params);
        r.device_name = device.name().to_owned();
        r
    }

    /// A renderer with a custom synthesis engine and flow provider.
    pub fn with_engines(synthesizer: &'a dyn Synthesizer, flows: &'a dyn FlowProvider, params: VideoParams) -> Self {
        Self {
            synthesizer,
            flows,
            device_name: String::from("custom"),
            params,
            options: RenderOptions::default(),
            progress: None,
            cancel: CancelToken::new(),
        }
    }

    /// Sets the memory and parallelism options.
    pub fn options(mut self, options: RenderOptions) -> Self {
        self.options = options;
        self
    }

    /// Calls `f` with progress reports (from several threads).
    pub fn on_progress(mut self, f: &'a (dyn Fn(&Progress) + Sync)) -> Self {
        self.progress = Some(f);
        self
    }

    /// Stops the render when `token` is cancelled.
    pub fn cancel_token(mut self, token: CancelToken) -> Self {
        self.cancel = token;
        self
    }

    /// The look parameters.
    pub fn params(&self) -> &VideoParams {
        &self.params
    }

    /// Calls the progress callback; a panic in it becomes an error.
    fn report(&self, p: Progress) -> Result<(), VideoError> {
        match self.progress {
            Some(f) => catch_unwind(AssertUnwindSafe(|| f(&p))).map_err(|e| panicked("the progress callback", &*e)),
            None => Ok(()),
        }
    }

    /// Renders `shot` into `sink`. Every frame of the range is written exactly once, in no
    /// particular order.
    pub fn render(&self, shot: &Shot<'_>, sink: &dyn FrameSink) -> Result<RenderStats, VideoError> {
        let start = Instant::now();
        self.params.validate()?;
        let (first, last) = shot.range.unwrap_or((shot.frames.first(), shot.frames.last()));
        if shot.frames.is_empty() || !shot.frames.contains(first) || !shot.frames.contains(last) {
            return invalid(format!("frames {first}-{last} are not all in the input"));
        }
        if let Some(m) = shot.mask
            && (!m.contains(first) || !m.contains(last))
        {
            return invalid(format!("the mask sequence does not cover frames {first}-{last}"));
        }
        let mut keys = shot.keyframes.clone();
        keys.sort_by_key(|k| k.frame);
        let plan = Plan::new(first, last, &keys.iter().map(|k| k.frame).collect::<Vec<_>>())?;

        let o = &self.options;
        // Declared before the stores, so that it is dropped after them.
        let spill = SpillDirs::new(o.spill_dir.as_ref());
        let mut frames = FrameCache::new(shot.frames, o.frame_cache_bytes, spill.store("frames"));
        let size = frames.get(plan.keyframes[0])?.size();
        for k in &keys {
            if k.style.channels() != 4 || k.style.size() != size {
                return invalid(format!(
                    "keyframe {} must be an RGBA image of the frame size {}x{}",
                    k.frame, size.0, size.1
                ));
            }
        }
        let resources = o.resources(&plan, size, shot.mask.is_some());
        let parallel = resources.parallel;
        frames.set_budget(resources.frame_cache_bytes);

        let cpu = ComputeDevice::cpu();
        let flow_provider: &dyn FlowProvider = match o.flow_backend {
            FlowBackend::Device => self.flows,
            FlowBackend::Cpu => &cpu,
        };
        // Keyframes first: a chain waits for its keyframe's frame, which must already be taken.
        let mut tasks: VecDeque<Task> = (0..plan.keyframes.len()).map(Task::Keyframe).collect();
        tasks.extend((0..plan.chains.len()).map(Task::Chain));
        let job = Job {
            size,
            masks: shot.mask.map(|m| FrameCache::new(m, resources.frame_cache_bytes, spill.store("masks"))),
            styles: keys.into_iter().map(|k| Arc::new(k.style)).collect(),
            identity_map: encode_positions(&PositionField::identity(size.0, size.1), size),
            flows: FlowBank::new(flow_provider, size.0, size.1, &self.params.flow, parallel.min(4), resources.flow_cache_flows)?,
            buffers: SpillStore::new(resources.buffer_bytes, spill.store("passes")),
            keyframe_results: (0..plan.keyframes.len()).map(|_| OnceLock::new()).collect(),
            tasks: Mutex::new(tasks),
            stop: AtomicBool::new(false),
            error: Mutex::new(None),
            synthesized: AtomicUsize::new(0),
            written: AtomicUsize::new(0),
            synthesis_nanos: AtomicU64::new(0),
            guide_nanos: AtomicU64::new(0),
            frames,
            plan,
        };
        self.report(Progress::Started {
            frames: job.plan.num_frames(),
            syntheses: job.plan.num_syntheses(),
            chains: job.plan.chains.len(),
            parallel,
        })?;

        let (tx, rx) = sync_channel::<(usize, Arc<Image>)>(2 * parallel);
        let rx = Mutex::new(rx);
        let (job_ref, rx) = (&job, &rx);
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(move || self.writer(job_ref, rx, sink));
            }
            for _ in 0..parallel {
                let tx = tx.clone();
                s.spawn(move || self.worker(job_ref, tx));
            }
            drop(tx);
        });

        let written = job.written.load(Ordering::Relaxed);
        let stats = RenderStats {
            device: self.device_name.clone(),
            frames: written,
            syntheses: job.synthesized.load(Ordering::Relaxed),
            chains: job.plan.chains.len(),
            parallel,
            resources,
            seconds: start.elapsed().as_secs_f64(),
            synthesis_seconds: job.synthesis_nanos.load(Ordering::Relaxed) as f64 * 1e-9,
            guide_seconds: job.guide_nanos.load(Ordering::Relaxed) as f64 * 1e-9,
            flow: job.flows.stats(),
            frame_cache: job.frames.stats(),
            buffers: job.buffers.stats(),
        };
        let expected = job.plan.num_frames();
        let error = job.error.lock().unwrap_or_else(|e| e.into_inner()).take();
        // Dropping the stores deletes their spill files and directories, then the render's own.
        drop(job);
        drop(spill);
        if self.cancel.is_cancelled() {
            return Err(VideoError::Cancelled);
        }
        if let Some(e) = error {
            return Err(e);
        }
        if written != expected {
            return invalid(format!("internal error: {written} of {expected} frames were written"));
        }
        Ok(stats)
    }

    /// Hands finished frames to the sink (two of these run, so encoding overlaps).
    fn writer(&self, job: &Job<'_>, rx: &Mutex<Receiver<(usize, Arc<Image>)>>, sink: &dyn FrameSink) {
        loop {
            let next = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
            let Ok((frame, image)) = next else { return };
            if job.stop.load(Ordering::Relaxed) {
                continue; // keep draining so that workers never block
            }
            // A panicking sink must not end this thread: workers would block on the channel.
            let written = catch_unwind(AssertUnwindSafe(|| sink.write(frame, &image)))
                .unwrap_or_else(|e| Err(panicked("the frame sink", &*e)))
                .and_then(|()| {
                    let done = job.written.fetch_add(1, Ordering::Relaxed) + 1;
                    self.report(Progress::Written { frame, done, total: job.plan.num_frames() })
                });
            if let Err(e) = written {
                job.fail(e);
            }
        }
    }

    fn worker(&self, job: &Job<'_>, tx: SyncSender<(usize, Arc<Image>)>) {
        while !job.stop.load(Ordering::Relaxed) {
            let Some(task) = job.next_task() else { break };
            // A panic (in a frame source, the flow estimator, ...) stops the render with an error.
            let result = catch_unwind(AssertUnwindSafe(|| match task {
                Task::Keyframe(i) => self.keyframe(job, i, &tx),
                Task::Chain(c) => self.chain(job, &job.plan.chains[c], &tx),
            }))
            .unwrap_or_else(|e| Err(panicked("a render task", &*e)));
            if let Err(e) = result {
                job.fail(e);
            }
        }
    }

    /// Stops between frames when cancelled or when another thread failed.
    fn check(&self, job: &Job<'_>) -> Result<(), VideoError> {
        if self.cancel.is_cancelled() {
            job.stop.store(true, Ordering::Relaxed);
        }
        if job.stop.load(Ordering::Relaxed) { Err(VideoError::Cancelled) } else { Ok(()) }
    }

    /// The synthesis input for frame `t` of a chain from keyframe `key` (an index into the
    /// plan's keyframes).
    fn input(
        &self,
        job: &Job<'_>,
        key: usize,
        t: usize,
        temporal: Option<Image>,
        positions: Image,
        confidence: Option<Image>,
    ) -> Result<SynthesisInput, VideoError> {
        let w = &self.params.weights;
        let style = &job.styles[key];
        let key_video = job.frames.get(job.plan.keyframes[key])?;
        let mut guides = vec![GuideLayer::new((*key_video).clone(), (*job.frames.get(t)?).clone()).with_weight(w.video)];
        if let Some(warped) = temporal.filter(|_| w.temporal > 0.0) {
            guides.push(GuideLayer::new((**style).clone(), warped).with_weight(w.temporal));
        }
        if w.mapping > 0.0 {
            let mut layer = GuideLayer::new(job.identity_map.clone(), positions).with_weight(w.mapping);
            if let Some(c) = confidence {
                layer = layer.with_modulation(c);
            }
            guides.push(layer);
        }
        if let Some(masks) = job.masks.as_ref().filter(|_| w.mask > 0.0) {
            let mask = masks.get(t)?;
            if mask.size() != job.size {
                return invalid(format!("mask {t} does not have the frame size"));
            }
            guides.push(GuideLayer::new(alpha_as_gray(style), (*mask).clone()).with_weight(w.mask));
        }
        SynthesisInput::from_rgba_layers(style, Some(w.style), &guides).map_err(|e| VideoError::InvalidInput(e.to_string()))
    }

    fn synthesize(&self, job: &Job<'_>, input: &SynthesisInput, frame: usize, key: usize) -> Result<Arc<Image>, VideoError> {
        let start = Instant::now();
        let out = catch_unwind(AssertUnwindSafe(|| self.synthesizer.synthesize(input, &self.params.synthesis)))
            .unwrap_or_else(|e| Err(panicked("the synthesis engine", &*e)))?;
        job.synthesis_nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let done = job.synthesized.fetch_add(1, Ordering::Relaxed) + 1;
        self.report(Progress::Synthesized { frame, keyframe: job.plan.keyframes[key], done, total: job.plan.num_syntheses() })?;
        Ok(Arc::new(out.image))
    }

    /// Synthesizes keyframe `key`'s own frame, which starts the chains on both sides.
    fn keyframe(&self, job: &Job<'_>, key: usize, tx: &SyncSender<(usize, Arc<Image>)>) -> Result<(), VideoError> {
        let slot = &job.keyframe_results[key];
        let _guard = KeyframeGuard(slot);
        self.check(job)?;
        let k = job.plan.keyframes[key];
        let input = self.input(job, key, k, None, job.identity_map.clone(), None)?;
        let out = self.synthesize(job, &input, k, key)?;
        let _ = slot.set(Some(Arc::clone(&out)));
        let image = match self.params.keyframe_output {
            KeyframeOutput::Synthesized => out,
            KeyframeOutput::Painting => {
                let style = &job.styles[key];
                Arc::new(io::select_channels(style, io::eval_num_channels(style)))
            }
        };
        emit(tx, k, image)
    }

    /// Runs one chain from its keyframe's frame to its end.
    fn chain(&self, job: &Job<'_>, chain: &Chain, tx: &SyncSender<(usize, Arc<Image>)>) -> Result<(), VideoError> {
        let k = job.plan.keyframes[chain.keyframe];
        let Some(key_frame) = job.keyframe_results[chain.keyframe].wait().clone() else {
            return Err(VideoError::Cancelled); // the keyframe failed and reported why
        };
        let mut prev_gray: GrayImage = job.frames.gray(k)?;
        let mut tracker = Tracker::new(BankEstimator { bank: &job.flows }, prev_gray.clone(), self.params.tracking)?;
        let mut prev = to_rgba(&key_frame);
        for t in chain.frames() {
            self.check(job)?;
            let start = Instant::now();
            let gray = job.frames.gray(t)?;
            tracker.step(&gray)?;
            let positions = encode_positions(tracker.positions(), job.size);
            let confidence = self.params.gate_mapping.then(|| confidence_weights(tracker.confidence()));
            let temporal = (self.params.weights.temporal > 0.0)
                .then(|| job.flows.flow(&gray, &prev_gray).map(|flow| warp_rgba(&prev, &flow)))
                .transpose()?;
            job.guide_nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let input = self.input(job, chain.keyframe, t, temporal, positions, confidence)?;
            let out = self.synthesize(job, &input, t, chain.keyframe)?;
            prev = to_rgba(&out);
            prev_gray = gray;
            match chain.output {
                ChainOutput::Direct => emit(tx, t, out)?,
                ChainOutput::Blend { interval, pass } => {
                    let other = match pass {
                        Pass::A => Pass::B,
                        Pass::B => Pass::A,
                    };
                    let key = PassKey { interval, pass, frame: t };
                    let partner = PassKey { interval, pass: other, frame: t };
                    if let Some(found) = job.buffers.insert_or_take(key, Arc::clone(&out), &partner)? {
                        let (a, b) = if pass == Pass::A { (&out, &found) } else { (&found, &out) };
                        let alpha = job.plan.intervals[interval].alpha(t);
                        emit(tx, t, Arc::new(blend(self.params.blend, a, b, alpha)))?;
                    }
                }
            }
        }
        Ok(())
    }
}
