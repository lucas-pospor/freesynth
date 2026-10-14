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

use freesynth_core::{
    GuideLayer, Image, Image16, Nnf, Params, PerLevel, SynthesisExtras, SynthesisInput, WarmStart, io,
};
use freesynth_flow::track::{ConfidenceMap, PositionField, Tracker};
use freesynth_flow::{FlowField, GrayImage};
use rayon::prelude::*;

use crate::blend::{Blender, BuiltinBlender, PassFrame};
use crate::device::{ComputeDevice, FlowProvider, Synthesizer};
use crate::error::{VideoError, invalid, panicked};
use crate::flowbank::{BankEstimator, FlowBank, FlowStats};
use crate::frames::{FrameCache, FrameSink, FrameSource};
use crate::guides::{
    alpha_as_gray, confidence_weights, encode_positions, flow_consistency_weights, to_rgba, warp_nnf, warp_rgba,
};
use crate::params::{FlowBackend, KeyframeOutput, RenderOptions, VideoParams, WarmStartSettings};
use crate::plan::{Chain, ChainOutput, Pass, Plan};
use crate::progress::{CancelToken, Progress};
use crate::resources::{ResourcePlan, ShotFormat};
use crate::store::{SpillKey, SpillStore, StoreStats};

/// A painted version of one frame of the shot.
#[derive(Clone, Debug)]
pub struct Keyframe {
    /// The frame it stylizes.
    pub frame: usize,
    /// The painting (RGBA, the frame's size). Transparent pixels make the output RGBA too.
    pub style: Image,
}

/// A keyframe painting at 16 bits per channel, for 16-bit output
/// ([`Renderer::keyframes16`]).
#[derive(Clone, Debug)]
pub struct Keyframe16 {
    /// The frame it stylizes (a frame of one of the shot's [`Keyframe`]s).
    pub frame: usize,
    /// The painting (RGBA, the frame's size). The shot's 8-bit keyframe should be this painting
    /// rounded to 8 bits ([`Image16::to_image`]): the search runs on that one.
    pub style: Image16,
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

/// A synthesized frame: 8 bits per channel, or 16 for a render with 16-bit keyframes.
#[derive(Clone, Debug)]
enum Frame {
    Eight(Arc<Image>),
    Sixteen(Arc<Image16>),
}

impl Frame {
    fn channels(&self) -> usize {
        match self {
            Frame::Eight(i) => i.channels(),
            Frame::Sixteen(i) => i.channels(),
        }
    }

    fn size(&self) -> (usize, usize) {
        match self {
            Frame::Eight(i) => i.size(),
            Frame::Sixteen(i) => i.size(),
        }
    }
}

/// Packs a pass frame for the pass buffer (an 8-bit image of bytes): the frame's samples (one
/// byte each, or two little-endian bytes for 16-bit frames) followed, when `with_errors`, by
/// two bytes holding each pixel's error as a little-endian `u16` in 1/256 grey levels (missing
/// errors are stored as 0, so both passes of a frame carry the same layout).
fn pack_pass(frame: &Frame, errors: Option<&[f32]>, with_errors: bool) -> Image {
    if let (Frame::Eight(image), false) = (frame, with_errors) {
        return (**image).clone();
    }
    let ((w, h), nc) = (frame.size(), frame.channels());
    let sample_bytes = if matches!(frame, Frame::Sixteen(_)) { 2 } else { 1 };
    let stride = nc * sample_bytes + if with_errors { 2 } else { 0 };
    let mut data = Vec::with_capacity(w * h * stride);
    for i in 0..w * h {
        match frame {
            Frame::Eight(image) => data.extend_from_slice(&image.data()[i * nc..(i + 1) * nc]),
            Frame::Sixteen(image) => {
                data.extend(image.data()[i * nc..(i + 1) * nc].iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        if with_errors {
            let e = errors.map_or(0.0, |e| e[i]);
            data.extend_from_slice(&((e * 256.0).round().clamp(0.0, 65535.0) as u16).to_le_bytes());
        }
    }
    Image::from_raw(w, h, stride, data)
}

/// Reverses [`pack_pass`].
fn unpack_pass(stored: &Image, with_errors: bool, sixteen_bit: bool) -> (Frame, Option<Vec<f32>>) {
    if !with_errors && !sixteen_bit {
        return (Frame::Eight(Arc::new(stored.clone())), None);
    }
    let (w, h) = stored.size();
    let stride = stored.channels();
    let sample_bytes = stride - if with_errors { 2 } else { 0 };
    let mut errors = with_errors.then(|| Vec::with_capacity(w * h));
    let px = stored.data().chunks_exact(stride);
    let frame = if sixteen_bit {
        let mut data = Vec::with_capacity(w * h * sample_bytes / 2);
        for p in px {
            data.extend(p[..sample_bytes].as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b)));
            if let Some(e) = errors.as_mut() {
                e.push(f32::from(u16::from_le_bytes([p[sample_bytes], p[sample_bytes + 1]])) / 256.0);
            }
        }
        Frame::Sixteen(Arc::new(Image16::from_raw(w, h, sample_bytes / 2, data)))
    } else {
        let mut data = Vec::with_capacity(w * h * sample_bytes);
        for p in px {
            data.extend_from_slice(&p[..sample_bytes]);
            if let Some(e) = errors.as_mut() {
                e.push(f32::from(u16::from_le_bytes([p[sample_bytes], p[sample_bytes + 1]])) / 256.0);
            }
        }
        Frame::Eight(Arc::new(Image::from_raw(w, h, sample_bytes, data)))
    };
    (frame, errors)
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

/// What a keyframe's own synthesis leaves for the chains starting from it: the frame (the first
/// temporal guide) and, with a warm start, its final field.
struct KeyResult {
    image: Image,
    nnf: Option<Nnf>,
}

/// Sets a keyframe's result to "failed" if its task ends without a result, so that chains
/// waiting for it do not wait forever.
struct KeyframeGuard<'a>(&'a OnceLock<Option<Arc<KeyResult>>>);

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
    /// For 16-bit output: the 16-bit paintings with the channels the engine keeps of the 8-bit
    /// ones, in plan order.
    styles16: Option<Vec<Arc<Image16>>>,
    /// The mapping guide of the keyframe itself (identity positions).
    identity_map: Image,
    flows: FlowBank,
    /// Combines the two passes of a frame.
    blender: &'s dyn Blender,
    buffers: SpillStore<PassKey>,
    keyframe_results: Vec<OnceLock<Option<Arc<KeyResult>>>>,
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

/// Blends the two passes of a frame (packed by [`pack_pass`]) with the render's blender and
/// writes the result to `sink`. The passes are unpacked only here, so that frames queued for
/// the writers stay in their compact form.
fn blend_and_write(
    job: &Job<'_>,
    sink: &dyn FrameSink,
    frame: usize,
    (a, b): (&Image, &Image),
    alpha: f32,
) -> Result<(), VideoError> {
    let (with_errors, sixteen_bit) = (job.blender.needs_errors(), job.styles16.is_some());
    let ((a, ea), (b, eb)) = (unpack_pass(a, with_errors, sixteen_bit), unpack_pass(b, with_errors, sixteen_bit));
    let (ea, eb, blender) = (ea.as_deref(), eb.as_deref(), job.blender);
    match (a, b) {
        (Frame::Eight(a), Frame::Eight(b)) => {
            let out =
                blender.blend(PassFrame { image: &*a, errors: ea }, PassFrame { image: &*b, errors: eb }, alpha)?;
            sink.write(frame, &out)
        }
        (Frame::Sixteen(a), Frame::Sixteen(b)) => {
            let out =
                blender.blend16(PassFrame { image: &*a, errors: ea }, PassFrame { image: &*b, errors: eb }, alpha)?;
            sink.write16(frame, &out)
        }
        _ => invalid("internal error: the passes of a frame differ in bit depth"),
    }
}

/// What a chain hands to the writers for one frame.
enum Output {
    /// The finished frame.
    Ready(Frame),
    /// The two passes of the frame (A from the interval's first keyframe, B from its last) as
    /// packed for the pass buffer ([`pack_pass`]), and the position in the interval, to be
    /// blended by the writer.
    Blend { a: Arc<Image>, b: Arc<Image>, alpha: f32 },
}

/// Sends a finished frame to the writers.
fn emit(tx: &SyncSender<(usize, Output)>, frame: usize, image: Frame) -> Result<(), VideoError> {
    tx.send((frame, Output::Ready(image))).map_err(|_| VideoError::Cancelled)
}

/// One synthesis's results as the renderer uses them.
struct Synthesized {
    /// The 8-bit frame (the next frame's temporal guide).
    image: Arc<Image>,
    /// The frame to output: `image`, or its 16-bit counterpart.
    output: Frame,
    /// Root-mean-square patch errors, when asked for.
    errors: Option<Vec<f32>>,
    /// The final field, when a warm start needs it.
    nnf: Option<Nnf>,
}

/// Turns the engine's per-pixel patch errors (weighted sums of squared differences over the
/// patch) into root-mean-square differences in grey levels ([`PassFrame::errors`]): each is
/// divided by the total weight its patch actually had, then square-rooted.
///
/// That weight is the sum, over the patch's pixels (with the engine's clamped addressing at the
/// border), of the style weights plus each guide channel's weight times its modulation at that
/// pixel ([`SynthesisInput::target_modulation`]). Dividing by the nominal weight instead would
/// shrink the errors wherever a guide is gated off, more so the further a pass gets from its
/// keyframe.
fn rms_patch_errors(errors: &mut [f32], input: &SynthesisInput, patch: usize) {
    let styles: f32 = input.style_weights.iter().sum();
    let guides: f32 = input.guide_weights.iter().sum();
    let (w, h) = input.target_guide.size();
    let area = (patch * patch) as f32;
    let Some(modulation) = input.target_modulation.as_ref().filter(|_| w * h == errors.len()) else {
        let scale = 1.0 / (area * (styles + guides).max(f32::MIN_POSITIVE));
        errors.iter_mut().for_each(|v| *v = (*v * scale).max(0.0).sqrt());
        return;
    };
    // Each pixel's own weight.
    let own: Vec<f32> = modulation
        .data()
        .par_chunks_exact(modulation.channels())
        .map(|px| styles + px.iter().zip(&input.guide_weights).map(|(m, g)| g * f32::from(*m) / 255.0).sum::<f32>())
        .collect();
    // Summed over the patch, separably (clamping each axis is clamping the patch).
    let r = patch / 2;
    let rows: Vec<f32> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (0..patch).map(|o| own[y * w + (x + o).saturating_sub(r).min(w - 1)]).sum()
        })
        .collect();
    errors.par_iter_mut().enumerate().for_each(|(i, v)| {
        let (x, y) = (i % w, i / w);
        let total: f32 = (0..patch).map(|o| rows[(y + o).saturating_sub(r).min(h - 1) * w + x]).sum();
        *v = (*v / total.max(f32::MIN_POSITIVE)).max(0.0).sqrt();
    });
}

/// The flow from a chain's frame to its previous frame and, with temporal gating, the flow back.
type FlowsToPrev = (Arc<FlowField>, Option<Arc<FlowField>>);

/// The mapping guide of one synthesis: tracked positions with their confidence (`None`: full
/// confidence).
struct Mapping<'p> {
    positions: &'p PositionField,
    confidence: Option<&'p ConfidenceMap>,
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
    blender: Option<&'a dyn Blender>,
    keyframes16: Option<&'a [Keyframe16]>,
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
            blender: None,
            keyframes16: None,
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

    /// Blends the passes of each interval with `blender` instead of the built-in blend that
    /// [`VideoParams::blend`] selects.
    pub fn blender(mut self, blender: &'a dyn Blender) -> Self {
        self.blender = Some(blender);
        self
    }

    /// Renders 16-bit frames from these 16-bit keyframe paintings (one for each of the shot's
    /// keyframes): the search runs on the shot's 8-bit keyframes as usual, and each synthesis's
    /// final field is voted again from the 16-bit painting
    /// ([`SynthesisExtras::style16`]); passes are blended at 16 bits ([`Blender::blend16`]) and
    /// frames go to [`FrameSink::write16`].
    pub fn keyframes16(mut self, keyframes: &'a [Keyframe16]) -> Self {
        self.keyframes16 = Some(keyframes);
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
        let styles16 = match self.keyframes16 {
            Some(k16) => Some(
                keys.iter()
                    .map(|k| {
                        let Some(p) = k16.iter().find(|p| p.frame == k.frame) else {
                            return invalid(format!("keyframe {} has no 16-bit painting", k.frame));
                        };
                        if p.style.channels() != 4 || p.style.size() != size {
                            return invalid(format!(
                                "the 16-bit painting of keyframe {} must be RGBA of the frame size {}x{}",
                                k.frame, size.0, size.1
                            ));
                        }
                        Ok(Arc::new(io::select_channels16(&p.style, io::eval_num_channels(&k.style))))
                    })
                    .collect::<Result<Vec<_>, VideoError>>()?,
            ),
            None => None,
        };
        let builtin = BuiltinBlender { mode: self.params.blend, settings: self.params.blending };
        let blender = self.blender.unwrap_or(&builtin);
        let format = ShotFormat {
            size,
            has_mask: shot.mask.is_some(),
            blend_bytes: blender.working_bytes(size.0, size.1),
            sixteen_bit: styles16.is_some(),
        };
        let resources = o.resources(&plan, &format);
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
            blender,
            masks: shot.mask.map(|m| FrameCache::new(m, resources.frame_cache_bytes, spill.store("masks"))),
            styles: keys.into_iter().map(|k| Arc::new(k.style)).collect(),
            styles16,
            identity_map: encode_positions(&PositionField::identity(size.0, size.1), size),
            flows: FlowBank::new(
                flow_provider,
                size.0,
                size.1,
                &self.params.flow,
                parallel.min(4),
                resources.flow_cache_flows,
            )?,
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

        let (tx, rx) = sync_channel::<(usize, Output)>(2 * parallel);
        let rx = Mutex::new(rx);
        let (job_ref, rx) = (&job, &rx);
        std::thread::scope(|s| {
            for _ in 0..resources.writers.max(1) {
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

    /// Blends frames whose passes are both done and hands finished frames to the sink
    /// ([`ResourcePlan::writers`] of these run, normally two, so that blending and
    /// encoding overlap with each other and with the synthesis).
    fn writer(&self, job: &Job<'_>, rx: &Mutex<Receiver<(usize, Output)>>, sink: &dyn FrameSink) {
        loop {
            let next = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
            let Ok((frame, output)) = next else { return };
            if job.stop.load(Ordering::Relaxed) {
                continue; // keep draining so that workers never block
            }
            // A panicking blender or sink must not end this thread: workers would block on the
            // channel.
            let written = catch_unwind(AssertUnwindSafe(|| match output {
                Output::Ready(Frame::Eight(image)) => sink.write(frame, &image),
                Output::Ready(Frame::Sixteen(image)) => sink.write16(frame, &image),
                Output::Blend { a, b, alpha } => blend_and_write(job, sink, frame, (&a, &b), alpha),
            }))
            .unwrap_or_else(|e| Err(panicked("the blend or the frame sink", &*e)))
            .and_then(|()| {
                let done = job.written.fetch_add(1, Ordering::Relaxed) + 1;
                self.report(Progress::Written { frame, done, total: job.plan.num_frames() })
            });
            if let Err(e) = written {
                job.fail(e);
            }
        }
    }

    fn worker(&self, job: &Job<'_>, tx: SyncSender<(usize, Output)>) {
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
        temporal: Option<(Image, Option<Image>)>,
        mapping: &Mapping<'_>,
    ) -> Result<SynthesisInput, VideoError> {
        let w = &self.params.weights;
        let style = &job.styles[key];
        let key_video = job.frames.get(job.plan.keyframes[key])?;
        let mut guides =
            vec![GuideLayer::new((*key_video).clone(), (*job.frames.get(t)?).clone()).with_weight(w.video)];
        if let Some((warped, weights)) = temporal.filter(|_| w.temporal > 0.0) {
            let mut layer = GuideLayer::new((**style).clone(), warped).with_weight(w.temporal);
            if let Some(m) = weights {
                layer = layer.with_modulation(m);
            }
            guides.push(layer);
        }
        if w.mapping > 0.0 {
            let positions = encode_positions(mapping.positions, job.size);
            let mut layer = GuideLayer::new(job.identity_map.clone(), positions).with_weight(w.mapping);
            if let Some(c) = mapping.confidence {
                layer = layer.with_modulation(confidence_weights(c));
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
        SynthesisInput::from_rgba_layers(style, Some(w.style), &guides)
            .map_err(|e| VideoError::InvalidInput(e.to_string()))
    }

    /// The extras of a synthesis for keyframe `key`: the 16-bit style (for 16-bit output) and
    /// the warm start.
    fn extras(&self, job: &Job<'_>, key: usize, warm: Option<Nnf>) -> SynthesisExtras {
        let levels = self.params.warm_start.map_or(1, |s| s.levels);
        SynthesisExtras {
            warm_start: warm.map(|nnf| WarmStart { nnf, levels }),
            style16: job.styles16.as_ref().map(|s| (*s[key]).clone()),
        }
    }

    /// The synthesis parameters of a warm-started frame: the warm start's search/vote
    /// iterations at the levels it runs.
    fn warm_params(&self, input: &SynthesisInput, s: WarmStartSettings) -> Result<Params, VideoError> {
        let base = &self.params.synthesis;
        let sched =
            base.schedule(input, self.synthesizer.engine()).map_err(|e| VideoError::InvalidInput(e.to_string()))?;
        let first = sched.num_levels.saturating_sub(s.levels);
        let iters = (0..sched.num_levels)
            .map(|l| if l >= first { s.search_vote_iters } else { sched.search_vote_iters[l] })
            .collect();
        Ok(Params { search_vote_iters: Some(PerLevel::Levels(iters)), ..base.clone() })
    }

    /// Synthesizes `input` for `frame` of keyframe `key`. With `errors`, also returns the
    /// engine's per-pixel patch errors as root-mean-square differences (see [`PassFrame::errors`]).
    fn synthesize(
        &self,
        job: &Job<'_>,
        input: &SynthesisInput,
        extras: &SynthesisExtras,
        frame: usize,
        key: usize,
        errors: bool,
    ) -> Result<Synthesized, VideoError> {
        let start = Instant::now();
        let base = match (self.params.warm_start, &extras.warm_start) {
            (Some(s), Some(_)) => self.warm_params(input, s)?,
            _ => self.params.synthesis.clone(),
        };
        let params = Params { return_errors: errors, ..base };
        let out = catch_unwind(AssertUnwindSafe(|| self.synthesizer.synthesize_with(input, &params, extras)))
            .unwrap_or_else(|e| Err(panicked("the synthesis engine", &*e)))?;
        job.synthesis_nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let done = job.synthesized.fetch_add(1, Ordering::Relaxed) + 1;
        self.report(Progress::Synthesized {
            frame,
            keyframe: job.plan.keyframes[key],
            done,
            total: job.plan.num_syntheses(),
        })?;
        let errors = out.errors.filter(|_| errors).map(|mut e| {
            let patch = if params.extra_pass_3x3 { 3 } else { params.patch_size };
            rms_patch_errors(&mut e, input, patch);
            e
        });
        let image = Arc::new(out.image);
        let output = match (&extras.style16, out.image16) {
            (None, _) => Frame::Eight(Arc::clone(&image)),
            (Some(_), Some(i)) => Frame::Sixteen(Arc::new(i)),
            // An engine without 16-bit support: the exact widening of its 8-bit result.
            (Some(_), None) => Frame::Sixteen(Arc::new(Image16::from_image(&image))),
        };
        let nnf = self.needs_fields().then_some(out.nnf);
        Ok(Synthesized { image, output, errors, nnf })
    }

    /// Synthesizes keyframe `key`'s own frame, which starts the chains on both sides.
    fn keyframe(&self, job: &Job<'_>, key: usize, tx: &SyncSender<(usize, Output)>) -> Result<(), VideoError> {
        let slot = &job.keyframe_results[key];
        let _guard = KeyframeGuard(slot);
        self.check(job)?;
        let k = job.plan.keyframes[key];
        let identity = PositionField::identity(job.size.0, job.size.1);
        let mapping = Mapping { positions: &identity, confidence: None };
        let input = self.input(job, key, k, None, &mapping)?;
        let extras = self.extras(job, key, None);
        let out = self.synthesize(job, &input, &extras, k, key, false)?;
        let _ = slot.set(Some(Arc::new(KeyResult { image: (*out.image).clone(), nnf: out.nnf })));
        let image = match (self.params.keyframe_output, &job.styles16) {
            (KeyframeOutput::Synthesized, _) => out.output,
            (KeyframeOutput::Painting, Some(styles16)) => Frame::Sixteen(Arc::clone(&styles16[key])),
            (KeyframeOutput::Painting, None) => {
                let style = &job.styles[key];
                Frame::Eight(Arc::new(io::select_channels(style, io::eval_num_channels(style))))
            }
        };
        emit(tx, k, image)
    }

    /// Whether chains keep each frame's final field (for a warm start).
    fn needs_fields(&self) -> bool {
        self.params.warm_start.is_some()
    }

    /// The flow from a chain's frame (`gray`) to its previous frame (`prev_gray`), and with
    /// temporal gating the flow back, when the temporal guide or the warm start needs them.
    fn flows_to_prev(
        &self,
        job: &Job<'_>,
        prev_gray: &GrayImage,
        gray: &GrayImage,
    ) -> Result<Option<FlowsToPrev>, VideoError> {
        let temporal = self.params.weights.temporal > 0.0;
        if temporal && self.params.gate_temporal {
            let (to_prev, from_prev) = job.flows.pair(gray, prev_gray)?;
            return Ok(Some((to_prev, Some(from_prev))));
        }
        if temporal || self.params.warm_start.is_some() {
            return Ok(Some((job.flows.flow(gray, prev_gray)?, None)));
        }
        Ok(None)
    }

    /// The temporal guide of a chain's frame: the chain's previous output (`prev`) warped to the
    /// frame by `to_prev`, with its per-pixel weights when the guide is gated (`from_prev` is the
    /// flow back).
    fn temporal_guide(
        &self,
        prev: &Image,
        to_prev: &FlowField,
        from_prev: Option<&FlowField>,
    ) -> (Image, Option<Image>) {
        let p = &self.params;
        let weights = from_prev
            .map(|from_prev| flow_consistency_weights(to_prev, from_prev, &p.tracking.check, p.temporal_gate_erosion));
        (warp_rgba(prev, to_prev), weights)
    }

    /// Runs one chain from its keyframe's frame to its end.
    fn chain(&self, job: &Job<'_>, chain: &Chain, tx: &SyncSender<(usize, Output)>) -> Result<(), VideoError> {
        let k = job.plan.keyframes[chain.keyframe];
        let Some(key_result) = job.keyframe_results[chain.keyframe].wait().clone() else {
            return Err(VideoError::Cancelled); // the keyframe failed and reported why
        };
        let mut prev_gray: GrayImage = job.frames.gray(k)?;
        let mut tracker = Tracker::new(BankEstimator { bank: &job.flows }, prev_gray.clone(), self.params.tracking)?;
        let mut prev = to_rgba(&key_result.image);
        let mut prev_nnf = key_result.nnf.clone();
        for t in chain.frames() {
            self.check(job)?;
            let start = Instant::now();
            let gray = job.frames.gray(t)?;
            tracker.step(&gray)?;
            let flows = self.flows_to_prev(job, &prev_gray, &gray)?;
            let temporal = match &flows {
                Some((to_prev, from_prev)) if self.params.weights.temporal > 0.0 => {
                    Some(self.temporal_guide(&prev, to_prev, from_prev.as_deref()))
                }
                _ => None,
            };
            // The warm start: the previous frame's field, moved along the flow to this frame.
            let warm = match (&prev_nnf, &flows) {
                (Some(nnf), Some((to_prev, _))) if self.params.warm_start.is_some() => Some(warp_nnf(nnf, to_prev)),
                _ => None,
            };
            job.guide_nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let mapping = Mapping {
                positions: tracker.positions(),
                confidence: self.params.gate_mapping.then(|| tracker.confidence()),
            };
            let input = self.input(job, chain.keyframe, t, temporal, &mapping)?;
            let extras = self.extras(job, chain.keyframe, warm);
            let blended = matches!(chain.output, ChainOutput::Blend { .. });
            let synthesized =
                self.synthesize(job, &input, &extras, t, chain.keyframe, blended && job.blender.needs_errors())?;
            let Synthesized { image, output: out, errors, nnf } = synthesized;
            prev = to_rgba(&image);
            prev_nnf = nnf;
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
                    let mine = Arc::new(pack_pass(&out, errors.as_deref(), job.blender.needs_errors()));
                    if let Some(theirs) = job.buffers.insert_or_take(key, Arc::clone(&mine), &partner)? {
                        let (a, b) = if pass == Pass::A { (mine, theirs) } else { (theirs, mine) };
                        let alpha = job.plan.intervals[interval].alpha(t);
                        tx.send((t, Output::Blend { a, b, alpha })).map_err(|_| VideoError::Cancelled)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_frames_survive_the_buffer() {
        let image = Image::from_raw(3, 2, 3, (0..18).map(|v| v * 13).collect());
        let image16 = Image16::from_raw(3, 2, 2, (0..12).map(|v| v * 5003 + 7).collect());
        let errors: Vec<f32> = (0..6).map(|i| i as f32 * 1.25).collect();
        for frame in [Frame::Eight(Arc::new(image)), Frame::Sixteen(Arc::new(image16))] {
            let sixteen = matches!(frame, Frame::Sixteen(_));
            for with_errors in [false, true] {
                let stored = pack_pass(&frame, Some(&errors), with_errors);
                let (back, back_errors) = unpack_pass(&stored, with_errors, sixteen);
                match (&frame, &back) {
                    (Frame::Eight(a), Frame::Eight(b)) => assert_eq!(a, b),
                    (Frame::Sixteen(a), Frame::Sixteen(b)) => assert_eq!(a, b),
                    _ => panic!("bit depth changed"),
                }
                assert_eq!(back_errors, with_errors.then(|| errors.clone()));
            }
        }
    }

    #[test]
    fn patch_errors_are_normalised_by_the_effective_weight() {
        let (w, h, ng, patch) = (9usize, 7usize, 4usize, 5usize);
        let guide = Image::from_raw(w, h, ng, vec![0; w * h * ng]);
        let modulation: Vec<u8> = (0..w * h * ng).map(|i| ((i * 37) % 256) as u8).collect();
        let mut input = SynthesisInput {
            source_style: Image::from_raw(w, h, 3, vec![0; w * h * 3]),
            source_guide: guide.clone(),
            target_guide: guide,
            target_modulation: None,
            style_weights: vec![1.0 / 3.0; 3],
            guide_weights: vec![2.0, 2.0, 2.0, 0.5],
            modulated_range: None,
        };
        let raw: Vec<f32> = (0..w * h).map(|i| 100.0 + i as f32).collect();
        let nominal: Vec<f32> = raw.iter().map(|e| (e / (25.0 * 7.5)).sqrt()).collect();
        let close = |e: &[f32]| e.iter().zip(&nominal).for_each(|(a, b)| assert!((a - b).abs() < 1e-4 * b, "{a} {b}"));
        // Without modulation: the nominal weight.
        let mut e = raw.clone();
        rms_patch_errors(&mut e, &input, patch);
        close(&e);
        // Full modulation everywhere is the same.
        input.target_modulation = Some(Image::from_raw(w, h, ng, vec![255; w * h * ng]));
        let mut e = raw.clone();
        rms_patch_errors(&mut e, &input, patch);
        close(&e);
        // Per-pixel modulation: the clamped patch sum.
        input.target_modulation = Some(Image::from_raw(w, h, ng, modulation.clone()));
        let own = |x: usize, y: usize| {
            let i = y * w + x;
            let guides: f32 = (0..ng).map(|k| input.guide_weights[k] * f32::from(modulation[i * ng + k]) / 255.0).sum();
            1.0 + guides
        };
        let mut e = raw.clone();
        rms_patch_errors(&mut e, &input, patch);
        for y in 0..h {
            for x in 0..w {
                let mut total = 0.0;
                for oy in -2i32..=2 {
                    for ox in -2i32..=2 {
                        let yy = (y as i32 + oy).clamp(0, h as i32 - 1) as usize;
                        total += own((x as i32 + ox).clamp(0, w as i32 - 1) as usize, yy);
                    }
                }
                let want = (raw[y * w + x] / total).sqrt();
                assert!((e[y * w + x] - want).abs() < 1e-4 * want, "({x}, {y}): {} vs {want}", e[y * w + x]);
            }
        }
        // Gating a guide off raises the error above the nominal normalisation.
        input.target_modulation = Some(Image::from_raw(w, h, ng, vec![0; w * h * ng]));
        let mut e = raw.clone();
        rms_patch_errors(&mut e, &input, patch);
        e.iter().zip(&nominal).for_each(|(a, b)| assert!((a / b - 7.5f32.sqrt()).abs() < 1e-3));
    }
}
