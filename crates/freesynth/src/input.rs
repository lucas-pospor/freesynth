//! Opening a project's input sequences: image files, or video files decoded once into a frame
//! cache of PNG files.
//!
//! Keyframe chains read frames in both directions and several times, and a re-render after a
//! keyframe edit reads them all again. Seeking in a compressed video for each of those reads
//! would be slow, so a video input is decoded once, sequentially, into numbered PNG files in the
//! render cache (`decoded/<role>-<hash>/`). Only the frames the project's range can read are
//! written (frames before `media.first` are decoded and dropped inside ffmpeg, which keeps the
//! numbering exact). The cache is keyed by the file's contents and the colour conversion, so it
//! is reused until the video changes or the range grows beyond it.
//!
//! The first render of a video does not wait for the whole decode: [`Opener::open`] returns at
//! once with a [`DecodeJob`] that the render runs on its own thread, and the input's frames
//! become readable as the decoder writes them (a read of a frame not decoded yet waits for it).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Condvar, Mutex};

use freesynth_core::Image;
use freesynth_media::{
    BitDepth, ColourTags, Ffmpeg, MediaError, VideoDecoder, VideoInfo, is_video_path, probe, save_png,
};
use freesynth_video::{CancelToken, FrameSource, ImageSequence, VideoError};
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::event::Event;
use crate::hashing::{FileHashes, FileStamp, Hash, Hasher};

/// An opened input sequence.
pub(crate) struct Input {
    /// The input's path (resolved): a pattern, a directory or a video file.
    pub source: PathBuf,
    /// The frames, numbered as the project sees them.
    pub seq: Frames,
    /// Content key of each frame (`keys[t - seq.first()]`).
    keys: Vec<Hash>,
    /// The video's description, for video inputs.
    pub video: Option<VideoInfo>,
    /// The colour tags the video was decoded with.
    pub colour: Option<ColourTags>,
    /// Frame number of a video's first frame (only part of the video may be decoded: `seq`
    /// can start later).
    pub start_number: usize,
    /// The decode still to run, for a video input whose frame cache is not complete.
    pub decode: Option<DecodeJob>,
}

impl Input {
    /// The files this input reads: its image files, or its video file.
    pub fn files(&self) -> Vec<PathBuf> {
        match &self.video {
            Some(_) => vec![self.source.clone()],
            None => {
                let seq = &self.seq;
                (seq.first()..=seq.last()).filter_map(|t| seq.path(t)).map(Path::to_path_buf).collect()
            }
        }
    }

    /// The content key of frame `t` (a frame outside the sequence gets a fixed "missing" key).
    pub fn key(&self, t: usize) -> Hash {
        t.checked_sub(self.seq.first())
            .and_then(|i| self.keys.get(i))
            .copied()
            .unwrap_or_else(|| Hasher::new("missing-frame").finish())
    }
}

/// The frames of an input: image files, possibly still being written by a [`DecodeJob`].
pub(crate) struct Frames {
    files: ImageSequence,
    /// The decode writing the files, while one runs.
    decoding: Option<Arc<DecodeState>>,
}

impl Frames {
    /// The file of frame `t`.
    pub fn path(&self, t: usize) -> Option<&Path> {
        self.files.path(t)
    }
}

impl FrameSource for Frames {
    fn first(&self) -> usize {
        self.files.first()
    }

    fn len(&self) -> usize {
        self.files.len()
    }

    fn load(&self, frame: usize) -> Result<Image, VideoError> {
        if let Some(state) = &self.decoding
            && let Some(i) = frame.checked_sub(self.files.first()).filter(|&i| i < self.files.len())
        {
            state.wait(i)?;
        }
        self.files.load(frame)
    }
}

/// How a decode ended.
#[derive(Clone, Debug, PartialEq)]
enum DecodeEnd {
    /// Every frame the decoder returned was written; this many.
    Done(usize),
    /// The decode failed with this message.
    Failed(String),
    /// The decode was cancelled or aborted.
    Stopped,
}

/// Which frames of a running decode exist, for readers to wait on.
pub(crate) struct DecodeState {
    /// Per frame: written yet. Then how the decode ended (`None` while it runs).
    state: Mutex<(Vec<bool>, Option<DecodeEnd>)>,
    changed: Condvar,
    /// Set when the render ends early: the decode stops.
    abort: AtomicBool,
}

impl DecodeState {
    fn new(frames: usize) -> Self {
        Self { state: Mutex::new((vec![false; frames], None)), changed: Condvar::new(), abort: AtomicBool::new(false) }
    }

    fn mark(&self, i: usize) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = s.0.get_mut(i) {
            *slot = true;
        }
        self.changed.notify_all();
    }

    /// Stops the decode and releases every reader waiting for a frame.
    fn abort(&self) {
        self.abort.store(true, Ordering::Relaxed);
        let _guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.changed.notify_all();
    }

    fn end(&self, result: DecodeEnd) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).1 = Some(result);
        self.changed.notify_all();
    }

    /// Waits until frame `i` (an index from 0) is written.
    fn wait(&self, i: usize) -> Result<(), VideoError> {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if s.0.get(i).copied().unwrap_or(false) {
                return Ok(());
            }
            match &s.1 {
                Some(DecodeEnd::Done(n)) => {
                    return Err(VideoError::InvalidInput(format!(
                        "frame {i} of the video does not exist: the decoder returned {n} frames, fewer than the \
                         container lists (render again to use the frames that exist)"
                    )));
                }
                Some(DecodeEnd::Failed(msg)) => {
                    return Err(VideoError::InvalidInput(format!("decoding the video failed: {msg}")));
                }
                Some(DecodeEnd::Stopped) => return Err(VideoError::Cancelled),
                None => {}
            }
            if self.abort.load(Ordering::Relaxed) {
                return Err(VideoError::Cancelled);
            }
            s = self.changed.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// What a finished frame cache directory records about its video.
#[derive(Serialize, Deserialize)]
struct DecodedIndex {
    video: Hash,
    colour: ColourTags,
    info: VideoInfo,
    /// Index of the first decoded frame.
    #[serde(default)]
    start: usize,
    /// Number of frames decoded (files `start..start + frames`).
    frames: usize,
    /// The decode reached the end of the video (no later frame exists).
    #[serde(default)]
    reached_end: bool,
}

impl DecodedIndex {
    /// True when the cache holds every frame of `range` (indexes from 0, the end `None` for
    /// "to the end of the video").
    fn covers(&self, range: (usize, Option<usize>)) -> bool {
        let (a, b) = range;
        let end = self.start + self.frames; // one past the last cached frame
        self.start <= a
            && match b {
                Some(b) => b < end || self.reached_end,
                None => self.reached_end,
            }
    }
}

/// A video decode into the frame cache, to run on its own thread during the render
/// ([`run`](Self::run)). Frames become readable through the input's [`Frames`] as they are
/// written; the cache is marked complete (its `index.json`) only when every frame is there.
pub(crate) struct DecodeJob {
    ff: Ffmpeg,
    path: PathBuf,
    info: VideoInfo,
    tags: ColourTags,
    video_hash: Hash,
    dir: PathBuf,
    /// Index of the first frame to keep, and how many frames to decode.
    skip: usize,
    count: usize,
    state: Arc<DecodeState>,
    threads: usize,
}

/// Ends a decode's state if the decode thread unwinds, so that readers waiting for frames are
/// released instead of waiting forever.
struct EndOnUnwind<'a>(&'a DecodeState);

impl Drop for EndOnUnwind<'_> {
    fn drop(&mut self) {
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.1.is_none() {
            s.1 = Some(DecodeEnd::Failed("the video decoder panicked".into()));
            self.0.changed.notify_all();
        }
    }
}

impl DecodeJob {
    /// Stops the decode (the render ended early); frames not decoded yet fail to load.
    pub fn abort(&self) {
        self.state.abort();
    }

    /// Decodes every frame into PNG files, encoding on several threads, then writes the cache
    /// index. On failure (or when aborted or cancelled) the partial cache is removed.
    pub fn run(&self, report: &(dyn Fn(&Event) + Sync), cancel: &CancelToken) -> Result<(), Error> {
        let _guard = EndOnUnwind(&self.state);
        let result = self.decode(report, cancel);
        let end = match &result {
            Ok(n) => DecodeEnd::Done(*n),
            Err(e) if e.is_cancelled() => DecodeEnd::Stopped,
            Err(e) => DecodeEnd::Failed(e.to_string()),
        };
        self.state.end(end);
        let count = match result {
            Ok(count) => count,
            Err(e) => {
                let _ = fs::remove_dir_all(&self.dir);
                return Err(e);
            }
        };
        let reached_end = count < self.count || self.skip + self.count >= self.info.frame_count;
        let index = DecodedIndex {
            video: self.video_hash,
            colour: self.tags.clone(),
            info: self.info.clone(),
            start: self.skip,
            frames: count,
            reached_end,
        };
        let bytes = serde_json::to_vec(&index).expect("indexes serialize");
        freesynth_media::write_atomic(&self.dir.join("index.json"), &bytes)?;
        Ok(())
    }

    /// Returns the number of frames kept (at most as many as asked for).
    fn decode(&self, report: &(dyn Fn(&Event) + Sync), cancel: &CancelToken) -> Result<usize, Error> {
        let (path, info, dir, skip) = (&self.path, &self.info, &self.dir, self.skip);
        let what = if skip == 0 && self.count == info.frame_count {
            format!("{} frames", info.frame_count)
        } else {
            format!("frames {}-{} of {}", skip, skip + self.count - 1, info.frame_count)
        };
        report(&Event::Status(format!("decoding '{}' ({what})", path.display())));
        let decoder = VideoDecoder::open_range(&self.ff, path, info, &self.tags, skip, Some(self.count))?;
        let workers = self.threads.clamp(1, 8);
        let (tx, rx) = sync_channel::<(usize, Image)>(workers * 2);
        let rx = Mutex::new(rx);
        let written = AtomicUsize::new(0);
        let failure: Mutex<Option<Error>> = Mutex::new(None);
        let total = self.count;
        let stopped = || cancel.is_cancelled() || self.state.abort.load(Ordering::Relaxed);
        let mut decoded = 0;
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| {
                    loop {
                        let next = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let Ok((i, image)) = next else { return };
                        match save_png(&image, &frame_file(dir, skip + i), BitDepth::Eight) {
                            Ok(()) => {
                                self.state.mark(i);
                                let done = written.fetch_add(1, Ordering::Relaxed) + 1;
                                report(&Event::Decoded { done, total: total.max(done) });
                            }
                            Err(e) => {
                                failure.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(e.into());
                            }
                        }
                    }
                });
            }
            for frame in decoder {
                if stopped() {
                    failure
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .get_or_insert(Error::Render(VideoError::Cancelled));
                    break;
                }
                if failure.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
                    break;
                }
                match frame {
                    Ok(image) => {
                        // Frames beyond the count asked for are not part of the input.
                        if decoded < total && tx.send((decoded, image)).is_err() {
                            break;
                        }
                        decoded += 1;
                    }
                    Err(e) => {
                        failure.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(e.into());
                        break;
                    }
                }
            }
            drop(tx);
        });
        if let Some(e) = failure.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e);
        }
        if decoded == 0 {
            return Err(MediaError::Unsupported(format!("no frame could be decoded from '{}'", path.display())).into());
        }
        if decoded < total {
            report(&Event::Warning(format!(
                "'{}': decoded {} frames, the container lists {}",
                path.display(),
                skip + decoded,
                info.frame_count
            )));
        }
        Ok(decoded.min(total))
    }
}

/// Loads a sequence given as a pattern (with a `#` run) or a directory.
pub(crate) fn image_sequence(path: &Path) -> Result<ImageSequence, VideoError> {
    let is_pattern = path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains('#'));
    if is_pattern { ImageSequence::from_pattern(path) } else { ImageSequence::from_dir(path) }
}

/// Context for opening inputs.
pub(crate) struct Opener<'a> {
    pub cache_dir: &'a Path,
    pub files: &'a mut FileHashes,
    pub ffmpeg: &'a dyn Fn() -> Result<Ffmpeg, Error>,
    pub report: &'a (dyn Fn(&Event) + Sync),
    pub threads: usize,
}

impl Opener<'_> {
    /// Opens the input at `path` (resolved): an image pattern or directory, or a video file
    /// whose first frame is number `start_number`. `role` names the cache directory
    /// (`frames`, `mask`). `colour` overrides the video's tags. `range` limits the frames a
    /// video input decodes (`first` and `last` frame numbers, `None` for its ends).
    pub fn open(
        &mut self,
        path: &Path,
        role: &str,
        start_number: usize,
        colour: Option<&ColourTags>,
        range: (Option<usize>, Option<usize>),
    ) -> Result<Input, Error> {
        if is_video_path(path) {
            return self.open_video(path, role, start_number, colour, range);
        }
        let seq = image_sequence(path)?;
        let paths: Vec<PathBuf> =
            (seq.first()..=seq.last()).map(|t| seq.path(t).expect("in range").to_path_buf()).collect();
        let keys = self.files.hash_all(&paths, self.threads)?;
        Ok(Input {
            source: path.to_owned(),
            seq: Frames { files: seq, decoding: None },
            keys,
            video: None,
            colour: None,
            start_number: 0,
            decode: None,
        })
    }

    fn open_video(
        &mut self,
        path: &Path,
        role: &str,
        start_number: usize,
        colour: Option<&ColourTags>,
        range: (Option<usize>, Option<usize>),
    ) -> Result<Input, Error> {
        let ff = (self.ffmpeg)()?;
        let video_hash = self.files.hash(path)?;
        let info = probe_cached(&ff, path)?;
        // Project tags override the file's; the rest get the usual defaults.
        let mut tags = info.colour.clone();
        if let Some(c) = colour {
            tags.matrix = c.matrix.clone().or(tags.matrix);
            tags.range = c.range.clone().or(tags.range);
            tags.primaries = c.primaries.clone().or(tags.primaries);
            tags.transfer = c.transfer.clone().or(tags.transfer);
        }
        let tags = tags.completed(info.width, info.height, &info.pix_fmt);
        if tags.is_hdr() {
            (self.report)(&Event::Warning(format!(
                "'{}' is HDR ({}); it is processed as display-referred video",
                path.display(),
                tags.describe()
            )));
        }
        if info.is_vfr() {
            (self.report)(&Event::Warning(format!(
                "'{}' has a variable frame rate; frames are numbered in display order and a video output keeps \
                 their timing",
                path.display()
            )));
        }
        // The frames the render can read, as indexes from 0. A range outside the video is
        // reported by the caller, against the whole video.
        let skip = range.0.map_or(0, |f| f.saturating_sub(start_number)).min(info.frame_count - 1);
        let end = range.1.map(|l| l.saturating_sub(start_number)).filter(|&e| e + 1 < info.frame_count);
        let decode_key = Hasher::new("decode").hash(&video_hash).str(&format!("{tags:?}")).finish();
        let parent = self.cache_dir.join("decoded");
        let dir = parent.join(format!("{role}-{}", decode_key.short()));
        let index: Option<DecodedIndex> = fs::read(dir.join("index.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<DecodedIndex>(&b).ok())
            .filter(|i| {
                i.video == video_hash
                    && i.colour == tags
                    && i.covers((skip, end))
                    && frames_present(&dir, i.start, i.frames)
            });
        let (info, start, count, decode) = match index {
            Some(i) => (i.info, i.start, i.frames, None),
            None => {
                let count = end.map_or(info.frame_count, |e| e + 1).max(skip + 1) - skip;
                let bytes = count as f64 * (info.width * info.height) as f64 * DECODED_BYTES_PER_PIXEL;
                if bytes > DECODE_WARNING_BYTES {
                    (self.report)(&Event::Warning(format!(
                        "decoding {count} frames of '{}' needs about {:.0} GB in '{}'; set media.first and media.last \
                         to decode only the frames you render",
                        path.display(),
                        bytes / 1e9,
                        parent.display()
                    )));
                }
                remove_role_dirs(&parent, role);
                fs::create_dir_all(&dir).map_err(|source| Error::Io { path: dir.clone(), source })?;
                let job = DecodeJob {
                    ff,
                    path: path.to_owned(),
                    info: info.clone(),
                    tags: tags.clone(),
                    video_hash,
                    dir: dir.clone(),
                    skip,
                    count,
                    state: Arc::new(DecodeState::new(count)),
                    threads: self.threads,
                };
                (info, skip, count, Some(job))
            }
        };
        let paths: Vec<PathBuf> = (start..start + count).map(|i| frame_file(&dir, i)).collect();
        let keys = (start..start + count)
            .map(|i| Hasher::new("video-frame").hash(&decode_key).u64(i as u64).finish())
            .collect();
        let seq = Frames {
            files: ImageSequence::from_paths(start_number + start, paths),
            decoding: decode.as_ref().map(|job| Arc::clone(&job.state)),
        };
        Ok(Input { source: path.to_owned(), seq, keys, video: Some(info), colour: Some(tags), start_number, decode })
    }
}

/// Typical size of a decoded frame in the PNG frame cache, per pixel (measured on camera
/// footage; flat synthetic video compresses much better).
const DECODED_BYTES_PER_PIXEL: f64 = 1.8;

/// Frame caches estimated larger than this get a warning before the decode starts.
const DECODE_WARNING_BYTES: f64 = 4e9;

/// [`probe`] with the results remembered for the process, by path and file stamp: checking a
/// project before a render and the render itself probe once.
pub(crate) fn probe_cached(ff: &Ffmpeg, path: &Path) -> Result<VideoInfo, Error> {
    static PROBES: Mutex<Option<HashMap<(PathBuf, FileStamp), VideoInfo>>> = Mutex::new(None);
    let stamp = FileStamp::of(path)?;
    let key = (path.to_owned(), stamp);
    if let Some(info) = PROBES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&key)) {
        return Ok(info.clone());
    }
    let info = probe(ff, path)?;
    if info.frame_count == 0 {
        return Err(MediaError::Unsupported(format!("'{}' holds no frames", path.display())).into());
    }
    PROBES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().insert(key, info.clone());
    Ok(info)
}

/// Runs `body` while the decodes of `inputs` run on their own threads, then waits for them.
/// When `body` fails, the decodes are stopped. A decode failure is returned in preference to
/// the body's error (a render failing to read a frame is a consequence of it).
pub(crate) fn with_decodes<T>(
    inputs: &[&Input],
    report: &(dyn Fn(&Event) + Sync),
    cancel: &CancelToken,
    body: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    let jobs: Vec<&DecodeJob> = inputs.iter().filter_map(|i| i.decode.as_ref()).collect();
    if jobs.is_empty() {
        return body();
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = jobs.iter().map(|job| s.spawn(move || job.run(report, cancel))).collect();
        let result = body();
        if result.is_err() {
            jobs.iter().for_each(|job| job.abort());
        }
        let mut decode_error = None;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    decode_error.get_or_insert(e);
                }
                Err(_) => {
                    decode_error.get_or_insert(Error::Project("the video decoder panicked".into()));
                }
            }
        }
        match (result, decode_error) {
            (Ok(v), None) => Ok(v),
            // Stopped because the render failed: report the render's error.
            (Err(e), Some(d)) if d.is_cancelled() => Err(e),
            (_, Some(d)) => Err(d),
            (Err(e), None) => Err(e),
        }
    })
}

/// The cache file of decoded frame `i`.
fn frame_file(dir: &Path, i: usize) -> PathBuf {
    dir.join(format!("{i:06}.png"))
}

/// True when the frame files `start..start + count` of a finished decode are still there.
fn frames_present(dir: &Path, start: usize, count: usize) -> bool {
    count > 0 && frame_file(dir, start).is_file() && frame_file(dir, start + count - 1).is_file()
}

/// Removes earlier decodes of the same role (an older version of the video).
fn remove_role_dirs(parent: &Path, role: &str) {
    let Ok(entries) = fs::read_dir(parent) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_str().is_some_and(|n| n.starts_with(&format!("{role}-"))) {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

/// The frame numbers an input holds: every frame of a video file (also those not decoded).
pub(crate) fn frame_range(input: &Input) -> (usize, usize) {
    let s = &input.seq;
    match &input.video {
        Some(v) => (input.start_number, input.start_number + v.frame_count - 1),
        None => (s.first(), s.last()),
    }
}

/// Number of frames a sequence covers, for messages.
pub(crate) fn describe(input: &Input) -> String {
    let s = &input.seq;
    let (first, last) = frame_range(input);
    match &input.video {
        Some(v) => format!(
            "{} frames ({first}-{last}), {}x{}, {} fps, {}",
            v.frame_count,
            v.width,
            v.height,
            v.frame_rate,
            input.colour.as_ref().map(ColourTags::describe).unwrap_or_default()
        ),
        None => format!("{} frames ({}-{})", s.len(), s.first(), s.last()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_wait_for_decoded_frames() {
        let state = Arc::new(DecodeState::new(3));
        std::thread::scope(|s| {
            let reader = s.spawn(|| (state.wait(2), state.wait(0)));
            state.mark(0);
            state.mark(2);
            let (a, b) = reader.join().unwrap();
            assert!(a.is_ok() && b.is_ok());
        });
        // A frame the decoder never produced fails once the decode ends.
        let short = DecodeState::new(3);
        short.mark(0);
        short.end(DecodeEnd::Done(1));
        assert!(short.wait(0).is_ok());
        let err = short.wait(1).unwrap_err().to_string();
        assert!(err.contains("returned 1 frames"), "{err}");
        let failed = DecodeState::new(2);
        failed.end(DecodeEnd::Failed("broken file".into()));
        let stopped = DecodeState::new(2);
        stopped.end(DecodeEnd::Stopped);
        assert!(matches!(stopped.wait(0), Err(VideoError::Cancelled)));
        assert!(failed.wait(1).unwrap_err().to_string().contains("broken file"));
        // An aborted decode releases its readers.
        let aborted = Arc::new(DecodeState::new(2));
        std::thread::scope(|s| {
            let reader = s.spawn(|| aborted.wait(1));
            std::thread::sleep(std::time::Duration::from_millis(20));
            aborted.abort();
            assert!(matches!(reader.join().unwrap(), Err(VideoError::Cancelled)));
        });
    }
}
