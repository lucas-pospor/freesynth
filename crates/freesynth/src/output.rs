//! Writing output frames and the output video.
//!
//! Rendered frames are always written as image files: to the project's output sequence, or to
//! the render cache when the project only asks for a video. The video is then encoded from
//! those files (from lossless PNG copies in the cache when the output frames are JPEG), in frame
//! order, by a thread that starts with the render and feeds each frame to
//! ffmpeg as soon as it and all frames before it exist. Frames reach the sink in no particular
//! order (the two passes of an interval meet in its middle), so part of the encode necessarily
//! waits for the end of the render.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use freesynth_core::{Image, Image16};
use freesynth_media::{
    BitDepth, EncodeSettings, Ffmpeg, MediaError, VideoEncoder, encode_files, load_rgba16, save_frame, save_png16,
};
use freesynth_video::{FrameSink, VideoError, pattern_path};

use crate::cache::{FrameKind, Manifest};
use crate::error::Error;
use crate::event::Event;
use crate::hashing::{FileStamp, Hash};

/// Where output frames go and in which format.
pub(crate) struct OutputFrames {
    /// A path with a `#` run for the frame number.
    pattern: PathBuf,
    pub depth: BitDepth,
}

impl OutputFrames {
    /// Frames into `spec` (a pattern with `#`, or a directory receiving `NNNNN.png`). Nothing
    /// is created ([`create_dir`](Self::create_dir)).
    pub fn parse(spec: &Path, depth: BitDepth) -> Result<Self, Error> {
        let is_pattern = spec.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains('#'));
        let pattern = if is_pattern { spec.to_owned() } else { spec.join("#####.png") };
        pattern_path(&pattern, 0)?;
        Ok(Self { pattern, depth })
    }

    /// Creates the frames' folder; `what` names it in the error message (`output.frames`).
    pub fn create_dir(&self, what: &str) -> Result<(), Error> {
        if let Some(dir) = self.pattern.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)
                .map_err(|e| Error::Project(format!("cannot create the folder of {what} '{}': {e}", dir.display())))?;
        }
        Ok(())
    }

    /// The file of frame `t`.
    pub fn path(&self, t: usize) -> PathBuf {
        pattern_path(&self.pattern, t).expect("the pattern was checked")
    }

    /// The file format (lower-case extension).
    pub fn format(&self) -> String {
        self.pattern.extension().and_then(|e| e.to_str()).unwrap_or("png").to_ascii_lowercase()
    }
}

/// Which output frames exist, for the encoder thread to wait on.
pub(crate) struct Ready {
    first: usize,
    state: Mutex<(Vec<bool>, bool)>,
    changed: Condvar,
}

impl Ready {
    /// Frames `first..=last`, none ready.
    pub fn new(first: usize, last: usize) -> Self {
        Self { first, state: Mutex::new((vec![false; last - first + 1], false)), changed: Condvar::new() }
    }

    /// Marks frame `t` as written.
    pub fn mark(&self, t: usize) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = t.checked_sub(self.first).and_then(|i| s.0.get_mut(i)) {
            *slot = true;
        }
        self.changed.notify_all();
    }

    /// Wakes every waiter for good (the render stopped).
    pub fn abort(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
        self.changed.notify_all();
    }

    /// Waits until frame `t` is written; false when the render was aborted first.
    pub fn wait(&self, t: usize) -> bool {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if s.0.get(t - self.first).copied().unwrap_or(true) {
                return true;
            }
            if s.1 {
                return false;
            }
            s = self.changed.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// How often the manifest is saved while frames are written, so that an interrupted render
/// keeps most of its work.
const SAVE_INTERVAL: Duration = Duration::from_secs(2);

/// The sink of a project render: writes the frames the plan wants (others are rendered only
/// as context and dropped), records them in the manifest and tells the encoder.
pub(crate) struct ProjectSink<'a> {
    pub target: &'a OutputFrames,
    /// The file key of each frame to write.
    pub wanted: &'a BTreeMap<usize, Hash>,
    /// Lossless copies to write too (the video's source when the output frames are lossy),
    /// with their file keys.
    pub master: Option<(&'a OutputFrames, &'a BTreeMap<usize, Hash>)>,
    pub manifest: &'a Mutex<Manifest>,
    pub cache_dir: &'a Path,
    pub ready: &'a Ready,
    pub written: &'a AtomicUsize,
    pub report: &'a (dyn Fn(&Event) + Sync),
    pub last_save: &'a Mutex<Instant>,
}

impl FrameSink for ProjectSink<'_> {
    fn write(&self, frame: usize, image: &Image) -> Result<(), VideoError> {
        self.store(frame, &|path, depth| save_frame(image, path, depth))
    }

    fn write16(&self, frame: usize, image: &Image16) -> Result<(), VideoError> {
        self.store(frame, &|path, depth| {
            let is_png = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("png"));
            if depth == BitDepth::Sixteen && is_png {
                let (w, h) = image.size();
                save_png16(w, h, image.channels(), image.data(), path)
            } else {
                save_frame(&image.to_image(), path, depth)
            }
        })
    }
}

/// Writes a frame's image to a path at a bit depth.
type Saver<'a> = dyn Fn(&Path, BitDepth) -> Result<(), MediaError> + 'a;

impl ProjectSink<'_> {
    /// Writes frame `frame` (when the plan wants it) with `save`, records it and tells the
    /// encoder.
    fn store(&self, frame: usize, save: &Saver<'_>) -> Result<(), VideoError> {
        let Some(key) = self.wanted.get(&frame) else { return Ok(()) };
        let save = |frames: &OutputFrames| {
            let path = frames.path(frame);
            save(&path, frames.depth)
                .map_err(|e| VideoError::InvalidInput(format!("cannot write frame {frame}: {e}")))?;
            let stamp = FileStamp::of(&path).map_err(|e| VideoError::InvalidInput(e.to_string()))?;
            Ok::<_, VideoError>((path, stamp))
        };
        let (path, stamp) = save(self.target)?;
        let master = match self.master {
            Some((frames, keys)) => Some((save(frames)?, keys[&frame])),
            None => None,
        };
        {
            let mut m = self.manifest.lock().unwrap_or_else(|e| e.into_inner());
            m.record(FrameKind::Output, frame, *key, &path, stamp);
            if let Some(((path, stamp), key)) = master {
                m.record(FrameKind::Master, frame, key, &path, stamp);
            }
            let mut last = self.last_save.lock().unwrap_or_else(|e| e.into_inner());
            if last.elapsed() >= SAVE_INTERVAL {
                *last = Instant::now();
                // Best effort: the final save reports errors.
                let _ = m.save(self.cache_dir);
            }
        }
        self.ready.mark(frame);
        let done = self.written.fetch_add(1, Ordering::Relaxed) + 1;
        (self.report)(&Event::Written { frame, done, total: self.wanted.len() });
        Ok(())
    }
}

/// The video to encode.
pub(crate) struct VideoJob<'a> {
    pub ff: Ffmpeg,
    pub path: PathBuf,
    /// The frames encoded (lossless files).
    pub frames: &'a OutputFrames,
    /// The frame size.
    pub size: (usize, usize),
    pub first: usize,
    pub last: usize,
    pub settings: EncodeSettings,
    /// Per-frame display durations for variable-frame-rate output (then the encode runs from
    /// the files after the render).
    pub durations: Option<Vec<f64>>,
}

/// An output frame loaded for the encoder.
enum Loaded {
    Eight(Image),
    Sixteen(Vec<u16>),
}

/// Loads the output frame at `path`.
fn load_frame(path: &Path, depth: BitDepth) -> Result<Loaded, Error> {
    Ok(match depth {
        BitDepth::Eight => Loaded::Eight(
            freesynth_core::io::load_rgba(path)
                .map_err(|e| MediaError::Image { path: path.to_owned(), message: e.to_string() })?,
        ),
        BitDepth::Sixteen => Loaded::Sixteen(load_rgba16(path)?.2),
    })
}

/// Hands a loaded frame to the encoder.
fn write_frame(enc: &mut VideoEncoder, frame: &Loaded) -> Result<(), Error> {
    match frame {
        Loaded::Eight(image) => enc.write(image)?,
        Loaded::Sixteen(data) => enc.write_rgba16(data)?,
    }
    Ok(())
}

/// Threads loading output frames ahead of the encoder. Frames reach the sink out of order
/// (with two keyframes, the frames next to them come last), so much of the encode runs after
/// the render, and loading the files one by one would then slow it down.
const PREFETCH_THREADS: usize = 4;
/// How many frames are loaded ahead of the encoder at most.
const PREFETCH_AHEAD: usize = 8;

/// Frames loaded ahead of the encoder, by frame number.
struct Prefetch {
    state: Mutex<PrefetchState>,
    changed: Condvar,
}

struct PrefetchState {
    /// The next frame a loader takes.
    next: usize,
    /// The next frame the encoder takes.
    wanted: usize,
    loaded: BTreeMap<usize, Result<Loaded, Error>>,
    /// The encoder stopped: loaders take no more frames.
    stop: bool,
}

impl Prefetch {
    fn lock(&self) -> std::sync::MutexGuard<'_, PrefetchState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&self, guard: std::sync::MutexGuard<'a, PrefetchState>) -> std::sync::MutexGuard<'a, PrefetchState> {
        self.changed.wait(guard).unwrap_or_else(|e| e.into_inner())
    }

    fn stop(&self) {
        self.lock().stop = true;
        self.changed.notify_all();
    }
}

impl VideoJob<'_> {
    /// Encodes the frames in order as they become ready. Returns the encoder's notes. The
    /// encoder starts at once, so that a failing one fails early.
    pub fn stream(&self, ready: &Ready, report: &(dyn Fn(&Event) + Sync)) -> Result<Vec<String>, Error> {
        let total = self.last - self.first + 1;
        let (w, h) = self.size;
        let mut enc = VideoEncoder::create(&self.ff, &self.path, w, h, &self.settings)?;
        let notes = enc.notes().to_vec();
        // Frame files are read on several threads, ahead of the encoder.
        let prefetch = Prefetch {
            state: Mutex::new(PrefetchState {
                next: self.first,
                wanted: self.first,
                loaded: BTreeMap::new(),
                stop: false,
            }),
            changed: Condvar::new(),
        };
        let loader = || {
            loop {
                let t = {
                    let mut s = prefetch.lock();
                    while !s.stop && s.next <= self.last && s.next >= s.wanted + PREFETCH_AHEAD {
                        s = prefetch.wait(s);
                    }
                    if s.stop || s.next > self.last {
                        return;
                    }
                    s.next += 1;
                    s.next - 1
                };
                let frame = if ready.wait(t) {
                    load_frame(&self.frames.path(t), self.frames.depth)
                } else {
                    Err(Error::Render(VideoError::Cancelled))
                };
                prefetch.lock().loaded.insert(t, frame);
                prefetch.changed.notify_all();
            }
        };
        let mut encode = || -> Result<(), Error> {
            for t in self.first..=self.last {
                let frame = {
                    let mut s = prefetch.lock();
                    loop {
                        if let Some(f) = s.loaded.remove(&t) {
                            s.wanted = t + 1;
                            break f;
                        }
                        s = prefetch.wait(s);
                    }
                };
                prefetch.changed.notify_all();
                write_frame(&mut enc, &frame?)?;
                report(&Event::Encoded { done: t - self.first + 1, total });
            }
            Ok(())
        };
        let encoded = std::thread::scope(|s| {
            for _ in 0..PREFETCH_THREADS.min(total) {
                s.spawn(loader);
            }
            let result = encode();
            prefetch.stop();
            result
        });
        encoded?;
        enc.finish()?;
        Ok(notes)
    }

    /// Encodes from the files with per-frame durations (variable frame rate).
    pub fn encode_files(&self, durations: &[f64], list_dir: &Path) -> Result<Vec<String>, Error> {
        let files: Vec<(PathBuf, f64)> =
            (self.first..=self.last).zip(durations).map(|(t, &d)| (self.frames.path(t), d)).collect();
        let (w, h) = self.size;
        Ok(encode_files(&self.ff, &files, &self.path, w, h, &self.settings, list_dir)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_waits_and_aborts() {
        let ready = Ready::new(10, 12);
        std::thread::scope(|s| {
            let waiter = s.spawn(|| (ready.wait(10), ready.wait(12)));
            ready.mark(10);
            ready.mark(12);
            assert_eq!(waiter.join().unwrap(), (true, true));
        });
        let ready = Ready::new(0, 1);
        std::thread::scope(|s| {
            let waiter = s.spawn(|| ready.wait(1));
            ready.abort();
            assert!(!waiter.join().unwrap());
        });
    }

    #[test]
    fn output_patterns() {
        let dir = crate::test_util::scratch("output-patterns");
        let o = OutputFrames::parse(&dir.join("seq"), BitDepth::Eight).unwrap();
        assert_eq!(o.path(7), dir.join("seq").join("00007.png"));
        o.create_dir("output.frames").unwrap();
        assert!(dir.join("seq").is_dir());
        let o = OutputFrames::parse(&dir.join("f_[###].jpg"), BitDepth::Eight).unwrap();
        assert_eq!((o.path(3), o.format()), (dir.join("f_003.jpg"), "jpg".to_owned()));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
