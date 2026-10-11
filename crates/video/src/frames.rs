//! Frame sources, the decoded-frame cache and output sinks.
//!
//! A shot's frames are addressed by their frame number. A [`FrameSource`] produces them as
//! 8-bit RGBA images: from numbered image files ([`ImageSequence`]) or from memory
//! ([`MemoryFrames`]). Keyframe chains read frames in both directions and several times, so the
//! renderer decodes each frame once into a [`FrameCache`] (bounded, spilling to disk) and serves
//! every later read from it. Results go to a [`FrameSink`].

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use freesynth_core::{Image, io};
use freesynth_flow::GrayImage;

use crate::error::{VideoError, invalid, io_error};
use crate::store::{SpillStore, StoreStats};

/// Numbered input frames as 8-bit RGBA images.
pub trait FrameSource: Send + Sync {
    /// Number of the first frame.
    fn first(&self) -> usize;

    /// Number of frames; they are numbered `first()` to `first() + len() - 1`.
    fn len(&self) -> usize;

    /// True when the source has no frame.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of the last frame (`first()` for an empty source).
    fn last(&self) -> usize {
        self.first() + self.len().saturating_sub(1)
    }

    /// True when `frame` is one of the source's frames.
    fn contains(&self, frame: usize) -> bool {
        frame >= self.first() && frame - self.first() < self.len()
    }

    /// Loads frame `frame` as RGBA.
    fn load(&self, frame: usize) -> Result<Image, VideoError>;
}

/// File extensions [`ImageSequence`] picks up (compared without case).
const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "bmp", "tga"];

/// Numbered image files, one per frame.
#[derive(Clone, Debug)]
pub struct ImageSequence {
    first: usize,
    paths: Vec<PathBuf>,
}

/// Splits a file stem into its text before the trailing digits and the digits.
fn trailing_number(stem: &str) -> Option<(&str, &str)> {
    let digits = stem.len() - stem.bytes().rev().take_while(u8::is_ascii_digit).count();
    (digits < stem.len()).then(|| stem.split_at(digits))
}

impl ImageSequence {
    /// Numbered frames from explicit paths, `paths[i]` being frame `first + i`.
    pub fn from_paths(first: usize, paths: Vec<PathBuf>) -> Self {
        Self { first, paths }
    }

    /// Every image file in `dir` whose name ends in a frame number (`frame_0042.png`,
    /// `042.jpg`). The numbers must form one contiguous range and the files must share one
    /// name prefix and extension.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, VideoError> {
        let dir = dir.as_ref();
        let mut found: BTreeMap<usize, PathBuf> = BTreeMap::new();
        let mut naming: Option<(String, String)> = None;
        for entry in fs::read_dir(dir).map_err(io_error(dir))? {
            let path = entry.map_err(io_error(dir))?.path();
            let (Some(stem), Some(ext)) = (path.file_stem().and_then(|s| s.to_str()), path.extension().and_then(|s| s.to_str()))
            else {
                continue;
            };
            if !IMAGE_EXTENSIONS.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                continue;
            }
            let Some((prefix, digits)) = trailing_number(stem) else {
                continue;
            };
            let this = (prefix.to_owned(), ext.to_ascii_lowercase());
            match &naming {
                None => naming = Some(this),
                Some(n) if *n != this => {
                    return invalid(format!("'{}' holds differently named image sequences", dir.display()));
                }
                Some(_) => {}
            }
            let number: usize = digits.parse().map_err(|_| VideoError::InvalidInput(format!("frame number too large in '{}'", path.display())))?;
            if found.insert(number, path).is_some() {
                return invalid(format!("frame {number} appears twice in '{}'", dir.display()));
            }
        }
        Self::from_numbered(found, &dir.display().to_string())
    }

    /// The files matching `pattern`, a path whose file name contains a run of `#` characters
    /// (optionally in brackets: `video/[####].png`) standing for the zero-padded frame number.
    /// The frame range is whatever exists on disk; it must be contiguous.
    pub fn from_pattern(pattern: impl AsRef<Path>) -> Result<Self, VideoError> {
        let pattern = pattern.as_ref();
        let (dir, prefix, width, suffix) = split_pattern(pattern)?;
        let dir_or_cwd = if dir.as_os_str().is_empty() { Path::new(".") } else { dir.as_path() };
        let mut found = BTreeMap::new();
        for entry in fs::read_dir(dir_or_cwd).map_err(io_error(dir_or_cwd))? {
            let path = entry.map_err(io_error(dir_or_cwd))?.path();
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
            let Some(digits) = name.strip_prefix(prefix.as_str()).and_then(|r| r.strip_suffix(suffix.as_str())) else {
                continue;
            };
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let Ok(number) = digits.parse::<usize>() else { continue };
            // The same zero padding as the pattern (numbers may grow beyond it).
            if format!("{number:0width$}") == digits {
                found.insert(number, path);
            }
        }
        Self::from_numbered(found, &pattern.display().to_string())
    }

    fn from_numbered(found: BTreeMap<usize, PathBuf>, what: &str) -> Result<Self, VideoError> {
        let Some(&first) = found.keys().next() else {
            return invalid(format!("no numbered image files found for '{what}'"));
        };
        if let Some((i, (&number, _))) = found.iter().enumerate().find(|&(i, (&n, _))| n != first + i) {
            return invalid(format!("frame {} is missing for '{what}' (next is {number})", first + i));
        }
        Ok(Self { first, paths: found.into_values().collect() })
    }

    /// The file of frame `frame`.
    pub fn path(&self, frame: usize) -> Option<&Path> {
        frame.checked_sub(self.first).and_then(|i| self.paths.get(i)).map(PathBuf::as_path)
    }
}

/// Splits a frame pattern into directory, file name prefix, number width and suffix.
fn split_pattern(pattern: &Path) -> Result<(PathBuf, String, usize, String), VideoError> {
    let name = pattern.file_name().and_then(|s| s.to_str()).ok_or_else(|| {
        VideoError::InvalidInput(format!("frame pattern '{}' has no file name", pattern.display()))
    })?;
    let Some(end) = name.rfind('#') else {
        return invalid(format!("frame pattern '{}' has no '#' for the frame number", pattern.display()));
    };
    let start = name[..end].trim_end_matches('#').len();
    let width = end + 1 - start;
    let (mut prefix, mut suffix) = (&name[..start], &name[end + 1..]);
    if prefix.ends_with('[') && suffix.starts_with(']') {
        prefix = &prefix[..prefix.len() - 1];
        suffix = &suffix[1..];
    }
    let dir = pattern.parent().map(Path::to_path_buf).unwrap_or_default();
    Ok((dir, prefix.to_owned(), width, suffix.to_owned()))
}

/// The path of frame `frame` for a pattern accepted by [`ImageSequence::from_pattern`].
pub fn pattern_path(pattern: impl AsRef<Path>, frame: usize) -> Result<PathBuf, VideoError> {
    let (dir, prefix, width, suffix) = split_pattern(pattern.as_ref())?;
    Ok(dir.join(format!("{prefix}{frame:0width$}{suffix}")))
}

impl FrameSource for ImageSequence {
    fn first(&self) -> usize {
        self.first
    }

    fn len(&self) -> usize {
        self.paths.len()
    }

    fn load(&self, frame: usize) -> Result<Image, VideoError> {
        let path = self.path(frame).ok_or_else(|| VideoError::InvalidInput(format!("frame {frame} is not in the sequence")))?;
        io::load_rgba(path).map_err(|source| VideoError::Load { path: path.to_owned(), source })
    }
}

/// Frames held in memory (RGBA).
#[derive(Clone, Debug)]
pub struct MemoryFrames {
    first: usize,
    frames: Vec<Arc<Image>>,
}

impl MemoryFrames {
    /// `frames[i]` is frame `first + i`; every frame must be RGBA.
    pub fn new(first: usize, frames: Vec<Image>) -> Result<Self, VideoError> {
        if frames.iter().any(|f| f.channels() != 4) {
            return invalid("memory frames must be RGBA");
        }
        Ok(Self { first, frames: frames.into_iter().map(Arc::new).collect() })
    }
}

impl FrameSource for MemoryFrames {
    fn first(&self) -> usize {
        self.first
    }

    fn len(&self) -> usize {
        self.frames.len()
    }

    fn load(&self, frame: usize) -> Result<Image, VideoError> {
        frame
            .checked_sub(self.first)
            .and_then(|i| self.frames.get(i))
            .map(|f| (**f).clone())
            .ok_or_else(|| VideoError::InvalidInput(format!("frame {frame} is not in the source")))
    }
}

/// Decodes each frame of a source once and keeps the results within a memory budget, spilling
/// the rest to disk. All frames must have the same size.
pub struct FrameCache<'a> {
    source: &'a dyn FrameSource,
    store: SpillStore<usize>,
    /// One lock per frame, so that a frame is decoded once even when two chains ask at once.
    decoding: Vec<Mutex<()>>,
    size: Mutex<Option<(usize, usize)>>,
}

impl<'a> FrameCache<'a> {
    /// A cache over `source` holding up to `budget` bytes in memory, spilling to `spill_dir`
    /// (nothing is spilled without one).
    pub fn new(source: &'a dyn FrameSource, budget: usize, spill_dir: Option<PathBuf>) -> Self {
        Self {
            source,
            store: SpillStore::new(budget, spill_dir),
            decoding: (0..source.len()).map(|_| Mutex::new(())).collect(),
            size: Mutex::new(None),
        }
    }

    /// Changes the memory budget. It applies from the next decoded frame on.
    pub fn set_budget(&mut self, budget: usize) {
        self.store.set_budget(budget);
    }

    /// The source.
    pub fn source(&self) -> &dyn FrameSource {
        self.source
    }

    /// Frame `frame` (RGBA), decoded on first use.
    pub fn get(&self, frame: usize) -> Result<Arc<Image>, VideoError> {
        if !self.source.contains(frame) {
            return invalid(format!("frame {frame} is outside the shot ({}-{})", self.source.first(), self.source.last()));
        }
        if let Some(image) = self.store.get(&frame)? {
            return Ok(image);
        }
        let _decoding = self.decoding[frame - self.source.first()].lock().unwrap_or_else(|e| e.into_inner());
        if let Some(image) = self.store.get(&frame)? {
            return Ok(image);
        }
        let image = self.source.load(frame)?;
        if image.channels() != 4 {
            return invalid(format!("frame {frame} is not RGBA"));
        }
        {
            let mut size = self.size.lock().unwrap_or_else(|e| e.into_inner());
            match *size {
                None => *size = Some(image.size()),
                Some(s) if s != image.size() => {
                    return invalid(format!(
                        "frame {frame} is {}x{}, other frames are {}x{}",
                        image.width(),
                        image.height(),
                        s.0,
                        s.1
                    ));
                }
                Some(_) => {}
            }
        }
        let image = Arc::new(image);
        self.store.insert(frame, Arc::clone(&image))?;
        Ok(image)
    }

    /// The Rec. 601 luma of frame `frame`, for optical flow.
    pub fn gray(&self, frame: usize) -> Result<GrayImage, VideoError> {
        let image = self.get(frame)?;
        Ok(GrayImage::from_rgba8(image.width(), image.height(), image.data())?)
    }

    /// The cache's memory and spill counters.
    pub fn stats(&self) -> StoreStats {
        self.store.stats()
    }
}

/// Receives finished output frames. Frames arrive in no particular order, each exactly once.
pub trait FrameSink: Send + Sync {
    /// Stores frame `frame`.
    fn write(&self, frame: usize, image: &Image) -> Result<(), VideoError>;
}

/// Writes frames as numbered image files. The format follows the extension (PNG, JPEG, BMP or
/// TGA).
#[derive(Clone, Debug)]
pub struct ImageSequenceWriter {
    pattern: PathBuf,
}

impl ImageSequenceWriter {
    /// Writes frame `t` to `pattern` with its `#` run replaced by the zero-padded frame number
    /// (see [`ImageSequence::from_pattern`]). Creates the directory.
    pub fn new(pattern: impl Into<PathBuf>) -> Result<Self, VideoError> {
        let pattern = pattern.into();
        split_pattern(&pattern)?;
        if let Some(dir) = pattern.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir).map_err(io_error(dir))?;
        }
        Ok(Self { pattern })
    }

    /// Writes PNG files named `NNNNN.png` into `dir`.
    pub fn into_dir(dir: impl AsRef<Path>) -> Result<Self, VideoError> {
        Self::new(dir.as_ref().join("#####.png"))
    }

    /// The file frame `frame` is written to.
    pub fn path(&self, frame: usize) -> PathBuf {
        pattern_path(&self.pattern, frame).expect("the pattern was checked")
    }
}

impl FrameSink for ImageSequenceWriter {
    fn write(&self, frame: usize, image: &Image) -> Result<(), VideoError> {
        let path = self.path(frame);
        io::save(image, &path).map_err(|source| VideoError::Save { path, source })
    }
}

/// Collects frames in memory (for tests and hosts that encode frames themselves).
#[derive(Debug, Default)]
pub struct MemorySink {
    frames: Mutex<BTreeMap<usize, Image>>,
}

impl MemorySink {
    pub fn new() -> Self {
        Self::default()
    }

    /// The frames received so far, by frame number.
    pub fn into_frames(self) -> BTreeMap<usize, Image> {
        self.frames.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

impl FrameSink for MemorySink {
    fn write(&self, frame: usize, image: &Image) -> Result<(), VideoError> {
        let previous = self.frames.lock().unwrap_or_else(|e| e.into_inner()).insert(frame, image.clone());
        if previous.is_some() {
            return invalid(format!("frame {frame} was written twice"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::scratch_dir;

    fn rgba(w: usize, h: usize, v: u8) -> Image {
        Image::from_raw(w, h, 4, vec![v; w * h * 4])
    }

    #[test]
    fn patterns() {
        let (dir, prefix, width, suffix) = split_pattern(Path::new("a/b/frame_[####].png")).unwrap();
        assert_eq!((dir, prefix.as_str(), width, suffix.as_str()), (PathBuf::from("a/b"), "frame_", 4, ".png"));
        let (_, prefix, width, suffix) = split_pattern(Path::new("##.jpg")).unwrap();
        assert_eq!((prefix.as_str(), width, suffix.as_str()), ("", 2, ".jpg"));
        assert_eq!(pattern_path("out/[###].png", 7).unwrap(), PathBuf::from("out/007.png"));
        assert_eq!(pattern_path("out/x#.png", 123).unwrap(), PathBuf::from("out/x123.png"));
        assert!(split_pattern(Path::new("out/frame.png")).is_err());
        assert_eq!(trailing_number("frame_0042"), Some(("frame_", "0042")));
        assert_eq!(trailing_number("frame"), None);
    }

    #[test]
    fn sequences_from_dir_and_pattern() {
        let dir = scratch_dir("frames-seq");
        fs::create_dir_all(&dir).unwrap();
        for t in 3..6 {
            io::save_png(&rgba(4, 3, t as u8), dir.join(format!("f{t:03}.png"))).unwrap();
        }
        fs::write(dir.join("notes.txt"), "x").unwrap();
        let seq = ImageSequence::from_dir(&dir).unwrap();
        assert_eq!((seq.first(), seq.len(), seq.last()), (3, 3, 5));
        assert_eq!(seq.load(4).unwrap(), rgba(4, 3, 4));
        assert!(seq.load(6).is_err());
        let seq = ImageSequence::from_pattern(dir.join("f[###].png")).unwrap();
        assert_eq!((seq.first(), seq.len()), (3, 3));
        // A gap is an error.
        fs::remove_file(dir.join("f004.png")).unwrap();
        assert!(ImageSequence::from_dir(&dir).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn writer_and_cache() {
        let dir = scratch_dir("frames-writer");
        let writer = ImageSequenceWriter::new(dir.join("out").join("o_[####].png")).unwrap();
        writer.write(12, &rgba(5, 5, 9)).unwrap();
        let path = dir.join("out").join("o_0012.png");
        assert_eq!(writer.path(12), path);
        assert_eq!(io::load_rgba(&path).unwrap(), rgba(5, 5, 9));

        let frames = MemoryFrames::new(10, (0..4).map(|v| rgba(8, 8, v)).collect()).unwrap();
        // Room for one frame only: the others spill.
        let cache = FrameCache::new(&frames, 8 * 8 * 4, Some(dir.join("spill")));
        for t in 10..14 {
            assert_eq!(*cache.get(t).unwrap(), rgba(8, 8, (t - 10) as u8));
        }
        assert_eq!(*cache.get(10).unwrap(), rgba(8, 8, 0));
        assert!(cache.stats().spilled >= 3);
        assert!(cache.get(14).is_err());
        assert_eq!(cache.gray(11).unwrap().get(0, 0), 1.0);
        drop(cache);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cache_rejects_mixed_sizes() {
        let frames = MemoryFrames::new(0, vec![rgba(8, 8, 0), rgba(9, 8, 0)]).unwrap();
        let cache = FrameCache::new(&frames, 1 << 20, None);
        cache.get(0).unwrap();
        assert!(cache.get(1).is_err());
    }

    #[test]
    fn memory_sink_rejects_duplicates() {
        let sink = MemorySink::new();
        sink.write(1, &rgba(2, 2, 0)).unwrap();
        assert!(sink.write(1, &rgba(2, 2, 0)).is_err());
        assert_eq!(sink.into_frames().len(), 1);
    }
}
