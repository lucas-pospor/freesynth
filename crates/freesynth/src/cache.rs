//! The render cache: which output frames are up to date, and which parts of a shot must be
//! rendered again.
//!
//! Every output frame gets a key, a hash of everything it depends on. A keyframe chain carries
//! its state from frame to frame (the previous output feeds the temporal guide, tracked
//! positions accumulate), so a chain frame's key is a hash chain: the key of the frame before
//! it in the chain, plus this frame's inputs. The chain starts from a seed that covers the look
//! settings, the keyframe painting and the keyframe's video frame. A blended frame's key
//! combines the keys of both passes.
//!
//! After a render, [`Manifest`] remembers each written frame's key, path and file stamp. On the
//! next render, a frame is reused when its key, path and file are unchanged; the rest are
//! *dirty*. Paths are stored relative to the project's directory, so that a project folder
//! moved as a whole keeps its cache. Dirty frames are grouped into the smallest set of shot segments that reproduce
//! them ([`dirty_runs`]): editing one keyframe dirties its own chains and the blends next to
//! it, which is the range between its two neighbouring keyframes, and nothing else.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use freesynth_video::plan::{ChainOutput, Pass, Plan};
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::hashing::{FileHashes, FileStamp, Hash, Hasher};

/// Version of the manifest layout; a different version is treated as an empty cache. (Changes
/// to how frames are rendered are covered by the render revision in the frame keys instead,
/// see `OUTPUT_REVISION` in the render module.)
const MANIFEST_VERSION: u32 = 2;

/// The key of every output frame of `plan`.
///
/// - `look`: hash of everything that applies to every frame (parameters, device kind);
/// - `paintings[i]`: hash of the painting of `plan.keyframes[i]`;
/// - `frame(t)` and `mask(t)`: hashes of input frame `t` and its mask frame.
pub fn frame_keys(
    plan: &Plan,
    look: &Hash,
    paintings: &[Hash],
    frame: &dyn Fn(usize) -> Hash,
    mask: &dyn Fn(usize) -> Option<Hash>,
) -> BTreeMap<usize, Hash> {
    let seeds: Vec<Hash> = plan
        .keyframes
        .iter()
        .zip(paintings)
        .map(|(&k, painting)| {
            Hasher::new("seed").hash(look).hash(painting).u64(k as u64).hash(&frame(k)).opt(mask(k).as_ref()).finish()
        })
        .collect();
    let mut keys = BTreeMap::new();
    for (&k, seed) in plan.keyframes.iter().zip(&seeds) {
        keys.insert(k, Hasher::new("keyframe").hash(seed).finish());
    }
    let mut passes: BTreeMap<(usize, usize), [Option<Hash>; 2]> = BTreeMap::new();
    for chain in &plan.chains {
        let mut state = seeds[chain.keyframe];
        for t in chain.frames() {
            state = Hasher::new("chain").hash(&state).u64(t as u64).hash(&frame(t)).opt(mask(t).as_ref()).finish();
            match chain.output {
                ChainOutput::Direct => {
                    keys.insert(t, state);
                }
                ChainOutput::Blend { interval, pass } => {
                    let slot = passes.entry((interval, t)).or_default();
                    slot[usize::from(pass == Pass::B)] = Some(state);
                }
            }
        }
    }
    for ((_, t), [a, b]) in passes {
        let (a, b) = (a.expect("pass A covers its interval"), b.expect("pass B covers its interval"));
        keys.insert(t, Hasher::new("blend").hash(&a).hash(&b).finish());
    }
    keys
}

/// Groups dirty frames into the shot segments to render: each run `(first, last)` is a frame
/// range whose ends are keyframes or the shot's ends, rendered with the keyframes inside it.
///
/// The shot splits into segments that share their keyframe ends: the lead-in before the first
/// keyframe, each keyframe's own frame, each interval between neighbouring keyframes and the
/// lead-out after the last. A segment is needed when it produces a dirty frame; needed
/// segments that touch are merged into one run.
pub fn dirty_runs(plan: &Plan, dirty: &BTreeSet<usize>) -> Vec<(usize, usize)> {
    let (first, last) = (*plan.frames.start(), *plan.frames.end());
    let keys = &plan.keyframes;
    // (range, frames it produces)
    let mut segments: Vec<((usize, usize), (usize, usize))> = Vec::new();
    if keys[0] > first {
        segments.push(((first, keys[0]), (first, keys[0] - 1)));
    }
    for (i, &k) in keys.iter().enumerate() {
        segments.push(((k, k), (k, k)));
        if let Some(&next) = keys.get(i + 1)
            && next - k >= 2
        {
            segments.push(((k, next), (k + 1, next - 1)));
        }
    }
    let n = keys[keys.len() - 1];
    if n < last {
        segments.push(((n, last), (n + 1, last)));
    }
    let mut needed: Vec<(usize, usize)> = segments
        .into_iter()
        .filter(|(_, (a, b))| dirty.range(*a..=*b).next().is_some())
        .map(|(range, _)| range)
        .collect();
    needed.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (a, b) in needed {
        match runs.last_mut() {
            Some(run) if a <= run.1 => run.1 = run.1.max(b),
            _ => runs.push((a, b)),
        }
    }
    runs
}

/// The frames whose output depends on keyframe `k`'s chains: its own frame and the frames up
/// to (not including) its neighbouring keyframes, or the shot's ends.
pub fn keyframe_reach(plan: &Plan, k: usize) -> Option<(usize, usize)> {
    let i = plan.keyframes.iter().position(|&f| f == k)?;
    let lo = if i == 0 { *plan.frames.start() } else { plan.keyframes[i - 1] + 1 };
    let hi = plan.keyframes.get(i + 1).map_or(*plan.frames.end(), |&next| next - 1);
    Some((lo, hi))
}

/// What the cache knows about one written output frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameRecord {
    /// The frame's key ([`frame_keys`] combined with the output format).
    pub key: Hash,
    /// Where it was written, relative to the project's directory.
    pub path: PathBuf,
    /// The file's stamp right after writing (a changed stamp means someone replaced it).
    pub stamp: FileStamp,
}

/// What the cache knows about the encoded output video.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoRecord {
    pub key: Hash,
    /// The video file, relative to the project's directory.
    pub path: PathBuf,
    pub stamp: FileStamp,
}

/// The record of a project's last renders, stored as `manifest.json` in the cache directory.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    version: u32,
    /// Written output frames, by frame number.
    frames: BTreeMap<usize, FrameRecord>,
    /// Lossless copies of the output frames the video is encoded from, when the output frames
    /// are in a lossy format (JPEG), by frame number.
    #[serde(default)]
    masters: BTreeMap<usize, FrameRecord>,
    /// The output video, once encoded.
    video: Option<VideoRecord>,
    /// Remembered input file hashes.
    pub files: FileHashes,
    /// The project's directory, which stored paths are relative to.
    #[serde(skip)]
    base: PathBuf,
}

/// Which records of the manifest a frame file belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// The project's output frames.
    Output,
    /// The lossless copies the video is encoded from.
    Master,
}

impl Manifest {
    /// The manifest file in `cache_dir`.
    pub fn path(cache_dir: &Path) -> PathBuf {
        cache_dir.join("manifest.json")
    }

    /// Loads the manifest of `cache_dir` for the project in directory `base`; a missing,
    /// unreadable or outdated one gives an empty manifest (everything renders again).
    pub fn load(cache_dir: &Path, base: &Path) -> Self {
        let mut m = fs::read(Self::path(cache_dir))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Manifest>(&bytes).ok())
            .filter(|m| m.version == MANIFEST_VERSION)
            .unwrap_or_else(|| Self { version: MANIFEST_VERSION, ..Self::default() });
        m.base = base.to_owned();
        m
    }

    /// Writes the manifest atomically.
    pub fn save(&self, cache_dir: &Path) -> Result<(), Error> {
        fs::create_dir_all(cache_dir).map_err(|source| Error::Io { path: cache_dir.to_owned(), source })?;
        let mut this = self.clone();
        this.version = MANIFEST_VERSION;
        let bytes = serde_json::to_vec(&this).expect("manifests serialize");
        let path = Self::path(cache_dir);
        freesynth_media::write_atomic(&path, &bytes).map_err(Error::Media)
    }

    /// `path` as stored: relative to the project's directory.
    fn stored(&self, path: &Path) -> PathBuf {
        crate::paths::relative_to(path, &self.base)
    }

    fn records(&self, kind: FrameKind) -> &BTreeMap<usize, FrameRecord> {
        match kind {
            FrameKind::Output => &self.frames,
            FrameKind::Master => &self.masters,
        }
    }

    /// True when frame `t` was written to `path` with `key` and the file is unchanged since.
    pub fn is_current(&self, kind: FrameKind, t: usize, key: &Hash, path: &Path) -> bool {
        self.records(kind).get(&t).is_some_and(|r| {
            &r.key == key && r.path == self.stored(path) && FileStamp::of(path).is_ok_and(|stamp| stamp == r.stamp)
        })
    }

    /// Records that frame `t` was written to `path` with `key`.
    pub fn record(&mut self, kind: FrameKind, t: usize, key: Hash, path: &Path, stamp: FileStamp) {
        let record = FrameRecord { key, path: self.stored(path), stamp };
        match kind {
            FrameKind::Output => self.frames.insert(t, record),
            FrameKind::Master => self.masters.insert(t, record),
        };
    }

    /// Forgets frame `t` (about to be written again).
    pub fn forget(&mut self, t: usize) {
        self.frames.remove(&t);
        self.masters.remove(&t);
    }

    /// True when the video at `path` was encoded with `key` and is unchanged since.
    pub fn video_is_current(&self, key: &Hash, path: &Path) -> bool {
        self.video.as_ref().is_some_and(|r| {
            &r.key == key && r.path == self.stored(path) && FileStamp::of(path).is_ok_and(|stamp| stamp == r.stamp)
        })
    }

    /// Records that the video at `path` was encoded with `key`.
    pub fn record_video(&mut self, key: Hash, path: &Path, stamp: FileStamp) {
        self.video = Some(VideoRecord { key, path: self.stored(path), stamp });
    }
}

/// An exclusive lock on a cache directory, so that two renders of one project do not write
/// the same files at once. Released on drop.
pub struct CacheLock {
    path: PathBuf,
}

impl CacheLock {
    /// Takes the lock of `cache_dir`. A lock left by a process that no longer runs is taken
    /// over (checked on Linux; elsewhere remove the `lock` file by hand).
    pub fn acquire(cache_dir: &Path) -> Result<Self, Error> {
        fs::create_dir_all(cache_dir).map_err(|source| Error::Io { path: cache_dir.to_owned(), source })?;
        let path = cache_dir.join("lock");
        for _ in 0..2 {
            match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    use std::io::Write;
                    let _ = write!(f, "{}", std::process::id());
                    return Ok(Self { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let owner = fs::read_to_string(&path).ok().and_then(|s| s.trim().parse::<u32>().ok());
                    let stale = owner.is_some_and(|pid| {
                        pid != std::process::id()
                            && cfg!(target_os = "linux")
                            && !Path::new(&format!("/proc/{pid}")).exists()
                    });
                    if stale {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    return Err(Error::Project(format!(
                        "another render of this project is running (process {}); if not, delete '{}'",
                        owner.map_or_else(|| "?".to_owned(), |p| p.to_string()),
                        path.display()
                    )));
                }
                Err(source) => return Err(Error::Io { path, source }),
            }
        }
        Err(Error::Project(format!("cannot take the render lock '{}'", path.display())))
    }
}

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(t: usize) -> Hash {
        Hasher::new("frame").u64(t as u64).finish()
    }

    fn keys_for(plan: &Plan, paintings: &[Hash], changed_frame: Option<usize>) -> BTreeMap<usize, Hash> {
        let frame = |t: usize| if Some(t) == changed_frame { Hasher::new("edited").finish() } else { h(t) };
        frame_keys(plan, &Hasher::new("look").finish(), paintings, &frame, &|_| None)
    }

    fn changed(a: &BTreeMap<usize, Hash>, b: &BTreeMap<usize, Hash>) -> BTreeSet<usize> {
        a.iter().filter(|(t, k)| b.get(t) != Some(k)).map(|(&t, _)| t).collect()
    }

    #[test]
    fn every_frame_gets_a_key() {
        let plan = Plan::new(0, 30, &[5, 15, 16, 25]).unwrap();
        let paintings: Vec<Hash> = (0..4).map(|i| Hasher::new("p").u64(i).finish()).collect();
        let keys = keys_for(&plan, &paintings, None);
        assert_eq!(keys.len(), 31);
        assert_eq!(keys.keys().copied().collect::<Vec<_>>(), (0..=30).collect::<Vec<_>>());
        let distinct: BTreeSet<_> = keys.values().collect();
        assert_eq!(distinct.len(), 31);
    }

    #[test]
    fn editing_a_keyframe_dirties_only_its_reach() {
        let plan = Plan::new(0, 40, &[0, 10, 20, 30]).unwrap();
        let paintings: Vec<Hash> = (0..4).map(|i| Hasher::new("p").u64(i).finish()).collect();
        let before = keys_for(&plan, &paintings, None);
        let mut edited = paintings.clone();
        edited[2] = Hasher::new("p").str("repainted").finish();
        let after = keys_for(&plan, &edited, None);
        let dirty = changed(&before, &after);
        // Keyframe 20's chains reach 11..=29 (its intervals with 10 and 30); 10 and 30 stay.
        assert_eq!(dirty, (11..=29).collect());
        assert_eq!(keyframe_reach(&plan, 20), Some((11, 29)));
        assert_eq!(dirty_runs(&plan, &dirty), [(10, 30)]);
        // The last keyframe also owns the lead-out.
        let mut edited = paintings.clone();
        edited[3] = Hasher::new("p").str("x").finish();
        let dirty = changed(&before, &keys_for(&plan, &edited, None));
        assert_eq!(dirty, (21..=40).collect());
        assert_eq!(dirty_runs(&plan, &dirty), [(20, 40)]);
    }

    #[test]
    fn an_input_frame_change_follows_the_chains() {
        let plan = Plan::new(0, 20, &[0, 20]).unwrap();
        let paintings = [Hasher::new("a").finish(), Hasher::new("b").finish()];
        let before = keys_for(&plan, &paintings, None);
        let after = keys_for(&plan, &paintings, Some(5));
        // Pass A changes from 5 onwards, pass B from 5 downwards: every blended frame.
        assert_eq!(changed(&before, &after), (1..=19).collect());
        // A keyframe's own video frame changes its whole reach.
        let after = keys_for(&plan, &paintings, Some(20));
        assert_eq!(changed(&before, &after), (1..=20).collect());
    }

    #[test]
    fn runs_merge_touching_segments() {
        let plan = Plan::new(0, 50, &[10, 20, 30, 40]).unwrap();
        let set = |v: &[usize]| v.iter().copied().collect::<BTreeSet<usize>>();
        assert_eq!(dirty_runs(&plan, &set(&[])), []);
        assert_eq!(dirty_runs(&plan, &set(&[3])), [(0, 10)]);
        assert_eq!(dirty_runs(&plan, &set(&[20])), [(20, 20)]);
        assert_eq!(dirty_runs(&plan, &set(&[15, 25])), [(10, 30)]);
        assert_eq!(dirty_runs(&plan, &set(&[15, 35])), [(10, 20), (30, 40)]);
        assert_eq!(dirty_runs(&plan, &set(&[45, 0])), [(0, 10), (40, 50)]);
        let all: BTreeSet<usize> = (0..=50).collect();
        assert_eq!(dirty_runs(&plan, &all), [(0, 50)]);
        // Adjacent keyframes have no interval segment.
        let plan = Plan::new(0, 3, &[1, 2]).unwrap();
        assert_eq!(dirty_runs(&plan, &set(&[2])), [(2, 2)]);
        assert_eq!(dirty_runs(&plan, &set(&[0, 3])), [(0, 1), (2, 3)]);
    }

    #[test]
    fn manifest_round_trip_and_lock() {
        let dir = crate::test_util::scratch("manifest");
        let out_kind = FrameKind::Output;
        assert_eq!(Manifest::load(&dir, &dir).frames.len(), 0, "a missing manifest is empty");
        fs::create_dir_all(&dir).unwrap();
        let out = dir.join("f.png");
        fs::write(&out, b"png").unwrap();
        let key = h(1);
        let mut m = Manifest::load(&dir, &dir);
        m.record(out_kind, 1, key, &out, FileStamp::of(&out).unwrap());
        assert_eq!(m.frames[&1].path, PathBuf::from("f.png"), "stored relative to the project");
        m.save(&dir).unwrap();
        let back = Manifest::load(&dir, &dir);
        assert_eq!(back, Manifest { version: MANIFEST_VERSION, ..m });
        assert!(back.is_current(out_kind, 1, &key, &out));
        assert!(!back.is_current(FrameKind::Master, 1, &key, &out));
        assert!(!back.is_current(out_kind, 1, &h(2), &out));
        assert!(!back.is_current(out_kind, 2, &key, &out));

        // The project folder moved: the same relative path in the new place is current.
        let moved = crate::test_util::scratch("manifest-moved");
        fs::rename(&dir, &moved).unwrap();
        let there = Manifest::load(&moved, &moved);
        assert!(there.is_current(out_kind, 1, &key, &moved.join("f.png")));
        fs::rename(&moved, &dir).unwrap();

        fs::write(&out, b"replaced").unwrap();
        assert!(!back.is_current(out_kind, 1, &key, &out), "a replaced file is not current");

        let lock = CacheLock::acquire(&dir).unwrap();
        assert!(CacheLock::acquire(&dir).is_err());
        drop(lock);
        drop(CacheLock::acquire(&dir).unwrap());
        fs::remove_dir_all(dir).unwrap();
    }
}
