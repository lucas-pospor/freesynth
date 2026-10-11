//! A bounded in-memory image store that spills to disk.
//!
//! The pipeline keeps two kinds of frames for a while: decoded input frames (read by several
//! keyframe chains, in both directions) and synthesized frames waiting for the other pass of their
//! interval. Both can outgrow the machine's memory for long or large shots: one interval of 4K
//! frames is several gigabytes. A [`SpillStore`] holds images up to a byte budget and writes the
//! least recently used ones to raw files in a spill directory beyond it, reading them back on
//! demand. Disk writes happen outside the store's lock, so other threads are not held up by them.

use std::collections::HashMap;
use std::fs;
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use freesynth_core::Image;

use crate::error::{VideoError, io_error};

/// Magic bytes at the start of a spill file, followed by width, height and channels (`u32`
/// little-endian each) and the raw pixel data.
const MAGIC: &[u8; 6] = b"FSRAW1";

/// A key of a [`SpillStore`]: identifies an image and names its spill file.
pub trait SpillKey: Clone + Eq + Hash + Send + Sync {
    /// A file name stem unique among the store's keys (no directory separators).
    fn file_stem(&self) -> String;
}

impl SpillKey for usize {
    fn file_stem(&self) -> String {
        format!("{self:06}")
    }
}

/// Counters of a [`SpillStore`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreStats {
    /// Largest number of bytes held in memory at once.
    pub peak_bytes: usize,
    /// Images written to the spill directory.
    pub spilled: usize,
    /// Bytes written to the spill directory.
    pub spilled_bytes: u64,
    /// Images read back from the spill directory.
    pub reloaded: usize,
}

/// An image held in memory.
struct Resident {
    image: Arc<Image>,
    last_use: u64,
    /// The same image is also in the spill directory, so it can be dropped without a write.
    on_disk: bool,
}

struct Inner<K> {
    resident: HashMap<K, Resident>,
    /// Images being written to disk (still counted in `used` and still readable).
    writing: HashMap<K, Arc<Image>>,
    /// Images only on disk.
    disk: HashMap<K, PathBuf>,
    used: usize,
    clock: u64,
    stats: StoreStats,
}

/// What [`SpillStore::lookup`] found for a key.
enum Found {
    Memory(Arc<Image>),
    Disk(PathBuf),
    Missing,
}

/// Images by key, kept in memory up to a byte budget and spilled to disk beyond it.
///
/// Without a spill directory the budget is not enforced and everything stays in memory.
/// Spill files are deleted when their image is removed and when the store is dropped.
///
/// An image being written out still occupies memory until the write finishes, and other threads
/// may insert meanwhile, so the memory held can briefly exceed the budget by about one image per
/// thread using the store.
pub struct SpillStore<K: SpillKey> {
    dir: Option<PathBuf>,
    budget: usize,
    inner: Mutex<Inner<K>>,
}

fn image_bytes(image: &Image) -> usize {
    image.data().len()
}

impl<K: SpillKey> SpillStore<K> {
    /// A store that keeps up to `budget` bytes in memory and spills to `dir` (created on first
    /// use) beyond that. With `dir` = `None`, nothing is ever spilled.
    pub fn new(budget: usize, dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            budget,
            inner: Mutex::new(Inner {
                resident: HashMap::new(),
                writing: HashMap::new(),
                disk: HashMap::new(),
                used: 0,
                clock: 0,
                stats: StoreStats::default(),
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner<K>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Changes the memory budget. It applies from the next insertion on.
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
    }

    /// The memory budget in bytes.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// The store's counters so far.
    pub fn stats(&self) -> StoreStats {
        self.lock().stats
    }

    /// Bytes currently held in memory.
    pub fn memory_bytes(&self) -> usize {
        self.lock().used
    }

    /// Number of images in the store, in memory or on disk.
    pub fn len(&self) -> usize {
        let g = self.lock();
        g.resident.len() + g.writing.len() + g.disk.len()
    }

    /// True when the store holds no image.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds (or replaces) an image, spilling older images if the budget is exceeded.
    pub fn insert(&self, key: K, image: Arc<Image>) -> Result<(), VideoError> {
        let mut g = self.lock();
        self.forget(&mut g, &key);
        Self::put(&mut g, key, image, false);
        self.enforce(g)
    }

    /// Returns the image for `key`, reading it back from disk if it was spilled (it is then
    /// kept in memory again, which may spill others).
    pub fn get(&self, key: &K) -> Result<Option<Arc<Image>>, VideoError> {
        let mut g = self.lock();
        let path = match Self::lookup(&mut g, key) {
            Found::Memory(image) => return Ok(Some(image)),
            Found::Missing => return Ok(None),
            Found::Disk(path) => path,
        };
        drop(g);
        let image = Arc::new(read_raw(&path)?);
        let mut g = self.lock();
        g.stats.reloaded += 1;
        // Another thread may have removed or reloaded it meanwhile.
        if g.disk.remove(key).is_some() {
            Self::put(&mut g, key.clone(), Arc::clone(&image), true);
            self.enforce(g)?;
        }
        Ok(Some(image))
    }

    /// Removes the image for `key` and returns it (read back from disk if it was spilled).
    pub fn take(&self, key: &K) -> Result<Option<Arc<Image>>, VideoError> {
        let mut g = self.lock();
        match Self::lookup(&mut g, key) {
            Found::Memory(image) => {
                self.forget(&mut g, key);
                Ok(Some(image))
            }
            Found::Missing => Ok(None),
            Found::Disk(path) => {
                g.disk.remove(key);
                g.stats.reloaded += 1;
                drop(g);
                Self::read_and_delete(&path).map(Some)
            }
        }
    }

    /// Atomically: if `partner` is in the store, removes and returns it without adding `image`;
    /// otherwise adds `image` under `key` and returns `None`. Two threads that each insert one of
    /// two partner images this way never both miss, and never both find, the other.
    pub fn insert_or_take(&self, key: K, image: Arc<Image>, partner: &K) -> Result<Option<Arc<Image>>, VideoError> {
        let mut g = self.lock();
        match Self::lookup(&mut g, partner) {
            Found::Missing => {
                self.forget(&mut g, &key);
                Self::put(&mut g, key, image, false);
                self.enforce(g)?;
                Ok(None)
            }
            Found::Memory(found) => {
                self.forget(&mut g, partner);
                Ok(Some(found))
            }
            Found::Disk(path) => {
                g.disk.remove(partner);
                g.stats.reloaded += 1;
                drop(g);
                Self::read_and_delete(&path).map(Some)
            }
        }
    }

    /// Reads a spill file that is no longer referenced, then deletes it.
    fn read_and_delete(path: &Path) -> Result<Arc<Image>, VideoError> {
        let image = read_raw(path);
        let _ = fs::remove_file(path);
        image.map(Arc::new)
    }

    fn lookup(g: &mut Inner<K>, key: &K) -> Found {
        g.clock += 1;
        let clock = g.clock;
        if let Some(r) = g.resident.get_mut(key) {
            r.last_use = clock;
            return Found::Memory(Arc::clone(&r.image));
        }
        if let Some(image) = g.writing.get(key) {
            return Found::Memory(Arc::clone(image));
        }
        match g.disk.get(key) {
            Some(path) => Found::Disk(path.clone()),
            None => Found::Missing,
        }
    }

    fn put(g: &mut Inner<K>, key: K, image: Arc<Image>, on_disk: bool) {
        g.clock += 1;
        g.used += image_bytes(&image);
        g.stats.peak_bytes = g.stats.peak_bytes.max(g.used);
        let last_use = g.clock;
        g.resident.insert(key, Resident { image, last_use, on_disk });
    }

    /// Drops every trace of `key` (a write in flight finds the key gone and deletes its file).
    fn forget(&self, g: &mut Inner<K>, key: &K) {
        if let Some(r) = g.resident.remove(key) {
            g.used -= image_bytes(&r.image);
            if r.on_disk && let Some(path) = self.path_for(key) {
                let _ = fs::remove_file(path);
            }
        }
        if let Some(image) = g.writing.remove(key) {
            g.used -= image_bytes(&image);
        }
        if let Some(path) = g.disk.remove(key) {
            let _ = fs::remove_file(path);
        }
    }

    fn path_for(&self, key: &K) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join(format!("{}.raw", key.file_stem())))
    }

    /// Spills least recently used images until the budget holds. Writes happen without the lock.
    fn enforce(&self, mut g: MutexGuard<'_, Inner<K>>) -> Result<(), VideoError> {
        if self.dir.is_none() {
            return Ok(());
        }
        let mut victims = Vec::new();
        // Bytes that the writes below will free.
        let mut freeing = 0;
        while g.used - freeing > self.budget {
            let Some(key) = g.resident.iter().min_by_key(|(_, r)| r.last_use).map(|(k, _)| k.clone()) else {
                break;
            };
            let r = g.resident.remove(&key).expect("key was just found");
            if r.on_disk {
                // Clean copy: dropping it frees the memory.
                g.used -= image_bytes(&r.image);
                let path = self.path_for(&key).expect("spilling needs a directory");
                g.disk.insert(key, path);
            } else {
                freeing += image_bytes(&r.image);
                g.writing.insert(key.clone(), Arc::clone(&r.image));
                victims.push((key, r.image));
            }
        }
        drop(g);
        if victims.is_empty() {
            return Ok(());
        }
        let dir = self.dir.as_ref().expect("checked above");
        fs::create_dir_all(dir).map_err(io_error(dir))?;
        let mut result = Ok(());
        for (key, image) in victims {
            let path = self.path_for(&key).expect("spilling needs a directory");
            let written = write_raw(&path, &image);
            let mut g = self.lock();
            let still_wanted = g.writing.remove(&key).is_some();
            if still_wanted {
                g.used -= image_bytes(&image);
            }
            match written {
                Ok(()) if still_wanted => {
                    g.stats.spilled += 1;
                    g.stats.spilled_bytes += image_bytes(&image) as u64;
                    g.disk.insert(key, path);
                }
                Ok(()) => {
                    let _ = fs::remove_file(&path);
                }
                Err(e) => {
                    // Keep the image in memory rather than losing it.
                    if still_wanted {
                        Self::put(&mut g, key, image, false);
                    }
                    let _ = fs::remove_file(&path);
                    if result.is_ok() {
                        result = Err(e);
                    }
                }
            }
        }
        result
    }
}

impl<K: SpillKey> Drop for SpillStore<K> {
    fn drop(&mut self) {
        let g = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        let mut paths: Vec<PathBuf> = g.disk.drain().map(|(_, p)| p).collect();
        for (k, r) in g.resident.drain() {
            if r.on_disk && let Some(p) = self.dir.as_ref().map(|d| d.join(format!("{}.raw", k.file_stem()))) {
                paths.push(p);
            }
        }
        for p in paths {
            let _ = fs::remove_file(p);
        }
        if let Some(dir) = &self.dir {
            // Only succeeds when empty, which is what we want.
            let _ = fs::remove_dir(dir);
        }
    }
}

/// Writes an image as a raw spill file.
fn write_raw(path: &Path, image: &Image) -> Result<(), VideoError> {
    let file = fs::File::create(path).map_err(io_error(path))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    let header = [image.width() as u32, image.height() as u32, image.channels() as u32];
    w.write_all(MAGIC).map_err(io_error(path))?;
    for v in header {
        w.write_all(&v.to_le_bytes()).map_err(io_error(path))?;
    }
    w.write_all(image.data()).map_err(io_error(path))?;
    w.flush().map_err(io_error(path))
}

/// Reads a raw spill file written by [`write_raw`].
fn read_raw(path: &Path) -> Result<Image, VideoError> {
    let file = fs::File::open(path).map_err(io_error(path))?;
    let mut r = BufReader::with_capacity(1 << 20, file);
    let mut magic = [0u8; 6];
    r.read_exact(&mut magic).map_err(io_error(path))?;
    let mut header = [0u8; 12];
    r.read_exact(&mut header).map_err(io_error(path))?;
    let field = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().expect("4 bytes")) as usize;
    let (w, h, c) = (field(0), field(1), field(2));
    let len = w.checked_mul(h).and_then(|n| n.checked_mul(c));
    let (true, Some(len)) = (&magic == MAGIC && (1..=8).contains(&c), len) else {
        return Err(VideoError::Io {
            path: path.to_owned(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, "not a spill file"),
        });
    };
    let mut data = vec![0u8; len];
    r.read_exact(&mut data).map_err(io_error(path))?;
    Ok(Image::from_raw(w, h, c, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::scratch_dir;

    fn image(seed: u8, w: usize, h: usize) -> Arc<Image> {
        let data = (0..w * h * 4).map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed)).collect();
        Arc::new(Image::from_raw(w, h, 4, data))
    }

    #[test]
    fn keeps_everything_within_the_budget() {
        let store = SpillStore::new(1 << 20, Some(scratch_dir("store-budget")));
        for i in 0..4 {
            store.insert(i, image(i as u8, 16, 16)).unwrap();
        }
        assert_eq!(store.stats().spilled, 0);
        assert_eq!(store.memory_bytes(), 4 * 16 * 16 * 4);
        assert_eq!(*store.get(&2).unwrap().unwrap(), *image(2, 16, 16));
    }

    #[test]
    fn spills_least_recently_used_and_reads_back() {
        let dir = scratch_dir("store-spill");
        let frame = 32 * 32 * 4;
        let store = SpillStore::new(2 * frame, Some(dir.clone()));
        for i in 0..5usize {
            store.insert(i, image(i as u8, 32, 32)).unwrap();
            assert!(store.memory_bytes() <= 2 * frame);
        }
        let s = store.stats();
        assert_eq!(s.spilled, 3);
        assert_eq!(s.spilled_bytes, 3 * frame as u64);
        assert!(dir.join("000000.raw").exists());
        assert_eq!(store.len(), 5);
        // Reading a spilled image brings it back (and spills another one).
        assert_eq!(*store.get(&0).unwrap().unwrap(), *image(0, 32, 32));
        assert_eq!(store.stats().reloaded, 1);
        assert!(store.memory_bytes() <= 2 * frame);
        // Taking removes the file.
        let taken = store.take(&1).unwrap().unwrap();
        assert_eq!(*taken, *image(1, 32, 32));
        assert!(!dir.join("000001.raw").exists());
        assert_eq!(store.take(&1).unwrap(), None);
        assert_eq!(store.len(), 4);
        drop(store);
        assert!(!dir.exists(), "spill directory is removed when the store is dropped");
    }

    #[test]
    fn insert_or_take_pairs_partners() {
        let store = SpillStore::new(32 * 32 * 4, Some(scratch_dir("store-partner")));
        assert_eq!(store.insert_or_take(1, image(1, 32, 32), &2).unwrap(), None);
        // Push key 1 to disk.
        store.insert(9, image(9, 32, 32)).unwrap();
        assert_eq!(store.stats().spilled, 1);
        let partner = store.insert_or_take(2, image(2, 32, 32), &1).unwrap().unwrap();
        assert_eq!(*partner, *image(1, 32, 32));
        assert_eq!(store.get(&2).unwrap(), None, "the second image is not stored");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn without_a_directory_nothing_spills() {
        let store = SpillStore::new(10, None);
        for i in 0..3usize {
            store.insert(i, image(0, 8, 8)).unwrap();
        }
        assert_eq!(store.stats().spilled, 0);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn rejects_foreign_files() {
        let dir = scratch_dir("store-foreign");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.raw");
        fs::write(&path, b"not an image at all").unwrap();
        assert!(read_raw(&path).is_err());
        let _ = fs::remove_dir_all(dir);
    }
}
