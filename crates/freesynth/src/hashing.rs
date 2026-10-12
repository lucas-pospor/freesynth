//! Content hashes for the render cache.
//!
//! Every input that can change a rendered frame is identified by a BLAKE3 hash: file contents,
//! settings, and the hashes of upstream frames. Hashing a long image sequence on every render
//! would cost seconds, so file hashes are remembered together with the file's size and
//! modification time ([`FileHashes`]) and recomputed only when either changes.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// A 256-bit content hash.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hash(pub [u8; 32]);

impl Hash {
    /// The hash in lower-case hex.
    pub fn to_hex(self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Parses 64 hex digits.
    pub fn from_hex(text: &str) -> Option<Self> {
        if text.len() != 64 || !text.is_ascii() {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(Self(out))
    }

    /// The first 16 hex digits (for directory names).
    pub fn short(&self) -> String {
        self.to_hex()[..16].to_owned()
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash({})", self.short())
    }
}

impl Serialize for Hash {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Hash {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Hash::from_hex(&text).ok_or_else(|| serde::de::Error::custom("bad hash"))
    }
}

/// Builds a hash from tagged parts. Each part is length-prefixed, so different splits of the
/// same bytes never collide.
pub struct Hasher(blake3::Hasher);

impl Hasher {
    /// A hasher for values of kind `domain` (hashes of different kinds never coincide).
    pub fn new(domain: &str) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"freesynth-cache-1\0");
        let mut this = Self(h);
        this.bytes(domain.as_bytes());
        this
    }

    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.0.update(&(bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
        self
    }

    pub fn str(&mut self, text: &str) -> &mut Self {
        self.bytes(text.as_bytes())
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.bytes(&value.to_le_bytes())
    }

    pub fn hash(&mut self, hash: &Hash) -> &mut Self {
        self.bytes(&hash.0)
    }

    pub fn opt(&mut self, hash: Option<&Hash>) -> &mut Self {
        match hash {
            Some(h) => self.u64(1).hash(h),
            None => self.u64(0),
        }
    }

    pub fn finish(&self) -> Hash {
        Hash(*self.0.finalize().as_bytes())
    }
}

/// Hashes a file's contents.
pub fn hash_file_contents(path: &Path) -> Result<Hash, Error> {
    let io = |source| Error::Io { path: path.to_owned(), source };
    let mut file = fs::File::open(path).map_err(io)?;
    let mut h = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(io)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(Hash(*h.finalize().as_bytes()))
}

/// A file's size and modification time, the cheap test for "unchanged".
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub struct FileStamp {
    pub size: u64,
    /// Modification time in nanoseconds since the Unix epoch.
    pub mtime_ns: i128,
}

impl FileStamp {
    /// The stamp of the file at `path`.
    pub fn of(path: &Path) -> Result<Self, Error> {
        let meta = fs::metadata(path).map_err(|source| Error::Io { path: path.to_owned(), source })?;
        let mtime_ns = meta
            .modified()
            .ok()
            .map(|t| match t.duration_since(UNIX_EPOCH) {
                Ok(d) => d.as_nanos() as i128,
                Err(e) => -(e.duration().as_nanos() as i128),
            })
            .unwrap_or(0);
        Ok(Self { size: meta.len(), mtime_ns })
    }
}

/// One remembered file hash.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Entry {
    stamp: FileStamp,
    hash: Hash,
}

/// A file hashed by [`FileHashes::hash_all`]: its index, stamp and hash.
type Hashed = (usize, FileStamp, Hash);

/// File hashes remembered with each file's stamp. A file is re-read only when its size or
/// modification time changed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileHashes {
    files: BTreeMap<PathBuf, Entry>,
}

impl FileHashes {
    /// The hash of `path`'s contents, from memory when its stamp is unchanged.
    pub fn hash(&mut self, path: &Path) -> Result<Hash, Error> {
        let stamp = FileStamp::of(path)?;
        if let Some(e) = self.files.get(path)
            && e.stamp == stamp
        {
            return Ok(e.hash);
        }
        let hash = hash_file_contents(path)?;
        self.files.insert(path.to_owned(), Entry { stamp, hash });
        Ok(hash)
    }

    /// Hashes many files, using up to `threads` threads for the ones that changed.
    pub fn hash_all(&mut self, paths: &[PathBuf], threads: usize) -> Result<Vec<Hash>, Error> {
        let mut out: Vec<Option<Hash>> = vec![None; paths.len()];
        let mut todo = Vec::new();
        for (i, p) in paths.iter().enumerate() {
            let stamp = FileStamp::of(p)?;
            match self.files.get(p) {
                Some(e) if e.stamp == stamp => out[i] = Some(e.hash),
                _ => todo.push((i, stamp)),
            }
        }
        if !todo.is_empty() {
            let threads = threads.clamp(1, todo.len());
            let chunk = todo.len().div_ceil(threads);
            let results: Vec<Result<Vec<Hashed>, Error>> = std::thread::scope(|s| {
                let handles: Vec<_> = todo
                    .chunks(chunk)
                    .map(|part| {
                        s.spawn(move || {
                            part.iter()
                                .map(|&(i, stamp)| Ok((i, stamp, hash_file_contents(&paths[i])?)))
                                .collect::<Result<Vec<_>, Error>>()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("hashing threads do not panic")).collect()
            });
            for part in results {
                for (i, stamp, hash) in part? {
                    self.files.insert(paths[i].clone(), Entry { stamp, hash });
                    out[i] = Some(hash);
                }
            }
        }
        Ok(out.into_iter().map(|h| h.expect("every path was hashed")).collect())
    }

    /// Forgets every file not in `keep` (so that the record does not grow forever).
    pub fn retain(&mut self, keep: &std::collections::HashSet<PathBuf>) {
        self.files.retain(|p, _| keep.contains(p));
    }

    /// Number of remembered files.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.files.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip_and_domains() {
        let a = Hasher::new("a").str("x").finish();
        assert_eq!(Hash::from_hex(&a.to_hex()), Some(a));
        assert_ne!(a, Hasher::new("b").str("x").finish());
        // Length prefixes keep different splits apart.
        assert_ne!(Hasher::new("a").str("ab").str("c").finish(), Hasher::new("a").str("a").str("bc").finish());
        assert_eq!(Hash::from_hex("zz"), None);
        assert_eq!(a.short().len(), 16);
    }

    #[test]
    fn remembered_hashes_follow_changes() {
        let dir = crate::test_util::scratch("hashing");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        std::fs::write(&p, b"one").unwrap();
        let mut hashes = FileHashes::default();
        let first = hashes.hash(&p).unwrap();
        assert_eq!(first, hash_file_contents(&p).unwrap());
        std::fs::write(&p, b"two!").unwrap();
        let second = hashes.hash(&p).unwrap();
        assert_ne!(first, second, "a size change is noticed");
        let all = hashes.hash_all(&[p.clone(), p.clone()], 2).unwrap();
        assert_eq!(all, [second, second]);
        assert_eq!(hashes.len(), 1);
        hashes.retain(&Default::default());
        assert_eq!(hashes.len(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
