//! Optical flow computed once per frame pair and shared by every keyframe chain.
//!
//! Each chain needs flows between its frames: the one-step flow that warps its previous output
//! (the temporal guide) and the flows to the anchors of its position field. Chains of
//! neighbouring keyframes ask for some of the same pairs. A [`FlowBank`] computes each pair at
//! most once while it is in use, keeps recent results in a small cache, and makes a thread that
//! asks for a pair another thread is computing wait for that result instead of computing it
//! again.
//!
//! Flows are keyed by frame *content* (a 128-bit hash of the samples), not by frame number:
//! the position tracker of `freesynth-flow` hands the estimator images, not indices, and the
//! flow between two images depends on nothing else.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use freesynth_flow::{DisParams, FlowError, FlowEstimator, FlowField, GrayImage};

use crate::device::FlowProvider;
use crate::error::VideoError;

/// Identifies the content of a frame: size plus a 128-bit hash of its samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ContentId {
    size: (usize, usize),
    hash: [u64; 2],
}

/// A fast non-cryptographic 128-bit hash of the image's sample bits (two independent
/// multiply-xorshift lanes). Collisions between real frames are not a practical concern.
pub(crate) fn content_id(image: &GrayImage) -> ContentId {
    const K: [u64; 2] = [0x9E37_79B9_7F4A_7C15, 0xC2B2_AE3D_27D4_EB4F];
    let mut h = [0x243F_6A88_85A3_08D3u64, 0x1319_8A2E_0370_7344];
    let data = image.data();
    let (pairs, rest) = data.as_chunks::<2>();
    for p in pairs {
        let word = u64::from(p[0].to_bits()) | (u64::from(p[1].to_bits()) << 32);
        for (lane, k) in h.iter_mut().zip(K) {
            *lane = (*lane ^ word).wrapping_mul(k).rotate_left(29);
        }
    }
    for &v in rest {
        for (lane, k) in h.iter_mut().zip(K) {
            *lane = (*lane ^ u64::from(v.to_bits())).wrapping_mul(k).rotate_left(29);
        }
    }
    let finish = |mut x: u64| {
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^ (x >> 33)
    };
    ContentId { size: image.size(), hash: [finish(h[0] ^ data.len() as u64), finish(h[1])] }
}

type Key = (ContentId, ContentId);

struct Cache {
    flows: HashMap<Key, (Arc<FlowField>, u64)>,
    /// Pairs some thread is computing right now.
    pending: HashSet<Key>,
    clock: u64,
    capacity: usize,
}

impl Cache {
    fn get(&mut self, key: &Key) -> Option<Arc<FlowField>> {
        self.clock += 1;
        let clock = self.clock;
        self.flows.get_mut(key).map(|(f, used)| {
            *used = clock;
            Arc::clone(f)
        })
    }

    fn put(&mut self, key: Key, flow: Arc<FlowField>) {
        self.clock += 1;
        self.flows.insert(key, (flow, self.clock));
        while self.flows.len() > self.capacity {
            let oldest = self.flows.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| *k).expect("not empty");
            self.flows.remove(&oldest);
        }
    }
}

/// Counters of a [`FlowBank`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlowStats {
    /// Flows computed.
    pub computed: usize,
    /// Requests answered from the cache (or by waiting for another thread's computation).
    pub reused: usize,
    /// Total time spent computing flows, summed over threads, in seconds.
    pub seconds: f64,
}

/// A pool of flow estimators with a shared cache of recent results (see the module docs).
pub struct FlowBank {
    estimators: Mutex<Vec<Box<dyn FlowEstimator + Send>>>,
    estimator_free: Condvar,
    cache: Mutex<Cache>,
    cache_changed: Condvar,
    computed: AtomicUsize,
    reused: AtomicUsize,
    nanos: AtomicU64,
}

/// Returns an estimator to the pool when dropped.
struct Lease<'a> {
    bank: &'a FlowBank,
    estimator: Option<Box<dyn FlowEstimator + Send>>,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(e) = self.estimator.take() {
            lock(&self.bank.estimators).push(e);
            self.bank.estimator_free.notify_one();
        }
    }
}

/// Clears pairs from the pending set if their computation ends without a result (an error
/// or a panic), so that threads waiting for them wake up instead of waiting forever.
struct PendingGuard<'a> {
    bank: &'a FlowBank,
    keys: &'a [Key],
    armed: bool,
}

impl PendingGuard<'_> {
    /// Clears the pending pairs under the caller's lock and wakes the waiting threads.
    fn release(&mut self, cache: &mut Cache) {
        for k in self.keys {
            cache.pending.remove(k);
        }
        self.armed = false;
        self.bank.cache_changed.notify_all();
    }
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let bank = self.bank;
            self.release(&mut lock(&bank.cache));
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl FlowBank {
    /// A bank of `estimators` estimators for `width` x `height` frames from `provider`, caching
    /// up to `capacity` flows.
    pub fn new(
        provider: &dyn FlowProvider,
        width: usize,
        height: usize,
        params: &DisParams,
        estimators: usize,
        capacity: usize,
    ) -> Result<Self, VideoError> {
        let estimators =
            (0..estimators.max(1)).map(|_| provider.flow_estimator(width, height, params)).collect::<Result<_, _>>()?;
        Ok(Self {
            estimators: Mutex::new(estimators),
            estimator_free: Condvar::new(),
            cache: Mutex::new(Cache {
                flows: HashMap::new(),
                pending: HashSet::new(),
                clock: 0,
                capacity: capacity.max(2),
            }),
            cache_changed: Condvar::new(),
            computed: AtomicUsize::new(0),
            reused: AtomicUsize::new(0),
            nanos: AtomicU64::new(0),
        })
    }

    /// The counters so far.
    pub fn stats(&self) -> FlowStats {
        FlowStats {
            computed: self.computed.load(Ordering::Relaxed),
            reused: self.reused.load(Ordering::Relaxed),
            seconds: self.nanos.load(Ordering::Relaxed) as f64 * 1e-9,
        }
    }

    fn lease(&self) -> Lease<'_> {
        let mut pool = lock(&self.estimators);
        loop {
            if let Some(e) = pool.pop() {
                return Lease { bank: self, estimator: Some(e) };
            }
            pool = self.estimator_free.wait(pool).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The flow from `frame0` to `frame1`: `frame0(x) ~ frame1(x + flow(x))`.
    pub fn flow(&self, frame0: &GrayImage, frame1: &GrayImage) -> Result<Arc<FlowField>, FlowError> {
        let (a, b) = (content_id(frame0), content_id(frame1));
        let mut found = self.fetch(&[(a, b)], |e| Ok(vec![e.compute(frame0, frame1)?]))?;
        Ok(found.pop().expect("one flow requested"))
    }

    /// The flows from `frame0` to `frame1` and back.
    pub fn pair(&self, frame0: &GrayImage, frame1: &GrayImage) -> Result<(Arc<FlowField>, Arc<FlowField>), FlowError> {
        let (a, b) = (content_id(frame0), content_id(frame1));
        let mut found = self.fetch(&[(a, b), (b, a)], |e| {
            let (f, g) = e.compute_pair(frame0, frame1)?;
            Ok(vec![f, g])
        })?;
        let g = found.pop().expect("two flows requested");
        let f = found.pop().expect("two flows requested");
        Ok((f, g))
    }

    /// Returns the flows for `keys`, computing them all with `compute` unless every one is
    /// cached. Waits while another thread computes any of them.
    fn fetch(
        &self,
        keys: &[Key],
        compute: impl FnOnce(&mut dyn FlowEstimator) -> Result<Vec<FlowField>, FlowError>,
    ) -> Result<Vec<Arc<FlowField>>, FlowError> {
        let mut cache = lock(&self.cache);
        while keys.iter().any(|k| cache.pending.contains(k)) {
            cache = self.cache_changed.wait(cache).unwrap_or_else(|e| e.into_inner());
        }
        let cached: Option<Vec<_>> = keys.iter().map(|k| cache.get(k)).collect();
        if let Some(flows) = cached {
            self.reused.fetch_add(keys.len(), Ordering::Relaxed);
            return Ok(flows);
        }
        cache.pending.extend(keys.iter().copied());
        drop(cache);
        let mut pending = PendingGuard { bank: self, keys, armed: true };

        let start = Instant::now();
        let result = {
            let mut lease = self.lease();
            compute(lease.estimator.as_deref_mut().expect("leased"))
        };
        self.nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut cache = lock(&self.cache);
        pending.release(&mut cache);
        let flows: Vec<Arc<FlowField>> = result?.into_iter().map(Arc::new).collect();
        self.computed.fetch_add(flows.len(), Ordering::Relaxed);
        for (k, f) in keys.iter().zip(&flows) {
            cache.put(*k, Arc::clone(f));
        }
        Ok(flows)
    }
}

/// A [`FlowEstimator`] that answers from a [`FlowBank`], so that a position tracker shares
/// flows with the rest of the pipeline.
pub(crate) struct BankEstimator<'a> {
    pub bank: &'a FlowBank,
}

impl FlowEstimator for BankEstimator<'_> {
    fn compute(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<FlowField, FlowError> {
        Ok((*self.bank.flow(frame0, frame1)?).clone())
    }

    fn compute_pair(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<(FlowField, FlowField), FlowError> {
        let (f, g) = self.bank.pair(frame0, frame1)?;
        Ok(((*f).clone(), (*g).clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::ComputeDevice;

    fn frame(shift: usize) -> GrayImage {
        let (w, h) = (48, 40);
        let data = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w + shift) as f32, (i / w) as f32);
                128.0 + 60.0 * (x * 0.3).sin() * (y * 0.25).cos() + 30.0 * (x * 0.11 + y * 0.07).sin()
            })
            .collect();
        GrayImage::from_vec(w, h, data).unwrap()
    }

    #[test]
    fn content_ids_distinguish_frames() {
        assert_eq!(content_id(&frame(0)), content_id(&frame(0)));
        assert_ne!(content_id(&frame(0)), content_id(&frame(1)));
        let mut f = frame(0);
        f.data_mut()[17] += 0.5;
        assert_ne!(content_id(&f), content_id(&frame(0)));
    }

    #[test]
    fn each_pair_is_computed_once() {
        let device = ComputeDevice::cpu();
        let bank = FlowBank::new(&device, 48, 40, &DisParams::default(), 2, 8).unwrap();
        let (a, b, c) = (frame(0), frame(2), frame(4));
        let (ab, ba) = bank.pair(&a, &b).unwrap();
        assert_eq!(bank.stats().computed, 2);
        // Both directions are now cached.
        assert!(Arc::ptr_eq(&bank.flow(&a, &b).unwrap(), &ab));
        assert!(Arc::ptr_eq(&bank.flow(&b, &a).unwrap(), &ba));
        assert_eq!(bank.stats().computed, 2);
        assert_eq!(bank.stats().reused, 2);
        // A pair with one cached direction is recomputed as a pair.
        bank.flow(&c, &a).unwrap();
        bank.pair(&a, &c).unwrap();
        assert_eq!(bank.stats().computed, 5);
        // The flow is roughly the shift.
        let mid = ab.get(24, 20);
        assert!((mid[0] + 2.0).abs() < 0.5 && mid[1].abs() < 0.5, "{mid:?}");
    }

    #[test]
    fn concurrent_requests_share_one_computation() {
        let device = ComputeDevice::cpu();
        let bank = FlowBank::new(&device, 48, 40, &DisParams::default(), 4, 8).unwrap();
        let (a, b) = (frame(0), frame(3));
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| bank.flow(&a, &b).unwrap());
            }
        });
        assert_eq!(bank.stats().computed, 1);
        assert_eq!(bank.stats().reused, 3);
    }

    #[test]
    fn a_panicking_computation_does_not_block_waiting_threads() {
        let device = ComputeDevice::cpu();
        let bank = FlowBank::new(&device, 48, 40, &DisParams::default(), 2, 8).unwrap();
        let (a, b) = (frame(0), frame(3));
        let key = [(content_id(&a), content_id(&b))];
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = bank.fetch(&key, |_| panic!("estimator failure"));
        }));
        assert!(panicked.is_err());
        assert!(lock(&bank.cache).pending.is_empty(), "the pair is no longer pending");
        // A later request computes the pair instead of waiting for the failed computation.
        bank.flow(&a, &b).unwrap();
        assert_eq!(bank.stats().computed, 1);
        // So does an error.
        let (c, d) = (frame(5), frame(6));
        let key = [(content_id(&c), content_id(&d))];
        assert!(bank.fetch(&key, |_| Err(FlowError::InvalidInput("test".into()))).is_err());
        assert!(lock(&bank.cache).pending.is_empty());
    }

    #[test]
    fn cache_is_bounded() {
        let device = ComputeDevice::cpu();
        let bank = FlowBank::new(&device, 48, 40, &DisParams::default(), 1, 2).unwrap();
        let frames: Vec<_> = (0..4).map(frame).collect();
        for f in &frames[1..] {
            bank.flow(&frames[0], f).unwrap();
        }
        assert_eq!(lock(&bank.cache).flows.len(), 2);
        bank.flow(&frames[0], &frames[1]).unwrap();
        assert_eq!(bank.stats().computed, 4, "the oldest flow was evicted and recomputed");
    }
}
