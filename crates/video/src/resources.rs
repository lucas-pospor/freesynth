//! How much memory a render may use, and how it is split between parallel chains and caches.
//!
//! A render's memory goes to three places: the keyframe chains running at the same time (each
//! holds its engine input, its tracker state, guide images and flows in flight), the decoded
//! frame cache and the buffer of synthesized frames waiting for their blend partner. The two
//! caches can spill to disk; the chains cannot. [`RenderOptions::resources`] turns the total
//! [`RenderOptions::memory_budget`] into a number of parallel chains and cache budgets for one
//! shot, so that a shot with many keyframes or large frames stays within the budget instead of
//! running every chain at once.

use std::fs;
use std::path::PathBuf;

use crate::params::RenderOptions;
use crate::plan::Plan;

/// Estimated memory of one running chain per frame pixel, in bytes. Measured peaks: about
/// 85 B/px at 960x540 and 145 B/px at 3840x2160 (engine input, tracker anchor and previous
/// states, guides, flows in flight); rounded up.
pub const CHAIN_BYTES_PER_PIXEL: usize = 150;

/// Chains run at the same time unless [`RenderOptions::parallel_chains`] asks for more. Two
/// chains fill a GPU much better than one; beyond four the gain is small and memory grows.
pub const MAX_DEFAULT_PARALLEL: usize = 4;

/// Writer threads of a render: they blend the two passes of a frame and hand finished frames to
/// the sink, so that blending and encoding overlap with the synthesis. A render runs this many
/// unless the memory budget cannot hold their blends next to one chain ([`ResourcePlan::writers`]).
pub const WRITERS: usize = 2;

/// Frames each cache keeps in memory at least, whatever the budget (when the shot has them).
pub const MIN_RESIDENT_FRAMES: usize = 8;

/// The default total memory budget: half of the memory available to the process (physical
/// memory, or a lower cgroup limit), at most 4 GiB. 4 GiB when the amount cannot be found.
pub fn default_memory_budget() -> usize {
    const CAP: usize = 4 << 30;
    available_memory().map_or(CAP, |m| (m / 2).min(CAP))
}

/// Memory available to the process: physical memory, lowered by a cgroup v2 limit (Linux).
fn available_memory() -> Option<usize> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
    let total = line.split_whitespace().nth(1)?.parse::<usize>().ok()?.checked_mul(1024)?;
    Some(cgroup_limit().map_or(total, |limit| limit.min(total)))
}

/// The memory limit of this process's cgroup (v2), if one is set.
fn cgroup_limit() -> Option<usize> {
    let cgroup = fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cgroup.lines().find_map(|l| l.strip_prefix("0::"))?;
    let file = PathBuf::from("/sys/fs/cgroup").join(path.trim_start_matches('/')).join("memory.max");
    fs::read_to_string(file).ok()?.trim().parse().ok()
}

/// What a shot's frames look like, as far as memory goes ([`RenderOptions::resources`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShotFormat {
    /// Frame width and height.
    pub size: (usize, usize),
    /// Whether the shot has a mask sequence.
    pub has_mask: bool,
    /// Working memory of one blend ([`Blender::working_bytes`](crate::Blender::working_bytes)).
    pub blend_bytes: usize,
    /// Whether synthesized frames are kept at 16 bits per channel (a render with 16-bit
    /// keyframes, [`Renderer::keyframes16`](crate::Renderer::keyframes16)).
    pub sixteen_bit: bool,
}

/// How one render uses memory: resolved from [`RenderOptions`] for a shot by
/// [`RenderOptions::resources`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourcePlan {
    /// Chains (worker threads) run at the same time.
    pub parallel: usize,
    /// Writer threads (blending and output): [`WRITERS`], or 1 when the budget is tight.
    pub writers: usize,
    /// Bytes of decoded frames each input sequence (frames, masks) keeps in memory.
    pub frame_cache_bytes: usize,
    /// Bytes of synthesized frames waiting for their blend partner kept in memory.
    pub buffer_bytes: usize,
    /// Flow fields kept for reuse.
    pub flow_cache_flows: usize,
    /// Estimated peak memory of the render, in bytes. Above the budget only when the budget
    /// cannot hold one chain plus the minimum caches, or when there is no spill directory and
    /// the shot's frames alone exceed it.
    pub estimated_bytes: usize,
}

impl RenderOptions {
    /// Resolves the parallelism and cache budgets for `plan` with frames of the given
    /// [`ShotFormat`].
    ///
    /// With [`parallel_chains`](Self::parallel_chains) unset, the renderer runs as many chains
    /// as the shot has (up to [`MAX_DEFAULT_PARALLEL`] and the number of CPU threads) as long as
    /// they fit in [`memory_budget`](Self::memory_budget) next to the flow cache and minimal
    /// frame caches. The caches then share what is left, up to their own limits
    /// ([`frame_cache_bytes`](Self::frame_cache_bytes), [`buffer_bytes`](Self::buffer_bytes)).
    /// Without a spill directory nothing can spill, so the caches count with everything they
    /// will hold. The renderer's writer threads ([`WRITERS`], or one when the budget cannot hold
    /// two writers' blends next to one chain) blend one frame at a time each, with
    /// [`ShotFormat::blend_bytes`] of working memory; the pass pairs queued for them count as well.
    pub fn resources(&self, plan: &Plan, format: &ShotFormat) -> ResourcePlan {
        let ShotFormat { size: (width, height), has_mask, blend_bytes, sixteen_bit } = *format;
        let pixels = width * height;
        let frame = pixels * 4;
        // A waiting pass frame: up to 4 channels (of 1 or 2 bytes) plus 2 bytes of patch error
        // for the adaptive blend.
        let pass_frame = pixels * if sixteen_bit { 10 } else { 6 };
        let flow = (pixels * 8).max(1);
        let per_chain = pixels * CHAIN_BYTES_PER_PIXEL;
        // Blends run in the writer threads, at most one per writer at a time, each with the
        // blend's working memory and its two passes unpacked (up to 4 samples of 1 or 2 bytes
        // plus a 4-byte error per pixel). Pass pairs waiting for a writer stay packed: up to
        // `2 p` in the queue, one more per chain blocked on the full queue and one per writer.
        let unpacked_pair = 2 * pixels * if sixteen_bit { 12 } else { 8 };
        let blends = |p: usize, writers: usize| {
            if plan.num_blended_frames() == 0 {
                return 0;
            }
            writers * (blend_bytes + unpacked_pair) + (3 * p + writers) * 2 * pass_frame
        };
        let sequences = 1 + usize::from(has_mask);
        let spills = self.spill_dir.is_some();
        // What the caches would hold if nothing were spilled, capped by their own limits.
        let frames_want = (plan.num_frames() * frame).min(if spills { self.frame_cache_bytes } else { usize::MAX });
        let buffers_want =
            (plan.num_blended_frames() * pass_frame).min(if spills { self.buffer_bytes } else { usize::MAX });
        // What they need at least.
        let floor = |want: usize, frame: usize| if spills { want.min(MIN_RESIDENT_FRAMES * frame) } else { want };
        let caches_floor = sequences * floor(frames_want, frame) + floor(buffers_want, pass_frame);
        let flows = |p: usize| (self.flow_cache_bytes / flow).max(3 * p.min(MAX_DEFAULT_PARALLEL)).max(2);
        let chains_with = |p: usize, writers: usize| p * per_chain + flows(p) * flow + blends(p, writers);
        // One writer when two writers' blends do not fit next to one chain.
        let writers = if chains_with(1, WRITERS) + caches_floor > self.memory_budget { 1 } else { WRITERS };
        let chains = |p: usize| chains_with(p, writers);

        let tasks = (plan.chains.len() + plan.keyframes.len()).max(1);
        let parallel = match self.parallel_chains {
            Some(n) => n.clamp(1, tasks),
            None => {
                let threads = std::thread::available_parallelism().map_or(MAX_DEFAULT_PARALLEL, usize::from);
                let useful = plan.chains.len().max(plan.keyframes.len()).clamp(1, tasks);
                let limit = useful.min(MAX_DEFAULT_PARALLEL).min(threads).max(1);
                (1..=limit).rev().find(|&p| chains(p) + caches_floor <= self.memory_budget).unwrap_or(1)
            }
        };

        // The caches share what the chains leave, down to their floors.
        let left = self.memory_budget.saturating_sub(chains(parallel));
        let (frame_cache, buffers) = if !spills || sequences * frames_want + buffers_want <= left {
            (frames_want, buffers_want)
        } else {
            let share = left / (sequences + 1);
            let (f, b) = if frames_want <= share {
                (frames_want, buffers_want.min(left - sequences * frames_want))
            } else if buffers_want <= share {
                (frames_want.min((left - buffers_want) / sequences), buffers_want)
            } else {
                // Equal shares; when that is below the pass buffer's floor, the frame caches
                // make up for it.
                let b = share.max(floor(buffers_want, pass_frame));
                (left.saturating_sub(b) / sequences, b)
            };
            (f.max(floor(frames_want, frame)), b.max(floor(buffers_want, pass_frame)))
        };
        ResourcePlan {
            parallel,
            writers,
            frame_cache_bytes: if spills { frame_cache } else { self.frame_cache_bytes },
            buffer_bytes: if spills { buffers } else { self.buffer_bytes },
            flow_cache_flows: flows(parallel),
            estimated_bytes: chains(parallel) + sequences * frame_cache + buffers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(size: (usize, usize), has_mask: bool, blend_bytes: usize) -> ShotFormat {
        ShotFormat { size, has_mask, blend_bytes, sixteen_bit: false }
    }

    fn options(budget: usize) -> RenderOptions {
        RenderOptions { memory_budget: budget, spill_dir: Some("spill".into()), ..RenderOptions::default() }
    }

    fn threads() -> usize {
        std::thread::available_parallelism().map_or(MAX_DEFAULT_PARALLEL, usize::from)
    }

    #[test]
    fn default_budget_is_bounded() {
        let b = default_memory_budget();
        assert!(b > 0 && b <= 4 << 30, "{b}");
        assert_eq!(RenderOptions::default().memory_budget, b);
    }

    #[test]
    fn many_keyframes_do_not_run_every_chain_at_once() {
        // 61 frames, 11 keyframes: 20 chains.
        let keys: Vec<usize> = (0..=60).step_by(6).collect();
        let plan = Plan::new(0, 60, &keys).unwrap();
        assert_eq!(plan.chains.len(), 20);
        let r = options(4 << 30).resources(&plan, &fmt((960, 540), false, 0));
        assert_eq!(r.parallel, MAX_DEFAULT_PARALLEL.min(threads()));
        assert!(r.estimated_bytes <= 4 << 30, "{r:?}");
        // At 4K the budget, not the cap, decides; the estimate stays within it.
        let r4k = options(4 << 30).resources(&plan, &fmt((3840, 2160), false, 0));
        assert!(r4k.parallel >= 1 && r4k.parallel <= 2, "{r4k:?}");
        assert!(r4k.estimated_bytes <= 4 << 30, "{r4k:?}");
        assert!(r4k.flow_cache_flows <= 3 * MAX_DEFAULT_PARALLEL, "{r4k:?}");
        // A smaller budget runs fewer chains.
        let small = options(512 << 20).resources(&plan, &fmt((1920, 1080), false, 0));
        assert_eq!(small.parallel, 1, "{small:?}");
    }

    #[test]
    fn small_shots_keep_everything_in_memory() {
        // Two keyframes, 100 frames of 960x540: two chains, no cache pressure.
        let plan = Plan::new(0, 99, &[0, 99]).unwrap();
        let r = options(4 << 30).resources(&plan, &fmt((960, 540), false, 0));
        assert_eq!(r.parallel, 2.min(threads()));
        assert_eq!(r.frame_cache_bytes, 100 * 960 * 540 * 4);
        assert_eq!(r.buffer_bytes, 98 * 960 * 540 * 6);
        let wide = options(4 << 30).resources(&plan, &ShotFormat { sixteen_bit: true, ..fmt((960, 540), false, 0) });
        assert_eq!(wide.buffer_bytes, 98 * 960 * 540 * 10);
        assert_eq!(r.flow_cache_flows, (256 << 20) / (960 * 540 * 8));
    }

    #[test]
    fn caches_shrink_to_fit_the_budget_at_4k() {
        let plan = Plan::new(0, 99, &[0, 99]).unwrap();
        let frame = 3840 * 2160 * 4;
        let r = options(4 << 30).resources(&plan, &fmt((3840, 2160), true, 0));
        assert!(r.parallel >= 1);
        assert!(r.frame_cache_bytes >= MIN_RESIDENT_FRAMES * frame && r.frame_cache_bytes < 1 << 30, "{r:?}");
        assert!(r.buffer_bytes >= MIN_RESIDENT_FRAMES * frame * 3 / 2 && r.buffer_bytes <= 1 << 30, "{r:?}");
        assert!(r.estimated_bytes <= 4 << 30, "{r:?}");
    }

    #[test]
    fn blend_memory_is_reserved_for_the_writers() {
        let plan = Plan::new(0, 99, &[0, 99]).unwrap();
        let blend = 3840 * 2160 * 96;
        let plain = options(4 << 30).resources(&plan, &fmt((3840, 2160), false, 0));
        let with = options(4 << 30).resources(&plan, &fmt((3840, 2160), false, blend));
        assert!(with.parallel <= plain.parallel, "{with:?} vs {plain:?}");
        assert!(with.estimated_bytes <= 4 << 30, "{with:?}");
        let chains = |r: &ResourcePlan| r.parallel * 3840 * 2160 * CHAIN_BYTES_PER_PIXEL;
        // At 4K two writers' blends do not fit next to a chain in 4 GiB: one writer.
        assert_eq!((plain.writers, with.writers), (WRITERS, 1), "{plain:?} {with:?}");
        // The pass pairs queued for the writers count too: at least `2 p + writers` packed pairs.
        let queued = (2 * with.parallel + with.writers) * 2 * 3840 * 2160 * 6;
        assert!(with.estimated_bytes >= chains(&with) + with.writers * blend + queued, "{with:?}");
        // With room, two writers.
        let roomy = options(16 << 30).resources(&plan, &fmt((3840, 2160), false, blend));
        assert_eq!(roomy.writers, WRITERS, "{roomy:?}");
        assert!(roomy.estimated_bytes >= chains(&roomy) + WRITERS * blend, "{roomy:?}");
        // A shot without blended frames (one keyframe) needs no blend memory.
        let single = Plan::new(0, 99, &[0]).unwrap();
        let a = options(4 << 30).resources(&single, &fmt((3840, 2160), false, 0));
        let b = options(4 << 30).resources(&single, &fmt((3840, 2160), false, blend));
        assert_eq!(a, b);
    }

    #[test]
    fn explicit_settings_win() {
        let plan = Plan::new(0, 60, &(0..=60).step_by(6).collect::<Vec<_>>()).unwrap();
        let o = RenderOptions { parallel_chains: Some(7), frame_cache_bytes: 1000, ..options(4 << 30) };
        let r = o.resources(&plan, &fmt((64, 48), false, 0));
        assert_eq!(r.parallel, 7);
        assert_eq!(r.frame_cache_bytes, 1000);
        assert_eq!(r.flow_cache_flows, (256 << 20) / (64 * 48 * 8));
        // Without a spill directory the caches cannot spill, so they count in full.
        let o = RenderOptions { spill_dir: None, ..options(16 << 20) };
        let r = o.resources(&plan, &fmt((640, 480), false, 0));
        assert_eq!(r.parallel, 1);
        assert!(r.estimated_bytes > 61 * 640 * 480 * 4);
    }
}
