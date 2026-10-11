//! PatchMatch with the uniformity (occupancy) term (`patchmatch` in `ebsynth_cpu.cpp`).
//!
//! For every target pixel we keep the centre of its best source patch (the NNF) and its patch
//! error `E`. `Omega` counts, for every source pixel, how many matched source patches cover it.
//! A candidate replaces the current match when
//!
//! ```text
//! E(cand) + lambda * occ(cand)  <  E(cur) + lambda * occ(cur),
//! occ(p) = (sum of Omega over the patch at p / patch_size^2) / omega_best
//! ```
//!
//! where `omega_best` is the occupancy of a perfectly uniform assignment. This discourages
//! overusing the same source region (Jamriska et al., "Stylizing Video by Example", 2019).
//!
//! Each iteration scans the target in alternating directions doing propagation (from the left and
//! upper neighbour, or right and lower on odd passes) and random search at exponentially shrinking
//! radii. Like the original, the target is split into horizontal bands (one per thread unless
//! [`Params::num_bands`](crate::Params::num_bands) says otherwise), each
//! scanned sequentially.
//!
//! Sharing `Omega` between bands is inherently racy. [`OmegaSync::Shared`] reproduces the
//! original (all bands read and update one shared array immediately; results vary slightly from
//! run to run). [`OmegaSync::PerTile`] gives each band a private copy for the duration of an
//! iteration and merges the changes afterwards, which is deterministic for a given band count.
//!
//! Occupancy counts are `i32` (a cell counts at most one patch per target pixel), but patch sums
//! over `patch_size^2` cells are accumulated in `i64`: with a tiny source and a huge target they
//! exceed `i32::MAX` (the original's `int` sums overflow there).

use std::sync::atomic::{AtomicI32, Ordering::Relaxed};

use rayon::prelude::*;

use crate::image::Nnf;
use crate::patch::{EXACT, FAST, Kernel, MODULATED, PatchSpace};
use crate::rng::GlibcRand;
use crate::synth::OmegaSync;

/// Minimum band height before the scan uses fewer bands than threads (as in the original).
const MIN_TILE_HEIGHT: usize = 8;

/// Settings for one [`patchmatch`] call.
pub(crate) struct PatchMatchConfig {
    pub lambda: f32,
    pub num_iters: usize,
    /// Number of bands the target is split into (the original's thread count).
    pub num_bands: usize,
    pub sync: OmegaSync,
    pub kernel: Kernel,
}

/// An accepted move of one target pixel's match, from the old to the new source patch centre.
type Move = ([i32; 2], [i32; 2]);

/// Reusable per-level storage for the occupancy map.
#[derive(Default)]
pub(crate) struct OmegaWorkspace {
    global: Vec<AtomicI32>,
    /// Band-private copies ([`OmegaSync::PerTile`]).
    copies: Vec<Vec<i32>>,
    /// Per band, the moves accepted during the current iteration ([`OmegaSync::PerTile`]).
    logs: Vec<Vec<Move>>,
    /// Whether `global` and `copies` hold the exact occupancy of the field left by the last
    /// [`OmegaSync::PerTile`] call, so the next call on an unchanged field can reuse them.
    per_tile_valid: bool,
    /// Scratch maps for the box filters of [`init_omega`] and [`merge_logs`].
    hist: Vec<i32>,
    rows: Vec<i32>,
}

/// Read/update access to an occupancy map.
trait OmegaAccess {
    /// Sum of the counts covered by the patch centred at `c`.
    fn patch_sum(&self, c: [i32; 2]) -> i64;
    /// Adds `delta` to every count covered by the patch centred at `c`.
    fn add_patch(&mut self, c: [i32; 2], delta: i32);
    /// Moves one pixel's match from the patch at `from` to the patch at `to`.
    #[inline(always)]
    fn move_patch(&mut self, from: [i32; 2], to: [i32; 2]) {
        self.add_patch(to, 1);
        self.add_patch(from, -1);
    }
}

/// The occupancy map shared by all bands.
struct SharedOmega<'a> {
    cells: &'a [AtomicI32],
    width: usize,
    patch: usize,
}

impl OmegaAccess for SharedOmega<'_> {
    #[inline(always)]
    fn patch_sum(&self, c: [i32; 2]) -> i64 {
        let r = (self.patch / 2) as i32;
        let mut i = (c[1] - r) as usize * self.width + (c[0] - r) as usize;
        let mut sum = 0i64;
        for _ in 0..self.patch {
            for cell in &self.cells[i..i + self.patch] {
                sum += cell.load(Relaxed) as i64;
            }
            i += self.width;
        }
        sum
    }

    #[inline(always)]
    fn add_patch(&mut self, c: [i32; 2], delta: i32) {
        let r = (self.patch / 2) as i32;
        let mut i = (c[1] - r) as usize * self.width + (c[0] - r) as usize;
        for _ in 0..self.patch {
            for cell in &self.cells[i..i + self.patch] {
                // A deliberately non-atomic read-modify-write, mirroring the original's plain
                // `+=` from several threads: concurrent updates may occasionally be lost, which
                // only perturbs the uniformity heuristic. Relaxed loads/stores keep this free of
                // undefined behaviour while compiling to plain moves.
                cell.store(cell.load(Relaxed) + delta, Relaxed);
            }
            i += self.width;
        }
    }
}

/// A band-private occupancy map that also logs the band's moves.
struct LocalOmega<'a> {
    cells: &'a mut [i32],
    width: usize,
    patch: usize,
    log: &'a mut Vec<Move>,
}

impl OmegaAccess for LocalOmega<'_> {
    #[inline(always)]
    fn patch_sum(&self, c: [i32; 2]) -> i64 {
        let r = (self.patch / 2) as i32;
        let mut i = (c[1] - r) as usize * self.width + (c[0] - r) as usize;
        let mut sum = 0i64;
        for _ in 0..self.patch {
            sum += self.cells[i..i + self.patch].iter().map(|&v| v as i64).sum::<i64>();
            i += self.width;
        }
        sum
    }

    #[inline(always)]
    fn move_patch(&mut self, from: [i32; 2], to: [i32; 2]) {
        add_patch_cells(self.cells, self.width, self.patch, to, 1);
        add_patch_cells(self.cells, self.width, self.patch, from, -1);
        self.log.push((from, to));
    }

    #[inline(always)]
    fn add_patch(&mut self, c: [i32; 2], delta: i32) {
        add_patch_cells(self.cells, self.width, self.patch, c, delta);
    }
}

/// Adds `delta` to the `patch x patch` cells centred at `c` of a `width`-wide map.
#[inline(always)]
fn add_patch_cells(cells: &mut [i32], width: usize, patch: usize, c: [i32; 2], delta: i32) {
    let r = (patch / 2) as i32;
    let mut i = (c[1] - r) as usize * width + (c[0] - r) as usize;
    for _ in 0..patch {
        for cell in &mut cells[i..i + patch] {
            *cell += delta;
        }
        i += width;
    }
}

/// Per-call constants shared by all bands.
struct ScanParams<'a> {
    lambda: f32,
    omega_best: f32,
    use_omega: bool,
    /// Random-search radii, largest first.
    radii: &'a [i32],
}

/// One horizontal band of the target, with its slice of the NNF and errors.
struct Band<'a> {
    y0: usize,
    y1: usize,
    nnf: &'a mut [[i32; 2]],
    err: &'a mut [f32],
    /// Snapshot of the NNF row just above / below the band (owned by the neighbouring band).
    above: Option<Vec<[i32; 2]>>,
    below: Option<Vec<[i32; 2]>>,
}

/// `RANDI` of the original: a cheap 32-bit hash step used for random search.
#[inline(always)]
fn randi(u: u32) -> u32 {
    18000u32.wrapping_mul(u & 65535).wrapping_add(u >> 16)
}

/// Runs `num_iters` PatchMatch iterations, refining `nnf` in place.
///
/// On return `errors` holds the patch error (without the uniformity term) of every match. When
/// given, `mask` restricts the search to target pixels whose mask value is non-zero.
///
/// `ws` carries the occupancy maps between calls. With [`OmegaSync::PerTile`] they are reused
/// when the next call has the same sizes, so a caller that changes the field between calls must
/// pass a fresh workspace (the synthesis driver uses one per pyramid level).
pub(crate) fn patchmatch(
    space: &PatchSpace<'_>,
    nnf: &mut Nnf,
    errors: &mut Vec<f32>,
    cfg: &PatchMatchConfig,
    rng: &mut GlibcRand,
    mask: Option<&[u8]>,
    ws: &mut OmegaWorkspace,
) {
    match cfg.kernel {
        Kernel::Fast => patchmatch_k::<FAST>(space, nnf, errors, cfg, rng, mask, ws),
        Kernel::Exact => patchmatch_k::<EXACT>(space, nnf, errors, cfg, rng, mask, ws),
        Kernel::Modulated => patchmatch_k::<MODULATED>(space, nnf, errors, cfg, rng, mask, ws),
    }
}

/// Patch errors of the current matches of `nnf` (`nnfError`), computed with `kernel`.
pub(crate) fn nnf_errors(space: &PatchSpace<'_>, nnf: &Nnf, errors: &mut Vec<f32>, kernel: Kernel) {
    match kernel {
        Kernel::Fast => nnf_errors_k::<FAST>(space, nnf, errors),
        Kernel::Exact => nnf_errors_k::<EXACT>(space, nnf, errors),
        Kernel::Modulated => nnf_errors_k::<MODULATED>(space, nnf, errors),
    }
}

/// Patch errors of the current matches (`nnfError`).
fn nnf_errors_k<const K: u8>(space: &PatchSpace<'_>, nnf: &Nnf, errors: &mut Vec<f32>) {
    let w = nnf.width();
    errors.resize(w * nnf.height(), 0.0);
    if w == 0 {
        return;
    }
    errors.par_chunks_mut(w).zip(nnf.data().par_chunks(w)).enumerate().for_each(|(y, (e, n))| {
        for (x, (e, &[sx, sy])) in e.iter_mut().zip(n).enumerate() {
            *e = space.error::<K>(x as i32, y as i32, sx, sy, f32::MAX);
        }
    });
}

fn patchmatch_k<const K: u8>(
    space: &PatchSpace<'_>,
    nnf: &mut Nnf,
    errors: &mut Vec<f32>,
    cfg: &PatchMatchConfig,
    rng: &mut GlibcRand,
    mask: Option<&[u8]>,
    ws: &mut OmegaWorkspace,
) {
    let (tw, th) = (space.tw, space.th);
    let (sw, sh) = (space.sw, space.sh);
    let patch = space.patch;

    nnf_errors_k::<K>(space, nnf, errors);

    // Random-search radii: max(sw, sh), then halving down to 1.
    let mut radii = vec![sw.max(sh) as i32];
    while *radii.last().unwrap() > 1 {
        let k = radii.len() as i32;
        radii.push((0.5f64.powi(k) * radii[0] as f64) as i32);
    }

    let threads = cfg.num_bands.max(1);
    let num_tiles = if th.div_ceil(threads) > MIN_TILE_HEIGHT { threads } else { th.div_ceil(MIN_TILE_HEIGHT).max(1) };
    let tile_height = th / num_tiles;

    let omega_best = ((tw * th) as f32 / (sw * sh) as f32) * (patch * patch) as f32;
    // With lambda == 0 (the 3x3 polishing pass) occupancy never influences a decision, so the
    // map is neither built nor maintained.
    let use_omega = cfg.lambda != 0.0 && cfg.num_iters > 0;
    let per_tile = use_omega && cfg.sync == OmegaSync::PerTile && num_tiles > 1;

    // In PerTile mode the maps left by the previous call are exact for the field it left, so
    // when this call starts from that field (same level, nothing changed it in between) they
    // are reused instead of rebuilt.
    let reuse = per_tile && ws.per_tile_valid && ws.copies.len() == num_tiles && ws.global.len() == sw * sh;
    ws.per_tile_valid = false;
    if use_omega && !reuse {
        init_omega(ws, nnf, sw, sh, patch);
        if per_tile {
            ws.copies.resize_with(num_tiles, Vec::new);
            ws.copies.truncate(num_tiles);
            for c in &mut ws.copies {
                c.resize(sw * sh, 0);
            }
            sync_copies(&ws.global, &mut ws.copies);
        }
    }
    if per_tile {
        ws.logs.resize_with(num_tiles, Vec::new);
        ws.logs.truncate(num_tiles);
    }

    let scan = ScanParams { lambda: cfg.lambda, omega_best, use_omega, radii: &radii };

    for iter in 0..cfg.num_iters {
        let iter_seed = rng.next_i32() as u32;
        let odd = iter % 2 == 0;

        // Split the field into bands, snapshotting the rows just outside each band.
        let bounds: Vec<(usize, usize)> = (0..num_tiles)
            .map(|t| {
                let y0 = t * tile_height;
                let y1 = if t == num_tiles - 1 { th } else { (y0 + tile_height).min(th) };
                (y0, y1)
            })
            .collect();
        let row = |y: usize| nnf.data()[y * tw..(y + 1) * tw].to_vec();
        let snapshots: Vec<_> = bounds
            .iter()
            .map(|&(y0, y1)| ((y0 > 0).then(|| row(y0 - 1)), (y1 < th).then(|| row(y1))))
            .collect();
        let mut bands = Vec::with_capacity(num_tiles);
        let mut rest_n = nnf.data_mut();
        let mut rest_e = &mut errors[..];
        for (&(y0, y1), (above, below)) in bounds.iter().zip(snapshots) {
            let (n, rn) = rest_n.split_at_mut((y1 - y0) * tw);
            let (e, re) = rest_e.split_at_mut((y1 - y0) * tw);
            rest_n = rn;
            rest_e = re;
            bands.push(Band { y0, y1, nnf: n, err: e, above, below });
        }

        if per_tile {
            bands.into_par_iter().zip(ws.copies.par_iter_mut()).zip(ws.logs.par_iter_mut()).for_each(
                |((mut band, copy), log)| {
                    log.clear();
                    let mut omega = LocalOmega { cells: copy, width: sw, patch, log };
                    scan_band_dispatch::<_, K>(space, &scan, &mut band, &mut omega, odd, iter_seed, mask);
                },
            );
            merge_logs(ws, sw, sh, patch);
        } else {
            let global = &ws.global[..];
            bands.into_par_iter().for_each(|mut band| {
                let mut omega = SharedOmega { cells: global, width: sw, patch };
                scan_band_dispatch::<_, K>(space, &scan, &mut band, &mut omega, odd, iter_seed, mask);
            });
        }
    }
    ws.per_tile_valid = per_tile;
}

/// Rebuilds the shared occupancy map from the current field.
///
/// `Omega(p)` counts the matched patches covering source pixel `p`, i.e. the number of patch
/// centres within `r = patch / 2` of `p` in both axes: a box sum of the histogram of the field's
/// patch centres. All centres lie in `[r, size - r)`, so no patch is clipped and the clipped box
/// sum is exact. Integer sums are order-independent, so the result is deterministic.
fn init_omega(ws: &mut OmegaWorkspace, nnf: &Nnf, sw: usize, sh: usize, patch: usize) {
    let n = sw * sh;
    if ws.global.len() != n {
        ws.global = (0..n).map(|_| AtomicI32::new(0)).collect();
    }
    if n == 0 {
        return;
    }
    // Histogram of patch centres.
    ws.hist.clear();
    ws.hist.resize(n, 0);
    for &[x, y] in nnf.data() {
        ws.hist[y as usize * sw + x as usize] += 1;
    }
    box_filter(&ws.hist, &mut ws.rows, &ws.global, sw, sh, patch / 2, false);
}

/// Writes (or, with `accumulate`, adds) the `(2r + 1) x (2r + 1)` box sums of the `sw x sh` map
/// `hist`, clipped to the map, into `out`. `rows` is scratch space.
fn box_filter(hist: &[i32], rows: &mut Vec<i32>, out: &[AtomicI32], sw: usize, sh: usize, r: usize, accumulate: bool) {
    // Horizontal box sums (running window over [x - r, x + r], clipped to the row).
    rows.resize(sw * sh, 0);
    rows.par_chunks_mut(sw).zip(hist.par_chunks(sw)).for_each(|(o, inp)| {
        let mut s: i32 = inp[..r.min(sw)].iter().sum();
        for x in 0..sw {
            if x + r < sw {
                s += inp[x + r];
            }
            o[x] = s;
            if x >= r {
                s -= inp[x - r];
            }
        }
    });
    // Vertical box sums of the row sums.
    let rows = &rows[..];
    out.par_chunks(sw).enumerate().for_each_init(Vec::new, |acc: &mut Vec<i32>, (y, o)| {
        let (y0, y1) = (y.saturating_sub(r), (y + r).min(sh - 1));
        acc.clear();
        acc.extend_from_slice(&rows[y0 * sw..(y0 + 1) * sw]);
        for yy in y0 + 1..=y1 {
            for (a, &v) in acc.iter_mut().zip(&rows[yy * sw..(yy + 1) * sw]) {
                *a += v;
            }
        }
        for (cell, &v) in o.iter().zip(acc.iter()) {
            cell.store(if accumulate { cell.load(Relaxed) + v } else { v }, Relaxed);
        }
    });
}

const MERGE_CHUNK: usize = 1 << 14;

/// Groups the band copies into per-chunk slices so chunks can be processed in parallel.
fn chunked(copies: &mut [Vec<i32>]) -> Vec<Vec<&mut [i32]>> {
    let mut per_chunk: Vec<Vec<&mut [i32]>> = Vec::new();
    for copy in copies.iter_mut() {
        for (ci, s) in copy.chunks_mut(MERGE_CHUNK).enumerate() {
            if per_chunk.len() <= ci {
                per_chunk.push(Vec::new());
            }
            per_chunk[ci].push(s);
        }
    }
    per_chunk
}

/// Copies the global map into every band copy.
fn sync_copies(global: &[AtomicI32], copies: &mut [Vec<i32>]) {
    chunked(copies).into_par_iter().enumerate().for_each(|(ci, mut slices)| {
        let g = &global[ci * MERGE_CHUNK..];
        for s in slices.iter_mut() {
            for (d, c) in s.iter_mut().zip(g) {
                *d = c.load(Relaxed);
            }
        }
    });
}

/// Applies the moves the bands accepted during an iteration to the global map and to every band
/// copy, so that all of them again hold the occupancy of the whole field.
///
/// Integer additions commute, so applying the logs in any order gives the same, deterministic
/// result. When few moves were accepted, replaying them is much cheaper than a pass over the
/// maps; when a map would receive more updates than it has cells, the global map instead gets
/// the box filter of the histogram of moved patch centres (as in [`init_omega`]) and the band
/// copies are copied from it.
fn merge_logs(ws: &mut OmegaWorkspace, sw: usize, sh: usize, patch: usize) {
    let n = sw * sh;
    let cost = |moves: usize| moves.saturating_mul(2 * patch * patch);
    let total: usize = ws.logs.iter().map(Vec::len).sum();
    if total == 0 {
        return;
    }
    if cost(total) < n {
        let mut global = SharedOmega { cells: &ws.global, width: sw, patch };
        for &(from, to) in ws.logs.iter().flatten() {
            global.move_patch(from, to);
        }
    } else {
        // Box filter of the histogram of moved patch centres (+1 at the new, -1 at the old).
        ws.hist.clear();
        ws.hist.resize(n, 0);
        for &([fx, fy], [tx, ty]) in ws.logs.iter().flatten() {
            ws.hist[ty as usize * sw + tx as usize] += 1;
            ws.hist[fy as usize * sw + fx as usize] -= 1;
        }
        box_filter(&ws.hist, &mut ws.rows, &ws.global, sw, sh, patch / 2, true);
    }
    let (global, logs) = (&ws.global[..], &ws.logs[..]);
    ws.copies.par_iter_mut().enumerate().for_each(|(b, copy)| {
        if cost(total - logs[b].len()) < n {
            for (_, log) in logs.iter().enumerate().filter(|&(o, _)| o != b) {
                for &(from, to) in log {
                    add_patch_cells(copy, sw, patch, to, 1);
                    add_patch_cells(copy, sw, patch, from, -1);
                }
            }
        } else {
            for (d, c) in copy.iter_mut().zip(global) {
                *d = c.load(Relaxed);
            }
        }
    });
}

/// The current match of the pixel being processed.
struct Current {
    n: [i32; 2],
    err: f32,
    /// Occupancy sum over the patch at `n`, computed on first use and then kept up to date
    /// incrementally. With a band-private map this always equals a fresh `patch_sum`; with the
    /// shared map it may miss concurrent updates by other bands, like any racy read would.
    omega_sum: Option<i64>,
}

/// Runs [`scan_band`] compiled for the best instruction set the CPU supports.
///
/// Rust never contracts or reorders floating-point operations, so every variant computes
/// bit-identical results; only the instruction selection (e.g. AVX2 vector width) differs.
fn scan_band_dispatch<O: OmegaAccess, const K: u8>(
    sp: &PatchSpace<'_>,
    scan: &ScanParams<'_>,
    band: &mut Band<'_>,
    omega: &mut O,
    odd: bool,
    iter_seed: u32,
    mask: Option<&[u8]>,
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: the required CPU features were detected at run time just above.
        unsafe { scan_band_avx2::<O, K>(sp, scan, band, omega, odd, iter_seed, mask) };
        return;
    }
    scan_band::<O, K>(sp, scan, band, omega, odd, iter_seed, mask);
}

/// [`scan_band`] with AVX2 and FMA enabled (everything it calls is inlined).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn scan_band_avx2<O: OmegaAccess, const K: u8>(
    sp: &PatchSpace<'_>,
    scan: &ScanParams<'_>,
    band: &mut Band<'_>,
    omega: &mut O,
    odd: bool,
    iter_seed: u32,
    mask: Option<&[u8]>,
) {
    scan_band::<O, K>(sp, scan, band, omega, odd, iter_seed, mask);
}

/// Sequential PatchMatch scan of one band (the body of the original's OpenMP region).
#[inline(always)]
fn scan_band<O: OmegaAccess, const K: u8>(
    sp: &PatchSpace<'_>,
    scan: &ScanParams<'_>,
    band: &mut Band<'_>,
    omega: &mut O,
    odd: bool,
    iter_seed: u32,
    mask: Option<&[u8]>,
) {
    let w = sp.tw as i32;
    let h = sp.th as i32;
    let hp = (sp.patch / 2) as i32;
    let (sbw, sbh) = (sp.sw as i32, sp.sh as i32);
    let (y0, y1) = (band.y0 as i32, band.y1 as i32);
    let q: i32 = if odd { 1 } else { -1 };
    let (ys, ye) = if odd { (y0, y1) } else { (y1 - 1, y0 - 1) };
    let (xs, xe) = if odd { (0, w) } else { (w - 1, -1) };

    let mut y = ys;
    while y != ye {
        let row_base = ((y - y0) * w) as usize;
        let mut x = xs;
        while x != xe {
            let li = row_base + x as usize;
            if mask.is_none_or(|m| m[(y * w + x) as usize] != 0) {
                let mut cur = Current { n: band.nnf[li], err: band.err[li], omega_sum: None };
                // Propagation from the horizontal neighbour.
                if if odd { x > 0 } else { x < w - 1 } {
                    let mut n = band.nnf[(li as i32 - q) as usize];
                    n[0] += q;
                    if if odd { n[0] < sbw - hp } else { n[0] >= hp } {
                        try_patch::<O, K>(sp, scan, omega, x, y, n, &mut cur);
                    }
                }
                // Propagation from the vertical neighbour (possibly in the next band).
                if if odd { y > 0 } else { y < h - 1 } {
                    let ny = y - q;
                    let mut n = if ny < y0 {
                        band.above.as_ref().expect("missing row above band")[x as usize]
                    } else if ny >= y1 {
                        band.below.as_ref().expect("missing row below band")[x as usize]
                    } else {
                        band.nnf[((ny - y0) * w + x) as usize]
                    };
                    n[1] += q;
                    if if odd { n[1] < sbh - hp } else { n[1] >= hp } {
                        try_patch::<O, K>(sp, scan, omega, x, y, n, &mut cur);
                    }
                }
                // Random search around the post-propagation match, smallest radius first.
                let mut seed = randi(((x | (y << 11)) as u32) ^ iter_seed);
                let pix0 = cur.n;
                for &rad in scan.radii.iter().rev() {
                    let tl = [(pix0[0] - rad).max(hp), (pix0[1] - rad).max(hp)];
                    let br = [(pix0[0] + rad).min(sbw - hp), (pix0[1] + rad).min(sbh - hp)];
                    let rx = randi(seed);
                    let ry = randi(rx);
                    seed = ry;
                    let (rw, rh) = (br[0] - tl[0], br[1] - tl[1]);
                    if rw <= 0 || rh <= 0 {
                        continue;
                    }
                    // rx, ry < 2^31, so this equals the original's signed modulo.
                    let n = [tl[0] + (rx % rw as u32) as i32, tl[1] + (ry % rh as u32) as i32];
                    try_patch::<O, K>(sp, scan, omega, x, y, n, &mut cur);
                }
                band.nnf[li] = cur.n;
                band.err[li] = cur.err;
            }
            x += q;
        }
        y += q;
    }
}

/// Considers source patch `cand` for target pixel `(x, y)` (`tryPatch`).
#[inline(always)]
fn try_patch<O: OmegaAccess, const K: u8>(
    sp: &PatchSpace<'_>,
    scan: &ScanParams<'_>,
    omega: &mut O,
    x: i32,
    y: i32,
    cand: [i32; 2],
    cur: &mut Current,
) {
    // With lambda >= 0, re-evaluating the current match can never change anything: it
    // reproduces the same error and occupancy, so the strict comparison below would fail (or,
    // in the exact kernel's border case, "replace" the match by itself). Skipping it is exact
    // and saves most of the propagation work once the field is coherent. With lambda < 0 the
    // threshold lies below the current error, so the original's re-evaluation stops early with
    // a partial error that can be accepted (and replaces E); that case is evaluated as usual.
    if cand == cur.n && scan.lambda >= 0.0 {
        return;
    }
    if !scan.use_omega {
        let new_err = sp.error::<K>(x, y, cand[0], cand[1], cur.err);
        if new_err < cur.err {
            cur.n = cand;
            cur.err = new_err;
        }
        return;
    }
    let pp = (sp.patch * sp.patch) as f32;
    let cur_sum = *cur.omega_sum.get_or_insert_with(|| omega.patch_sum(cur.n));
    let cur_occ = (cur_sum as f32 / pp) / scan.omega_best;
    let lambda = scan.lambda;
    let threshold = if K == EXACT {
        // The 2019 build (gcc -O3 -march=native) contracts this into a fused multiply-add.
        lambda.mul_add(cur_occ, cur.err)
    } else {
        cur.err + lambda * cur_occ
    };
    let new_err = sp.error::<K>(x, y, cand[0], cand[1], threshold);
    // The occupancy term is non-negative, so a candidate whose error alone exceeds the
    // threshold is rejected without reading the occupancy map (rounding is monotonic, so this
    // decides exactly as the full comparison would).
    if new_err > threshold && scan.lambda >= 0.0 {
        return;
    }
    let new_sum = omega.patch_sum(cand);
    let new_occ = (new_sum as f32 / pp) / scan.omega_best;
    let total = if K == EXACT {
        // Fused as well, except after a border (clamped) patch evaluation.
        if sp.is_interior(x, y) { lambda.mul_add(new_occ, new_err) } else { new_err + lambda * new_occ }
    } else {
        new_err + lambda * new_occ
    };
    if total < threshold {
        omega.move_patch(cur.n, cand);
        // The candidate's patch gained one count per pixel, minus the pixels it shares with the
        // old patch, which lost one.
        let p = sp.patch as i32;
        let overlap = (p - (cand[0] - cur.n[0]).abs()).max(0) * (p - (cand[1] - cur.n[1]).abs()).max(0);
        cur.omega_sum = Some(new_sum + (p * p - overlap) as i64);
        cur.n = cand;
        cur.err = new_err;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnf::nnf_init_random;
    use crate::patch::pack;

    #[test]
    fn randi_matches_macro() {
        // RANDI(u) = 18000 * (u & 65535) + (u >> 16)
        assert_eq!(randi(0), 0);
        assert_eq!(randi(1), 18000);
        assert_eq!(randi(0xFFFF_FFFF), 18000 * 65535 + 65535);
    }

    fn run(sync: OmegaSync, threads: usize) -> (Nnf, Vec<f32>) {
        let (w, h) = (40usize, 36usize);
        let n = w * h;
        let style: Vec<u8> = (0..n).map(|i| ((i % w) * 6) as u8).collect();
        let guide: Vec<u8> = (0..n).map(|i| ((i / w) * 7) as u8).collect();
        let packed = pack(&style, 1, &guide, 1, n);
        let weights = [1.0f32, 1.0];
        let space = PatchSpace::new((w, h), (w, h), 1, 5, &packed, &packed, &weights, None);
        let mut rng = GlibcRand::default();
        let mut nnf = nnf_init_random((w, h), (w, h), 5, &mut rng);
        let mut errors = Vec::new();
        let cfg = PatchMatchConfig { lambda: 10.0, num_iters: 6, num_bands: threads, sync, kernel: Kernel::Fast };
        patchmatch(&space, &mut nnf, &mut errors, &cfg, &mut rng, None, &mut OmegaWorkspace::default());
        (nnf, errors)
    }

    #[test]
    fn converges_towards_identity_on_self_match() {
        let (nnf, errors) = run(OmegaSync::PerTile, 4);
        // Source == target with distinctive gradients: the vast majority of interior pixels
        // should find their own position.
        let (w, h) = (nnf.width(), nnf.height());
        let mut exact = 0;
        let mut total = 0;
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                total += 1;
                if nnf.get(x, y) == [x as i32, y as i32] {
                    exact += 1;
                }
            }
        }
        assert!(exact * 10 >= total * 9, "{exact}/{total} pixels matched themselves");
        assert!(errors.iter().all(|e| e.is_finite() && *e >= 0.0));
    }

    #[test]
    fn per_tile_mode_is_deterministic() {
        let a = run(OmegaSync::PerTile, 4);
        let b = run(OmegaSync::PerTile, 4);
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    /// Brute-force occupancy: for every source pixel, the number of matched patches covering it.
    fn brute_omega(nnf: &Nnf, sw: usize, sh: usize, patch: usize) -> Vec<i32> {
        let r = (patch / 2) as i32;
        let mut o = vec![0; sw * sh];
        for &[x, y] in nnf.data() {
            for yy in y - r..=y + r {
                for xx in x - r..=x + r {
                    o[yy as usize * sw + xx as usize] += 1;
                }
            }
        }
        o
    }

    fn load(cells: &[AtomicI32]) -> Vec<i32> {
        cells.iter().map(|c| c.load(Relaxed)).collect()
    }

    #[test]
    fn init_omega_matches_brute_force() {
        let mut rng = GlibcRand::new(5);
        for (patch, (sw, sh), (tw, th)) in [(3, (9, 7), (20, 13)), (5, (23, 31), (17, 40)), (7, (15, 15), (3, 50))] {
            let nnf = nnf_init_random((tw, th), (sw, sh), patch, &mut rng);
            let mut ws = OmegaWorkspace::default();
            init_omega(&mut ws, &nnf, sw, sh, patch);
            assert_eq!(load(&ws.global), brute_omega(&nnf, sw, sh, patch), "patch {patch}");
        }
    }

    #[test]
    fn per_tile_maps_stay_exact_after_merging() {
        let (w, h) = (40usize, 36usize);
        let n = w * h;
        let style: Vec<u8> = (0..n).map(|i| ((i % w) * 6) as u8).collect();
        let guide: Vec<u8> = (0..n).map(|i| ((i / w) * 7) as u8).collect();
        let packed = pack(&style, 1, &guide, 1, n);
        let weights = [1.0f32, 1.0];
        let space = PatchSpace::new((w, h), (w, h), 1, 5, &packed, &packed, &weights, None);
        let mut rng = GlibcRand::default();
        let mut nnf = nnf_init_random((w, h), (w, h), 5, &mut rng);
        let mut errors = Vec::new();
        let mut ws = OmegaWorkspace::default();
        let cfg = PatchMatchConfig { lambda: 10.0, num_iters: 3, num_bands: 4, sync: OmegaSync::PerTile, kernel: Kernel::Fast };
        // Two calls: the second reuses the maps the first left behind.
        for _ in 0..2 {
            patchmatch(&space, &mut nnf, &mut errors, &cfg, &mut rng, None, &mut ws);
            assert!(ws.per_tile_valid);
            assert_eq!(load(&ws.global), brute_omega(&nnf, w, h, 5));
            let exact = brute_omega(&nnf, w, h, 5);
            assert!(ws.copies.iter().all(|c| *c == exact));
        }
    }

    #[test]
    fn patch_sums_do_not_overflow() {
        let big = i32::MAX / 4;
        let cells: Vec<AtomicI32> = (0..49).map(|_| AtomicI32::new(big)).collect();
        let shared = SharedOmega { cells: &cells, width: 7, patch: 5 };
        assert_eq!(shared.patch_sum([3, 3]), 25 * big as i64);
        let mut copy = vec![big + 1; 49];
        let mut log = Vec::new();
        let local = LocalOmega { cells: &mut copy, width: 7, patch: 5, log: &mut log };
        assert_eq!(local.patch_sum([3, 3]), 25 * (big as i64 + 1));
    }

    #[test]
    fn single_band_modes_agree() {
        let a = run(OmegaSync::PerTile, 1);
        let b = run(OmegaSync::Shared, 1);
        assert_eq!(a.0, b.0);
    }
}
