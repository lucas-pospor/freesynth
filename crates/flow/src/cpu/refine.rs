//! Variational refinement of one level (Kroeger et al. 2016, section 2.3), as in `vr.wgsl`.
//!
//! The energy, its linearisation and the solver are those of the GPU kernel, including its
//! tiling: each dispatch relaxes 32x16 tiles independently, with a two-pixel border read once
//! at the start of the dispatch (block Gauss-Seidel), runs up to two fixed-point iterations of
//! red-black over-relaxation, and every other dispatch shifts the tile grid by half a tile. The
//! tiles keep the CPU version close to the GPU one, and they are a good fit for the CPU too:
//! a tile and its coefficients (about 66 KB) stay in the core's caches, and tiles run in
//! parallel.
//!
//! Inside a tile, pixels are stored by column parity (even and odd columns apart). The tile
//! origin is always even, so one row of one red-black colour is 16 contiguous values, and all
//! four neighbours of those pixels are 16 contiguous values of the other parity: the sweeps
//! are plain vector arithmetic.
//!
//! The data terms depend only on the flow that the level starts from, so they are computed once
//! per level instead of once per dispatch as on the GPU. They are stored in blocks of 16x8
//! pixels, which both tile grids are aligned to, so that a tile reads four contiguous blocks.

use rayon::prelude::*;

use super::lanes::{F16, Lanes, opaque};
use super::{Isa, clamp_index, dispatch, split_coordinate};
use crate::params::DisParams;

const TILE_W: usize = 32;
const TILE_H: usize = 16;
const HALO: usize = 2;
/// Tile rows plus the border.
const SPAN_H: usize = TILE_H + 2 * HALO;
/// Columns of one parity in a tile row plus the border.
const HALF: usize = (TILE_W + 2 * HALO) / 2;
/// Storage of one span row: `HALF` values from index `OFF`, so that the tile's own 16 values
/// (indices 1 to 16) start a cache line and load in one piece.
const ROW: usize = 48;
const OFF: usize = 15;
/// Pixels of one parity in a tile row.
const LANES: usize = TILE_W / 2;
/// Tile grid shift of every other dispatch.
const SHIFT: (usize, usize) = (TILE_W / 2, TILE_H / 2);
/// Fixed-point iterations per dispatch.
const PER_DISPATCH: usize = 2;
/// Stored values per pixel: the three linearised data terms (each `a du + b dv + c`), then the
/// flow they are linearised around.
const NF: usize = 11;
/// Width and height of a block of data terms.
const BLOCK_W: usize = 16;
const BLOCK_H: usize = 8;
/// Values per parity in a block row.
const BLOCK_LANES: usize = BLOCK_W / 2;
/// Values in a block: `[field][column parity][row][column / 2]`.
const BLOCK_LEN: usize = NF * 2 * BLOCK_H * BLOCK_LANES;

/// A block of data terms, aligned to a cache line (its size is a multiple of 64 bytes, so every
/// row of eight values in it is aligned to 32 bytes).
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
struct Block([f32; BLOCK_LEN]);

const ZERO_BLOCK: Block = Block([0.0; BLOCK_LEN]);

/// Frame 1's value and derivatives at a pixel: `I`, `Ix`, `Iy`, `Ixy`, `Ixx`, `Iyy`, then two
/// unused values, aligned so that the eight values load in one piece.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(32))]
struct Derivs([f32; 8]);

/// Buffers reused between flows.
#[derive(Debug, Default)]
pub(super) struct Scratch {
    /// Frame 1 and its derivatives per pixel.
    d1: Vec<Derivs>,
    /// Data terms in blocks, with a ring of zero blocks around the image so that every tile
    /// finds its four blocks.
    terms: Vec<Block>,
    /// The other refinement iterate.
    iterate: Vec<[f32; 2]>,
}

/// Constants of one level's refinement.
#[derive(Clone, Copy)]
struct Consts {
    w: usize,
    h: usize,
    /// Blocks per row of `terms`, the ring included.
    blocks_x: usize,
    half_alpha: f32,
    delta: f32,
    gamma: f32,
    omega: f32,
    zeta2: f32,
    eps2: f32,
    inner: usize,
}

/// Refines `flow` (`w` x `h`) in place.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine_level(
    s: &mut Scratch,
    i0: &[f32],
    i1: &[f32],
    w: usize,
    h: usize,
    flow: &mut Vec<[f32; 2]>,
    p: &DisParams,
    isa: Isa,
) {
    let c = Consts {
        w,
        h,
        blocks_x: w.div_ceil(BLOCK_W) + 2,
        half_alpha: 0.5 * p.vr_alpha,
        delta: p.vr_delta,
        gamma: p.vr_gamma,
        omega: p.vr_omega,
        zeta2: p.vr_zeta * p.vr_zeta,
        eps2: p.vr_epsilon * p.vr_epsilon,
        inner: p.vr_inner_iterations,
    };
    derivatives(&mut s.d1, i1, w, h, isa);
    let blocks_y = h.div_ceil(BLOCK_H) + 2;
    s.terms.resize(blocks_y * c.blocks_x, ZERO_BLOCK);
    {
        let (d1, flow) = (&s.d1[..], &flow[..]);
        let (first, rest) = s.terms.split_at_mut(c.blocks_x);
        let (inner, last) = rest.split_at_mut((blocks_y - 2) * c.blocks_x);
        first.fill(ZERO_BLOCK);
        last.fill(ZERO_BLOCK);
        inner.par_chunks_mut(c.blocks_x).enumerate().for_each_init(Vec::new, |rows, (by, blocks)| {
            // The ring blocks at both ends of the block row.
            blocks[0] = ZERO_BLOCK;
            blocks[c.blocks_x - 1] = ZERO_BLOCK;
            for r in 0..BLOCK_H {
                let y = by * BLOCK_H + r;
                if y >= h {
                    // Rows below the image in the last block row.
                    for b in blocks.iter_mut() {
                        for fq in 0..NF * 2 {
                            b.0[(fq * BLOCK_H + r) * BLOCK_LANES..][..BLOCK_LANES].fill(0.0);
                        }
                    }
                    continue;
                }
                padded_rows(i0, w, h, y, rows);
                let (c, rows) = (&c, &rows[..]);
                dispatch!(isa, (c: &Consts, rows: &[f32], d1: &[Derivs], flow: &[[f32; 2]], y: usize, r: usize, blocks: &mut [Block]) {
                    terms_row(c, rows, d1, flow, y, r, blocks)
                });
            }
        });
    }
    s.iterate.resize(w * h, [0.0; 2]);
    let iterations = p.vr_iterations;
    for k in 0..iterations.div_ceil(PER_DISPATCH) {
        let outer = PER_DISPATCH.min(iterations - k * PER_DISPATCH);
        let shift = if k % 2 == 1 { SHIFT } else { (0, 0) };
        sweep_tiles(&c, &s.terms, flow, &mut s.iterate, shift, outer, isa);
        std::mem::swap(flow, &mut s.iterate);
    }
}

/// Pixels per chunk of the per-row loops below.
const CHUNK: usize = 16;

/// Rows `y - 1`, `y` and `y + 1` of `img` (clamped), each padded so that entry `i` is column
/// `i - 1` clamped to the image, and long enough for whole chunks plus the neighbours.
fn padded_rows(img: &[f32], w: usize, h: usize, y: usize, buf: &mut Vec<f32>) {
    let len = w.next_multiple_of(CHUNK) + 2;
    buf.resize(3 * len, 0.0);
    for (r, src_y) in [y.saturating_sub(1), y, (y + 1).min(h - 1)].into_iter().enumerate() {
        let src = &img[src_y * w..(src_y + 1) * w];
        let dst = &mut buf[r * len..(r + 1) * len];
        dst[1..=w].copy_from_slice(src);
        dst[0] = src[0];
        dst[w + 1..].fill(src[w - 1]);
    }
}

/// Derivatives by central differences at the 16 pixels from column `x`, from three padded rows
/// (`padded_rows`): `[I, Ix, Iy, Ixy, Ixx, Iyy]`.
#[inline(always)]
fn derivatives16(rows: &[f32], x: usize) -> [F16; 6] {
    let len = rows.len() / 3;
    let at = |r: usize, dx: usize| F16::load(&rows[r * len + x + dx..]);
    let (v0, v1, v2) = (at(0, 0), at(0, 1), at(0, 2));
    let (v3, v4, v5) = (at(1, 0), at(1, 1), at(1, 2));
    let (v6, v7, v8) = (at(2, 0), at(2, 1), at(2, 2));
    let (half, quarter, two) = (F16::splat(0.5), F16::splat(0.25), F16::splat(2.0));
    [
        v4,
        half * (v5 - v3),
        half * (v7 - v1),
        quarter * ((v8 - v6) - (v2 - v0)),
        v5 - two * v4 + v3,
        v7 - two * v4 + v1,
    ]
}

/// Fills the interleaved plane of frame 1's value and derivatives (`I`, `Ix`, `Iy`, `Ixy`,
/// `Ixx`, `Iyy`; central differences, clamped at the border).
fn derivatives(d1: &mut Vec<Derivs>, i1: &[f32], w: usize, h: usize, isa: Isa) {
    d1.resize(w * h, Derivs::default());
    d1.par_chunks_mut(w).enumerate().for_each_init(Vec::new, |rows, (y, out)| {
        padded_rows(i1, w, h, y, rows);
        let rows = &rows[..];
        dispatch!(isa, (rows: &[f32], w: usize, out: &mut [Derivs]) {
            for x in (0..w).step_by(CHUNK) {
                let v = derivatives16(rows, x);
                for (k, o) in out[x..].iter_mut().take(CHUNK).enumerate() {
                    *o = Derivs([v[0].0[k], v[1].0[k], v[2].0[k], v[3].0[k], v[4].0[k], v[5].0[k], 0.0, 0.0]);
                }
            }
        });
    });
}

/// Data terms of row `y`, written to row `r` of a row of blocks: brightness constancy and the
/// two gradient constancy terms, each linearised around the level's starting flow and divided
/// by the norm of its spatial gradient (plus `zeta`). Spatial derivatives average both frames
/// (frame 1 warped by the flow, its derivatives blended bilinearly from the four surrounding
/// pixels), temporal ones are differences. All terms are zero where the flow leaves the frame,
/// and for the columns right of the image in the last block. `rows` are frame 0's rows around
/// `y` (`padded_rows`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn terms_row(c: &Consts, rows: &[f32], d1: &[Derivs], flow: &[[f32; 2]], y: usize, r: usize, blocks: &mut [Block]) {
    let (w, h) = (c.w, c.h);
    for x0 in (0..w).step_by(CHUNK) {
        let n = CHUNK.min(w - x0);
        // Frame 1's value and derivatives at the warped positions (one gather per lane), and
        // whether the warped position is inside the frame.
        let mut warped = [[0.0f32; CHUNK]; 6];
        let mut inside = [0.0f32; CHUNK];
        let (mut fu, mut fv) = ([0.0f32; CHUNK], [0.0f32; CHUNK]);
        for k in 0..n {
            let u = flow[y * w + x0 + k];
            (fu[k], fv[k]) = (u[0], u[1]);
            let (wx, wy) = ((x0 + k) as f32 + u[0], y as f32 + u[1]);
            if !(wx >= 0.0 && wy >= 0.0 && wx <= (w - 1) as f32 && wy <= (h - 1) as f32) {
                continue;
            }
            inside[k] = 1.0;
            let (bx, ax) = split_coordinate(wx);
            let (by, ay) = split_coordinate(wy);
            let (x0, y0) = (bx as usize, by as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let at = |i: usize| Lanes(d1[i].0);
            let top = Lanes::mix(at(y0 * w + x0), at(y0 * w + x1), ax);
            let bottom = Lanes::mix(at(y1 * w + x0), at(y1 * w + x1), ax);
            let v = Lanes::mix(top, bottom, ay).0;
            for q in 0..6 {
                warped[q][k] = v[q];
            }
        }
        let [i, wix, wiy, wixy, wixx, wiyy] = warped.map(Lanes);
        let d0 = derivatives16(rows, x0);
        let half = F16::splat(0.5);
        let ix = half * (d0[1] + wix);
        let iy = half * (d0[2] + wiy);
        let iz = i - d0[0];
        let ixy = half * (d0[3] + wixy);
        let ixx = half * (d0[4] + wixx);
        let iyy = half * (d0[5] + wiyy);
        let ixz = wix - d0[1];
        let iyz = wiy - d0[2];
        let (one, zeta2, inside) = (F16::splat(1.0), F16::splat(c.zeta2), Lanes(inside));
        // Out-of-frame lanes (and lanes right of the image) get zero terms: their warped values
        // are zero, but the frame 0 derivatives are not.
        let nb = inside * (one / ix.mul_add(ix, iy.mul_add(iy, zeta2)).sqrt());
        let ngx = inside * (one / ixx.mul_add(ixx, ixy.mul_add(ixy, zeta2)).sqrt());
        let ngy = inside * (one / ixy.mul_add(ixy, iyy.mul_add(iyy, zeta2)).sqrt());
        let terms = [ix * nb, iy * nb, iz * nb, ixx * ngx, ixy * ngx, ixz * ngx, ixy * ngy, iyy * ngy, iyz * ngy, Lanes(fu), Lanes(fv)];
        // The chunk is one block column (block 0 is the ring); lane k is column x0 + k, so
        // even lanes go to parity 0 and odd lanes to parity 1.
        let block = &mut blocks[x0 / BLOCK_W + 1].0;
        for (f, v) in terms.into_iter().enumerate() {
            for q in 0..2 {
                let lanes = Lanes::<BLOCK_LANES>::from_fn(|m| v.0[2 * m + q]);
                lanes.store(&mut block[((f * 2 + q) * BLOCK_H + r) * BLOCK_LANES..]);
            }
        }
    }
}

/// Values of a tile and its border, by column parity: `[parity][row][column / 2]`.
type Span = [[[f32; ROW]; SPAN_H]; 2];
/// One value per tile pixel: `[parity][row][column / 2]`.
type PerPixel = [[[f32; LANES]; TILE_H]; 2];

/// Working set of one tile. Aligned to a cache line: the span arrays are whole multiples of 64
/// bytes, so every per-pixel row (16 values) starts a cache line and loads it in one piece.
#[derive(Clone)]
#[repr(C, align(64))]
struct Tile {
    /// Flow over the tile and its border.
    u: Span,
    v: Span,
    /// Smoothness weights over the tile and a one-pixel border.
    sw: Span,
    /// Data terms and starting flow of the tile pixels.
    terms: [PerPixel; NF],
    /// Over-relaxation factor, 0 for positions outside the image.
    omega: PerPixel,
    /// The current fixed-point iteration's linear system, solved for the new flow:
    /// `U = g + M (wl U_left + wr U_right + wu U_up + wd U_down)` with `M` symmetric.
    gu: PerPixel,
    gv: PerPixel,
    m11: PerPixel,
    m12: PerPixel,
    m22: PerPixel,
    wl: PerPixel,
    wr: PerPixel,
    wu: PerPixel,
    wd: PerPixel,
}

const _: () = assert!(std::mem::size_of::<Span>().is_multiple_of(64) && std::mem::size_of::<PerPixel>().is_multiple_of(64));

impl Tile {
    fn boxed() -> Box<Self> {
        const SPAN: Span = [[[0.0; ROW]; SPAN_H]; 2];
        const LANES0: PerPixel = [[[0.0; LANES]; TILE_H]; 2];
        Box::new(Self {
            u: SPAN,
            v: SPAN,
            sw: SPAN,
            terms: [LANES0; NF],
            omega: LANES0,
            gu: LANES0,
            gv: LANES0,
            m11: LANES0,
            m12: LANES0,
            m22: LANES0,
            wl: LANES0,
            wr: LANES0,
            wu: LANES0,
            wd: LANES0,
        })
    }
}

/// Column groups per band of tiles: the unit of parallel work is one group of a band.
const GROUPS: usize = 2;

thread_local! {
    /// Each thread's tile working set, reused across tiles and dispatches.
    static TILE: std::cell::RefCell<Option<Box<Tile>>> = const { std::cell::RefCell::new(None) };
}

/// One dispatch: relaxes every tile of the grid shifted by `shift`, reading `input` and writing
/// `output`.
fn sweep_tiles(c: &Consts, terms: &[Block], input: &[[f32; 2]], output: &mut [[f32; 2]], shift: (usize, usize), outer: usize, isa: Isa) {
    let (w, h) = (c.w, c.h);
    let tiles_x = (w + shift.0).div_ceil(TILE_W);
    let per_group = tiles_x.div_ceil(GROUPS);
    // Rows of tile row `ty` are oy..oy + TILE_H with oy = ty * TILE_H - shift.1; the first tile
    // row may start above the image. Each band is split into groups of whole tiles; a group
    // owns its part of every row of the band.
    let first_rows = (TILE_H - shift.1).min(h);
    let (head, tail) = output.split_at_mut(first_rows * w);
    let mut tasks = Vec::new();
    for (ty, band) in std::iter::once(head).chain(tail.chunks_mut(TILE_H * w)).enumerate() {
        let mut groups: Vec<Vec<&mut [[f32; 2]]>> = (0..GROUPS).map(|_| Vec::with_capacity(TILE_H)).collect();
        for mut row in band.chunks_mut(w) {
            for (g, group) in groups.iter_mut().enumerate() {
                let end = ((g + 1) * per_group * TILE_W).saturating_sub(shift.0).min(w);
                let (seg, rest) = row.split_at_mut(end.saturating_sub(w - row.len()));
                group.push(seg);
                row = rest;
            }
        }
        tasks.extend(groups.into_iter().enumerate().map(|(g, rows)| (ty, g, rows)));
    }
    tasks.into_par_iter().for_each(|(ty, g, mut rows)| {
        let oy = (ty * TILE_H) as isize - shift.1 as isize;
        TILE.with_borrow_mut(|tile| {
            let tile = &mut **tile.get_or_insert_with(Tile::boxed);
            // Columns of this group, from the first column it owns.
            let x0 = (g * per_group * TILE_W).saturating_sub(shift.0);
            let mut x_lo = x0;
            for tx in g * per_group..((g + 1) * per_group).min(tiles_x) {
                let ox = (tx * TILE_W) as isize - shift.0 as isize;
                let x_hi = ((tx + 1) * TILE_W - shift.0).min(w);
                let rows = &mut rows[..];
                let cols = x_lo..x_hi;
                dispatch!(isa, (tile: &mut Tile, c: &Consts, terms: &[Block], input: &[[f32; 2]], ox: isize, oy: isize, outer: usize, rows: &mut [&mut [[f32; 2]]], cols: std::ops::Range<usize>, x0: usize) {
                    relax_tile(tile, c, terms, input, ox, oy, outer, rows, cols, x0)
                });
                x_lo = x_hi;
            }
        });
    });
}

/// Loads, relaxes and stores one tile with origin `(ox, oy)` (both even). `rows` are the tile's
/// rows that lie inside the image; entry `x - x0` of a row is image column `x`, and the tile
/// writes columns `cols`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn relax_tile(
    t: &mut Tile,
    c: &Consts,
    terms: &[Block],
    input: &[[f32; 2]],
    ox: isize,
    oy: isize,
    outer: usize,
    rows: &mut [&mut [[f32; 2]]],
    cols: std::ops::Range<usize>,
    x0: usize,
) {
    let (w, h) = (c.w, c.h);
    // Flow over the tile and its border, clamped to the image.
    let sx = ox - HALO as isize;
    let interior = sx >= 0 && sx as usize + 2 * HALF <= w;
    for cy in 0..SPAN_H {
        let row = &input[clamp_index(oy - HALO as isize + cy as isize, h) * w..][..w];
        if interior {
            let row: &[[f32; 2]; 2 * HALF] = row[sx as usize..sx as usize + 2 * HALF].try_into().expect("span fits");
            for k in 0..HALF {
                [t.u[0][cy][OFF + k], t.v[0][cy][OFF + k]] = row[2 * k];
                [t.u[1][cy][OFF + k], t.v[1][cy][OFF + k]] = row[2 * k + 1];
            }
        } else {
            for cx in 0..2 * HALF {
                let [u, v] = row[clamp_index(sx + cx as isize, w)];
                t.u[cx & 1][cy][OFF + (cx >> 1)] = u;
                t.v[cx & 1][cy][OFF + (cx >> 1)] = v;
            }
        }
    }
    // Data terms from the tile's four blocks (block coordinates are shifted by the ring);
    // positions outside the image have zero terms and are never updated.
    let (bx0, by0) = ((ox / BLOCK_W as isize + 1) as usize, (oy / BLOCK_H as isize + 1) as usize);
    for (half_y, by) in [by0, by0 + 1].into_iter().enumerate() {
        for (half_x, bx) in [bx0, bx0 + 1].into_iter().enumerate() {
            let block = &terms[by * c.blocks_x + bx].0;
            for f in 0..NF {
                for q in 0..2 {
                    let src = &block[(f * 2 + q) * BLOCK_H * BLOCK_LANES..];
                    for r in 0..BLOCK_H {
                        let lanes = Lanes::<BLOCK_LANES>::load(&src[r * BLOCK_LANES..]);
                        lanes.store(&mut t.terms[f][q][half_y * BLOCK_H + r][half_x * BLOCK_LANES..]);
                    }
                }
            }
        }
    }
    let inside = ox >= 0 && oy >= 0 && ox as usize + TILE_W <= w && oy as usize + TILE_H <= h;
    if inside {
        t.omega = [[[c.omega; LANES]; TILE_H]; 2];
    } else {
        for ty in 0..TILE_H {
            let y = oy + ty as isize;
            for q in 0..2 {
                for (i, o) in t.omega[q][ty].iter_mut().enumerate() {
                    let x = ox + (2 * i + q) as isize;
                    *o = if (0..h as isize).contains(&y) && (0..w as isize).contains(&x) { c.omega } else { 0.0 };
                }
            }
        }
    }
    for _ in 0..outer {
        smoothness_weights(t, c.eps2);
        linearise(t, c, ox, oy);
        for _ in 0..c.inner {
            for colour in 0..opaque(2) {
                relax_colour(t, colour);
            }
        }
    }
    // Store the tile pixels inside the image.
    for (k, row) in rows.iter_mut().enumerate() {
        let ty = (oy.max(0) - oy) as usize + k;
        let cy = ty + HALO;
        if cols.len() == TILE_W {
            // A whole tile row: interleave the two parities back.
            let out: &mut [[f32; 2]; TILE_W] = (&mut row[cols.start - x0..cols.end - x0]).try_into().expect("tile row");
            for i in 0..LANES {
                out[2 * i] = [t.u[0][cy][OFF + i + 1], t.v[0][cy][OFF + i + 1]];
                out[2 * i + 1] = [t.u[1][cy][OFF + i + 1], t.v[1][cy][OFF + i + 1]];
            }
        } else {
            for x in cols.clone() {
                let cx = (x as isize - ox) as usize;
                row[x - x0] = [t.u[cx & 1][cy][OFF + cx / 2 + 1], t.v[cx & 1][cy][OFF + cx / 2 + 1]];
            }
        }
    }
}

/// Smoothness weights `Psi'` from central differences of the flow, over the tile and a one-pixel
/// border. The span repeats the edge value outside the image, so on the outermost row and column
/// of the image the difference is half the one-sided one, as in the GPU kernel.
#[inline(always)]
fn smoothness_weights(t: &mut Tile, eps2: f32) {
    // 17 columns of each parity: span column cx = 2 k + q for k = i + 1 - q, whose horizontal
    // neighbours are entries i and i + 1 of the other parity.
    type L = Lanes<{ HALF - 1 }>;
    let half = L::splat(0.5);
    for cy in 1..opaque(SPAN_H - 1) {
        for q in 0..opaque(2) {
            let (o, k0) = (1 - q, 1 - q);
            let dxu = half * (L::load(&t.u[o][cy][OFF + 1..]) - L::load(&t.u[o][cy][OFF..]));
            let dxv = half * (L::load(&t.v[o][cy][OFF + 1..]) - L::load(&t.v[o][cy][OFF..]));
            let dyu = half * (L::load(&t.u[q][cy + 1][OFF + k0..]) - L::load(&t.u[q][cy - 1][OFF + k0..]));
            let dyv = half * (L::load(&t.v[q][cy + 1][OFF + k0..]) - L::load(&t.v[q][cy - 1][OFF + k0..]));
            let s = dxu.mul_add(dxu, dxv * dxv) + dyu.mul_add(dyu, dyv.mul_add(dyv, L::splat(eps2)));
            (half / s.sqrt()).store(&mut t.sw[q][cy][OFF + k0..]);
        }
    }
}

/// Evaluates the robust weights at the current flow and sets up every tile pixel's linear
/// system, `(A + W) du = b + sum_n w_n (U_n - u0)` with `W` the sum of the edge weights, solved
/// for the new flow.
#[inline(always)]
fn linearise(t: &mut Tile, c: &Consts, ox: isize, oy: isize) {
    let splat = F16::splat;
    let half_alpha = splat(c.half_alpha);
    for q in 0..opaque(2) {
        let o = 1 - q;
        // Edge weights to neighbours outside the image are zero: 0/1 factors per lane.
        let left_ok = F16::from_fn(|i| if ox + (2 * i + q) as isize > 0 { 1.0 } else { 0.0 });
        let right_ok = F16::from_fn(|i| if ox + (2 * i + q) as isize + 1 < c.w as isize { 1.0 } else { 0.0 });
        for ty in 0..opaque(TILE_H) {
            let cy = ty + HALO;
            let y = oy + ty as isize;
            let up_ok = splat(if y > 0 { 1.0 } else { 0.0 });
            let down_ok = splat(if y + 1 < c.h as isize { 1.0 } else { 0.0 });
            let (su, sv) = (F16::load(&t.u[q][cy][OFF + 1..]), F16::load(&t.v[q][cy][OFF + 1..]));
            let ws = F16::load(&t.sw[q][cy][OFF + 1..]);
            let (sl, sr) = (F16::load(&t.sw[o][cy][OFF + q..]), F16::load(&t.sw[o][cy][OFF + q + 1..]));
            let (s_up, s_down) = (F16::load(&t.sw[q][cy - 1][OFF + 1..]), F16::load(&t.sw[q][cy + 1][OFF + 1..]));
            let tm = &t.terms;
            let (t0, t1, t2) = (Lanes(tm[0][q][ty]), Lanes(tm[1][q][ty]), Lanes(tm[2][q][ty]));
            let (t3, t4, t5) = (Lanes(tm[3][q][ty]), Lanes(tm[4][q][ty]), Lanes(tm[5][q][ty]));
            let (t6, t7, t8) = (Lanes(tm[6][q][ty]), Lanes(tm[7][q][ty]), Lanes(tm[8][q][ty]));
            let (u0, v0) = (Lanes(tm[9][q][ty]), Lanes(tm[10][q][ty]));
            let (du, dv) = (su - u0, sv - v0);
            let ei = t0.mul_add(du, t1.mul_add(dv, t2));
            let egx = t3.mul_add(du, t4.mul_add(dv, t5));
            let egy = t6.mul_add(du, t7.mul_add(dv, t8));
            let eps2 = splat(c.eps2);
            let pi = splat(c.delta * 0.5) / ei.mul_add(ei, eps2).sqrt();
            let pg = splat(c.gamma * 0.5) / egx.mul_add(egx, egy.mul_add(egy, eps2)).sqrt();
            let a11 = (pi * t0).mul_add(t0, pg * t3.mul_add(t3, t6 * t6));
            let a12 = (pi * t0).mul_add(t1, pg * t3.mul_add(t4, t6 * t7));
            let a22 = (pi * t1).mul_add(t1, pg * t4.mul_add(t4, t7 * t7));
            let bu = -(pi * t0).mul_add(t2, pg * t3.mul_add(t5, t6 * t8));
            let bv = -(pi * t1).mul_add(t2, pg * t4.mul_add(t5, t7 * t8));
            let wl = half_alpha * ((ws + sl) * left_ok);
            let wr = half_alpha * ((ws + sr) * right_ok);
            let wu = half_alpha * ((ws + s_up) * up_ok);
            let wd = half_alpha * ((ws + s_down) * down_ok);
            let wsum = wl + wr + wu + wd;
            let (m11, m22) = (a11 + wsum, a22 + wsum);
            let det = m11.mul_add(m22, -(a12 * a12));
            let inv_det = det.map(|d| if d > 1e-12 { 1.0 / d } else { 0.0 });
            let (i11, i12, i22) = (m22 * inv_det, -a12 * inv_det, m11 * inv_det);
            let (bu, bv) = (wsum.mul_add(-u0, bu), wsum.mul_add(-v0, bv));
            t.gu[q][ty] = i11.mul_add(bu, i12.mul_add(bv, u0)).0;
            t.gv[q][ty] = i12.mul_add(bu, i22.mul_add(bv, v0)).0;
            t.m11[q][ty] = i11.0;
            t.m12[q][ty] = i12.0;
            t.m22[q][ty] = i22.0;
            t.wl[q][ty] = wl.0;
            t.wr[q][ty] = wr.0;
            t.wu[q][ty] = wu.0;
            t.wd[q][ty] = wd.0;
        }
    }
}

/// One over-relaxed update of every tile pixel of one colour (`(x + y) % 2`).
#[inline(always)]
fn relax_colour(t: &mut Tile, colour: usize) {
    for ty in 0..opaque(TILE_H) {
        let cy = ty + HALO;
        // The tile origin is even, so the colour of span column cx in span row cy is
        // (cx + cy) % 2.
        let q = (colour + cy) & 1;
        let o = 1 - q;
        let (wl, wr, wu, wd) = (Lanes(t.wl[q][ty]), Lanes(t.wr[q][ty]), Lanes(t.wu[q][ty]), Lanes(t.wd[q][ty]));
        let su = wd.mul_add(
            F16::load(&t.u[q][cy + 1][OFF + 1..]),
            wu.mul_add(F16::load(&t.u[q][cy - 1][OFF + 1..]), wr.mul_add(F16::load(&t.u[o][cy][OFF + q + 1..]), wl * F16::load(&t.u[o][cy][OFF + q..]))),
        );
        let sv = wd.mul_add(
            F16::load(&t.v[q][cy + 1][OFF + 1..]),
            wu.mul_add(F16::load(&t.v[q][cy - 1][OFF + 1..]), wr.mul_add(F16::load(&t.v[o][cy][OFF + q + 1..]), wl * F16::load(&t.v[o][cy][OFF + q..]))),
        );
        let (m11, m12, m22) = (Lanes(t.m11[q][ty]), Lanes(t.m12[q][ty]), Lanes(t.m22[q][ty]));
        let nu = m11.mul_add(su, m12.mul_add(sv, Lanes(t.gu[q][ty])));
        let nv = m12.mul_add(su, m22.mul_add(sv, Lanes(t.gv[q][ty])));
        let om = Lanes(t.omega[q][ty]);
        let (cu, cv) = (F16::load(&t.u[q][cy][OFF + 1..]), F16::load(&t.v[q][cy][OFF + 1..]));
        om.mul_add(nu - cu, cu).store(&mut t.u[q][cy][OFF + 1..]);
        om.mul_add(nv - cv, cv).store(&mut t.v[q][cy][OFF + 1..]);
    }
}
