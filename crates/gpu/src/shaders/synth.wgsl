// FreeSynth synthesis kernels: a WGSL port of the 2019 ebsynth CUDA engine (ebsynth_cuda.cu).
//
// This file is a template: the host substitutes the double-brace placeholders (channel layout,
// patch radius, modulation, workgroup size) before compiling, so loops over words and patch
// pixels have constant trip counts.
//
// Images are stored with 8-bit channels packed four per u32 word, `*_W` words per pixel (the
// last word zero-padded). The source style and guide are interleaved in one buffer; the target
// style, guide and modulation have their own buffers.
//
// Mask convention: 0 = pixel still being optimised, 1 = converged (the CUDA engine used 255 and
// 0). A zero-filled mask therefore means "search everywhere".

const STYLE_W: u32 = {{STYLE_W}}u;   // words per style pixel
const GUIDE_W: u32 = {{GUIDE_W}}u;   // words per guide pixel
const SRC_W: u32 = STYLE_W + GUIDE_W; // words per packed source pixel
const R: i32 = {{R}};                // patch radius (patch size = 2R + 1)
const MODULATED: bool = {{MODULATED}};
const WG: u32 = {{WG}}u;
// Cache the target patch neighbourhood of each workgroup in workgroup memory.
const USE_TILE: bool = {{USE_TILE}};

const PATCH: i32 = 2 * R + 1;
// Target tile: the workgroup's pixels plus an R-pixel apron, with every word a patch error
// reads from the target side (style, guide, modulation). The checkerboard kernel's workgroups
// cover 2 WG x WG pixels and use a (2 WG + 2R) x TILE tile; its pipelines are built from a module
// of their own whose TILE_LEN (and USE_TILE) is sized for that tile, the other kernels' from one
// sized for TILE x TILE.
const TILE: i32 = i32(WG) + 2 * R;
const TILE_WORDS: u32 = STYLE_W + GUIDE_W + select(0u, GUIDE_W, MODULATED);
const TILE_LEN: u32 = {{TILE_LEN}}u;
const FLT_MAX: f32 = 3.40282347e+38;

// Offset of the propagation pass (4, 2 or 1 in the CUDA engine); one pipeline per value.
override PROP_R: i32 = 1;
// Colour of the pixels a checkerboard half-pass updates: those with (x + y) % 2 == CB_COLOR.
override CB_COLOR: u32 = 0u;

struct Params {
    tw: i32, th: i32,           // target (level) size
    sw: i32, sh: i32,           // source (level) size
    prev_tw: i32, prev_th: i32, // size of the previous (coarser) level, for NNF upscaling
    patch_size: i32,            // patch size used for NNF upscale clamping
    use_omega: u32,             // uniformity term enabled (lambda != 0)
    lambda: f32,
    omega_best: f32,            // occupancy of a perfectly uniform assignment
    stop_threshold: u32,
    seed: u32,
    round_votes: u32,           // 1: round votes to nearest, 0: truncate (2019)
    vote_norm: f32,             // patch_size^2 * style channels (weighted voting)
    _pad0: u32, _pad1: u32,
    style_w: array<vec4<f32>, STYLE_W>,
    guide_w: array<vec4<f32>, GUIDE_W>,
}

@group(0) @binding(0) var<uniform> P: Params;
// Source style and guide, packed per pixel (style words first, SRC_W words per pixel) so that
// one source pixel is a single contiguous read.
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> tgt_guide: array<u32>;
@group(0) @binding(3) var<storage, read> tgt_mod: array<u32>;
// Current target style (read by the patch distance) and the vote output.
@group(0) @binding(4) var<storage, read_write> style_cur: array<u32>;
@group(0) @binding(5) var<storage, read_write> style_new: array<u32>;
// Current NNF and the output of passes that must not update it in place.
@group(0) @binding(6) var<storage, read_write> nnf_cur: array<vec2<i32>>;
@group(0) @binding(7) var<storage, read_write> nnf_new: array<vec2<i32>>;
@group(0) @binding(8) var<storage, read_write> err: array<f32>;
@group(0) @binding(9) var<storage, read_write> omega: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> mask: array<u32>;
@group(0) @binding(11) var<storage, read_write> mask_tmp: array<u32>;
@group(0) @binding(12) var<storage, read_write> rng: array<u32>;

// ---------------------------------------------------------------------------------------------
// Helpers

fn bytes(v: u32) -> vec4<f32> {
    return vec4<f32>(unpack4xU8(v));
}

/// PCG-RXS-M-XS 32-bit step (replaces the CUDA engine's 64-bit PCG; WGSL has no u64).
fn pcg_next(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_hash(v: u32) -> u32 {
    var s = v;
    return pcg_next(&s);
}

/// Global invocation id as a pixel inside the current target level, or (-1, -1).
fn target_pixel(gid: vec3<u32>) -> vec2<i32> {
    let p = vec2<i32>(gid.xy);
    if (p.x >= P.tw || p.y >= P.th) {
        return vec2<i32>(-1, -1);
    }
    return p;
}

// ---------------------------------------------------------------------------------------------
// Patch distance (PatchSSD_Split / PatchSSD_Split_Modulation)

var<workgroup> tile: array<u32, TILE_LEN>;
/// Tile index of the top-left pixel of the current invocation's target patch.
var<private> tile_base: u32;
/// Pixels per tile row.
var<private> tile_stride: u32;

/// Fills `tile` with the (clamped) target pixels of the `size` rectangle whose top-left pixel is
/// `origin`. Must be called in uniform control flow by every invocation.
fn load_tile_rect(origin: vec2<i32>, size: vec2<i32>, lid: vec3<u32>) {
    tile_stride = u32(size.x);
    for (var k = lid.y * WG + lid.x; k < u32(size.x * size.y) * TILE_WORDS; k += WG * WG) {
        let p = i32(k / TILE_WORDS);
        let w = k % TILE_WORDS;
        let g = clamp(origin + vec2<i32>(p % size.x, p / size.x), vec2<i32>(0, 0), vec2<i32>(P.tw - 1, P.th - 1));
        let gi = u32(g.y * P.tw + g.x);
        var v: u32;
        if (w < STYLE_W) {
            v = style_cur[gi * STYLE_W + w];
        } else if (w < STYLE_W + GUIDE_W) {
            v = tgt_guide[gi * GUIDE_W + w - STYLE_W];
        } else {
            v = tgt_mod[gi * GUIDE_W + w - STYLE_W - GUIDE_W];
        }
        tile[k] = v;
    }
    workgroupBarrier();
}

/// Fills `tile` with the (clamped) target pixels around the workgroup and records the
/// invocation's patch position. Must be called in uniform control flow by every invocation.
fn load_tile(wid: vec3<u32>, lid: vec3<u32>) {
    if (!USE_TILE) { return; }
    load_tile_rect(vec2<i32>(wid.xy * WG) - vec2<i32>(R, R), vec2<i32>(TILE, TILE), lid);
    tile_base = lid.y * u32(TILE) + lid.x;
}

/// Error between target pixel `ti` (tile index) and source pixel `si`.
fn pixel_error_tile(ti: u32, si: u32) -> f32 {
    var e = 0.0;
    let o = ti * TILE_WORDS;
    for (var k = 0u; k < STYLE_W; k++) {
        let d = bytes(tile[o + k]) - bytes(src[si * SRC_W + k]);
        e += dot(P.style_w[k], d * d);
    }
    for (var k = 0u; k < GUIDE_W; k++) {
        let d = bytes(tile[o + STYLE_W + k]) - bytes(src[si * SRC_W + STYLE_W + k]);
        var w = P.guide_w[k];
        if (MODULATED) {
            w *= bytes(tile[o + STYLE_W + GUIDE_W + k]) / 255.0;
        }
        e += dot(w, d * d);
    }
    return e;
}

/// Error between target pixel `ti` (global index) and source pixel `si`.
fn pixel_error(ti: u32, si: u32) -> f32 {
    var e = 0.0;
    for (var k = 0u; k < STYLE_W; k++) {
        let d = bytes(style_cur[ti * STYLE_W + k]) - bytes(src[si * SRC_W + k]);
        e += dot(P.style_w[k], d * d);
    }
    for (var k = 0u; k < GUIDE_W; k++) {
        let d = bytes(tgt_guide[ti * GUIDE_W + k]) - bytes(src[si * SRC_W + STYLE_W + k]);
        var w = P.guide_w[k];
        if (MODULATED) {
            w *= bytes(tgt_mod[ti * GUIDE_W + k]) / 255.0;
        }
        e += dot(w, d * d);
    }
    return e;
}

/// Weighted SSD between the target patch at `t` and the source patch at `s`, with clamped
/// target addressing (the CUDA engine's texture reads). Stops after a patch row once the
/// partial sum exceeds `ebest`. Source patch centres always keep the whole patch inside the
/// source, so the source side needs no clamping.
fn patch_error(t: vec2<i32>, s: vec2<i32>, ebest: f32) -> f32 {
    var e = 0.0;
    var si = u32((s.y - R) * P.sw + (s.x - R));
    if (USE_TILE) {
        // The tile was loaded with clamped coordinates, so border patches need no special case.
        var ti = tile_base;
        for (var py = 0; py < PATCH; py++) {
            for (var px = 0u; px < u32(PATCH); px++) {
                e += pixel_error_tile(ti + px, si + px);
            }
            if (e > ebest) {
                return e;
            }
            ti += tile_stride;
            si += u32(P.sw);
        }
        return e;
    }
    for (var py = -R; py <= R; py++) {
        let ty = clamp(t.y + py, 0, P.th - 1);
        for (var px = -R; px <= R; px++) {
            let tx = clamp(t.x + px, 0, P.tw - 1);
            e += pixel_error(u32(ty * P.tw + tx), si + u32(px + R));
        }
        if (e > ebest) {
            return e;
        }
        si += u32(P.sw);
    }
    return e;
}

// ---------------------------------------------------------------------------------------------
// Occupancy map (Omega) for the uniformity term

/// Sum of the occupancy counts under the patch centred at `c`. Accumulated in f32, which is
/// exact below 2^24 and, unlike an i32 sum, cannot overflow when a tiny source meets a huge
/// target (a cell can count up to tw * th patches).
fn omega_sum(c: vec2<i32>) -> f32 {
    var sum = 0.0;
    var row = (c.y - R) * P.sw + (c.x - R);
    for (var oy = 0; oy < PATCH; oy++) {
        for (var ox = 0; ox < PATCH; ox++) {
            sum += f32(atomicLoad(&omega[row + ox]));
        }
        row += P.sw;
    }
    return sum;
}

fn omega_add(c: vec2<i32>, delta: i32) {
    var row = (c.y - R) * P.sw + (c.x - R);
    for (var oy = 0; oy < PATCH; oy++) {
        for (var ox = 0; ox < PATCH; ox++) {
            atomicAdd(&omega[row + ox], delta);
        }
        row += P.sw;
    }
}

// ---------------------------------------------------------------------------------------------
// PatchMatch

/// The current best match of the pixel being processed.
struct Best {
    n: vec2<i32>,
    e: f32,
    /// Occupancy sum over the patch at `n`; read on first use, then updated incrementally.
    osum: f32,
    osum_valid: bool,
}

/// `tryPatch`: replaces the match by `cand` if that lowers error + lambda * occupancy.
fn try_patch(t: vec2<i32>, cand: vec2<i32>, b: ptr<function, Best>) {
    // Re-evaluating the current match can never win the strict comparison below.
    if (all(cand == (*b).n)) {
        return;
    }
    if (P.use_omega == 0u) {
        let ne = patch_error(t, cand, (*b).e);
        if (ne < (*b).e) {
            (*b).n = cand;
            (*b).e = ne;
        }
        return;
    }
    let pp = f32(PATCH * PATCH);
    if (!(*b).osum_valid) {
        (*b).osum = omega_sum((*b).n);
        (*b).osum_valid = true;
    }
    let cur_occ = ((*b).osum / pp) / P.omega_best;
    let threshold = (*b).e + P.lambda * cur_occ;
    // Reading the candidate's occupancy first tightens the error bound the patch distance may
    // stop at: the candidate can only win if its error is below threshold - lambda * new_occ.
    let nsum = omega_sum(cand);
    let new_occ = (nsum / pp) / P.omega_best;
    let bound = threshold - P.lambda * new_occ;
    if (bound < 0.0) {
        return;
    }
    let ne = patch_error(t, cand, bound);
    if (ne + P.lambda * new_occ < threshold) {
        omega_add(cand, 1);
        omega_add((*b).n, -1);
        // The candidate's patch gained one count per pixel, minus those shared with the old
        // patch, which lost one.
        let ov = max(PATCH - abs(cand.x - (*b).n.x), 0) * max(PATCH - abs(cand.y - (*b).n.y), 0);
        (*b).osum = nsum + f32(PATCH * PATCH - ov);
        (*b).n = cand;
        (*b).e = ne;
    }
}

/// `tryNeighborsOffset`: candidate from the neighbour at offset `o`, shifted back by `o`.
fn try_neighbor(t: vec2<i32>, o: vec2<i32>, b: ptr<function, Best>) {
    let q = clamp(t + o, vec2<i32>(0, 0), vec2<i32>(P.tw - 1, P.th - 1));
    let n = nnf_cur[q.y * P.tw + q.x] - o;
    if (n.x >= R && n.x < P.sw - R && n.y >= R && n.y < P.sh - R) {
        try_patch(t, n, b);
    }
}

/// `krnlEvalErrorPass`: patch error of every current match.
@compute @workgroup_size(WG, WG)
fn eval_error(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    load_tile(wid, lid);
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    let i = t.y * P.tw + t.x;
    err[i] = patch_error(t, nnf_cur[i], FLT_MAX);
}

/// `krnlPropagationPass`: tries the matches of the four neighbours at distance PROP_R.
/// Reads `nnf_cur`, writes `nnf_new` (the host swaps them afterwards).
@compute @workgroup_size(WG, WG)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    load_tile(wid, lid);
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    let i = t.y * P.tw + t.x;
    var b = Best(nnf_cur[i], err[i], 0.0, false);
    if (mask[i] == 0u) {
        try_neighbor(t, vec2<i32>(-PROP_R, 0), &b);
        try_neighbor(t, vec2<i32>(PROP_R, 0), &b);
        try_neighbor(t, vec2<i32>(0, -PROP_R), &b);
        try_neighbor(t, vec2<i32>(0, PROP_R), &b);
    }
    err[i] = b.e;
    nnf_new[i] = b.n;
}

/// `tryRandomOffsetInRadius`: a uniformly random candidate within `r` of the current match.
fn try_random(t: vec2<i32>, r: i32, b: ptr<function, Best>, state: ptr<function, u32>) {
    let lo = max((*b).n - vec2<i32>(r, r), vec2<i32>(R, R));
    let hi = min((*b).n + vec2<i32>(r, r), vec2<i32>(P.sw - 1 - R, P.sh - 1 - R));
    let nx = lo.x + i32(pcg_next(state) % u32(hi.x - lo.x + 1));
    let ny = lo.y + i32(pcg_next(state) % u32(hi.y - lo.y + 1));
    try_patch(t, vec2<i32>(nx, ny), b);
}

/// Random search of one pixel at radii 1, 2, 4, ... < max(sw, sh) / 2, each centred on the best
/// match found so far.
fn random_search_pixel(t: vec2<i32>, b: ptr<function, Best>, state: ptr<function, u32>) {
    let max_r = max(P.sw, P.sh) / 2;
    for (var r = 1; r < max_r; r *= 2) {
        try_random(t, r, b, state);
    }
}

/// `krnlRandomSearchPass` for all radii in one dispatch (the CUDA engine issued one dispatch per
/// radius; each radius is still centred on the best match found so far). Updates `nnf_cur` in
/// place. Used by the compat search.
@compute @workgroup_size(WG, WG)
fn random_search(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    load_tile(wid, lid);
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    let i = t.y * P.tw + t.x;
    if (mask[i] != 0u) { return; }
    var b = Best(nnf_cur[i], err[i], 0.0, false);
    var state = rng[i];
    random_search_pixel(t, &b, &state);
    rng[i] = state;
    err[i] = b.e;
    nnf_cur[i] = b.n;
}

/// One checkerboard (red-black) half-pass of the improved search. The pixels of colour CB_COLOR
/// try the matches of their four direct neighbours, which all have the other colour and so stay
/// fixed during this pass (and were just updated by the previous one), then do their random
/// search. Updates `nnf_cur` in place.
///
/// Each workgroup covers 2 WG x WG pixels and each invocation takes the pixel of its row pair
/// that has the pass colour, so the grid is (ceil(tw / 2), th) invocations.
@compute @workgroup_size(WG, WG)
fn checkerboard(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let x0 = 2 * i32(wid.x * WG);
    let y0 = i32(wid.y * WG);
    if (USE_TILE) {
        load_tile_rect(vec2<i32>(x0 - R, y0 - R), vec2<i32>(2 * i32(WG) + 2 * R, TILE), lid);
    }
    let y = y0 + i32(lid.y);
    let x = x0 + 2 * i32(lid.x) + ((y + i32(CB_COLOR)) & 1);
    tile_base = lid.y * tile_stride + u32(x - x0);
    if (x >= P.tw || y >= P.th) { return; }
    let t = vec2<i32>(x, y);
    let i = y * P.tw + x;
    if (mask[i] != 0u) { return; }
    var b = Best(nnf_cur[i], err[i], 0.0, false);
    try_neighbor(t, vec2<i32>(-1, 0), &b);
    try_neighbor(t, vec2<i32>(1, 0), &b);
    try_neighbor(t, vec2<i32>(0, -1), &b);
    try_neighbor(t, vec2<i32>(0, 1), &b);
    var state = rng[i];
    random_search_pixel(t, &b, &state);
    rng[i] = state;
    err[i] = b.e;
    nnf_cur[i] = b.n;
}

// ---------------------------------------------------------------------------------------------
// Voting

/// `krnlVotePlain`: average of the source pixels the overlapping patches map onto each target
/// pixel. Patches that fall outside the target or source are skipped (as in the CPU engine).
/// Integer arithmetic gives exactly the CPU engine's truncated or rounded float average.
@compute @workgroup_size(WG, WG)
fn vote_plain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    var acc: array<vec4<u32>, STYLE_W>;
    var count = 0u;
    for (var py = -R; py <= R; py++) {
        let y = t.y + py;
        if (y < 0 || y >= P.th) { continue; }
        for (var px = -R; px <= R; px++) {
            let x = t.x + px;
            if (x < 0 || x >= P.tw) { continue; }
            let n = nnf_cur[y * P.tw + x] - vec2<i32>(px, py);
            if (n.x < 0 || n.x >= P.sw || n.y < 0 || n.y >= P.sh) { continue; }
            let si = u32(n.y * P.sw + n.x) * SRC_W;
            for (var k = 0u; k < STYLE_W; k++) {
                acc[k] += unpack4xU8(src[si + k]);
            }
            count += 1u;
        }
    }
    let o = u32(t.y * P.tw + t.x) * STYLE_W;
    let c = max(count, 1u);
    let half = select(0u, c / 2u, P.round_votes != 0u);
    for (var k = 0u; k < STYLE_W; k++) {
        style_new[o + k] = pack4xU8((acc[k] + vec4<u32>(half)) / c);
    }
}

/// Weighted sum of the style of all PATCH x PATCH patches overlapping target pixel `t`, with
/// clamped addressing; `weighted == false` gives every patch weight 1. Returns the weight sum.
fn vote_weighted_sum(t: vec2<i32>, weighted: bool, acc: ptr<function, array<vec4<f32>, STYLE_W>>) -> f32 {
    var wsum = 0.0;
    for (var k = 0u; k < STYLE_W; k++) {
        (*acc)[k] = vec4<f32>(0.0);
    }
    for (var py = -R; py <= R; py++) {
        let y = clamp(t.y + py, 0, P.th - 1);
        for (var px = -R; px <= R; px++) {
            let x = clamp(t.x + px, 0, P.tw - 1);
            let j = y * P.tw + x;
            let n = clamp(nnf_cur[j] - vec2<i32>(px, py), vec2<i32>(0, 0), vec2<i32>(P.sw - 1, P.sh - 1));
            var w = 1.0;
            if (weighted) {
                w = 1.0 / (1.0 + err[j] / P.vote_norm);
            }
            let si = u32(n.y * P.sw + n.x) * SRC_W;
            for (var k = 0u; k < STYLE_W; k++) {
                (*acc)[k] += w * bytes(src[si + k]);
            }
            wsum += w;
        }
    }
    return wsum;
}

/// `krnlVoteWeighted`: each overlapping patch weighted by 1 / (1 + normalised error). As in the
/// CUDA engine (clamped textures), all PATCH x PATCH patches always contribute: neighbours and
/// source positions outside the images are read at the nearest valid pixel. If all weights are
/// zero (infinite errors) the pixel falls back to equal weights, like the CPU engine.
@compute @workgroup_size(WG, WG)
fn vote_weighted(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    var acc: array<vec4<f32>, STYLE_W>;
    var wsum = vote_weighted_sum(t, true, &acc);
    if (!(wsum > 0.0 && wsum <= FLT_MAX)) {
        wsum = vote_weighted_sum(t, false, &acc);
    }
    let o = u32(t.y * P.tw + t.x) * STYLE_W;
    let inv = 1.0 / wsum;
    // When truncating, the small bias keeps exact averages from dropping one step because of the
    // GPU's approximate division.
    let bias = select(2e-4, 0.5, P.round_votes != 0u);
    for (var k = 0u; k < STYLE_W; k++) {
        style_new[o + k] = pack4xU8Clamp(vec4<u32>(acc[k] * inv + bias));
    }
}

// ---------------------------------------------------------------------------------------------
// Stop-threshold mask

/// `krnlEvalMask`: converged (1) where no style channel changed by `stop_threshold` or more
/// between the previous (`style_new`) and the current (`style_cur`) vote.
@compute @workgroup_size(WG, WG)
fn eval_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    let i = u32(t.y * P.tw + t.x);
    var max_diff = 0u;
    for (var k = 0u; k < STYLE_W; k++) {
        let a = unpack4xU8(style_cur[i * STYLE_W + k]);
        let b = unpack4xU8(style_new[i * STYLE_W + k]);
        let d = max(a, b) - min(a, b);
        max_diff = max(max_diff, max(max(d.x, d.y), max(d.z, d.w)));
    }
    mask_tmp[i] = select(0u, 1u, max_diff < P.stop_threshold);
}

/// `krnlDilateMask`: a pixel stays active if any pixel of its (clamped) patch window is active.
@compute @workgroup_size(WG, WG)
fn dilate_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    var converged = 1u;
    for (var py = -R; py <= R; py++) {
        let y = clamp(t.y + py, 0, P.th - 1);
        for (var px = -R; px <= R; px++) {
            let x = clamp(t.x + px, 0, P.tw - 1);
            converged &= mask_tmp[y * P.tw + x];
        }
    }
    mask[t.y * P.tw + t.x] = converged;
}

@compute @workgroup_size(WG, WG)
fn clear_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    mask[t.y * P.tw + t.x] = 0u;
}

// ---------------------------------------------------------------------------------------------
// Level setup

/// Zeroes the occupancy map (dispatched over the source level).
@compute @workgroup_size(WG, WG)
fn clear_omega(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = vec2<i32>(gid.xy);
    if (s.x >= P.sw || s.y >= P.sh) { return; }
    atomicStore(&omega[s.y * P.sw + s.x], 0);
}

/// Counts, for every source pixel, the matched patches covering it.
@compute @workgroup_size(WG, WG)
fn init_omega(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    omega_add(nnf_cur[t.y * P.tw + t.x], 1);
}

/// `nnfUpscale`: every fine pixel inherits its coarse parent's match, doubled and offset by its
/// position in the 2x2 block, clamped to [patch_size, size - patch_size - 1].
/// Reads the coarse field from `nnf_cur`, writes `nnf_new`.
@compute @workgroup_size(WG, WG)
fn upscale_nnf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    let c = clamp(t / 2, vec2<i32>(0, 0), vec2<i32>(P.prev_tw - 1, P.prev_th - 1));
    let v = nnf_cur[c.y * P.prev_tw + c.x] * 2 + (t % 2);
    let ps = P.patch_size;
    nnf_new[t.y * P.tw + t.x] = vec2<i32>(min(max(v.x, ps), P.sw - ps - 1), min(max(v.y, ps), P.sh - ps - 1));
}

/// Seeds the per-pixel random generators (dispatched over the finest target level).
@compute @workgroup_size(WG, WG)
fn init_rng(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = target_pixel(gid);
    if (t.x < 0) { return; }
    let i = u32(t.y * P.tw + t.x);
    rng[i] = pcg_hash(i ^ pcg_hash(P.seed ^ 0x9e3779b9u));
}
