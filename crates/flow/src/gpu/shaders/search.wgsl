// Inverse search: aligns every 8x8 patch of frame 0 with frame 1 by inverse-compositional
// Gauss-Newton iterations (Kroeger et al. 2016, section 2.1).
//
// Eight threads work on one patch, one per patch row, and a workgroup holds eight patches.
// Sums over the patch are reduced through workgroup memory; all eight threads of a patch
// compute the same displacement.
//
// PROPAGATE false: the first pass. Every patch starts from the coarser level's flow at its
// centre (zero on the coarsest level).
// PROPAGATE true: a propagation pass. Every patch compares the matching cost of its own
// displacement from the previous pass with the costs of its four grid neighbours'
// displacements. If the best neighbour costs less than L.margin times its own, the patch
// continues the search from that displacement for L.iters1 iterations; otherwise it keeps its
// result unchanged.
//
// In both cases the result falls back to the starting displacement when the search ends with
// a higher cost than it started with, or moves further than one patch size away. A patch stops
// iterating once a step is shorter than sqrt(L.tol2); the workgroup leaves the loop once all of
// its patches have stopped.

override PROPAGATE: bool = false;

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> i0: array<f32>;
@group(0) @binding(2) var<storage, read> i1: array<f32>;
@group(0) @binding(3) var<storage, read> flow_src: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> patch_in: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> patch_out: array<vec2<f32>>;

var<workgroup> red: array<vec4<f32>, 64>;
// Propagation passes: which patches of the workgroup continue the search, and whether any does.
var<workgroup> moving: array<u32, 8>;
var<workgroup> any_moving: u32;

// Sums `v` over the eight threads of this thread's patch. Must be called in uniform control
// flow by the whole workgroup.
fn patch_sum(li: u32, v: vec4<f32>) -> vec4<f32> {
    red[li] = v;
    workgroupBarrier();
    let b = li & ~7u;
    let s = ((red[b] + red[b + 1u]) + (red[b + 2u] + red[b + 3u])) + ((red[b + 4u] + red[b + 5u]) + (red[b + 6u] + red[b + 7u]));
    workgroupBarrier();
    return s;
}

// Per-thread state: the template row and its gradients.
var<private> t: array<f32, 8>;
var<private> gx: array<f32, 8>;
var<private> gy: array<f32, 8>;
var<private> px: u32;
var<private> py: u32;

// Row sums for displacement u: (sum gx r, sum gy r, sum r, sum r^2) with r the difference
// between frame 1 at the displaced position (bilinear, clamped) and the template.
fn row_residuals(u: vec2<f32>) -> vec4<f32> {
    let sx = f32(px) + u.x;
    let sy = f32(py) + u.y;
    let fx = floor(sx);
    let fy = floor(sy);
    let ax = sx - fx;
    let ay = sy - fy;
    let ix = i32(fx);
    let iy = i32(fy);
    let r0 = clamp_x(iy, L.h) * L.w;
    let r1 = clamp_x(iy + 1, L.h) * L.w;
    var c = clamp_x(ix, L.w);
    var prev = mix(i1[r0 + c], i1[r1 + c], ay);
    var acc = vec4<f32>(0.0);
    for (var k = 0u; k < PS; k++) {
        c = clamp_x(ix + i32(k) + 1, L.w);
        let next = mix(i1[r0 + c], i1[r1 + c], ay);
        let r = mix(prev, next, ax) - t[k];
        prev = next;
        acc += vec4<f32>(gx[k] * r, gy[k] * r, r, r * r);
    }
    return acc;
}

// Matching cost of a patch from its reduced residual sums.
fn cost(s: vec4<f32>) -> f32 {
    if (L.mean_norm != 0u) {
        return s.w - s.z * s.z / PATCH_PIXELS;
    }
    return s.w;
}

fn i0_at(x: i32, y: i32) -> f32 {
    return i0[clamp_x(y, L.h) * L.w + clamp_x(x, L.w)];
}

@compute @workgroup_size(64)
fn search(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let count = L.pw * L.ph;
    let p = min(((wg.y * nwg.x + wg.x) << 3u) + (li >> 3u), count - 1u);
    let pi = p % L.pw;
    let pj = p / L.pw;
    px = min(pi * L.stride, L.w - PS);
    py = min(pj * L.stride, L.h - PS) + (li & 7u);

    // Template row and Sobel gradients (scaled to intensity per pixel).
    let x = i32(px);
    let y = i32(py);
    var s1 = vec4<f32>(0.0);
    var s2 = vec4<f32>(0.0);
    for (var k = 0; k < 8; k++) {
        let xk = x + k;
        let a = i0_at(xk - 1, y - 1);
        let b = i0_at(xk, y - 1);
        let c = i0_at(xk + 1, y - 1);
        let d = i0_at(xk - 1, y);
        let f = i0_at(xk + 1, y);
        let g = i0_at(xk - 1, y + 1);
        let h = i0_at(xk, y + 1);
        let i = i0_at(xk + 1, y + 1);
        t[k] = i0_at(xk, y);
        gx[k] = 0.125 * ((c + 2.0 * f + i) - (a + 2.0 * d + g));
        gy[k] = 0.125 * ((g + 2.0 * h + i) - (a + 2.0 * b + c));
        s1 += vec4<f32>(gx[k] * gx[k], gx[k] * gy[k], gy[k] * gy[k], 0.0);
        s2 += vec4<f32>(gx[k], gy[k], 0.0, 0.0);
    }
    let hs = patch_sum(li, s1);
    let gs = patch_sum(li, s2);
    var hxx = hs.x;
    var hxy = hs.y;
    var hyy = hs.z;
    if (L.mean_norm != 0u) {
        hxx -= gs.x * gs.x / PATCH_PIXELS;
        hxy -= gs.x * gs.y / PATCH_PIXELS;
        hyy -= gs.y * gs.y / PATCH_PIXELS;
    }
    hxx += 1e-2;
    hyy += 1e-2;
    let det = hxx * hyy - hxy * hxy;
    let inv = vec3<f32>(hyy, -hxy, hxx) / det;

    // Starting displacement.
    var u0: vec2<f32>;
    var iters: u32;
    var keep = false;
    if (PROPAGATE) {
        iters = L.iters1;
        u0 = patch_in[p];
        var best = cost(patch_sum(li, row_residuals(u0)));
        // Grid neighbours; a missing neighbour repeats the patch's own value so that every
        // thread runs the same number of reductions.
        let left = select(p, p - 1u, pi > 0u);
        let right = select(p, p + 1u, pi + 1u < L.pw);
        let up = select(p, p - L.pw, pj > 0u);
        let down = select(p, p + L.pw, pj + 1u < L.ph);
        let candidates = array<u32, 4>(left, right, up, down);
        let own = u0;
        let own_cost = best;
        for (var n = 0u; n < 4u; n++) {
            let c = patch_in[candidates[n]];
            let e = cost(patch_sum(li, row_residuals(c)));
            if (e < best && e < L.margin * own_cost && any(c != own)) {
                best = e;
                u0 = c;
            }
        }
        // Without a better neighbour the patch keeps its result from the previous pass.
        keep = all(u0 == own);
    } else {
        iters = L.iters0;
        if (L.cw == 0u) {
            u0 = vec2<f32>(0.0);
        } else {
            let base = vec2<f32>(f32(px), f32(py - (li & 7u)));
            u0 = resample_src(base.x + 3.5, base.y + 3.5);
        }
    }

    // In a propagation pass, the iterations are skipped when no patch of the workgroup took a
    // neighbour's displacement (the common case once the field has settled).
    if (PROPAGATE) {
        if ((li & 7u) == 0u) {
            moving[li >> 3u] = select(1u, 0u, keep);
        }
        workgroupBarrier();
        if (li == 0u) {
            any_moving = moving[0] | moving[1] | moving[2] | moving[3] | moving[4] | moving[5] | moving[6] | moving[7];
        }
        if (workgroupUniformLoad(&any_moving) == 0u) {
            iters = 0u;
        }
    }

    // Gauss-Newton iterations. The cost at the start comes with the first evaluation.
    var u = u0;
    var start_cost = 0.0;
    var lost = keep;
    var converged = false;
    for (var it = 0u; it < iters; it++) {
        let s = patch_sum(li, row_residuals(u));
        if (it == 0u) {
            start_cost = cost(s);
        }
        var bx = s.x;
        var by = s.y;
        if (L.mean_norm != 0u) {
            bx -= gs.x * s.z / PATCH_PIXELS;
            by -= gs.y * s.z / PATCH_PIXELS;
        }
        // A search that runs further than a patch size is abandoned (reverted below). The
        // reductions keep running until every patch of the workgroup has stopped, so that the
        // workgroup stays in step.
        if (!lost && !converged) {
            let step = vec2<f32>(inv.x * bx + inv.y * by, inv.y * bx + inv.z * by);
            u -= step;
            let d = u - u0;
            lost = !(dot(d, d) <= PSF * PSF);
            converged = !lost && dot(step, step) < L.tol2;
        }
        if ((li & 7u) == 0u) {
            moving[li >> 3u] = select(1u, 0u, lost || converged);
        }
        workgroupBarrier();
        if (li == 0u) {
            any_moving = moving[0] | moving[1] | moving[2] | moving[3] | moving[4] | moving[5] | moving[6] | moving[7];
        }
        if (workgroupUniformLoad(&any_moving) == 0u) {
            break;
        }
    }
    if (iters > 0u) {
        let end_cost = cost(patch_sum(li, row_residuals(select(u, u0, lost))));
        if (lost || !(end_cost <= start_cost)) {
            u = u0;
        }
    }
    if ((li & 7u) == 0u && ((wg.y * nwg.x + wg.x) << 3u) + (li >> 3u) < count) {
        patch_out[p] = u;
    }
}
