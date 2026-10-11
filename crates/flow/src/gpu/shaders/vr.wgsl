// Variational refinement (Kroeger et al. 2016, section 2.3): OUTER fixed-point iterations of
//
//   E(U) = sum  delta Psi(E_I) + gamma Psi(E_G) + alpha Psi(|grad u|^2 + |grad v|^2),
//
// with Psi(s^2) = sqrt(s^2 + eps^2), U = u0 + du, and the brightness term E_I and gradient term
// E_G linearised around u0, the flow the level started from (densified or upsampled). Spatial
// derivatives average both frames (frame 1 warped by u0), temporal ones are differences, and
// each term is divided by the squared norm of its spatial gradient plus zeta^2, so that strong
// edges do not dominate. Each fixed-point iteration evaluates the robust weights Psi' at the
// current flow and keeps them fixed while L.inner red-black successive over-relaxation sweeps
// solve the resulting linear system.
//
// The linearised terms are computed from the two frames at the start of every dispatch
// instead of being stored: the frames stay in cache, while stored terms (a dozen floats per
// pixel) make the kernel bandwidth-bound on large frames.
//
// Each workgroup relaxes one 32x16 tile in workgroup memory; every thread owns two
// horizontally adjacent pixels, one of each colour, so that no thread idles during the
// red-black sweeps. Values outside the tile (a two-pixel border) are read once at the start
// and stay fixed during the dispatch, so tiles exchange information between dispatches only
// (block Gauss-Seidel). Consecutive dispatches shift the tile grid by (SHIFT_X, SHIFT_Y) so
// that tile borders move.

override SHIFT_X: u32 = 0u;
override SHIFT_Y: u32 = 0u;
// Fixed-point iterations per dispatch.
override OUTER: u32 = 1u;

const TILE_W: u32 = 32u;
const TILE_H: u32 = 16u;
const HALO: u32 = 2u;
// Flow cache: tile plus a two-pixel border.
const SPAN_W: u32 = 36u;
const SPAN_H: u32 = 20u;
// Smoothness weights: tile plus a one-pixel border.
const WSPAN_W: u32 = 34u;
const WSPAN_H: u32 = 18u;

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> i0: array<f32>;
@group(0) @binding(2) var<storage, read> i1: array<f32>;
// The flow the data terms are linearised around.
@group(0) @binding(3) var<storage, read> base: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> flow_in: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> flow_out: array<vec2<f32>>;

fn at0(x: i32, y: i32) -> f32 {
    return i0[clamp_x(y, L.h) * L.w + clamp_x(x, L.w)];
}

fn at1(x: i32, y: i32) -> f32 {
    return i1[clamp_x(y, L.h) * L.w + clamp_x(x, L.w)];
}

// Value and derivatives at the centre of a 3x3 neighbourhood (row-major), by central
// differences: (I, Ix, Iy, Ixy) and (Ixx, Iyy).
struct Derivs {
    a: vec4<f32>,
    b: vec2<f32>,
}

fn derivs(v: array<f32, 9>) -> Derivs {
    let ix = 0.5 * (v[5] - v[3]);
    let iy = 0.5 * (v[7] - v[1]);
    let ixy = 0.25 * ((v[8] - v[6]) - (v[2] - v[0]));
    let ixx = v[5] - 2.0 * v[4] + v[3];
    let iyy = v[7] - 2.0 * v[4] + v[1];
    return Derivs(vec4<f32>(v[4], ix, iy, ixy), vec2<f32>(ixx, iyy));
}

// The three normalised data terms at pixel (x, y), linearised around displacement u: each is a
// row (a, b, c) of the residual a du + b dv + c. Zero when x + u leaves the frame.
struct Terms {
    bright: vec3<f32>,
    grad_x: vec3<f32>,
    grad_y: vec3<f32>,
}

fn data_terms(x: i32, y: i32, u: vec2<f32>) -> Terms {
    let wx = f32(x) + u.x;
    let wy = f32(y) + u.y;
    if (!(wx >= 0.0 && wy >= 0.0 && wx <= f32(L.w - 1u) && wy <= f32(L.h - 1u))) {
        return Terms(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    }
    var n0: array<f32, 9>;
    for (var k = 0; k < 9; k++) {
        n0[k] = at0(x + k % 3 - 1, y + k / 3 - 1);
    }
    let d0 = derivs(n0);
    // Frame 1 around the warped position: the 4x4 block whose inner 2x2 pixels surround it;
    // derivatives at those four pixels, blended bilinearly.
    let bx = floor(wx);
    let by = floor(wy);
    let ax = wx - bx;
    let ay = wy - by;
    var blk: array<f32, 16>;
    for (var k = 0; k < 16; k++) {
        blk[k] = at1(i32(bx) + k % 4 - 1, i32(by) + k / 4 - 1);
    }
    var d1: array<Derivs, 4>;
    for (var q = 0; q < 4; q++) {
        let o = (q / 2) * 4 + q % 2;
        d1[q] = derivs(array<f32, 9>(
            blk[o], blk[o + 1], blk[o + 2],
            blk[o + 4], blk[o + 5], blk[o + 6],
            blk[o + 8], blk[o + 9], blk[o + 10],
        ));
    }
    let wa = mix(mix(d1[0].a, d1[1].a, ax), mix(d1[2].a, d1[3].a, ax), ay);
    let wb = mix(mix(d1[0].b, d1[1].b, ax), mix(d1[2].b, d1[3].b, ax), ay);

    let ix = 0.5 * (d0.a.y + wa.y);
    let iy = 0.5 * (d0.a.z + wa.z);
    let iz = wa.x - d0.a.x;
    let ixy = 0.5 * (d0.a.w + wa.w);
    let ixx = 0.5 * (d0.b.x + wb.x);
    let iyy = 0.5 * (d0.b.y + wb.y);
    let ixz = wa.y - d0.a.y;
    let iyz = wa.z - d0.a.z;
    return Terms(
        vec3<f32>(ix, iy, iz) * inverseSqrt(ix * ix + iy * iy + L.zeta2),
        vec3<f32>(ixx, ixy, ixz) * inverseSqrt(ixx * ixx + ixy * ixy + L.zeta2),
        vec3<f32>(ixy, iyy, iyz) * inverseSqrt(ixy * ixy + iyy * iyy + L.zeta2),
    );
}

// Flow over the tile and its border.
var<workgroup> su: array<vec2<f32>, 720>;
// Smoothness weights Psi' over the tile and a one-pixel border.
var<workgroup> sw: array<f32, 612>;

// One pixel: its data terms (fixed for the level) and the linear system of the current
// fixed-point iteration, (A + W) du = b + sum_n w_n (U_n - u0), with W the sum of the edge
// weights w_n to the left, right, upper and lower neighbours.
struct Pixel {
    inside: bool,
    // Image position.
    x: i32,
    y: i32,
    // Indices into `su` and `sw`.
    c: u32,
    cw: u32,
    u0: vec2<f32>,
    terms: Terms,
    b: vec2<f32>,
    // A + W: (m11, a12, m22) and the inverse determinant.
    m: vec3<f32>,
    inv_det: f32,
    wn: vec4<f32>,
}

// The pixel at tile position (tx, ty), with its data terms.
fn prepare(ox: i32, oy: i32, tx: u32, ty: u32) -> Pixel {
    var p: Pixel;
    p.x = ox + i32(tx);
    p.y = oy + i32(ty);
    p.inside = p.x >= 0 && p.y >= 0 && p.x < i32(L.w) && p.y < i32(L.h);
    p.c = (ty + HALO) * SPAN_W + tx + HALO;
    p.cw = (ty + 1u) * WSPAN_W + tx + 1u;
    p.u0 = vec2<f32>(0.0);
    p.terms = Terms(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    p.b = vec2<f32>(0.0);
    p.m = vec3<f32>(0.0);
    p.inv_det = 0.0;
    p.wn = vec4<f32>(0.0);
    if (p.inside) {
        p.u0 = base[u32(p.y) * L.w + u32(p.x)];
        p.terms = data_terms(p.x, p.y, p.u0);
    }
    return p;
}

// Evaluates the robust weights at the current flow and sets up the pixel's linear system.
fn linearise(pixel: Pixel) -> Pixel {
    var p = pixel;
    if (!p.inside) {
        return p;
    }
    let t0 = p.terms.bright;
    let t1 = p.terms.grad_x;
    let t2 = p.terms.grad_y;
    let d3 = vec3<f32>(su[p.c] - p.u0, 1.0);
    let ei = dot(t0, d3);
    let egx = dot(t1, d3);
    let egy = dot(t2, d3);
    let pi = L.delta * 0.5 * inverseSqrt(ei * ei + L.eps2);
    let pg = L.gamma * 0.5 * inverseSqrt(egx * egx + egy * egy + L.eps2);
    let a11 = pi * t0.x * t0.x + pg * (t1.x * t1.x + t2.x * t2.x);
    let a12 = pi * t0.x * t0.y + pg * (t1.x * t1.y + t2.x * t2.y);
    let a22 = pi * t0.y * t0.y + pg * (t1.y * t1.y + t2.y * t2.y);
    p.b = -vec2<f32>(
        pi * t0.x * t0.z + pg * (t1.x * t1.z + t2.x * t2.z),
        pi * t0.y * t0.z + pg * (t1.y * t1.z + t2.y * t2.z),
    );
    let ws = sw[p.cw];
    p.wn = 0.5 * L.alpha * vec4<f32>(
        select(0.0, ws + sw[p.cw - 1u], p.x > 0),
        select(0.0, ws + sw[p.cw + 1u], p.x + 1 < i32(L.w)),
        select(0.0, ws + sw[p.cw - WSPAN_W], p.y > 0),
        select(0.0, ws + sw[p.cw + WSPAN_W], p.y + 1 < i32(L.h)),
    );
    let wsum = p.wn.x + p.wn.y + p.wn.z + p.wn.w;
    p.m = vec3<f32>(a11 + wsum, a12, a22 + wsum);
    let det = p.m.x * p.m.z - a12 * a12;
    p.inv_det = select(0.0, 1.0 / det, det > 1e-12);
    return p;
}

// One over-relaxed update of a pixel's flow in `su`.
fn relax(p: Pixel) {
    let c = p.c;
    let rhs = p.b + p.wn.x * (su[c - 1u] - p.u0) + p.wn.y * (su[c + 1u] - p.u0) + p.wn.z * (su[c - SPAN_W] - p.u0)
        + p.wn.w * (su[c + SPAN_W] - p.u0);
    let du_new = p.inv_det * vec2<f32>(p.m.z * rhs.x - p.m.y * rhs.y, p.m.x * rhs.y - p.m.y * rhs.x);
    let du_old = su[c] - p.u0;
    su[c] = p.u0 + du_old + L.omega * (du_new - du_old);
}

// Smoothness weights Psi' from central differences of the cached flow. The cache repeats the
// edge value outside the image, so on the outermost row and column the difference is half the
// one-sided one (the CPU version does the same).
fn smoothness_weights(li: u32) {
    for (var k = li; k < WSPAN_W * WSPAN_H; k += 256u) {
        let c = (k / WSPAN_W + 1u) * SPAN_W + k % WSPAN_W + 1u;
        let dx = 0.5 * (su[c + 1u] - su[c - 1u]);
        let dy = 0.5 * (su[c + SPAN_W] - su[c - SPAN_W]);
        sw[k] = 0.5 * inverseSqrt(dot(dx, dx) + dot(dy, dy) + L.eps2);
    }
}

@compute @workgroup_size(16, 16)
fn refine(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    // Image coordinates of the tile's first pixel.
    let ox = i32(wg.x * TILE_W) - i32(SHIFT_X);
    let oy = i32(wg.y * TILE_H) - i32(SHIFT_Y);
    for (var k = li; k < SPAN_W * SPAN_H; k += 256u) {
        let x = clamp(ox - i32(HALO) + i32(k % SPAN_W), 0, i32(L.w) - 1);
        let y = clamp(oy - i32(HALO) + i32(k / SPAN_W), 0, i32(L.h) - 1);
        su[k] = flow_in[u32(y) * L.w + u32(x)];
    }
    // This thread's two pixels; the first has colour (x + y) & 1 == colour0, the second the
    // other colour.
    var pixels = array<Pixel, 2>(prepare(ox, oy, 2u * lid.x, lid.y), prepare(ox, oy, 2u * lid.x + 1u, lid.y));
    let colour0 = u32(ox + i32(2u * lid.x) + oy + i32(lid.y)) & 1u;
    for (var outer = 0u; outer < OUTER; outer++) {
        workgroupBarrier();
        smoothness_weights(li);
        workgroupBarrier();
        pixels[0] = linearise(pixels[0]);
        pixels[1] = linearise(pixels[1]);
        for (var it = 0u; it < L.inner; it++) {
            for (var phase = 0u; phase < 2u; phase++) {
                let p = pixels[phase ^ colour0];
                if (p.inside) {
                    relax(p);
                }
                workgroupBarrier();
            }
        }
    }
    for (var q = 0u; q < 2u; q++) {
        let p = pixels[q];
        if (p.inside) {
            flow_out[u32(p.y) * L.w + u32(p.x)] = su[p.c];
        }
    }
}
