// Densification (Kroeger et al. 2016, section 2.2): the flow at a pixel is the average of the
// displacements of all patches that cover it, each weighted by 1 / max(1, |d|), where d is the
// difference between frame 1 at the displaced position and frame 0 at the pixel.
//
// With FROM_COARSE set, there are no patches on this level: the flow is the coarser level's
// flow (`flow_src`), upsampled, which the refinement then improves.

override FROM_COARSE: bool = false;

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> i0: array<f32>;
@group(0) @binding(2) var<storage, read> i1: array<f32>;
// Patch displacements, or with FROM_COARSE the coarser level's flow.
@group(0) @binding(3) var<storage, read> flow_src: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> flow_out: array<vec2<f32>>;

fn at1(x: i32, y: i32) -> f32 {
    return i1[clamp_x(y, L.h) * L.w + clamp_x(x, L.w)];
}

// Bilinear sample of frame 1, clamped.
fn sample1(x: f32, y: f32) -> f32 {
    let fx = floor(x);
    let fy = floor(y);
    let ix = i32(fx);
    let iy = i32(fy);
    let ax = x - fx;
    let ay = y - fy;
    let top = mix(at1(ix, iy), at1(ix + 1, iy), ax);
    let bottom = mix(at1(ix, iy + 1), at1(ix + 1, iy + 1), ax);
    return mix(top, bottom, ay);
}

@compute @workgroup_size(16, 16)
fn densify(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= L.w || g.y >= L.h) {
        return;
    }
    let x = g.x;
    let y = g.y;
    let fx = f32(x);
    let fy = f32(y);
    if (FROM_COARSE) {
        flow_out[y * L.w + x] = resample_src(fx, fy);
        return;
    }
    let t = i0[y * L.w + x];

    // Patches with origin min(i * stride, w - 8) cover x when origin <= x < origin + 8. All
    // patches but the last have origin i * stride; the last one ends at the border.
    let s = L.stride;
    let ihi = min(x / s, L.pw - 1u);
    let ilo = select(0u, (x - (PS - 1u) + s - 1u) / s, x >= PS - 1u);
    let jhi = min(y / s, L.ph - 1u);
    let jlo = select(0u, (y - (PS - 1u) + s - 1u) / s, y >= PS - 1u);
    let extra_i = L.pw - 1u > ihi && x >= L.w - PS;
    let extra_j = L.ph - 1u > jhi && y >= L.h - PS;
    let ni = ihi + 1u - ilo + select(0u, 1u, extra_i);
    let nj = jhi + 1u - jlo + select(0u, 1u, extra_j);
    var acc = vec2<f32>(0.0);
    var wsum = 0.0;
    for (var kj = 0u; kj < nj; kj++) {
        let j = select(jlo + kj, L.ph - 1u, extra_j && kj == nj - 1u);
        let oy = min(j * s, L.h - PS);
        if (y < oy || y >= oy + PS) {
            continue;
        }
        for (var ki = 0u; ki < ni; ki++) {
            let i = select(ilo + ki, L.pw - 1u, extra_i && ki == ni - 1u);
            let ox = min(i * s, L.w - PS);
            if (x < ox || x >= ox + PS) {
                continue;
            }
            let u = flow_src[j * L.pw + i];
            let d = sample1(fx + u.x, fy + u.y) - t;
            let w = 1.0 / max(1.0, abs(d));
            acc += w * u;
            wsum += w;
        }
    }
    // Every pixel is covered by at least one patch.
    flow_out[y * L.w + x] = acc / max(wsum, 1e-20);
}
