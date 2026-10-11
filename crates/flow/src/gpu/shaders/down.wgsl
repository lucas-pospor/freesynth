// 2x2 box downsampling of both frames from the finer level (L.cw x L.ch) to this level.

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> src0: array<f32>;
@group(0) @binding(2) var<storage, read> src1: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst0: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst1: array<f32>;

@compute @workgroup_size(16, 16)
fn down(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= L.w || g.y >= L.h) {
        return;
    }
    // Level sizes are rounded down, so the 2x2 block is always inside the finer level.
    let a = (2u * g.y) * L.cw + 2u * g.x;
    let b = a + L.cw;
    let i = g.y * L.w + g.x;
    dst0[i] = 0.25 * (src0[a] + src0[a + 1u] + src0[b] + src0[b + 1u]);
    dst1[i] = 0.25 * (src1[a] + src1[a + 1u] + src1[b] + src1[b + 1u]);
}
