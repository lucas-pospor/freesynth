// Converts the two input frames to level-0 intensities.
//
// FORMAT 0: one f32 intensity per pixel. FORMAT 1: packed 8-bit RGBA (red in the low byte),
// converted to Rec. 601 luma in 0-255.

override FORMAT: u32 = 0u;

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> frame0: array<u32>;
@group(0) @binding(2) var<storage, read> frame1: array<u32>;
@group(0) @binding(3) var<storage, read_write> i0: array<f32>;
@group(0) @binding(4) var<storage, read_write> i1: array<f32>;

fn decode(word: u32) -> f32 {
    if (FORMAT == 0u) {
        return bitcast<f32>(word);
    }
    let c = unpack4x8unorm(word) * 255.0;
    return 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
}

@compute @workgroup_size(16, 16)
fn load(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= L.w || g.y >= L.h) {
        return;
    }
    let i = g.y * L.w + g.x;
    i0[i] = decode(frame0[i]);
    i1[i] = decode(frame1[i]);
}
