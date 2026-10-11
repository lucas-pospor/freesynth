// Upsamples the flow of the finest searched level (L.cw x L.ch) to full resolution.

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> flow_src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> flow_out: array<vec2<f32>>;

@compute @workgroup_size(16, 16)
fn upsample(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= L.w || g.y >= L.h) {
        return;
    }
    flow_out[g.y * L.w + g.x] = resample_src(f32(g.x), f32(g.y));
}
