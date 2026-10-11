// Flow resampling helpers. Included by kernels that declare `flow_src`, a flow field of size
// L.cw x L.ch.

// Bilinear sample of `flow_src` at continuous position (x, y), clamped to the field.
fn sample_src(x: f32, y: f32) -> vec2<f32> {
    let cx = clamp(x, 0.0, f32(L.cw) - 1.0);
    let cy = clamp(y, 0.0, f32(L.ch) - 1.0);
    let x0 = u32(cx);
    let y0 = u32(cy);
    let x1 = min(x0 + 1u, L.cw - 1u);
    let y1 = min(y0 + 1u, L.ch - 1u);
    let ax = cx - f32(x0);
    let ay = cy - f32(y0);
    let top = mix(flow_src[y0 * L.cw + x0], flow_src[y0 * L.cw + x1], ax);
    let bottom = mix(flow_src[y1 * L.cw + x0], flow_src[y1 * L.cw + x1], ax);
    return mix(top, bottom, ay);
}

// `flow_src` resampled at pixel centre (x, y) of this level and converted to this level's
// pixel units. Pixel centres are at integer coordinates on every level, so a level that is
// `scale` times coarser maps x to (x + 0.5) / scale - 0.5.
fn resample_src(x: f32, y: f32) -> vec2<f32> {
    let inv = 1.0 / L.scale;
    return L.scale * sample_src((x + 0.5) * inv - 0.5, (y + 0.5) * inv - 0.5);
}
