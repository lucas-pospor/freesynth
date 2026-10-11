// Bilinear pyramid resampling (`krnlResampleBilinear` of the 2019 CUDA engine).
//
// Output pixel (x, y) samples the input at (s*x, s*y), s = in_w / out_w, with clamped
// addressing. The arithmetic follows freesynth-core's CPU resampler (a product followed by
// three fused multiply-adds). Pixels are `words` u32 words of four 8-bit channels each.
//
// The host computes `s`: GPU division is approximate (WGSL allows 2.5 ulp), and a scale of
// 0.99999994 instead of 1 blends every pixel with its neighbour and truncates, so about half the
// channels of a full-resolution level would come out one step low.

struct ResampleParams {
    in_w: i32,
    in_h: i32,
    out_w: i32,
    out_h: i32,
    words: u32,
    scale: f32,                 // in_w / out_w, correctly rounded
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> RP: ResampleParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(16, 16)
fn resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= RP.out_w || y >= RP.out_h) { return; }
    let s = RP.scale;
    let fx = s * f32(x);
    let fy = s * f32(y);
    let ix = i32(fx);
    let iy = i32(fy);
    let u = fx - f32(ix);
    let t = fy - f32(iy);
    let x0 = clamp(ix, 0, RP.in_w - 1);
    let x1 = clamp(ix + 1, 0, RP.in_w - 1);
    let y0 = clamp(iy, 0, RP.in_h - 1);
    let y1 = clamp(iy + 1, 0, RP.in_h - 1);
    let w00 = vec4<f32>((1.0 - u) * (1.0 - t));
    let w10 = vec4<f32>(u * (1.0 - t));
    let w01 = vec4<f32>((1.0 - u) * t);
    let w11 = vec4<f32>(u * t);
    let n = RP.words;
    let p00 = u32(y0 * RP.in_w + x0) * n;
    let p10 = u32(y0 * RP.in_w + x1) * n;
    let p01 = u32(y1 * RP.in_w + x0) * n;
    let p11 = u32(y1 * RP.in_w + x1) * n;
    let o = u32(y * RP.out_w + x) * n;
    for (var k = 0u; k < n; k++) {
        var v = w00 * vec4<f32>(unpack4xU8(src[p00 + k]));
        v = fma(w10, vec4<f32>(unpack4xU8(src[p10 + k])), v);
        v = fma(w01, vec4<f32>(unpack4xU8(src[p01 + k])), v);
        v = fma(w11, vec4<f32>(unpack4xU8(src[p11 + k])), v);
        dst[o + k] = pack4xU8Clamp(vec4<u32>(v));
    }
}
