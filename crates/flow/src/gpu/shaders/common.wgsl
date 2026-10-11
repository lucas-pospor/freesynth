// Shared declarations of the DIS kernels. This file is prepended to every kernel source.

// Per-level constants. Not every kernel uses every field.
struct Level {
    // Size of this level.
    w: u32,
    h: u32,
    // Patch grid of this level.
    pw: u32,
    ph: u32,
    // Patch stride; Gauss-Newton iterations of the first search pass and after a patch takes
    // a neighbour's displacement; 1 when patch means are subtracted.
    stride: u32,
    iters0: u32,
    iters1: u32,
    mean_norm: u32,
    // Size of the other level a kernel reads: the coarser level for the search, the finer
    // level for downsampling, the finest searched level for the output.
    cw: u32,
    ch: u32,
    // Relaxation sweeps per fixed-point iteration of the variational refinement; squared
    // step length below which a patch's search stops.
    inner: u32,
    tol2: f32,
    // Variational refinement: smoothness, brightness and gradient weights, over-relaxation.
    alpha: f32,
    delta: f32,
    gamma: f32,
    omega: f32,
    // Flow scale from the other level to this one; squared regularisers of the data term
    // normalisation and of the robust penalty; propagation acceptance margin.
    scale: f32,
    zeta2: f32,
    eps2: f32,
    margin: f32,
};

const PS: u32 = 8u;
const PSF: f32 = 8.0;
const PATCH_PIXELS: f32 = 64.0;

fn clamp_x(x: i32, w: u32) -> u32 {
    return u32(clamp(x, 0, i32(w) - 1));
}
