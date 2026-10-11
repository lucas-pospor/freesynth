//! CPU backend: the same DIS as the GPU kernels, in Rust with rayon.
//!
//! [`CpuDis`] follows the GPU schedule stage by stage (pyramid, inverse search, propagation
//! passes, densification, tiled variational refinement, upsampling) with the same parameters,
//! the same arithmetic and the same order of the sums that matter, so that both backends agree
//! to within rounding on the same input. It serves as a fallback when no GPU is available and
//! as a reference for testing the GPU kernels.
//!
//! The hot loops are written over small fixed-size arrays so that the compiler vectorises them.
//! On x86-64 they are also compiled with AVX2 and with AVX-512 enabled, and the widest version
//! the CPU supports is picked at run time. All versions compute the same results: there is no
//! implicit floating-point contraction, and the explicit fused multiply-adds are exactly
//! rounded in every version (without hardware support they are slow, though).

mod lanes;
mod refine;
mod search;

use rayon::prelude::*;

use crate::error::{FlowError, invalid};
use crate::flo::FlowField;
use crate::image::GrayImage;
use crate::params::{DisParams, Plan};

/// Instruction sets the hot loops are compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Isa {
    /// Whatever the build target allows.
    Portable,
    /// AVX2 and FMA, detected at run time.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512 (foundation and vector-length extensions) on top of AVX2, detected at run time.
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Isa {
    /// The widest instruction set the CPU supports. `FREESYNTH_CPU_ISA` (`portable`, `avx2`)
    /// can lower it, for comparisons.
    fn detect() -> Self {
        let cap = std::env::var("FREESYNTH_CPU_ISA").ok();
        if cap.as_deref() == Some("portable") {
            return Isa::Portable;
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::is_x86_feature_detected as has;
            if has!("avx2") && has!("fma") {
                if has!("avx512f") && has!("avx512vl") && cap.as_deref() != Some("avx2") {
                    return Isa::Avx512;
                }
                return Isa::Avx2;
            }
        }
        Isa::Portable
    }
}

/// Compiles `$body` (a call of `#[inline(always)]` functions) for the build target and for each
/// [`Isa`] the build target can detect, and runs the version that `$isa` selects. `$args` name
/// the variables that the body uses.
macro_rules! dispatch {
    ($isa:expr, ($($arg:ident: $ty:ty),* $(,)?) $body:block) => {{
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        #[allow(clippy::too_many_arguments)]
        fn avx2($($arg: $ty),*) $body
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
        #[allow(clippy::too_many_arguments)]
        fn avx512($($arg: $ty),*) $body
        match $isa {
            // SAFETY (both arms): an `Isa` other than `Portable` is only created after its CPU
            // features were detected at run time.
            #[cfg(target_arch = "x86_64")]
            $crate::cpu::Isa::Avx2 => unsafe { avx2($($arg),*) },
            #[cfg(target_arch = "x86_64")]
            $crate::cpu::Isa::Avx512 => unsafe { avx512($($arg),*) },
            $crate::cpu::Isa::Portable => $body,
        }
    }};
}
pub(crate) use dispatch;

/// WGSL's `mix`: `a * (1 - t) + b * t` (the second product fused with the sum).
#[inline(always)]
pub(crate) fn mix(a: f32, b: f32, t: f32) -> f32 {
    b.mul_add(t, a * (1.0 - t))
}

/// Clamps a pixel index into `0..n`.
#[inline(always)]
pub(crate) fn clamp_index(i: isize, n: usize) -> usize {
    i.clamp(0, n as isize - 1) as usize
}

/// Rounds a coordinate down and splits it into an integer part and a fraction. The integer
/// part is limited to a range far outside any image, so that it cannot overflow later sums.
#[inline(always)]
pub(crate) fn split_coordinate(s: f32) -> (isize, f32) {
    let f = s.floor();
    (f.clamp(-(1 << 24) as f32, (1 << 24) as f32) as isize, s - f)
}

/// Bilinear sample of a flow field at continuous position `(x, y)`, clamped to the field.
#[inline(always)]
pub(crate) fn sample_flow(flow: &[[f32; 2]], w: usize, h: usize, x: f32, y: f32) -> [f32; 2] {
    let cx = x.clamp(0.0, w as f32 - 1.0);
    let cy = y.clamp(0.0, h as f32 - 1.0);
    let (x0, y0) = (cx as usize, cy as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (ax, ay) = (cx - x0 as f32, cy - y0 as f32);
    let lerp = |a: [f32; 2], b: [f32; 2], t: f32| [mix(a[0], b[0], t), mix(a[1], b[1], t)];
    let top = lerp(flow[y0 * w + x0], flow[y0 * w + x1], ax);
    let bottom = lerp(flow[y1 * w + x0], flow[y1 * w + x1], ax);
    lerp(top, bottom, ay)
}

/// A flow field of a coarser level, resampled at pixel centre `(x, y)` of a level that is
/// `scale` times finer and converted to that level's pixel units. Pixel centres are at integer
/// coordinates on every level, so `x` maps to `(x + 0.5) / scale - 0.5`.
#[inline(always)]
pub(crate) fn resample_flow(flow: &[[f32; 2]], w: usize, h: usize, scale: f32, x: f32, y: f32) -> [f32; 2] {
    let inv = 1.0 / scale;
    let s = sample_flow(flow, w, h, (x + 0.5) * inv - 0.5, (y + 0.5) * inv - 0.5);
    [scale * s[0], scale * s[1]]
}

/// One level of the image pyramid (empty images on level 0, which is the input), with the
/// flow computed on it.
#[derive(Debug, Default)]
struct Level {
    w: usize,
    h: usize,
    i0: Vec<f32>,
    i1: Vec<f32>,
    flow: Vec<[f32; 2]>,
}

/// Dense Inverse Search on the CPU, for frames of one size.
///
/// Uses the global rayon thread pool (run it inside [`rayon::ThreadPool::install`] to choose the
/// threads); the result does not depend on the number of threads. The environment variable
/// `FREESYNTH_CPU_ISA` (`portable` or `avx2`) restricts the instruction set picked at run time,
/// which is only useful for comparisons: every choice computes the same flow.
pub struct CpuDis {
    params: DisParams,
    plan: Plan,
    isa: Isa,
    levels: Vec<Level>,
    search: search::Scratch,
    refine: refine::Scratch,
}

impl std::fmt::Debug for CpuDis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuDis").field("params", &self.params).field("plan", &self.plan).field("isa", &self.isa).finish_non_exhaustive()
    }
}

impl CpuDis {
    /// Prepares the estimator for `width` x `height` frames. Buffers are allocated on first use
    /// and reused by later flows.
    pub fn new(width: usize, height: usize, params: &DisParams) -> Result<Self, FlowError> {
        let plan = params.plan(width, height)?;
        let levels = (0..=plan.coarsest)
            .map(|l| {
                let (w, h) = plan.level_size(l);
                Level { w, h, ..Level::default() }
            })
            .collect();
        Ok(Self {
            params: params.clone(),
            plan,
            isa: Isa::detect(),
            levels,
            search: search::Scratch::default(),
            refine: refine::Scratch::default(),
        })
    }

    /// Like [`CpuDis::new`], but restricted to portable code (no run-time selected SIMD
    /// version). Only useful for testing that both versions agree.
    #[doc(hidden)]
    pub fn new_portable(width: usize, height: usize, params: &DisParams) -> Result<Self, FlowError> {
        Ok(Self { isa: Isa::Portable, ..Self::new(width, height, params)? })
    }

    pub fn params(&self) -> &DisParams {
        &self.params
    }

    /// The pyramid levels in use.
    pub fn plan(&self) -> Plan {
        self.plan
    }

    /// Frame size this instance was built for.
    pub fn size(&self) -> (usize, usize) {
        (self.plan.width, self.plan.height)
    }

    /// Builds both pyramids from the input frames. Level 0 is the input itself and is not
    /// copied.
    fn build_pyramid(&mut self, frame0: &GrayImage, frame1: &GrayImage) {
        for l in 1..self.levels.len() {
            let (fine, coarse) = self.levels.split_at_mut(l);
            let (src, dst) = (&fine[l - 1], &mut coarse[0]);
            let sources = if l == 1 { [frame0.data(), frame1.data()] } else { [&src.i0[..], &src.i1[..]] };
            for (s, d) in sources.into_iter().zip([&mut dst.i0, &mut dst.i1]) {
                d.resize(dst.w * dst.h, 0.0);
                downsample(s, src.w, d, dst.w);
            }
        }
    }

    /// Computes the flow of every level from the coarsest to the output level.
    fn run_levels(&mut self, frame0: &GrayImage, frame1: &GrayImage) {
        let p = &self.params;
        let plan = self.plan;
        for l in (plan.output_level()..=plan.coarsest).rev() {
            let (fine, coarse) = self.levels.split_at_mut(l + 1);
            let level = &mut fine[l];
            let (i0, i1) = if l == 0 { (frame0.data(), frame1.data()) } else { (&level.i0[..], &level.i1[..]) };
            // The next coarser level has been computed in this call if it is in the plan.
            let coarser = coarse.first().filter(|_| l < plan.coarsest).map(|c| search::Coarse { flow: &c.flow, w: c.w, h: c.h });
            level.flow.resize(level.w * level.h, [0.0; 2]);
            if plan.is_searched(l) {
                let (pw, ph) = plan.patch_grid(l, p.patch_stride);
                let patches = search::search_level(&mut self.search, i0, i1, level.w, level.h, p, pw, ph, coarser, self.isa);
                search::densify(i0, i1, level.w, level.h, p.patch_stride, pw, ph, patches, &mut level.flow, self.isa);
            } else {
                let c = coarser.expect("levels below the finest searched one have a coarser level");
                upsample(c.flow, c.w, c.h, 2.0, &mut level.flow, level.w, self.isa);
            }
            if plan.is_refined(l) && p.vr_iterations > 0 {
                refine::refine_level(&mut self.refine, i0, i1, level.w, level.h, &mut level.flow, p, self.isa);
            }
        }
    }

    fn check_sizes(&self, frame0: &GrayImage, frame1: &GrayImage) -> Result<(), FlowError> {
        let (w, h) = self.size();
        if frame0.size() != (w, h) || frame1.size() != (w, h) {
            return invalid(format!("frames must be {w}x{h}"));
        }
        Ok(())
    }

    /// Exchanges the two pyramids, for the flow in the other direction.
    fn swap_pyramids(&mut self) {
        for level in &mut self.levels {
            std::mem::swap(&mut level.i0, &mut level.i1);
        }
    }

    /// Copies the flow of the output level out, upsampled to full resolution.
    fn output(&self) -> FlowField {
        let (w, h) = self.size();
        let out_level = self.plan.output_level();
        let level = &self.levels[out_level];
        // The level buffers stay allocated for the next flow; the result is copied out (in
        // parallel, which also spreads the page faults of the new allocation).
        let flow = if out_level == 0 {
            level.flow.par_iter().copied().collect()
        } else {
            let mut out = vec![[0.0; 2]; w * h];
            upsample(&level.flow, level.w, level.h, (1u32 << out_level) as f32, &mut out, w, self.isa);
            out
        };
        FlowField::from_vec(w, h, flow)
    }
}

impl crate::FlowEstimator for CpuDis {
    fn compute(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<FlowField, FlowError> {
        self.check_sizes(frame0, frame1)?;
        self.build_pyramid(frame0, frame1);
        self.run_levels(frame0, frame1);
        Ok(self.output())
    }

    /// Builds the two pyramids once for both directions.
    fn compute_pair(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<(FlowField, FlowField), FlowError> {
        self.check_sizes(frame0, frame1)?;
        self.build_pyramid(frame0, frame1);
        self.run_levels(frame0, frame1);
        let forward = self.output();
        self.swap_pyramids();
        self.run_levels(frame1, frame0);
        Ok((forward, self.output()))
    }
}

/// 2x2 box downsampling into `dst` (`dst_w` wide); level sizes are rounded down, so every block
/// is inside the source.
fn downsample(src: &[f32], src_w: usize, dst: &mut [f32], dst_w: usize) {
    dst.par_chunks_mut(dst_w).enumerate().for_each(|(y, row)| {
        let (a, b) = (&src[2 * y * src_w..], &src[(2 * y + 1) * src_w..]);
        for (x, d) in row.iter_mut().enumerate() {
            *d = 0.25 * (a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1]);
        }
    });
}

/// Resamples a coarser flow (`cw` x `ch`) onto `out` (`w` wide), `scale` times finer. Gives
/// the same values as [`resample_flow`] at every pixel; the sample positions, which only depend
/// on the column or the row, are computed once per column and once per row.
fn upsample(flow: &[[f32; 2]], cw: usize, ch: usize, scale: f32, out: &mut [[f32; 2]], w: usize, isa: Isa) {
    let h = out.len() / w;
    let columns = taps(w, cw, scale);
    let rows = taps(h, ch, scale);
    out.par_chunks_mut(w).zip(rows.par_iter()).for_each(|(row, &(y0, y1, ay))| {
        let (top, bottom) = (&flow[y0 * cw..][..cw], &flow[y1 * cw..][..cw]);
        let columns = &columns[..];
        // Dispatched for the fused multiply-adds of `mix`, which are slow without hardware
        // support.
        dispatch!(isa, (row: &mut [[f32; 2]], top: &[[f32; 2]], bottom: &[[f32; 2]], columns: &[(usize, usize, f32)], ay: f32, scale: f32) {
            for (o, &(x0, x1, ax)) in row.iter_mut().zip(columns) {
                let lerp = |a: [f32; 2], b: [f32; 2], t: f32| [mix(a[0], b[0], t), mix(a[1], b[1], t)];
                let s = lerp(lerp(top[x0], top[x1], ax), lerp(bottom[x0], bottom[x1], ax), ay);
                *o = [scale * s[0], scale * s[1]];
            }
        });
    });
}

/// For each of `n` pixel centres of a level `scale` times finer than one with `coarse` pixels
/// along the same axis: the two coarse pixels to interpolate between and the weight of the
/// second, as [`sample_flow`] computes them.
fn taps(n: usize, coarse: usize, scale: f32) -> Vec<(usize, usize, f32)> {
    let inv = 1.0 / scale;
    (0..n)
        .map(|i| {
            let c = ((i as f32 + 0.5) * inv - 0.5).clamp(0.0, coarse as f32 - 1.0);
            let i0 = c as usize;
            (i0, (i0 + 1).min(coarse - 1), c - i0 as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlowEstimator;
    use crate::params::Preset;

    fn texture(x: f32, y: f32) -> f32 {
        128.0 + 40.0 * (0.21 * x + 0.07 * y).sin() + 30.0 * (-0.05 * x + 0.17 * y + 1.1).sin() + 20.0 * (0.33 * x + 0.29 * y).sin()
    }

    fn shifted(w: usize, h: usize, shift: [f32; 2]) -> GrayImage {
        GrayImage::from_vec(w, h, (0..w * h).map(|i| texture((i % w) as f32 - shift[0], (i / w) as f32 - shift[1])).collect()).unwrap()
    }

    #[test]
    fn helpers() {
        assert_eq!(mix(2.0, 4.0, 0.25), 2.5);
        assert_eq!(clamp_index(-3, 5), 0);
        assert_eq!(clamp_index(9, 5), 4);
        assert_eq!(split_coordinate(-1.25), (-2, 0.75));
        let flow = [[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [2.0, 4.0]];
        assert_eq!(sample_flow(&flow, 2, 2, 0.5, 0.5), [1.0, 2.0]);
        // A 2x coarser field: pixel 0 of the fine level is at -0.25 on the coarse one.
        assert_eq!(resample_flow(&[[1.0, -1.0]], 1, 1, 2.0, 0.0, 0.0), [2.0, -2.0]);
        let mut small = vec![0.0; 2];
        downsample(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4, &mut small, 2);
        assert_eq!(small, [3.5, 5.5]);
    }

    #[test]
    fn recovers_a_translation() {
        let (w, h) = (96, 64);
        let truth = [1.75, -0.5];
        let (a, b) = (shifted(w, h, [0.0; 2]), shifted(w, h, truth));
        for preset in Preset::ALL {
            let mut dis = CpuDis::new(w, h, &DisParams::preset(preset)).unwrap();
            let flow = dis.compute(&a, &b).unwrap();
            let mask: Vec<bool> = (0..w * h).map(|i| (8..w - 8).contains(&(i % w)) && (8..h - 8).contains(&(i / w))).collect();
            let epe = flow.mean_epe(&FlowField::constant(w, h, truth), Some(&mask)).unwrap();
            assert!(epe < 0.15, "{preset}: EPE {epe}");
            // A second call reuses the buffers and gives the same result.
            assert_eq!(dis.compute(&a, &b).unwrap(), flow);
        }
    }

    #[test]
    fn upsampling_matches_resampling() {
        let (cw, ch) = (7, 5);
        let flow: Vec<[f32; 2]> = (0..cw * ch).map(|i| [(i as f32 * 0.37).sin() * 3.0, (i as f32 * 0.61).cos()]).collect();
        for (scale, w, h) in [(2.0, 14, 10), (2.0, 15, 11), (4.0, 29, 23), (1.0, 7, 5)] {
            let mut out = vec![[0.0; 2]; w * h];
            upsample(&flow, cw, ch, scale, &mut out, w, Isa::detect());
            for (i, o) in out.iter().enumerate() {
                assert_eq!(*o, resample_flow(&flow, cw, ch, scale, (i % w) as f32, (i / w) as f32), "scale {scale}, pixel {i}");
            }
        }
    }

    #[test]
    fn pair_equals_two_flows() {
        let (w, h) = (75, 53);
        let (a, b) = (shifted(w, h, [0.0; 2]), shifted(w, h, [1.3, -0.8]));
        for preset in Preset::ALL {
            let mut dis = CpuDis::new(w, h, &DisParams::preset(preset)).unwrap();
            let (forward, backward) = dis.compute_pair(&a, &b).unwrap();
            assert_eq!(forward, dis.compute(&a, &b).unwrap(), "{preset}");
            assert_eq!(backward, dis.compute(&b, &a).unwrap(), "{preset}");
        }
    }

    #[test]
    fn simd_and_portable_versions_agree() {
        let (w, h) = (75, 53);
        let (a, b) = (shifted(w, h, [0.0; 2]), shifted(w, h, [-2.3, 1.2]));
        let params = DisParams::preset(Preset::Medium);
        let fast = CpuDis::new(w, h, &params).unwrap().compute(&a, &b).unwrap();
        let portable = CpuDis::new_portable(w, h, &params).unwrap().compute(&a, &b).unwrap();
        assert_eq!(fast, portable);
    }

    #[test]
    fn small_and_odd_sizes() {
        for (w, h) in [(8, 8), (9, 33), (101, 67)] {
            let (a, b) = (shifted(w, h, [0.0; 2]), shifted(w, h, [0.5, 0.25]));
            for preset in Preset::ALL {
                let flow = CpuDis::new(w, h, &DisParams::preset(preset)).unwrap().compute(&a, &b).unwrap();
                assert!(flow.data().iter().all(|v| v[0].is_finite() && v[1].is_finite()), "{w}x{h} {preset}");
            }
        }
        assert!(CpuDis::new(7, 20, &DisParams::default()).is_err());
        let mut dis = CpuDis::new(16, 16, &DisParams::default()).unwrap();
        assert!(dis.compute(&GrayImage::new(16, 16), &GrayImage::new(16, 15)).is_err());
    }
}
