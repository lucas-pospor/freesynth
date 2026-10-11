//! Dense optical flow and position fields for FreeSynth.
//!
//! The estimator is Dense Inverse Search (DIS), written from Kroeger, Timofte, Dai, Van Gool,
//! "Fast Optical Flow using Dense Inverse Search", ECCV 2016:
//!
//! 1. an image pyramid (2x2 box filter per level);
//! 2. on every level from coarse to fine, 8x8 patches on a regular grid are aligned to the
//!    second frame by inverse-compositional Gauss-Newton iterations on mean-normalised
//!    intensities, starting from the coarser level's flow; a patch stops early once its steps
//!    become negligible;
//! 3. spatial propagation in parallel passes: every patch compares its result with its grid
//!    neighbours' and continues from a neighbour's displacement when that one fits clearly
//!    better, which repairs patches whose search went astray;
//! 4. densification: the flow at a pixel is the photometrically weighted average of all
//!    patches that cover it;
//! 5. variational refinement of the dense flow (normalised brightness and gradient constancy,
//!    robust smoothness), solved by red-black over-relaxation. It can continue below the
//!    finest searched level, refining the upsampled flow at full resolution.
//!
//! [`DisParams`] holds the parameters, with [`Preset`]s from fast to accurate. Estimators
//! implement [`FlowEstimator`]: they take the same parameters, consume [`GrayImage`]s and
//! produce [`FlowField`]s. There are two:
//!
//! - `gpu::GpuDis` (behind the default `gpu` feature), WGSL compute shaders on wgpu. It can
//!   also work on frames that are already on the GPU, on a device shared with the rest of the
//!   application, and record any number of flows into one submission.
//! - [`cpu::CpuDis`], the same algorithm in Rust on rayon threads, as a fallback without a GPU
//!   and as a reference: on the same input both compute the same flow up to rounding.
//!
//! [`track`] turns flows into position fields: for every pixel of a frame, its position in a
//! keyframe, with a confidence map that marks where that position cannot be trusted.
//!
//! Without the `gpu` feature the crate depends only on rayon.
//!
//! # Flow convention
//!
//! The flow from frame A to frame B at pixel `(x, y)` is the displacement `(u, v)` such that
//! the content at that pixel centre of A is at `(x + u, y + v)` in B, i.e.
//! `A(x, y) ~ B(x + u, y + v)`. Pixel centres are at integer coordinates. This is the
//! convention of the Middlebury `.flo` files ([`flo`]).
//!
//! ```no_run
//! # #[cfg(feature = "gpu")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use freesynth_flow::{DisParams, FlowEstimator, GrayImage, Preset, cpu::CpuDis, gpu::{GpuContext, GpuDis}};
//!
//! let a = GrayImage::from_luma8(64, 48, &[0; 64 * 48])?;
//! let b = a.clone();
//! let params = DisParams::preset(Preset::Medium);
//! // The GPU if there is one, the CPU otherwise.
//! let mut dis: Box<dyn FlowEstimator> = match GpuContext::new() {
//!     Ok(ctx) => Box::new(GpuDis::new(&ctx, 64, 48, &params)?),
//!     Err(_) => Box::new(CpuDis::new(64, 48, &params)?),
//! };
//! let flow = dis.compute(&a, &b)?;
//! flow.save("a_to_b.flo")?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "gpu"))]
//! # fn main() {}
//! ```

pub mod cpu;
mod error;
pub mod flo;
#[cfg(feature = "gpu")]
pub mod gpu;
mod image;
mod params;
pub mod track;

pub use crate::error::FlowError;
pub use crate::flo::{FLO_MAGIC, FlowField};
pub use crate::image::GrayImage;
pub use crate::params::{DisParams, PATCH_SIZE, Plan, Preset};

/// A dense flow estimator for frames of a fixed size.
pub trait FlowEstimator {
    /// The flow from `frame0` to `frame1`: `frame0(x) ~ frame1(x + flow(x))`.
    fn compute(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<FlowField, FlowError>;

    /// The flows from `frame0` to `frame1` and from `frame1` to `frame0`, equal to two calls of
    /// [`compute`](Self::compute). Estimators can share work between the two (the image
    /// pyramids, for example), so prefer this when both directions are needed.
    fn compute_pair(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<(FlowField, FlowField), FlowError> {
        Ok((self.compute(frame0, frame1)?, self.compute(frame1, frame0)?))
    }
}

impl<E: FlowEstimator + ?Sized> FlowEstimator for Box<E> {
    fn compute(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<FlowField, FlowError> {
        (**self).compute(frame0, frame1)
    }

    fn compute_pair(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<(FlowField, FlowField), FlowError> {
        (**self).compute_pair(frame0, frame1)
    }
}

impl<E: FlowEstimator + ?Sized> FlowEstimator for &mut E {
    fn compute(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<FlowField, FlowError> {
        (**self).compute(frame0, frame1)
    }

    fn compute_pair(&mut self, frame0: &GrayImage, frame1: &GrayImage) -> Result<(FlowField, FlowField), FlowError> {
        (**self).compute_pair(frame0, frame1)
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use std::path::PathBuf;

    /// A fresh scratch directory under the build directory (the system temp directory may be
    /// RAM-backed).
    pub fn scratch_dir(name: &str) -> PathBuf {
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"));
        let dir = target.join("tmp").join(format!("freesynth-flow-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
