//! Procedural test sequences for FreeSynth, with exact optical flow and ground truth.
//!
//! A scene is a panning, textured background with a few textured objects in front of it. The
//! objects translate, rotate, change scale and occlude each other. Because the geometry is
//! analytic, every frame comes with:
//!
//! - the input video frame, which serves as the guide;
//! - the **exact** forward and backward optical flow to the next frame (and to frames further
//!   away, on request), with occlusion masks;
//! - a ground-truth stylization of the frame, from one of several [`StyleFamily`] styles; the
//!   keyframe styles are the ground truth at the keyframes.
//!
//! Everything is generated from a seed, so scenes are not stored; they are rebuilt on demand,
//! and their pixels and flow values are reproducible bit for bit. Generated images are
//! released under CC0 1.0.
//!
//! ```no_run
//! use freesynth_scenes::{SceneParams, StyleFamily, write_scene};
//!
//! let params = SceneParams { family: StyleFamily::Painterly, seed: 3, ..SceneParams::default() };
//! let info = write_scene(&params, "orbit-painterly", std::path::Path::new("out/orbit-painterly"))?;
//! assert_eq!(info.frames, 30);
//! # Ok::<(), freesynth_scenes::Error>(())
//! ```
//!
//! # Choosing a style family
//!
//! In the [`StyleFamily::Rigid`] family the style is painted onto each surface, so the right
//! answer for a frame is the keyframe carried along by the motion. That is exactly what plain
//! flow-based advection does, so on this family advection looks better than it is: it can
//! match the ground truth while smearing in ways that the pixel metrics barely see. The
//! [`StyleFamily::Painterly`] and [`StyleFamily::Hatching`] families re-render the style from
//! the geometry in every frame (strokes keep their screen-space orientation and pixel size;
//! hatching lines stay fixed on screen), so rigid transport of a keyframe is measurably wrong
//! there. Judge a method on more than one family before reading anything into ground-truth
//! scores.
//!
//! # Flow and occlusion conventions
//!
//! The flow at pixel `(x, y)` moves the surface point seen at that pixel centre to
//! `(x + u, y + v)` in the other frame. A pixel is marked occluded when that point is hidden
//! or outside the other frame, and also when it, or a pixel that bilinear interpolation at
//! its destination would read, is an anti-aliased mix of two surfaces. Warping the other
//! frame with the flow therefore reproduces the first frame at every unmarked pixel, up to
//! interpolation error. See [`output`] for the directory layout.

mod error;
mod geometry;
pub mod manifest;
mod noise;
pub mod output;
mod rng;
mod scene;
mod strokes;
mod style;

pub use crate::error::Error;
pub use freesynth_flow::FlowField;
pub use crate::manifest::{Manifest, ManifestScene, Tier};
pub use crate::output::{Direction, SceneInfo, write_scene};
pub use crate::scene::{ExactFlow, Frame, LayerMap, Scene, SceneParams};
pub use crate::style::StyleFamily;
