//! Scene parameters, frame rendering and exact optical flow.

use freesynth_flow::FlowField;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::geometry::World;
use crate::rng::Rng;
use crate::style::{Look, Style, StyleFamily};

/// Sub-pixel sample offsets of the 3x3 supersampling grid. The centre sample is included.
const SUBSAMPLES: [f64; 3] = [-1.0 / 3.0, 0.0, 1.0 / 3.0];

/// Everything that determines a generated scene. The same parameters always give the same
/// pixels and flow values: all randomness comes from `seed`, and sines, cosines and the like
/// come from a portable software implementation instead of the platform's maths library.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SceneParams {
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Number of frames.
    pub frames: usize,
    /// Random seed. Scenes with the same seed and size share the input video, motion and flow
    /// whatever their style family, so families can be compared on identical footage.
    pub seed: u64,
    /// Ground-truth style family.
    pub family: StyleFamily,
    /// Frame indices whose ground truth is given as keyframe styles. Empty means the first and
    /// the last frame.
    pub keyframes: Vec<usize>,
    /// Number of moving objects in front of the background.
    pub objects: usize,
    /// Speed multiplier for every motion (camera pan, translation, rotation, scale changes).
    pub motion: f64,
    /// Frame gaps for which exact flow and occlusion masks are written (1 is always written).
    pub flow_gaps: Vec<usize>,
}

impl Default for SceneParams {
    fn default() -> Self {
        Self {
            width: 480,
            height: 270,
            frames: 30,
            seed: 1,
            family: StyleFamily::Rigid,
            keyframes: Vec::new(),
            objects: 3,
            motion: 1.0,
            flow_gaps: vec![1],
        }
    }
}

impl SceneParams {
    /// Checks that the parameters describe a scene that can be generated.
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::InvalidParams(msg));
        if !(16..=8192).contains(&self.width) || !(16..=8192).contains(&self.height) {
            return bad(format!("frame size {}x{} out of range (16 to 8192)", self.width, self.height));
        }
        if !(1..=100_000).contains(&self.frames) {
            return bad(format!("frame count {} out of range (1 to 100000)", self.frames));
        }
        if self.objects > 32 {
            return bad(format!("{} objects is too many (at most 32)", self.objects));
        }
        if !self.motion.is_finite() || self.motion < 0.0 {
            return bad(format!("motion {} must be a finite, non-negative number", self.motion));
        }
        if let Some(&k) = self.keyframes.iter().find(|&&k| k >= self.frames) {
            return bad(format!("keyframe {k} is not below the frame count {}", self.frames));
        }
        if let Some(&g) = self.flow_gaps.iter().find(|&&g| g == 0) {
            return bad(format!("flow gap {g} must be at least 1"));
        }
        Ok(())
    }

    /// The keyframe indices, sorted and without duplicates, with the default applied.
    pub fn resolved_keyframes(&self) -> Vec<usize> {
        let mut keys = if self.keyframes.is_empty() { vec![0, self.frames - 1] } else { self.keyframes.clone() };
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// The flow gaps to write: always 1, plus the requested ones below the frame count.
    pub fn resolved_flow_gaps(&self) -> Vec<usize> {
        let mut gaps: Vec<usize> = std::iter::once(1).chain(self.flow_gaps.iter().copied()).collect();
        gaps.retain(|&g| g < self.frames);
        gaps.sort_unstable();
        gaps.dedup();
        gaps
    }
}

/// Which layer each pixel shows, from the 3x3 supersampling of a rendered frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerMap {
    width: usize,
    height: usize,
    ids: Vec<u8>,
    pure: Vec<bool>,
}

impl LayerMap {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Layer seen at the pixel centre: 0 for the background, `1 + i` for object `i`.
    pub fn layer(&self, x: usize, y: usize) -> u8 {
        self.ids[y * self.width + x]
    }

    /// True when every sub-pixel sample of the pixel sees the same layer, i.e. the pixel is not
    /// an anti-aliased mix of two surfaces.
    pub fn is_pure(&self, x: usize, y: usize) -> bool {
        self.pure[y * self.width + x]
    }
}

/// One rendered frame.
#[derive(Clone, Debug)]
pub struct Frame {
    /// The input video frame (the guide).
    pub image: RgbImage,
    /// The ground-truth stylization of this frame.
    pub truth: RgbImage,
    /// Per-pixel layer ids.
    pub layers: LayerMap,
}

/// Exact optical flow between two frames, with its occlusion mask.
#[derive(Clone, Debug)]
pub struct ExactFlow {
    /// Displacement of the surface point seen at each pixel centre of the first frame.
    pub flow: FlowField,
    /// 255 where the flow cannot be used to warp, 0 elsewhere. A pixel is marked when its
    /// surface point is hidden in the second frame or leaves it, or when the pixel or the
    /// pixels around its destination are anti-aliased mixes of two surfaces (their colour
    /// does not belong to a single moving point).
    pub occlusion: GrayImage,
}

/// A procedural scene, ready to render any frame or flow on demand.
#[derive(Clone, Debug)]
pub struct Scene {
    params: SceneParams,
    world: World,
    look: Look,
    style: Style,
}

impl Scene {
    /// Builds the scene described by `params` (geometry, palettes, style textures).
    pub fn new(params: &SceneParams) -> Result<Self, Error> {
        params.validate()?;
        let (w, h) = (params.width, params.height);
        let world = World::random(&mut Rng::new(params.seed, 1), w, h, params.objects, params.motion);
        let look = Look::random(&mut Rng::new(params.seed, 2), &world, w, h);
        let style = Style::new(params.family, params.seed, &world, &look, params.frames, [w, h]);
        Ok(Self { params: params.clone(), world, look, style })
    }

    pub fn params(&self) -> &SceneParams {
        &self.params
    }

    /// Renders frame `t`: the input video, the ground truth and the layer map.
    pub fn render(&self, t: usize) -> Frame {
        let (w, h) = (self.params.width, self.params.height);
        let snap = self.world.at(t as f64);
        let frame_style = self.style.frame(&snap, &self.look, [w, h]);
        let mut image = vec![0u8; w * h * 3];
        let mut truth = vec![0u8; w * h * 3];
        let mut ids = vec![0u8; w * h];
        let mut pure = vec![false; w * h];
        image
            .par_chunks_mut(w * 3)
            .zip(truth.par_chunks_mut(w * 3))
            .zip(ids.par_chunks_mut(w).zip(pure.par_chunks_mut(w)))
            .enumerate()
            .for_each(|(y, ((img_row, truth_row), (id_row, pure_row)))| {
                for x in 0..w {
                    let (mut video, mut styled) = ([0.0f64; 3], [0.0f64; 3]);
                    let (mut first, mut same) = (None, true);
                    for dy in SUBSAMPLES {
                        for dx in SUBSAMPLES {
                            let (sx, sy) = (x as f64 + dx, y as f64 + dy);
                            let hit = snap.hit(sx, sy);
                            let v = self.look.video(&hit);
                            let s = self.style.shade(&snap, &self.look, &frame_style, &hit, sx, sy);
                            for c in 0..3 {
                                video[c] += v[c];
                                styled[c] += s[c];
                            }
                            match first {
                                None => first = Some(hit.layer),
                                Some(l) => same &= l == hit.layer,
                            }
                            if dx == 0.0 && dy == 0.0 {
                                id_row[x] = hit.layer as u8;
                            }
                        }
                    }
                    pure_row[x] = same;
                    for c in 0..3 {
                        img_row[x * 3 + c] = to_u8(video[c] / 9.0);
                        truth_row[x * 3 + c] = to_u8(styled[c] / 9.0);
                    }
                }
            });
        let as_image = |data| RgbImage::from_raw(w as u32, h as u32, data).expect("buffer size matches");
        Frame { image: as_image(image), truth: as_image(truth), layers: LayerMap { width: w, height: h, ids, pure } }
    }

    /// Exact flow from frame `from` to frame `to` (either order), given both frames' layer
    /// maps (from [`Scene::render`]).
    pub fn flow(&self, from: usize, to: usize, from_layers: &LayerMap, to_layers: &LayerMap) -> ExactFlow {
        let (w, h) = (self.params.width, self.params.height);
        let (s0, s1) = (self.world.at(from as f64), self.world.at(to as f64));
        let mut flow = FlowField::new(w, h);
        let mut occlusion = vec![0u8; w * h];
        flow.data_mut().par_chunks_mut(w).zip(occlusion.par_chunks_mut(w)).enumerate().for_each(
            |(y, (flow_row, occ_row))| {
                for x in 0..w {
                    let hit = s0.hit(x as f64, y as f64);
                    let [qx, qy] = s1.to_screen(hit.layer, hit.u, hit.v);
                    flow_row[x] = [(qx - x as f64) as f32, (qy - y as f64) as f32];
                    let layer = hit.layer as u8;
                    let inside = qx >= 0.0 && qy >= 0.0 && qx <= (w - 1) as f64 && qy <= (h - 1) as f64;
                    let usable = inside
                        && from_layers.is_pure(x, y)
                        && s1.hit(qx, qy).layer == hit.layer
                        && bilinear_support(qx, qy, w, h)
                            .all(|(nx, ny)| to_layers.layer(nx, ny) == layer && to_layers.is_pure(nx, ny));
                    occ_row[x] = if usable { 0 } else { 255 };
                }
            },
        );
        let occlusion = GrayImage::from_raw(w as u32, h as u32, occlusion).expect("buffer size matches");
        ExactFlow { flow, occlusion }
    }
}

/// The (up to four) pixels that bilinear interpolation at `(x, y)` reads. `(x, y)` must be
/// inside the frame.
fn bilinear_support(x: f64, y: f64, w: usize, h: usize) -> impl Iterator<Item = (usize, usize)> {
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    [(x0, y0), (x1, y0), (x0, y1), (x1, y1)].into_iter()
}

fn to_u8(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(family: StyleFamily) -> SceneParams {
        SceneParams { width: 64, height: 48, frames: 4, family, ..SceneParams::default() }
    }

    #[test]
    fn validates_parameters() {
        assert!(SceneParams::default().validate().is_ok());
        assert!(SceneParams { width: 8, ..tiny(StyleFamily::Rigid) }.validate().is_err());
        assert!(SceneParams { keyframes: vec![4], ..tiny(StyleFamily::Rigid) }.validate().is_err());
        assert!(SceneParams { flow_gaps: vec![0], ..tiny(StyleFamily::Rigid) }.validate().is_err());
        assert!(SceneParams { motion: f64::NAN, ..tiny(StyleFamily::Rigid) }.validate().is_err());
    }

    #[test]
    fn resolves_defaults() {
        let p = SceneParams { flow_gaps: vec![5, 2, 9], ..tiny(StyleFamily::Rigid) };
        assert_eq!(p.resolved_keyframes(), [0, 3]);
        assert_eq!(p.resolved_flow_gaps(), [1, 2]);
        let p = SceneParams { frames: 1, ..tiny(StyleFamily::Rigid) };
        assert_eq!(p.resolved_keyframes(), [0]);
        assert!(p.resolved_flow_gaps().is_empty());
    }

    #[test]
    fn families_share_the_input_video() {
        let a = Scene::new(&tiny(StyleFamily::Rigid)).unwrap().render(2);
        let b = Scene::new(&tiny(StyleFamily::Hatching)).unwrap().render(2);
        assert_eq!(a.image, b.image);
        assert_eq!(a.layers, b.layers);
        assert_ne!(a.truth, b.truth);
    }

    #[test]
    fn zero_motion_gives_zero_flow() {
        let p = SceneParams { motion: 0.0, ..tiny(StyleFamily::Rigid) };
        let scene = Scene::new(&p).unwrap();
        let (f0, f1) = (scene.render(0), scene.render(1));
        let flow = scene.flow(0, 1, &f0.layers, &f1.layers);
        let max = flow.flow.data().iter().map(|f| f[0].abs().max(f[1].abs())).fold(0.0, f32::max);
        assert!(max < 1e-6, "max displacement {max}");
        assert_eq!(f0.image, f1.image);
    }
}
