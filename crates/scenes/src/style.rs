//! The look of the input video and of the ground-truth stylizations.

use std::fmt;
use std::str::FromStr;

use libm::{cos, sin, sqrt};
use serde::{Deserialize, Serialize};

use crate::geometry::{Hit, Snapshot, World, size_factor};
use crate::noise::fbm;
use crate::rng::{Rng, mix64};
use crate::strokes::{Canvas, Stroke};

/// How the ground-truth stylization relates to the moving geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StyleFamily {
    /// Brush-stroke textures fixed in each object's own space, with outlines that scale with
    /// the object. The correct output is the keyframe transported rigidly along the motion,
    /// which is exactly what flow-based advection produces.
    #[default]
    Rigid,
    /// Painterly rendering re-done every frame: strokes are anchored to the surface (so they
    /// travel with it) but keep a screen-space orientation and a constant size in pixels, and
    /// outlines keep a constant width. When an object rotates or changes scale, the correct
    /// output is not a rigid transport of the keyframe.
    Painterly,
    /// Tonal hatching whose lines are fixed in screen space while the tone follows the
    /// shading of the moving objects, with constant-width outlines. Rigid transport of the
    /// keyframe drags the lines along and is wrong everywhere something moves.
    Hatching,
}

impl StyleFamily {
    pub const ALL: [StyleFamily; 3] = [StyleFamily::Rigid, StyleFamily::Painterly, StyleFamily::Hatching];

    pub fn as_str(self) -> &'static str {
        match self {
            StyleFamily::Rigid => "rigid",
            StyleFamily::Painterly => "painterly",
            StyleFamily::Hatching => "hatching",
        }
    }
}

impl fmt::Display for StyleFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for StyleFamily {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|f| f.as_str() == s)
            .ok_or_else(|| format!("unknown style family '{s}' (expected rigid, painterly or hatching)"))
    }
}

/// Colour schemes: input base colour and a four-colour stroke palette. Index 0 is reserved
/// for the background.
const SCHEMES: [([f64; 3], [[f32; 3]; 4]); 7] = [
    ([125.0, 120.0, 110.0], [[30.0, 40.0, 60.0], [50.0, 70.0, 90.0], [70.0, 110.0, 130.0], [90.0, 60.0, 40.0]]),
    ([210.0, 130.0, 70.0], [[200.0, 90.0, 40.0], [170.0, 60.0, 30.0], [230.0, 140.0, 80.0], [120.0, 40.0, 20.0]]),
    ([80.0, 150.0, 190.0], [[40.0, 120.0, 160.0], [60.0, 160.0, 200.0], [30.0, 80.0, 120.0], [120.0, 200.0, 230.0]]),
    ([110.0, 175.0, 90.0], [[90.0, 170.0, 60.0], [60.0, 130.0, 40.0], [140.0, 200.0, 110.0], [40.0, 90.0, 30.0]]),
    ([190.0, 90.0, 140.0], [[170.0, 60.0, 120.0], [210.0, 110.0, 160.0], [120.0, 40.0, 90.0], [230.0, 170.0, 200.0]]),
    ([200.0, 185.0, 80.0], [[220.0, 190.0, 60.0], [180.0, 150.0, 40.0], [240.0, 220.0, 130.0], [140.0, 110.0, 30.0]]),
    ([150.0, 110.0, 200.0], [[120.0, 80.0, 190.0], [160.0, 120.0, 220.0], [80.0, 50.0, 140.0], [200.0, 170.0, 240.0]]),
];

/// Per-layer surface description shared by the input video and the stylizations.
#[derive(Clone, Debug)]
struct Surface {
    base: [f64; 3],
    palette: [[f32; 3]; 4],
    detail_seed: u64,
}

/// The input video ("guide") and the surface tone that the styles follow.
#[derive(Clone, Debug)]
pub(crate) struct Look {
    surfaces: Vec<Surface>,
    /// Object semi-axis `a` per layer (unused for the background).
    axis: Vec<f64>,
    k: f64,
}

impl Look {
    pub fn random(rng: &mut Rng, world: &World, width: usize, height: usize) -> Self {
        let mut order: Vec<usize> = (1..SCHEMES.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let surfaces = (0..world.layers())
            .map(|l| {
                let scheme = if l == 0 { 0 } else { order[(l - 1) % order.len()] };
                Surface { base: SCHEMES[scheme].0, palette: SCHEMES[scheme].1, detail_seed: rng.next_u64() }
            })
            .collect();
        let axis = std::iter::once(1.0).chain(world.objects.iter().map(|o| o.a)).collect();
        Self { surfaces, axis, k: size_factor(width, height) }
    }

    /// Smooth shading of a surface point, roughly in `[0.5, 1.1]`. It lives in object space,
    /// so a point keeps its shade as it moves (the input video obeys brightness constancy).
    pub fn shade(&self, layer: usize, u: f64, v: f64, r: f64) -> f64 {
        if layer == 0 {
            0.85 + 0.15 * sin(u / (90.0 * self.k)) * cos(v / (70.0 * self.k))
        } else {
            0.65 + 0.35 * sqrt((1.0 - r).clamp(0.0, 1.0)) + 0.1 * (u / self.axis[layer])
        }
    }

    /// Ground colour under the painterly strokes of a layer.
    fn fill(&self, layer: usize) -> [f32; 3] {
        self.surfaces[layer].palette[0].map(|c| c * 0.8)
    }

    /// Colour of the input video at a hit: base colour, shading and fine texture.
    pub fn video(&self, hit: &Hit) -> [f64; 3] {
        let s = &self.surfaces[hit.layer];
        let shade = self.shade(hit.layer, hit.u, hit.v, hit.r);
        let fine = fbm(s.detail_seed, hit.u, hit.v, (6.0 * self.k).max(3.0), 3);
        let coarse = fbm(s.detail_seed ^ 0x5bd1_e995, hit.u, hit.v, (40.0 * self.k).max(12.0), 2);
        let f = shade * (0.78 + 0.30 * fine + 0.14 * coarse);
        s.base.map(|c| c * f)
    }
}

/// One family's per-scene data.
#[derive(Clone, Debug)]
pub(crate) enum Style {
    Rigid { textures: Vec<Canvas> },
    Painterly { particles: Vec<Vec<Particle>>, stroke_radius: f32, outline: f64 },
    Hatching { spacing: f64, angles: Vec<f64>, outline: f64 },
}

/// A painterly stroke anchored to a surface point.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Particle {
    u: f64,
    v: f64,
    colour: [f32; 3],
    half_length: f32,
    angle_jitter: f32,
    bristle_phase: f32,
}

/// Per-frame data of a style (the painterly canvases).
pub(crate) struct FrameStyle {
    canvases: Vec<Canvas>,
}

/// Resolution of the painterly canvases in canvas pixels per screen pixel. It matches the 3x3
/// supersampling of [`Scene::render`](crate::Scene::render), whose samples sit at offsets of
/// 1/3 pixel: each sample reads its own canvas pixel, so strokes and their bristle streaks are
/// anti-aliased like the input video and do not shimmer as they move by fractions of a pixel.
const CANVAS_SCALE: f64 = 3.0;

/// Screen-space stroke direction field of the painterly family (radians).
fn flow_angle(x: f64, y: f64, k: f64) -> f64 {
    0.6 + 0.5 * sin(x / (97.0 * k) + y / (131.0 * k)) + 0.3 * cos(y / (53.0 * k))
}

impl Style {
    pub fn new(family: StyleFamily, seed: u64, world: &World, look: &Look, frames: usize, size: [usize; 2]) -> Self {
        let k = size_factor(size[0], size[1]);
        let mut rng = Rng::new(seed, mix64(family as u64 + 101));
        match family {
            StyleFamily::Rigid => Style::Rigid { textures: rigid_textures(&mut rng, look, world.layers(), k) },
            StyleFamily::Painterly => {
                let radius = (2.6 * k).max(1.0);
                let particles = (0..world.layers())
                    .map(|l| painterly_particles(&mut rng, world, look, l, frames, size, radius, k))
                    .collect();
                Style::Painterly { particles, stroke_radius: radius as f32, outline: (1.6 * k).max(0.8) }
            }
            StyleFamily::Hatching => {
                let angles = (0..world.layers()).map(|_| rng.uniform(0.0, std::f64::consts::PI)).collect();
                Style::Hatching { spacing: (5.0 * k).max(2.5), angles, outline: (1.6 * k).max(0.8) }
            }
        }
    }

    /// Prepares the per-frame data for a snapshot (paints the painterly canvases).
    pub fn frame(&self, snap: &Snapshot<'_>, look: &Look, size: [usize; 2]) -> FrameStyle {
        let Style::Painterly { particles, stroke_radius, .. } = self else {
            return FrameStyle { canvases: Vec::new() };
        };
        let k = size_factor(size[0], size[1]);
        let scale = CANVAS_SCALE as f32;
        let n = CANVAS_SCALE as i32;
        let canvases = particles
            .iter()
            .enumerate()
            .map(|(layer, ps)| {
                let [x0, y0, x1, y1] = layer_bounds(snap, layer, size, *stroke_radius as f64);
                let fill = look.fill(layer);
                // Canvas pixel i is at screen position i / CANVAS_SCALE; cover every sample of
                // the screen pixels x0..=x1 (offsets of up to half a pixel).
                let (cx0, cy0) = (n * x0 - n / 2, n * y0 - n / 2);
                let (cw, ch) = (n * (x1 - x0 + 1), n * (y1 - y0 + 1));
                let mut canvas = Canvas::new(cx0, cy0, cw as usize, ch as usize, fill);
                let strokes: Vec<Stroke> = ps
                    .iter()
                    .filter_map(|p| {
                        let [x, y] = snap.to_screen(layer, p.u, p.v);
                        let reach = p.half_length as f64 + *stroke_radius as f64 + 1.0;
                        if x < (x0 as f64) - reach
                            || x > x1 as f64 + reach
                            || y < y0 as f64 - reach
                            || y > y1 as f64 + reach
                        {
                            return None;
                        }
                        let a = flow_angle(x, y, k) + p.angle_jitter as f64;
                        Some(Stroke {
                            centre: [x as f32 * scale, y as f32 * scale],
                            dir: [cos(a) as f32, sin(a) as f32],
                            half_length: p.half_length * scale,
                            radius: *stroke_radius * scale,
                            colour: p.colour,
                            opacity: 1.0,
                            bristle_phase: p.bristle_phase,
                        })
                    })
                    .collect();
                canvas.paint(&strokes, false);
                canvas
            })
            .collect();
        FrameStyle { canvases }
    }

    /// Ground-truth colour of one sample at screen position `(x, y)`.
    pub fn shade(&self, snap: &Snapshot<'_>, look: &Look, frame: &FrameStyle, hit: &Hit, x: f64, y: f64) -> [f64; 3] {
        let tone = look.shade(hit.layer, hit.u, hit.v, hit.r);
        match self {
            Style::Rigid { textures } => {
                let tex = &textures[hit.layer];
                let c = tex.sample_wrapped(hit.u, hit.v);
                let lit = 0.75 + 0.35 * tone;
                let mut col = c.map(|v| v as f64 * lit);
                if hit.layer > 0 {
                    // Outline inside the boundary, in object space: it scales with the object.
                    let edge = ((hit.r - 0.9) / 0.1).clamp(0.0, 1.0);
                    col = col.map(|v| v * (1.0 - 0.8 * edge) + 20.0 * edge);
                }
                col
            }
            Style::Painterly { outline, .. } => {
                let canvas = &frame.canvases[hit.layer];
                let (px, py) = ((x * CANVAS_SCALE).round() as i32, (y * CANVAS_SCALE).round() as i32);
                let c = canvas.get(px, py).unwrap_or_else(|| look.fill(hit.layer)).map(|v| v as f64);
                screen_outline(snap, hit, c, *outline)
            }
            Style::Hatching { spacing, angles, outline } => {
                let s = &look.surfaces[hit.layer];
                let paper = s.base.map(|c| 0.35 * c + 0.65 * 245.0);
                let ink = s.palette[3].map(|c| 0.45 * c as f64);
                let t = ((tone - 0.5) / 0.6).clamp(0.0, 1.0);
                let mut clear = 1.0;
                for (i, (threshold, turn)) in [(0.85, 0.0), (0.6, 0.5), (0.35, 0.25)].into_iter().enumerate() {
                    let strength = ((threshold - t) / 0.15).clamp(0.0, 1.0);
                    if strength <= 0.0 {
                        continue;
                    }
                    let a = angles[hit.layer] + turn * std::f64::consts::PI;
                    let proj = (x * cos(a) + y * sin(a)) / spacing + 0.37 * i as f64;
                    let d = ((proj - proj.floor()) - 0.5).abs() * spacing;
                    let half_width = 0.35 + 0.75 * strength;
                    clear *= 1.0 - (half_width + 0.5 - d).clamp(0.0, 1.0) * strength;
                }
                let c = std::array::from_fn(|c| paper[c] * clear + ink[c] * (1.0 - clear));
                screen_outline(snap, hit, c, *outline)
            }
        }
    }
}

/// Darkens `c` within `width` screen pixels of an object's outline (constant width).
fn screen_outline(snap: &Snapshot<'_>, hit: &Hit, c: [f64; 3], width: f64) -> [f64; 3] {
    if hit.layer == 0 {
        return c;
    }
    let d = snap.outline_distance(hit);
    let edge = (width + 0.5 - d).clamp(0.0, 1.0);
    c.map(|v| v * (1.0 - 0.85 * edge) + 18.0 * edge)
}

/// Integer screen bounds `[x0, y0, x1, y1]` (inclusive, clipped to the frame) of a layer.
fn layer_bounds(snap: &Snapshot<'_>, layer: usize, size: [usize; 2], margin: f64) -> [i32; 4] {
    let (w, h) = (size[0] as i32, size[1] as i32);
    let Some(pose) = snap.pose(layer) else {
        return [0, 0, w - 1, h - 1];
    };
    let extent = snap.object_extent(layer) * pose.scale + margin + 2.0;
    let [cx, cy] = pose.centre();
    let x0 = ((cx - extent).floor() as i32).clamp(0, w - 1);
    let y0 = ((cy - extent).floor() as i32).clamp(0, h - 1);
    let x1 = ((cx + extent).ceil() as i32).clamp(x0, w - 1);
    let y1 = ((cy + extent).ceil() as i32).clamp(y0, h - 1);
    [x0, y0, x1, y1]
}

/// Tileable stroke textures for the rigid family, one per layer.
fn rigid_textures(rng: &mut Rng, look: &Look, layers: usize, k: f64) -> Vec<Canvas> {
    (0..layers)
        .map(|layer| {
            let size = ((512.0 * k).round() as usize).max(48) * if layer == 0 { 2 } else { 1 };
            let count = if layer == 0 { 9000 } else { 2500 };
            let palette = look.surfaces[layer].palette;
            let length = (26.0 * k).max(3.0);
            let width = (6.0 * k).max(1.2);
            let mean_angle = rng.uniform(0.0, std::f64::consts::PI);
            let strokes: Vec<Stroke> = (0..count)
                .map(|_| {
                    let a = mean_angle + 0.35 * rng.normal();
                    let jitter: [f32; 3] = std::array::from_fn(|_| rng.uniform(-18.0, 18.0) as f32);
                    let c = palette[rng.below(4)];
                    Stroke {
                        centre: [rng.uniform(0.0, size as f64) as f32, rng.uniform(0.0, size as f64) as f32],
                        dir: [cos(a) as f32, sin(a) as f32],
                        half_length: (length * rng.uniform(0.25, 0.75)) as f32,
                        radius: (width * rng.uniform(0.3, 0.7)) as f32,
                        colour: std::array::from_fn(|i| (c[i] + jitter[i]).clamp(0.0, 255.0)),
                        opacity: 1.0,
                        bristle_phase: rng.uniform(0.0, std::f64::consts::TAU) as f32,
                    }
                })
                .collect();
            let mut canvas = Canvas::new(0, 0, size, size, palette[0]);
            canvas.paint(&strokes, true);
            canvas
        })
        .collect()
}

/// Stroke anchors for one layer of the painterly family: a jittered grid in object space that
/// covers everything the layer shows during the sequence, in a fixed random paint order.
#[allow(clippy::too_many_arguments)]
fn painterly_particles(
    rng: &mut Rng,
    world: &World,
    look: &Look,
    layer: usize,
    frames: usize,
    size: [usize; 2],
    radius: f64,
    k: f64,
) -> Vec<Particle> {
    let spacing = 1.1 * radius;
    let margin = 4.0 * radius + 20.0 * k;
    let [u0, v0, u1, v1] = if layer == 0 {
        // Background object space visited by the camera over the sequence.
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for t in [0.0, (frames.max(1) - 1) as f64] {
            let c = world.camera(t);
            for i in 0..2 {
                lo[i] = lo[i].min(c[i]);
                hi[i] = hi[i].max(c[i] + size[i] as f64);
            }
        }
        [lo[0] - margin, lo[1] - margin, hi[0] + margin, hi[1] + margin]
    } else {
        let obj = &world.objects[layer - 1];
        [-obj.a * 1.25, -obj.b * 1.25, obj.a * 1.25, obj.b * 1.25]
    };
    let palette = look.surfaces[layer].palette;
    let (nu, nv) = (((u1 - u0) / spacing).ceil() as usize, ((v1 - v0) / spacing).ceil() as usize);
    let mut keyed = Vec::with_capacity(nu * nv);
    for j in 0..nv {
        for i in 0..nu {
            let u = u0 + (i as f64 + rng.unit()) * spacing;
            let v = v0 + (j as f64 + rng.unit()) * spacing;
            let r = if layer == 0 { 0.0 } else { world.objects[layer - 1].radius(u, v) };
            let order = rng.next_u64();
            let tone = 0.7 + 0.4 * look.shade(layer, u, v, r.min(1.0));
            let c = palette[rng.below(4)];
            let jitter: [f64; 3] = std::array::from_fn(|_| rng.uniform(-14.0, 14.0));
            let particle = Particle {
                u,
                v,
                colour: std::array::from_fn(|i| ((c[i] as f64 + jitter[i]) * tone).clamp(0.0, 255.0) as f32),
                half_length: (radius * rng.uniform(1.5, 3.2)) as f32,
                angle_jitter: (0.25 * rng.normal()) as f32,
                bristle_phase: rng.uniform(0.0, std::f64::consts::TAU) as f32,
            };
            if r < 1.3 {
                keyed.push((order, particle));
            }
        }
    }
    keyed.sort_by_key(|&(order, _)| order);
    keyed.into_iter().map(|(_, p)| p).collect()
}
