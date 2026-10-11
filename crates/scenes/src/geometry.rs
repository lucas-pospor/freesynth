//! Scene geometry: a panning background plane and a few flat objects in front of it, each with
//! a smooth rigid-plus-scale motion. Everything is analytic, so the layer visible at any point
//! and the motion of any surface point are known exactly.
//!
//! Coordinates: screen positions are in pixels with pixel centres at integer coordinates.
//! Every layer has its own 2-D "object space" `(u, v)`: for the background, screen position
//! plus the camera offset; for an object, screen position relative to its centre, rotated by
//! minus its angle and divided by its scale. Surface properties (colour, texture, strokes) are
//! defined in object space, so a surface point keeps them while it moves.

use libm::{atan2, cos, sin, sqrt};

use crate::rng::Rng;

/// Outline of an object, as a normalised radius `r(u, v)` that is below 1 inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Ellipse,
    /// `|u/a|^4 + |v/b|^4 < 1`, a rounded rectangle.
    Superellipse,
    /// An ellipse with five lobes.
    Star,
}

impl Shape {
    const ALL: [Shape; 3] = [Shape::Ellipse, Shape::Superellipse, Shape::Star];
}

/// One moving object.
#[derive(Clone, Debug)]
pub(crate) struct Object {
    pub shape: Shape,
    /// Semi-axes in object units (pixels at scale 1).
    pub a: f64,
    pub b: f64,
    centre0: [f64; 2],
    drift: [f64; 2],
    swing: [f64; 2],
    swing_freq: f64,
    swing_phase: f64,
    wobble: [f64; 2],
    theta0: f64,
    omega: f64,
    scale_amp: f64,
    scale_freq: f64,
    scale_phase: f64,
}

impl Object {
    /// Normalised radius: `< 1` inside the object, `1` on its outline.
    pub fn radius(&self, u: f64, v: f64) -> f64 {
        let (nu, nv) = (u / self.a, v / self.b);
        match self.shape {
            Shape::Ellipse => sqrt(nu * nu + nv * nv),
            Shape::Superellipse => {
                let (u2, v2) = (nu * nu, nv * nv);
                sqrt(sqrt(u2 * u2 + v2 * v2))
            }
            Shape::Star => sqrt(nu * nu + nv * nv) / (0.75 + 0.25 * cos(5.0 * atan2(v, u))),
        }
    }

    /// Largest distance from the centre to the outline, in object units.
    pub fn extent(&self) -> f64 {
        self.a.max(self.b)
    }

    fn pose(&self, t: f64) -> Pose {
        let phase = self.swing_freq * t + self.swing_phase;
        let cx = self.centre0[0] + self.drift[0] * t + self.swing[0] * sin(phase) + self.wobble[0] * sin(0.13 * t);
        let cy = self.centre0[1] + self.drift[1] * t + self.swing[1] * cos(phase) + self.wobble[1] * cos(0.11 * t);
        let theta = self.theta0 + self.omega * t;
        let scale = 1.0 + self.scale_amp * sin(self.scale_freq * t + self.scale_phase);
        Pose { centre: [cx, cy], cos: cos(theta), sin: sin(theta), scale }
    }
}

/// Position, orientation and scale of an object at one instant.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pose {
    centre: [f64; 2],
    cos: f64,
    sin: f64,
    pub scale: f64,
}

impl Pose {
    pub fn to_local(self, x: f64, y: f64) -> [f64; 2] {
        let (dx, dy) = (x - self.centre[0], y - self.centre[1]);
        [(self.cos * dx + self.sin * dy) / self.scale, (-self.sin * dx + self.cos * dy) / self.scale]
    }

    pub fn to_screen(self, u: f64, v: f64) -> [f64; 2] {
        [
            self.centre[0] + self.scale * (self.cos * u - self.sin * v),
            self.centre[1] + self.scale * (self.sin * u + self.cos * v),
        ]
    }

    pub fn centre(self) -> [f64; 2] {
        self.centre
    }
}

/// The surface point seen at a screen position.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hit {
    /// 0 for the background, `1 + i` for object `i`.
    pub layer: usize,
    pub u: f64,
    pub v: f64,
    /// Normalised radius inside the object (0 for the background).
    pub r: f64,
    /// Screen pixels per object unit (1 for the background).
    pub scale: f64,
}

/// The static description of a scene's geometry and motion.
#[derive(Clone, Debug)]
pub(crate) struct World {
    camera_velocity: [f64; 2],
    pub objects: Vec<Object>,
}

/// Size factor relative to the 480x270 reference size, so scenes look alike at any size.
pub(crate) fn size_factor(width: usize, height: usize) -> f64 {
    sqrt((width * height) as f64 / (480.0 * 270.0))
}

impl World {
    /// Draws a random world. `motion` scales every speed (1 is the reference).
    pub fn random(rng: &mut Rng, width: usize, height: usize, objects: usize, motion: f64) -> Self {
        let (w, h) = (width as f64, height as f64);
        let k = size_factor(width, height);
        let camera_velocity = [rng.uniform(-2.0, 2.0) * k * motion, rng.uniform(-1.2, 1.2) * k * motion];
        let objects = (0..objects)
            .map(|i| Object {
                shape: Shape::ALL[i % Shape::ALL.len()],
                a: h * rng.uniform(0.20, 0.31),
                b: h * rng.uniform(0.15, 0.26),
                centre0: [w * rng.uniform(0.2, 0.8), h * rng.uniform(0.25, 0.75)],
                drift: [rng.uniform(-0.8, 0.8) * k * motion, rng.uniform(-0.4, 0.4) * k * motion],
                swing: [w * rng.uniform(0.06, 0.14), h * rng.uniform(0.04, 0.10)],
                swing_freq: std::f64::consts::TAU / rng.uniform(90.0, 150.0) * motion,
                swing_phase: rng.uniform(0.0, std::f64::consts::TAU),
                wobble: [12.0 * k * motion, 8.0 * k * motion],
                theta0: rng.uniform(0.0, std::f64::consts::PI),
                omega: rng.sign() * rng.uniform(0.012, 0.035) * motion,
                scale_amp: rng.uniform(0.04, 0.15),
                scale_freq: rng.uniform(0.04, 0.12) * motion,
                scale_phase: rng.uniform(0.0, std::f64::consts::TAU),
            })
            .collect();
        Self { camera_velocity, objects }
    }

    /// Number of layers, background included.
    pub fn layers(&self) -> usize {
        self.objects.len() + 1
    }

    /// Camera offset at time `t`: background object space is screen space plus this offset.
    pub fn camera(&self, t: f64) -> [f64; 2] {
        [self.camera_velocity[0] * t, self.camera_velocity[1] * t]
    }

    /// Poses of everything at time `t` (in frames).
    pub fn at(&self, t: f64) -> Snapshot<'_> {
        Snapshot { world: self, camera: self.camera(t), poses: self.objects.iter().map(|o| o.pose(t)).collect() }
    }
}

/// The world frozen at one instant.
pub(crate) struct Snapshot<'a> {
    world: &'a World,
    camera: [f64; 2],
    poses: Vec<Pose>,
}

impl Snapshot<'_> {
    /// The front-most surface point at a screen position. Later objects are in front.
    pub fn hit(&self, x: f64, y: f64) -> Hit {
        for (i, (obj, pose)) in self.world.objects.iter().zip(&self.poses).enumerate().rev() {
            let [u, v] = pose.to_local(x, y);
            let r = obj.radius(u, v);
            if r < 1.0 {
                return Hit { layer: i + 1, u, v, r, scale: pose.scale };
            }
        }
        Hit { layer: 0, u: x + self.camera[0], v: y + self.camera[1], r: 0.0, scale: 1.0 }
    }

    /// Screen position of the object-space point `(u, v)` of `layer`.
    pub fn to_screen(&self, layer: usize, u: f64, v: f64) -> [f64; 2] {
        match layer {
            0 => [u - self.camera[0], v - self.camera[1]],
            l => self.poses[l - 1].to_screen(u, v),
        }
    }

    /// Largest centre-to-outline distance of an object layer, in object units.
    pub fn object_extent(&self, layer: usize) -> f64 {
        self.world.objects[layer - 1].extent()
    }

    pub fn pose(&self, layer: usize) -> Option<&Pose> {
        layer.checked_sub(1).map(|i| &self.poses[i])
    }

    /// Distance in screen pixels from an object hit to that object's outline (approximate:
    /// first order in the normalised radius).
    pub fn outline_distance(&self, hit: &Hit) -> f64 {
        let obj = &self.world.objects[hit.layer - 1];
        let h = 0.25;
        let gu = (obj.radius(hit.u + h, hit.v) - obj.radius(hit.u - h, hit.v)) / (2.0 * h);
        let gv = (obj.radius(hit.u, hit.v + h) - obj.radius(hit.u, hit.v - h)) / (2.0 * h);
        let g = sqrt(gu * gu + gv * gv).max(1e-9);
        (1.0 - hit.r) * hit.scale / g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> World {
        World::random(&mut Rng::new(5, 0), 480, 270, 3, 1.0)
    }

    #[test]
    fn local_and_screen_are_inverse() {
        let w = world();
        let snap = w.at(7.5);
        for layer in 0..w.layers() {
            let (x, y) = (123.25, 87.5);
            let [u, v] = match snap.pose(layer) {
                Some(p) => p.to_local(x, y),
                None => [x + snap.camera[0], y + snap.camera[1]],
            };
            let [x2, y2] = snap.to_screen(layer, u, v);
            assert!((x2 - x).abs() < 1e-9 && (y2 - y).abs() < 1e-9);
        }
    }

    #[test]
    fn hits_the_front_object() {
        let w = world();
        let snap = w.at(0.0);
        let front = w.objects.len();
        let c = snap.poses[front - 1].centre();
        assert_eq!(snap.hit(c[0], c[1]).layer, front);
        // Far outside every object: background.
        assert_eq!(snap.hit(-1e5, -1e5).layer, 0);
    }

    #[test]
    fn outline_distance_of_a_circle() {
        let mut w = world();
        w.objects.truncate(1);
        let obj = &mut w.objects[0];
        obj.shape = Shape::Ellipse;
        obj.a = 40.0;
        obj.b = 40.0;
        let snap = w.at(0.0);
        let pose = snap.poses[0];
        let [x, y] = pose.to_screen(30.0, 0.0);
        let hit = snap.hit(x, y);
        assert_eq!(hit.layer, 1);
        assert!((snap.outline_distance(&hit) - 10.0 * pose.scale).abs() < 1e-3);
    }
}
