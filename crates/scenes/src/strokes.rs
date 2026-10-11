//! Anti-aliased brush strokes and the canvases they are painted on.

use libm::{sinf, sqrtf};
use rayon::prelude::*;

/// An RGB float canvas covering a rectangle of integer pixel positions.
#[derive(Clone, Debug)]
pub(crate) struct Canvas {
    x0: i32,
    y0: i32,
    width: usize,
    height: usize,
    data: Vec<[f32; 3]>,
}

impl Canvas {
    /// A canvas whose top-left pixel is `(x0, y0)`, filled with `fill`.
    pub fn new(x0: i32, y0: i32, width: usize, height: usize, fill: [f32; 3]) -> Self {
        Self { x0, y0, width, height, data: vec![fill; width * height] }
    }

    /// The pixel at absolute position `(x, y)`, or `None` outside the canvas.
    pub fn get(&self, x: i32, y: i32) -> Option<[f32; 3]> {
        let (cx, cy) = (x - self.x0, y - self.y0);
        if cx < 0 || cy < 0 || cx as usize >= self.width || cy as usize >= self.height {
            return None;
        }
        Some(self.data[cy as usize * self.width + cx as usize])
    }

    /// Bilinear sample with toroidal wrap-around; `(0, 0)` is the centre of the first pixel.
    /// Used for tileable textures (the canvas origin is ignored).
    pub fn sample_wrapped(&self, x: f64, y: f64) -> [f32; 3] {
        let (w, h) = (self.width as f64, self.height as f64);
        let (xw, yw) = (x.rem_euclid(w), y.rem_euclid(h));
        let (x0, y0) = (xw.floor() as usize % self.width, yw.floor() as usize % self.height);
        let (x1, y1) = ((x0 + 1) % self.width, (y0 + 1) % self.height);
        let (fx, fy) = ((xw - xw.floor()) as f32, (yw - yw.floor()) as f32);
        let px = |x: usize, y: usize| self.data[y * self.width + x];
        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| std::array::from_fn(|c| a[c] + (b[c] - a[c]) * t);
        lerp(lerp(px(x0, y0), px(x1, y0), fx), lerp(px(x0, y1), px(x1, y1), fx), fy)
    }

    /// Paints `strokes` in order. With `wrap`, the canvas is a torus (for tileable textures).
    /// Rows are painted in parallel bands; the result does not depend on the thread count.
    pub fn paint(&mut self, strokes: &[Stroke], wrap: bool) {
        const BAND: usize = 16;
        let (x0, y0, width, height) = (self.x0, self.y0, self.width, self.height);
        self.data.par_chunks_mut(BAND * width).enumerate().for_each(|(band, rows)| {
            let row_lo = (band * BAND) as i32 + y0;
            let row_hi = row_lo + (rows.len() / width) as i32; // exclusive
            for s in strokes {
                let [bx0, by0, bx1, by1] = s.bounds();
                // Copies of the stroke that touch this band (several near a torus seam).
                let shifts: &[i32] = if wrap { &[-1, 0, 1] } else { &[0] };
                for &sy in shifts {
                    let dy = sy * height as i32;
                    let (lo, hi) = ((by0 + dy).max(row_lo), (by1 + dy + 1).min(row_hi));
                    if lo >= hi {
                        continue;
                    }
                    for &sx in shifts {
                        let dx = sx * width as i32;
                        let (cl, ch) = ((bx0 + dx).max(x0), (bx1 + dx + 1).min(x0 + width as i32));
                        for y in lo..hi {
                            let row = &mut rows[(y - row_lo) as usize * width..][..width];
                            for x in cl..ch {
                                s.blend((x - dx) as f32, (y - dy) as f32, &mut row[(x - x0) as usize]);
                            }
                        }
                    }
                }
            }
        });
    }
}

/// One straight brush stroke: a capsule with soft edges and faint bristle streaks along it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stroke {
    pub centre: [f32; 2],
    /// Unit direction.
    pub dir: [f32; 2],
    pub half_length: f32,
    pub radius: f32,
    pub colour: [f32; 3],
    pub opacity: f32,
    /// Phase of the bristle streaks, in radians.
    pub bristle_phase: f32,
}

impl Stroke {
    /// Integer pixel bounds `[x0, y0, x1, y1]` (inclusive) of everything the stroke touches.
    pub fn bounds(&self) -> [i32; 4] {
        let ex = self.dir[0].abs() * self.half_length + self.radius + 1.0;
        let ey = self.dir[1].abs() * self.half_length + self.radius + 1.0;
        [
            (self.centre[0] - ex).floor() as i32,
            (self.centre[1] - ey).floor() as i32,
            (self.centre[0] + ex).ceil() as i32,
            (self.centre[1] + ey).ceil() as i32,
        ]
    }

    /// Blends the stroke into `dst` at pixel centre `(x, y)`.
    #[inline]
    fn blend(&self, x: f32, y: f32, dst: &mut [f32; 3]) {
        let (dx, dy) = (x - self.centre[0], y - self.centre[1]);
        let along = dx * self.dir[0] + dy * self.dir[1];
        let across = dy * self.dir[0] - dx * self.dir[1];
        let off = along - along.clamp(-self.half_length, self.half_length);
        let d = sqrtf(off * off + across * across);
        let coverage = (self.radius + 0.5 - d).clamp(0.0, 1.0) * self.opacity;
        if coverage <= 0.0 {
            return;
        }
        let streak = 1.0 + 0.12 * sinf(across / self.radius.max(0.5) * 4.7 + self.bristle_phase);
        for (d, &c) in dst.iter_mut().zip(&self.colour) {
            *d += (c * streak - *d) * coverage;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stroke(x: f32, y: f32) -> Stroke {
        Stroke {
            centre: [x, y],
            dir: [1.0, 0.0],
            half_length: 3.0,
            radius: 1.5,
            colour: [200.0; 3],
            opacity: 1.0,
            bristle_phase: 0.0,
        }
    }

    #[test]
    fn paints_inside_and_not_outside() {
        let mut c = Canvas::new(10, 20, 40, 40, [0.0; 3]);
        c.paint(&[stroke(30.0, 40.0)], false);
        assert!(c.get(30, 40).unwrap()[0] > 150.0);
        assert_eq!(c.get(30, 45).unwrap(), [0.0; 3]);
        assert_eq!(c.get(0, 0), None);
    }

    #[test]
    fn wraps_across_the_seam() {
        let mut c = Canvas::new(0, 0, 32, 32, [0.0; 3]);
        c.paint(&[stroke(0.0, 0.0)], true);
        // The stroke centred on the corner shows up on all four corners of the torus.
        for (x, y) in [(0, 0), (31, 0), (0, 31), (31, 31)] {
            assert!(c.get(x, y).unwrap()[0] > 50.0, "corner {x},{y}");
        }
        assert_eq!(c.get(16, 16).unwrap(), [0.0; 3]);
    }

    #[test]
    fn wrapped_sampling_is_periodic() {
        let mut c = Canvas::new(0, 0, 16, 8, [0.0; 3]);
        c.paint(&[stroke(5.0, 3.0)], true);
        let a = c.sample_wrapped(4.3, 2.6);
        let b = c.sample_wrapped(4.3 + 16.0 * 3.0, 2.6 - 8.0);
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-3);
        }
    }
}
