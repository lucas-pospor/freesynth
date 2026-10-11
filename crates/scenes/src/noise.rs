//! Procedural value noise, evaluated directly (no tables), so that it can be sampled at any
//! object-space position and gives the same value wherever the surface point appears.

use crate::rng::{hash2, unit_f64};

/// Quintic fade curve (`6t^5 - 15t^4 + 10t^3`): C2-continuous interpolation weights.
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Value noise in `[0, 1]` with unit lattice spacing.
pub fn value(seed: u64, x: f64, y: f64) -> f64 {
    let (xf, yf) = (x.floor(), y.floor());
    let (ix, iy) = (xf as i64, yf as i64);
    let (tx, ty) = (fade(x - xf), fade(y - yf));
    let corner = |dx: i64, dy: i64| unit_f64(hash2(seed, ix + dx, iy + dy));
    let top = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * tx;
    let bottom = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * tx;
    top + (bottom - top) * ty
}

/// Fractal sum of `octaves` value-noise layers, each at twice the frequency and half the
/// amplitude of the previous one, normalised to `[0, 1]`. `cell` is the lattice spacing of the
/// first octave.
pub fn fbm(seed: u64, x: f64, y: f64, cell: f64, octaves: u32) -> f64 {
    let (mut sum, mut norm, mut amp, mut freq) = (0.0, 0.0, 1.0, 1.0 / cell);
    for o in 0..octaves {
        sum += amp * value(seed.wrapping_add(o as u64 * 0x632b_e59b_d9b4_e019), x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_lattice_values_and_stays_in_range() {
        for i in 0..200 {
            let (x, y) = (i as f64 * 0.37 - 20.0, i as f64 * 0.71 - 50.0);
            let v = value(3, x, y);
            assert!((0.0..=1.0).contains(&v));
            assert!((0.0..=1.0).contains(&fbm(3, x, y, 4.0, 3)));
        }
        // At a lattice point the noise equals the corner hash.
        assert_eq!(value(9, 5.0, -2.0), unit_f64(hash2(9, 5, -2)));
    }

    #[test]
    fn is_continuous() {
        let (x, y) = (3.999_999, 1.5);
        assert!((value(1, x, y) - value(1, 4.0, y)).abs() < 1e-5);
    }
}
