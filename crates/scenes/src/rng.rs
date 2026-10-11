//! Small deterministic random number generation.
//!
//! Scenes must be reproducible from their seed alone, on every platform and with every version
//! of the dependencies, so the generator is defined here instead of coming from a crate:
//! SplitMix64 (Steele, Lea and Flood, 2014) for streams and a stateless 64-bit mix for lattice
//! hashing.

/// The SplitMix64 finaliser: a bijective 64-bit mix with good avalanche behaviour.
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Hashes a seed and two lattice coordinates to 64 random bits.
pub fn hash2(seed: u64, x: i64, y: i64) -> u64 {
    let h = mix64(seed ^ (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    mix64(h ^ (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f))
}

/// Maps 64 random bits to a float in `[0, 1)` using the top 53 bits.
pub fn unit_f64(bits: u64) -> f64 {
    (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// A SplitMix64 stream.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A stream for `seed`. Different `stream` values give independent streams for one seed.
    pub fn new(seed: u64, stream: u64) -> Self {
        Self { state: mix64(seed ^ mix64(stream.wrapping_add(0x6a09_e667_f3bc_c909))) }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    /// A float in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        unit_f64(self.next_u64())
    }

    /// A float in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// An integer in `0..n` (`n > 0`). The modulo bias is below 2^-40 for the sizes used here.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// `-1.0` or `1.0` with equal probability.
    pub fn sign(&mut self) -> f64 {
        if self.next_u64() >> 63 == 0 { -1.0 } else { 1.0 }
    }

    /// A standard normal sample (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.unit(); // in (0, 1]
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_values() {
        // SplitMix64 with state 0 advanced once: the published first output of the generator.
        let mut z = 0u64.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = mix64(z);
        assert_eq!(z, 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn streams_are_reproducible_and_distinct() {
        let a: Vec<u64> = (0..4).scan(Rng::new(7, 0), |r, _| Some(r.next_u64())).collect();
        let b: Vec<u64> = (0..4).scan(Rng::new(7, 0), |r, _| Some(r.next_u64())).collect();
        let c: Vec<u64> = (0..4).scan(Rng::new(7, 1), |r, _| Some(r.next_u64())).collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn unit_range_and_mean() {
        let mut r = Rng::new(1, 2);
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
            sum += u;
        }
        assert!((sum / n as f64 - 0.5).abs() < 0.01);
    }
}
