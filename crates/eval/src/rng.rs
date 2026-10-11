//! A small seeded random generator for pixel sampling and bootstrap resampling.

/// SplitMix64 (Steele, Lea and Flood, "Fast Splittable Pseudorandom Number Generators", 2014).
///
/// Fast, statistically adequate for resampling, and fully determined by its seed, so that every
/// sample set and confidence interval can be reproduced.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (Lemire's multiply-shift; the bias is below 2^-32 for the sizes
    /// used here). `n` must be positive.
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal deviate (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64(); // (0, 1]
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproducible_and_in_range() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
            assert!(a.below(7) < 7);
            let f = a.next_f64();
            assert!((0.0..1.0).contains(&f));
            b.below(7);
            b.next_f64();
        }
        assert_ne!(SplitMix64::new(1).next_u64(), SplitMix64::new(2).next_u64());
    }

    #[test]
    fn normal_moments() {
        let mut r = SplitMix64::new(7);
        let n = 100_000;
        let v: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let mean = v.iter().sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "{mean}");
        assert!((var - 1.0).abs() < 0.03, "{var}");
    }
}
