//! A bit-exact reimplementation of glibc's `rand()`.
//!
//! The 2019 engine draws its initial nearest-neighbour field and its per-iteration PatchMatch
//! seeds from the C library's `rand()`, never seeded (i.e. `srand(1)`). Reproducing glibc's
//! generator lets a single-threaded run match the original build bit for bit, which is how the
//! port is validated; it is also a perfectly adequate generator for this purpose.

/// glibc `random()`/`rand()` (additive feedback generator, `TYPE_3`: degree 31, separation 3).
#[derive(Clone, Debug)]
pub struct GlibcRand {
    state: [u32; 31],
    front: usize,
    rear: usize,
}

impl GlibcRand {
    /// Equivalent to `srand(seed)`. Like glibc, a seed of 0 is treated as 1.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 31];
        let seed = if seed == 0 { 1 } else { seed };
        state[0] = seed;
        let mut word = seed as i32;
        for s in state.iter_mut().skip(1) {
            // Park-Miller "minimal standard" step, computed without overflow (Schrage's method),
            // exactly as glibc's __srandom_r does.
            let hi = word / 127_773;
            let lo = word % 127_773;
            word = 16_807 * lo - 2_836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            *s = word as u32;
        }
        let mut rng = Self { state, front: 3, rear: 0 };
        for _ in 0..310 {
            rng.next_i32();
        }
        rng
    }

    /// The next value in `0..=i32::MAX`, exactly like `rand()`.
    pub fn next_i32(&mut self) -> i32 {
        let v = self.state[self.front].wrapping_add(self.state[self.rear]);
        self.state[self.front] = v;
        self.front = if self.front == 30 { 0 } else { self.front + 1 };
        self.rear = if self.rear == 30 { 0 } else { self.rear + 1 };
        (v >> 1) as i32
    }
}

impl Default for GlibcRand {
    /// The state of an unseeded C program (`srand(1)`).
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_glibc_default_sequence() {
        // First outputs of rand() in a glibc program that never calls srand().
        let mut r = GlibcRand::default();
        let expected = [1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335];
        for e in expected {
            assert_eq!(r.next_i32(), e);
        }
    }

    #[test]
    fn seed_zero_is_seed_one() {
        let mut a = GlibcRand::new(0);
        let mut b = GlibcRand::new(1);
        for _ in 0..100 {
            assert_eq!(a.next_i32(), b.next_i32());
        }
    }
}
