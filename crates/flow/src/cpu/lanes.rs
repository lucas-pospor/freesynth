//! Fixed-width groups of `f32` lanes for the hot loops.
//!
//! The operations work lane by lane on values (not on references into shared buffers), so the
//! compiler can see that nothing aliases and turns each operation into vector instructions of
//! whatever width the target allows. Every operation is the plain scalar operation in each lane
//! (fused multiply-adds included, which are exactly rounded everywhere), so results do not
//! depend on the instruction set.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// `n`, hidden from the optimiser.
///
/// Loops over the rows of a tile (and over column parities) use it as their bound so that the
/// compiler does not unroll them completely. After a complete unroll, the vectoriser tends to
/// group values from different rows into vectors, assembled one element at a time, which is
/// several times slower than vectorising the lanes of each row.
#[inline(always)]
pub(crate) fn opaque(n: usize) -> usize {
    std::hint::black_box(n)
}

/// `N` lanes of `f32`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub(crate) struct Lanes<const N: usize>(pub [f32; N]);

/// Eight lanes: the rows of a patch.
pub(crate) type F8 = Lanes<8>;
/// Sixteen lanes: one colour of a refinement tile row.
pub(crate) type F16 = Lanes<16>;

impl<const N: usize> Lanes<N> {
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self([v; N])
    }

    /// The first `N` values of `s`.
    #[inline(always)]
    pub fn load(s: &[f32]) -> Self {
        Self(s[..N].try_into().expect("slice holds N values"))
    }

    /// Writes the lanes to the first `N` values of `s`.
    #[inline(always)]
    pub fn store(self, s: &mut [f32]) {
        s[..N].copy_from_slice(&self.0);
    }

    /// Lanes from a function of the lane index.
    #[inline(always)]
    pub fn from_fn(f: impl Fn(usize) -> f32) -> Self {
        let mut r = [0.0; N];
        for (i, v) in r.iter_mut().enumerate() {
            *v = f(i);
        }
        Self(r)
    }

    /// Applies `f` to every lane.
    #[inline(always)]
    pub fn map(self, f: impl Fn(f32) -> f32) -> Self {
        let mut r = self.0;
        for v in &mut r {
            *v = f(*v);
        }
        Self(r)
    }

    #[inline(always)]
    fn zip(self, o: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        let mut r = self.0;
        for (v, w) in r.iter_mut().zip(&o.0) {
            *v = f(*v, *w);
        }
        Self(r)
    }

    /// Lane-wise `self * a + b` with a single rounding.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        let mut r = self.0;
        for ((v, a), b) in r.iter_mut().zip(&a.0).zip(&b.0) {
            *v = v.mul_add(*a, *b);
        }
        Self(r)
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        self.map(f32::sqrt)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        self.map(f32::abs)
    }

    /// Lane-wise `if self > o { self } else { o }`.
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        self.zip(o, |a, b| if a > b { a } else { b })
    }

    /// WGSL's `mix` with one weight for all lanes: `a * (1 - t) + b * t`.
    #[inline(always)]
    pub fn mix(a: Self, b: Self, t: f32) -> Self {
        b.mul_add(Self::splat(t), a * Self::splat(1.0 - t))
    }

    /// WGSL's `mix` with a weight per lane.
    #[inline(always)]
    pub fn mix_lanes(a: Self, b: Self, t: Self) -> Self {
        b.mul_add(t, a * (Self::splat(1.0) - t))
    }
}

impl F8 {
    /// Sum of the lanes in the order of the GPU kernels' reduction over a patch's rows.
    #[inline(always)]
    pub fn tree_sum(self) -> f32 {
        let v = self.0;
        ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))
    }
}

macro_rules! binary {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<const N: usize> $trait for Lanes<N> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, o: Self) -> Self {
                self.zip(o, |a, b| a $op b)
            }
        }
    };
}
binary!(Add, add, +);
binary!(Sub, sub, -);
binary!(Mul, mul, *);
binary!(Div, div, /);

impl<const N: usize> Neg for Lanes<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.map(|a| -a)
    }
}

impl<const N: usize> AddAssign for Lanes<N> {
    #[inline(always)]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_operations() {
        let a = F8::from_fn(|i| i as f32);
        let b = F8::splat(2.0);
        assert_eq!((a + b).0[3], 5.0);
        assert_eq!((a - b).0[0], -2.0);
        assert_eq!((a * b).0[7], 14.0);
        assert_eq!((a / b).0[1], 0.5);
        assert_eq!((-a).0[2], -2.0);
        assert_eq!(a.max(b).0, [2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(a.mul_add(b, b).0[3], 8.0);
        assert_eq!(a.tree_sum(), 28.0);
        let mut buf = [0.0; 10];
        a.store(&mut buf[1..]);
        assert_eq!(F8::load(&buf[1..]), a);
    }
}
