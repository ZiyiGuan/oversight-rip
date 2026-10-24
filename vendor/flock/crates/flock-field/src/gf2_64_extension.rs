// RMIP field64: quadratic extension adapted from the default Flock tower.
//! GF(2^128) as a quadratic extension of [`F128`].
//!
//! Write elements as `a_0 + a_1 u`, with
//!
//! ```text
//! u^2 + u + x^-1 = 0,
//! ```
//!
//! where `x` is the 64-bit polynomial-basis generator of `F128`. The absolute trace of
//! `x^-1` is one, so the quadratic is irreducible. Multiplication uses three
//! base-field products; multiplication by `x^-1` is a linear shift-and-fold.

use core::{
    mem::{align_of, size_of},
    ops::{Add, AddAssign, Mul, MulAssign},
};

use serde::{Deserialize, Serialize};

use crate::gf2_128::F128;

/// Multiply a base-field element by x^-1 for x^64+x^4+x^3+x+1.
#[inline]
pub const fn mul_by_x_inv(z: F128) -> F128 {
    let mask = 0u64.wrapping_sub(z.lo & 1);
    F128::from_u64((z.lo >> 1) ^ (0x8000_0000_0000_000d & mask))
}

/// Trace-one Artin--Schreier constant. Its trace is independently tested.
pub const QUADRATIC_NONRESIDUE: F128 = F128::from_u64(0x8000_0000_0000_000d);

/// An element `c_0 + c_1 u` of GF(2^128).
///
/// Protocol commitments do not commit this type directly. They split values
/// into the two canonical F128 coordinates and encode those coordinates as
/// separate base-field rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct F256 {
    pub c0: F128,
    pub c1: F128,
}

impl F256 {
    // RMIP: width-aware constants shared with the optional smaller tower.
    pub const BITS: usize = 128;
    pub const BYTES: usize = 16;
    pub const ZERO: Self = Self {
        c0: F128::ZERO,
        c1: F128::ZERO,
    };
    pub const ONE: Self = Self {
        c0: F128::ONE,
        c1: F128::ZERO,
    };
    pub const U: Self = Self {
        c0: F128::ZERO,
        c1: F128::ONE,
    };

    #[inline]
    pub const fn new(c0: F128, c1: F128) -> Self {
        Self { c0, c1 }
    }

    #[inline]
    pub const fn from_base(c0: F128) -> Self {
        Self { c0, c1: F128::ZERO }
    }

    /// Return the two canonical base-field coordinates `(c0, c1)`.
    #[inline]
    pub const fn coordinates(self) -> [F128; 2] {
        [self.c0, self.c1]
    }
}

impl From<F128> for F256 {
    #[inline]
    fn from(value: F128) -> Self {
        Self::from_base(value)
    }
}

impl Add for F256 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
        }
    }
}

impl AddAssign for F256 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.c0 += rhs.c0;
        self.c1 += rhs.c1;
    }
}

impl Mul for F256 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Karatsuba with u^2 = u + x^-1:
        // p0 = a0*b0, p1 = a1*b1, p2 = (a0+a1)*(b0+b1)
        // c0 = p0 + x^-1*p1
        // c1 = (a0*b1+a1*b0) + p1 = p2 + p0.
        let p0 = self.c0 * rhs.c0;
        let p1 = self.c1 * rhs.c1;
        let p2 = (self.c0 + self.c1) * (rhs.c0 + rhs.c1);
        Self {
            c0: p0 + mul_by_x_inv(p1),
            c1: p2 + p0,
        }
    }
}

impl MulAssign for F256 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Mul<F128> for F256 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F128) -> Self {
        Self {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}

impl MulAssign<F128> for F256 {
    #[inline]
    fn mul_assign(&mut self, rhs: F128) {
        self.c0 *= rhs;
        self.c1 *= rhs;
    }
}

impl Mul<F256> for F128 {
    type Output = F256;

    #[inline]
    fn mul(self, rhs: F256) -> F256 {
        rhs * self
    }
}

const _: [(); 16] = [(); size_of::<F256>()];
const _: [(); 8] = [(); align_of::<F256>()];

#[cfg(test)]
mod tests {
    use crate::{
        gf2_256::{F128, F256, QUADRATIC_NONRESIDUE, mul_by_x_inv},
        mul_by_x,
        test_rng::Rng,
    };

    fn absolute_trace(mut a: F128) -> F128 {
        let mut out = F128::ZERO;
        for _ in 0..F128::BITS {
            out += a;
            a *= a;
        }
        out
    }

    #[test]
    fn x_inverse_is_the_quadratic_nonresidue() {
        let mut rng = Rng(1);
        for _ in 0..64 {
            let a = rng.f128();
            assert_eq!(mul_by_x(mul_by_x_inv(a)), a);
            assert_eq!(mul_by_x_inv(mul_by_x(a)), a);
        }
        assert_eq!(mul_by_x(QUADRATIC_NONRESIDUE), F128::ONE);
        assert_eq!(absolute_trace(QUADRATIC_NONRESIDUE), F128::ONE);
    }

    #[test]
    fn quadratic_relation_and_karatsuba_match_reference() {
        assert_eq!(
            F256::U * F256::U,
            F256::U + F256::from(QUADRATIC_NONRESIDUE)
        );
        let mut rng = Rng(2);
        for _ in 0..64 {
            let a = rng.f256();
            let b = rng.f256();
            let reference = F256::new(
                a.c0 * b.c0 + mul_by_x_inv(a.c1 * b.c1),
                a.c0 * b.c1 + a.c1 * b.c0 + a.c1 * b.c1,
            );
            assert_eq!(a * b, reference);
        }
    }

    #[test]
    fn field_identities_and_frobenius_order() {
        let mut rng = Rng(3);
        for _ in 0..16 {
            let a = rng.f256();
            let b = rng.f256();
            let c = rng.f256();
            assert_eq!(a + F256::ZERO, a);
            assert_eq!(a * F256::ONE, a);
            assert_eq!(a * (b + c), a * b + a * c);

            let mut conjugate = a;
            for _ in 0..F128::BITS {
                conjugate *= conjugate;
            }
            assert_eq!(conjugate, F256::new(a.c0 + a.c1, a.c1));

            let mut full = conjugate;
            for _ in 0..F128::BITS {
                full *= full;
            }
            assert_eq!(full, a);
        }
    }

    #[test]
    fn coordinates_roundtrip() {
        let value = F256::new(F128::from_u64(1), F128::from_u64(3));
        let [c0, c1] = value.coordinates();
        assert_eq!(F256::new(c0, c1), value);
    }
}
