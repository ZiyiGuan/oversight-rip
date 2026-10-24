use crate::ghash;
use crate::BinaryField;

pub use crate::ghash::{F256Unreduced, F128};
pub use crate::reference::REDUCTION;

impl F128 {
    #[inline]
    pub const fn new(v: u128) -> Self {
        Self::from_parts(v as u64, (v >> 64) as u64)
    }

    #[inline]
    pub const fn to_u128(self) -> u128 {
        ((self.hi as u128) << 64) | self.lo as u128
    }

    pub const X: Self = Self::from_parts(2, 0);

    #[inline]
    pub fn mul_schoolbook(self, rhs: Self) -> Self {
        ghash::ghash_mul_schoolbook(self, rhs)
    }
}

impl BinaryField for F128 {
    const ZERO: Self = Self::from_parts(0, 0);
    const ONE: Self = Self::from_parts(1, 0);
    const KAPPA: usize = 128;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_parts(self.lo ^ rhs.lo, self.hi ^ rhs.hi)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn square(self) -> Self {
        ghash::ghash_square(self)
    }

    fn inv(self) -> Option<Self> {
        if self == <Self as BinaryField>::ZERO {
            return None;
        }
        let mut result = <Self as BinaryField>::ONE;
        let mut base = self;
        for i in 0..128 {
            if i > 0 {
                result *= base;
            }
            base = ghash::ghash_square(base);
        }
        Some(result)
    }

    fn pow(self, mut exp: u128) -> Self {
        let mut result = <Self as BinaryField>::ONE;
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            base = ghash::ghash_square(base);
            exp >>= 1;
        }
        result
    }

    #[inline]
    fn to_be_bytes(self) -> [u8; 16] {
        self.to_u128().to_be_bytes()
    }

    #[inline]
    fn from_be_bytes(b: [u8; 16]) -> Self {
        Self::new(u128::from_be_bytes(b))
    }
}

impl std::fmt::Debug for F128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "F128({:#034x})", self.to_u128())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ghash::mul_by_x;
    use crate::reference;

    fn rng(seed: u64) -> impl FnMut() -> u128 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let a = s;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((a as u128) << 64) | s as u128
        }
    }

    const EDGES: [u128; 8] = [0, 1, 2, 1 << 63, 1 << 64, 1 << 127, u128::MAX, REDUCTION];

    #[test]
    fn differential_agreement_with_the_portable_reference() {
        let mut next = rng(0x243f_6a88_85a3_08d3);
        for _ in 0..20_000 {
            let (a, b) = (next(), next());
            assert_eq!(
                (F128::new(a) * F128::new(b)).to_u128(),
                reference::mul(a, b),
                "mul {a:#x} * {b:#x}"
            );
            assert_eq!(
                F128::new(a).square().to_u128(),
                reference::square(a),
                "square {a:#x}"
            );
        }
        for a in EDGES {
            for b in EDGES {
                assert_eq!(
                    (F128::new(a) * F128::new(b)).to_u128(),
                    reference::mul(a, b),
                    "{a:#x}*{b:#x}"
                );
            }
            assert_eq!(
                F128::new(a).square().to_u128(),
                reference::square(a),
                "square {a:#x}"
            );
        }
    }

    #[test]
    fn every_variant_agrees() {
        let mut next = rng(0x9e37_79b9_7f4a_7c15);
        let check = |a: u128, b: u128| {
            let (x, y) = (F128::new(a), F128::new(b));
            let want = (x * y).to_u128();
            assert_eq!(
                x.mul_schoolbook(y).to_u128(),
                want,
                "schoolbook {a:#x}*{b:#x}"
            );
            assert_eq!(
                x.mul_unreduced(y).reduce().to_u128(),
                want,
                "deferred {a:#x}*{b:#x}"
            );
            assert_eq!(
                reference::mul(a, b),
                want,
                "reference schoolbook {a:#x}*{b:#x}"
            );
            assert_eq!(
                reference::mul_karatsuba(a, b),
                want,
                "reference karatsuba {a:#x}*{b:#x}"
            );
        };
        for _ in 0..20_000 {
            let (a, b) = (next(), next());
            check(a, b);
        }
        for a in EDGES {
            for b in EDGES {
                check(a, b);
            }
        }
    }

    #[test]
    fn deferred_reduction_is_reduction_of_the_sum() {
        let mut next = rng(0xbb67_ae85_84ca_a73b);
        for _ in 0..500 {
            let mut acc = F256Unreduced::ZERO;
            let mut sum = <F128 as BinaryField>::ZERO;
            for _ in 0..64 {
                let (a, b) = (F128::new(next()), F128::new(next()));
                acc ^= a.mul_unreduced(b);
                sum = sum.add(a * b);
            }
            assert_eq!(acc.reduce(), sum);
        }
    }

    #[test]
    fn mul_by_x_is_multiplication_by_x() {
        let mut next = rng(0x3c6e_f372_fe94_f82b);
        for _ in 0..20_000 {
            let a = F128::new(next());
            assert_eq!(mul_by_x(a), a * F128::X, "{a:?}");
        }
        for a in EDGES {
            assert_eq!(mul_by_x(F128::new(a)), F128::new(a) * F128::X, "{a:#x}");
        }
    }

    #[test]
    fn reduction_polynomial_is_correct() {
        assert_eq!(F128::X.pow(128), F128::new(REDUCTION));
        assert_eq!(F128::X.pow(127), F128::new(1u128 << 127));
    }

    #[test]
    fn multiplicative_order_divides_group_order() {
        for a in [
            F128::new(2),
            F128::new(3),
            F128::new(0xdead_beef),
            F128::new(u128::MAX),
        ] {
            assert_eq!(a.pow(u128::MAX), <F128 as BinaryField>::ONE, "{a:?}");
        }
    }

    #[test]
    fn every_high_bit_folds_to_the_right_power_of_x() {
        let times_x = |v: u128| (v << 1) ^ if v >> 127 == 1 { REDUCTION } else { 0 };
        let mut want = REDUCTION;
        for i in 0..128u32 {
            assert_eq!(
                reference::reduce(1u128 << i, 0),
                want,
                "reference x^(128+{i})"
            );
            assert_eq!(
                F128::X.pow(128 + i as u128).to_u128(),
                want,
                "adopted x^(128+{i})"
            );
            want = times_x(want);
        }
        assert_eq!(reference::reduce(0, 0x1234_5678), 0x1234_5678);
    }

    #[test]
    fn the_wire_format_did_not_move() {
        let mut next = rng(0xa54f_f53a_5f1d_36f1);
        for _ in 0..20_000 {
            let v = next();
            assert_eq!(F128::new(v).to_be_bytes(), v.to_be_bytes(), "{v:#x}");
            assert_eq!(F128::from_be_bytes(v.to_be_bytes()).to_u128(), v, "{v:#x}");
        }
    }
}
