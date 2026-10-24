//! RMIP experimental base field GF(2^64), modulo x^64+x^4+x^3+x+1.
//!
//! `F128` is the legacy API name only: the representation is exactly eight
//! bytes. This module is selected exclusively by the `field64` build feature.
//! Arithmetic is implemented independently of the 128-bit GHASH kernels.
//!
//! The Itoh–Tsujii chain and ARM immediate-product dispatch follow
//! leanEthereum/leanVM commit a0ca7858f2533ce51abdc20bb635c007e19b26b3,
//! crates/primitives/src/field/gf2_64.rs. Copyright (c) 2026 leanEthereum.
//! Upstream supplies MIT and Apache-2.0 licenses and credits binius64
//! (Apache-2.0) for GF64 arithmetic. See vendor/flock/PROVENANCE.md.

use core::ops::{Add, AddAssign, BitXor, BitXorAssign, Mul, MulAssign};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct F128 {
    pub lo: u64,
}

impl F128 {
    pub const BITS: usize = 64;
    pub const BYTES: usize = 8;
    pub const LOG_BITS: usize = 6;
    pub const ZERO: Self = Self { lo: 0 };
    pub const ONE: Self = Self { lo: 1 };

    /// Compatibility constructor. A nonzero high limb is an error, never
    /// silently discarded: packing and sampling must use the active width.
    #[inline]
    pub const fn new(lo: u64, hi: u64) -> Self {
        assert!(hi == 0, "GF(2^64) cannot represent a nonzero high limb");
        Self { lo }
    }

    #[inline]
    pub const fn from_u64(lo: u64) -> Self {
        Self { lo }
    }

    pub const fn from_u128(bits: u128) -> Self {
        assert!(bits >> 64 == 0, "GF(2^64) integer representation overflows");
        Self::from_u64(bits as u64)
    }

    /// Reduce a polynomial of degree below 128. This is deliberately named
    /// differently from checked integer packing, which must be injective.
    pub const fn from_polynomial(bits: u128) -> Self {
        let hi = (bits >> 64) as u64;
        let overflow = (hi >> 60) ^ (hi >> 61) ^ (hi >> 63);
        Self::from_u64(
            bits as u64
                ^ hi
                ^ (hi << 1)
                ^ (hi << 3)
                ^ (hi << 4)
                ^ overflow
                ^ (overflow << 1)
                ^ (overflow << 3)
                ^ (overflow << 4),
        )
    }

    pub const fn to_u128(self) -> u128 {
        self.lo as u128
    }

    pub const fn high_u64(self) -> u64 {
        0
    }

    pub const fn bit(self, index: usize) -> bool {
        assert!(index < Self::BITS, "field bit index out of range");
        ((self.lo >> index) & 1) != 0
    }

    pub const fn with_bit(self, index: usize, value: bool) -> Self {
        assert!(index < Self::BITS, "field bit index out of range");
        let mask = 1u64 << index;
        Self::from_u64((self.lo & !mask) | ((value as u64) << index))
    }

    pub const fn to_le_bytes(self) -> [u8; Self::BYTES] {
        self.lo.to_le_bytes()
    }

    pub const fn from_le_bytes(bytes: [u8; Self::BYTES]) -> Self {
        Self::from_u64(u64::from_le_bytes(bytes))
    }

    #[inline]
    pub const fn generator() -> Self {
        Self { lo: 2 }
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.lo == 0
    }

    #[inline]
    pub fn square(self) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            // SAFETY: the compile-time gate guarantees NEON and PMULL.
            Self::from_u64(unsafe { crate::f64_neon::mul_scalar(self.lo, self.lo) })
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            self.mul_unreduced(self).reduce()
        }
    }

    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> F256Unreduced {
        let (r0, r1) = clmul64(self.lo, rhs.lo);
        F256Unreduced { r0, r1 }
    }

    /// Itoh–Tsujii inversion: 63 squarings and eight other products.
    /// With t_k = x^(2^k-1), t_(a+b) = t_a^(2^b) * t_b and inv = t_63^2.
    /// Zero maps to zero, as in the default field.
    pub fn inv(self) -> Self {
        #[cfg(feature = "mul-count")]
        op_count::INVS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Use Mul rather than square() so the native inversion's internal
        // products remain visible to mul-count, as in the original chain.
        let sq = |mut v: Self, n: u32| {
            for _ in 0..n {
                v = v * v;
            }
            v
        };
        let t2 = sq(self, 1) * self;
        let t3 = sq(t2, 1) * self;
        let t6 = sq(t3, 3) * t3;
        let t12 = sq(t6, 6) * t6;
        let t24 = sq(t12, 12) * t12;
        let t48 = sq(t24, 24) * t24;
        let t60 = sq(t48, 12) * t12;
        let t63 = sq(t60, 3) * t3;
        sq(t63, 1)
    }
}

impl Add for F128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo ^ rhs.lo,
        }
    }
}

impl AddAssign for F128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
    }
}

impl Mul for F128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(feature = "mul-count")]
        op_count::MULS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            // SAFETY: the compile-time gate guarantees NEON and PMULL.
            Self::from_u64(unsafe { crate::f64_neon::mul_scalar(self.lo, rhs.lo) })
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            self.mul_unreduced(rhs).reduce()
        }
    }
}

impl MulAssign for F128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[inline]
pub const fn mul_by_x(z: F128) -> F128 {
    F128::from_u64((z.lo << 1) ^ (0x1b & 0u64.wrapping_sub(z.lo >> 63)))
}

/// Legacy name for the active field's 128-bit unreduced polynomial product.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F256Unreduced {
    pub r0: u64,
    pub r1: u64,
}

impl F256Unreduced {
    pub const ZERO: Self = Self { r0: 0, r1: 0 };

    /// Two folds using x^64 = x^4+x^3+x+1. The second overflow has degree
    /// below four, so the second fold cannot overflow the base word again.
    #[inline]
    pub fn reduce(self) -> F128 {
        let hi = self.r1;
        let overflow = (hi >> 60) ^ (hi >> 61) ^ (hi >> 63);
        F128::from_u64(
            self.r0
                ^ hi
                ^ (hi << 1)
                ^ (hi << 3)
                ^ (hi << 4)
                ^ overflow
                ^ (overflow << 1)
                ^ (overflow << 3)
                ^ (overflow << 4),
        )
    }
}

impl BitXor for F256Unreduced {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self {
            r0: self.r0 ^ rhs.r0,
            r1: self.r1 ^ rhs.r1,
        }
    }
}

impl BitXorAssign for F256Unreduced {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.r0 ^= rhs.r0;
        self.r1 ^= rhs.r1;
    }
}

#[inline]
fn clmul64(a: u64, b: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: the compile-time aes feature guarantees PMULL support.
        let product = unsafe { core::arch::aarch64::vmull_p64(a, b) };
        (product as u64, (product >> 64) as u64)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        use core::arch::x86_64::*;
        // SAFETY: pclmulqdq is enabled; SSE2 is baseline on x86_64.
        unsafe {
            let product =
                _mm_clmulepi64_si128::<0>(_mm_set_epi64x(0, a as i64), _mm_set_epi64x(0, b as i64));
            (
                _mm_cvtsi128_si64(product) as u64,
                _mm_cvtsi128_si64(_mm_unpackhi_epi64(product, product)) as u64,
            )
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    )))]
    software_clmul64(a, b)
}

#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))
))]
fn software_clmul64(a: u64, b: u64) -> (u64, u64) {
    let mut product = 0u128;
    for i in 0..64 {
        product ^= ((a as u128) << i) & 0u128.wrapping_sub(((b >> i) & 1) as u128);
    }
    (product as u64, (product >> 64) as u64)
}

#[cfg(feature = "mul-count")]
pub mod op_count {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static MULS: AtomicU64 = AtomicU64::new(0);
    pub static INVS: AtomicU64 = AtomicU64::new(0);
    /// Itoh–Tsujii's 63 self-products plus eight other products.
    pub const MULS_PER_INV: u64 = 71;
    pub fn reset() {
        MULS.store(0, Relaxed);
        INVS.store(0, Relaxed);
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Snapshot {
        pub native_muls: u64,
        pub invs: u64,
    }
    impl Snapshot {
        pub fn muls_excluding_inv(&self) -> u64 {
            self.native_muls.saturating_sub(self.invs * MULS_PER_INV)
        }
        pub fn circuit_constraints(&self) -> u64 {
            self.muls_excluding_inv() + self.invs
        }
    }
    pub fn snapshot() -> Snapshot {
        Snapshot {
            native_muls: MULS.load(Relaxed),
            invs: INVS.load(Relaxed),
        }
    }
    pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Snapshot) {
        reset();
        let out = f();
        (out, snapshot())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    /// Independent polynomial long division: no shared reduction code.
    fn reference_mul(a: u64, b: u64) -> u64 {
        let (lo, hi) = software_clmul64(a, b);
        let mut product = (lo as u128) | ((hi as u128) << 64);
        for i in (64..128).rev() {
            if ((product >> i) & 1) != 0 {
                product ^= ((1u128 << 64) | 0x1b) << (i - 64);
            }
        }
        product as u64
    }

    fn remainder(mut a: u128, b: u128) -> u128 {
        assert_ne!(b, 0);
        while a != 0 && a.leading_zeros() <= b.leading_zeros() {
            a ^= b << (b.leading_zeros() - a.leading_zeros());
        }
        a
    }

    #[test]
    fn modulus_is_irreducible() {
        // Rabin criterion for n=64: x^(2^64)=x and gcd(x^(2^32)-x,p)=1.
        let mut x = 2u64;
        for _ in 0..32 {
            x = reference_mul(x, x);
        }
        let mut a = (1u128 << 64) | 0x1b;
        let mut b = (x ^ 2) as u128;
        while b != 0 {
            (a, b) = (b, remainder(a, b));
        }
        assert_eq!(a, 1);
        for _ in 32..64 {
            x = reference_mul(x, x);
        }
        assert_eq!(x, 2);
    }

    #[test]
    fn hardware_and_deferred_arithmetic_match_independent_reference() {
        let mut rng = Rng(42);
        let mut acc = F256Unreduced::ZERO;
        let mut reference_acc = 0u64;
        for _ in 0..4096 {
            let a = rng.f128();
            let b = rng.f128();
            assert_eq!((a * b).lo, reference_mul(a.lo, b.lo));
            assert_eq!(a.square().lo, reference_mul(a.lo, a.lo));
            assert_eq!(mul_by_x(a), a * F128::generator());
            acc ^= a.mul_unreduced(b);
            reference_acc ^= reference_mul(a.lo, b.lo);
        }
        assert_eq!(acc.reduce().lo, reference_acc);
        assert_eq!((F128::from_u64(1 << 63) * F128::generator()).lo, 0x1b);
    }

    #[test]
    fn immediate_products_match_reference_at_reduction_spill_corners() {
        let corners = [0, 1, u64::MAX, 1 << 63, 0xf000_0000_0000_0000, 0x1b];
        for a in corners {
            for b in corners {
                let want = reference_mul(a, b);
                assert_eq!((F128::from_u64(a) * F128::from_u64(b)).lo, want);
            }
            assert_eq!(F128::from_u64(a).square().lo, reference_mul(a, a));
        }
    }

    fn reference_inverse(a: u64) -> u64 {
        let mut exponent = u64::MAX - 1;
        let mut base = a;
        let mut result = 1;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = reference_mul(result, base);
            }
            base = reference_mul(base, base);
            exponent >>= 1;
        }
        result
    }

    #[test]
    fn itoh_tsujii_matches_independent_exponentiation_including_zero() {
        let mut rng = Rng(0x6464_177);
        let values = [0, 1, u64::MAX, 1 << 63, 0xf000_0000_0000_0000, 0x1b]
            .into_iter()
            .chain((0..128).map(|_| rng.f128().lo));
        for a in values {
            assert_eq!(F128::from_u64(a).inv().lo, reference_inverse(a));
        }
    }

    /// Operation counters are global: run counter tests with one test thread.
    #[cfg(feature = "mul-count")]
    #[test]
    fn inv_mul_cost_is_stable() {
        for x in [F128::ZERO, F128::ONE, F128::from_u64(0x1234_5678_9abc_def0)] {
            let (inverse, counts) = op_count::measure(|| x.inv());
            assert_eq!(inverse.lo, reference_inverse(x.lo));
            assert_eq!(counts.invs, 1);
            assert_eq!(counts.native_muls, op_count::MULS_PER_INV);
            assert_eq!(counts.muls_excluding_inv(), 0);
            assert_eq!(counts.circuit_constraints(), 1);
        }
        let x = F128::from_u64(0x1234_5678_9abc_def0);
        let ((square, product), counts) = op_count::measure(|| (x.square(), x * x));
        assert_eq!(square, product);
        assert_eq!(
            counts.native_muls, 1,
            "square keeps the existing counter convention"
        );
    }

    #[test]
    fn inverses_frobenius_and_representation() {
        let mut rng = Rng(15);
        assert_eq!(core::mem::size_of::<F128>(), 8);
        assert_eq!(core::mem::size_of::<F256Unreduced>(), 16);
        for _ in 0..128 {
            let a = rng.nonzero();
            assert_eq!(a * a.inv(), F128::ONE);
            let mut x = a;
            for _ in 0..64 {
                x = x.square();
            }
            assert_eq!(x, a);
            assert_eq!(F128::from_le_bytes(a.to_le_bytes()), a);
            for i in 0..64 {
                assert_eq!(a.with_bit(i, true).bit(i), true);
                assert_eq!(a.with_bit(i, false).bit(i), false);
            }
        }
    }

    #[test]
    #[should_panic(expected = "nonzero high limb")]
    fn constructor_refuses_lossy_conversion() {
        let _ = F128::new(0, 1);
    }
}
