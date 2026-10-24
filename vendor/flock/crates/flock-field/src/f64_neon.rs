//! Two independent GF(2^64) elements per NEON register.
//!
//! RMIP: these kernels use x^64 = x^4 + x^3 + x + 1. The lanes are
//! separate field elements, never halves of a GF(2^128) element.
//!
//! Immediate PMULL reductions are adapted from leanEthereum/leanVM commit
//! a0ca7858f2533ce51abdc20bb635c007e19b26b3, primitives/field/gf2_64.rs
//! and primitives/field/neon.rs. Copyright (c) 2026 leanEthereum.
//! Upstream supplies MIT and Apache-2.0 licenses; its GF64 source credits
//! binius-zk/binius64 (Apache-2.0). See vendor/flock/PROVENANCE.md.

#[cfg(target_feature = "sha3")]
use core::arch::aarch64::veor3q_u64;
use core::arch::aarch64::{
    uint64x2_t, vdupq_n_u64, veorq_u64, vgetq_lane_u64, vmull_high_p64, vmull_p64,
    vreinterpretq_p64_u64, vreinterpretq_u64_p128, vshlq_n_u64, vshrq_n_u64, vuzp1q_u64,
    vuzp2q_u64, vzip1q_u64,
};

#[inline(always)]
unsafe fn xor3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    #[cfg(target_feature = "sha3")]
    unsafe {
        veor3q_u64(a, b, c)
    }
    #[cfg(not(target_feature = "sha3"))]
    unsafe {
        veorq_u64(a, veorq_u64(b, c))
    }
}

#[inline(always)]
unsafe fn mul_high(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    // SAFETY: this entire module is gated on NEON and PMULL.
    unsafe {
        vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ))
    }
}

/// Reduce two raw products `[lo0, hi0]`, `[lo1, hi1]` into a field pair.
/// Each high word is folded twice by 0x1b, entirely in vector registers.
/// This is a different layout from the deferred accumulators in [`reduce2`].
#[inline(always)]
unsafe fn reduce_product_pair(p0: uint64x2_t, p1: uint64x2_t) -> uint64x2_t {
    unsafe {
        let r = vdupq_n_u64(0x1b);
        let (t0, t1) = (mul_high(p0, r), mul_high(p1, r));
        let (u0, u1) = (mul_high(t0, r), mul_high(t1, r));
        let (mut e0, mut e1) = (xor3(p0, t0, u0), xor3(p1, t1, u1));
        // Retain upstream's empty compiler barrier: without it LLVM can
        // replace the two whole-register XORs with extra low-lane inserts.
        // It emits no instructions and has no memory or flag side effects.
        core::arch::asm!(
            "/* {0:v} {1:v} */",
            inout(vreg) e0,
            inout(vreg) e1,
            options(pure, nomem, nostack, preserves_flags)
        );
        vzip1q_u64(e0, e1)
    }
}

/// One immediate field product with both reduction folds kept in NEON.
/// The caller owns operation counting; squaring keeps its existing convention.
///
/// # Safety
/// The executing CPU must support NEON and PMULL (the module's cfg gate).
#[inline(always)]
pub(crate) unsafe fn mul_scalar(a: u64, b: u64) -> u64 {
    unsafe {
        let r = vdupq_n_u64(0x1b);
        let p = vreinterpretq_u64_p128(vmull_p64(a, b));
        let t = mul_high(p, r);
        let u = mul_high(t, r);
        vgetq_lane_u64::<0>(xor3(p, t, u))
    }
}

/// Two carryless products, with low and high limbs separated by lane.
///
/// # Safety
/// The executing CPU must support NEON and PMULL (the module's cfg gate).
#[inline(always)]
pub unsafe fn mul2_unreduced(a: uint64x2_t, b: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    // SAFETY: the caller and module gate guarantee NEON and PMULL.
    unsafe {
        let p0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b)));
        let p1 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ));
        (vuzp1q_u64(p0, p1), vuzp2q_u64(p0, p1))
    }
}

/// Reduce two degree-below-128 polynomials by the GF(2^64) modulus.
///
/// # Safety
/// The executing CPU must support NEON (the module's cfg gate).
#[inline(always)]
pub unsafe fn reduce2(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
    // SAFETY: NEON is guaranteed by the module gate and caller.
    unsafe {
        let overflow = veorq_u64(
            veorq_u64(vshrq_n_u64::<60>(hi), vshrq_n_u64::<61>(hi)),
            vshrq_n_u64::<63>(hi),
        );
        // Fold the high polynomial and its at-most-four-bit overflow together.
        let h = veorq_u64(hi, overflow);
        veorq_u64(
            veorq_u64(lo, h),
            veorq_u64(
                vshlq_n_u64::<1>(h),
                veorq_u64(vshlq_n_u64::<3>(h), vshlq_n_u64::<4>(h)),
            ),
        )
    }
}

/// Two independent field products, reduced in vector registers.
///
/// # Safety
/// The executing CPU must support NEON and PMULL (the module's cfg gate).
#[inline(always)]
pub unsafe fn mul2(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    #[cfg(feature = "mul-count")]
    crate::gf2_128::op_count::MULS.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: forwarded caller contract.
    unsafe {
        let p0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b)));
        let p1 = mul_high(a, b);
        reduce_product_pair(p0, p1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::arch::aarch64::{vld1q_u64, vst1q_u64, vzip2q_u64};

    fn reference_product(a: u64, b: u64) -> u128 {
        let mut p = 0u128;
        for bit in 0..64 {
            if (b >> bit) & 1 != 0 {
                p ^= (a as u128) << bit;
            }
        }
        p
    }

    fn reference_reduce(mut p: u128) -> u64 {
        for bit in (64..128).rev() {
            if (p >> bit) & 1 != 0 {
                p ^= ((1u128 << 64) | 0x1b) << (bit - 64);
            }
        }
        p as u64
    }

    fn check(a: [u64; 2], b: [u64; 2]) {
        let mut lo = [0; 2];
        let mut hi = [0; 2];
        let mut reduced = [0; 2];
        // SAFETY: the module gate guarantees instructions; arrays hold two lanes.
        unsafe {
            let av = vld1q_u64(a.as_ptr());
            let bv = vld1q_u64(b.as_ptr());
            let (l, h) = mul2_unreduced(av, bv);
            vst1q_u64(lo.as_mut_ptr(), l);
            vst1q_u64(hi.as_mut_ptr(), h);
            vst1q_u64(reduced.as_mut_ptr(), mul2(av, bv));
        }
        for lane in 0..2 {
            let p = reference_product(a[lane], b[lane]);
            assert_eq!((lo[lane] as u128) | ((hi[lane] as u128) << 64), p);
            assert_eq!(reduced[lane], reference_reduce(p));
        }
    }

    #[test]
    fn products_and_reduction_match_independent_polynomial_reference() {
        for a in 0..64 {
            for b in 0..64 {
                check([1 << a, !(1 << a)], [1 << b, !(1 << b)]);
            }
        }
        check([0, u64::MAX], [u64::MAX, 0]);
        let mut rng = crate::test_rng::Rng(319);
        for _ in 0..4096 {
            check(
                [rng.f128().lo, rng.f128().lo],
                [rng.f128().lo, rng.f128().lo],
            );
        }
    }

    #[test]
    fn deferred_lane_accumulators_preserve_products() {
        let mut rng = crate::test_rng::Rng(17);
        let mut reference = [0u128; 2];
        let mut actual = [0u64; 2];
        // SAFETY: the module gate guarantees instructions; arrays hold two lanes.
        unsafe {
            let mut lo = vdupq_n_u64(0);
            let mut hi = vdupq_n_u64(0);
            for _ in 0..257 {
                let a = [rng.f128().lo, rng.f128().lo];
                let b = [rng.f128().lo, rng.f128().lo];
                let (l, h) = mul2_unreduced(vld1q_u64(a.as_ptr()), vld1q_u64(b.as_ptr()));
                lo = veorq_u64(lo, l);
                hi = veorq_u64(hi, h);
                for lane in 0..2 {
                    reference[lane] ^= reference_product(a[lane], b[lane]);
                }
            }
            vst1q_u64(actual.as_mut_ptr(), reduce2(lo, hi));
        }
        assert_eq!(actual, reference.map(reference_reduce));
    }

    #[test]
    fn reduction_accepts_the_full_128_bit_polynomial_range() {
        let mut rng = crate::test_rng::Rng(811);
        for bit in 0..128 {
            let polynomials = [1u128 << bit, !(1u128 << bit)];
            check_reduction(polynomials);
        }
        for _ in 0..1024 {
            check_reduction(std::array::from_fn(|_| {
                (rng.f128().lo as u128) | ((rng.f128().lo as u128) << 64)
            }));
        }
    }

    /// Run global operation-counter tests with one test thread.
    #[cfg(feature = "mul-count")]
    #[test]
    fn immediate_pair_counts_two_field_products() {
        let a = [0x1234_5678_9abc_def0, u64::MAX];
        let b = [1 << 63, 0xf000_0000_0000_0000];
        let mut actual = [0; 2];
        let (_, counts) = crate::gf2_128::op_count::measure(|| {
            // SAFETY: module cfg supplies the instructions; each array holds two lanes.
            unsafe {
                vst1q_u64(
                    actual.as_mut_ptr(),
                    mul2(vld1q_u64(a.as_ptr()), vld1q_u64(b.as_ptr())),
                );
            }
        });
        assert_eq!(counts.native_muls, 2);
        assert_eq!(counts.invs, 0);
        assert_eq!(
            actual,
            core::array::from_fn(|i| reference_reduce(reference_product(a[i], b[i])))
        );
    }

    fn check_reduction(polynomials: [u128; 2]) {
        let low = polynomials.map(|p| p as u64);
        let high = polynomials.map(|p| (p >> 64) as u64);
        let mut actual = [0u64; 2];
        let mut immediate = [0u64; 2];
        // SAFETY: the module gate supplies NEON; every array holds two lanes.
        unsafe {
            vst1q_u64(
                actual.as_mut_ptr(),
                reduce2(vld1q_u64(low.as_ptr()), vld1q_u64(high.as_ptr())),
            );
            let lo = vld1q_u64(low.as_ptr());
            let hi = vld1q_u64(high.as_ptr());
            vst1q_u64(
                immediate.as_mut_ptr(),
                reduce_product_pair(vzip1q_u64(lo, hi), vzip2q_u64(lo, hi)),
            );
        }
        assert_eq!(actual, polynomials.map(reference_reduce));
        assert_eq!(immediate, polynomials.map(reference_reduce));
    }
}
