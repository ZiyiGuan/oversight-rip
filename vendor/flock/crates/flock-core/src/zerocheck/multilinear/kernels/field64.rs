//! Paired GF(2^64) AG folds and lookahead accumulation.
//!
//! Each NEON lane is a separate field value. The eight lookahead products
//! remain unreduced until the end of a chunk, just as in the scalar kernel.

#[cfg(test)]
use core::arch::aarch64::vld1q_u64;
use core::arch::aarch64::{
    uint64x2_t, vcombine_u64, vdupq_n_u64, veorq_u64, vextq_u64, vgetq_lane_u64, vld1_u64,
    vld2q_u64, vst1q_u64, vuzp1q_u64, vuzp2q_u64, vzip1q_u64,
};

use crate::field::{F128, F256Unreduced};
use flock_field::f64_neon::{mul2, mul2_unreduced};

struct LookaheadAcc {
    lo: [uint64x2_t; 4],
    hi: [uint64x2_t; 4],
}

impl LookaheadAcc {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe {
            Self {
                lo: [vdupq_n_u64(0); 4],
                hi: [vdupq_n_u64(0); 4],
            }
        }
    }

    #[inline(always)]
    unsafe fn add_product(&mut self, index: usize, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            let (lo, hi) = mul2_unreduced(a, b);
            self.lo[index] = veorq_u64(self.lo[index], lo);
            self.hi[index] = veorq_u64(self.hi[index], hi);
        }
    }

    /// `a01`/`a23` and `b01`/`b23` hold the four values indexed by x+2y.
    #[inline(always)]
    unsafe fn add(
        &mut self,
        a01: uint64x2_t,
        a23: uint64x2_t,
        b01: uint64x2_t,
        b23: uint64x2_t,
        weight: F128,
    ) {
        unsafe {
            let w = vdupq_n_u64(weight.lo);
            let e01 = mul2(w, a01);
            let e23 = mul2(w, a23);
            let sa0 = veorq_u64(e01, vextq_u64::<1>(e01, e01));
            let sb0 = veorq_u64(b01, vextq_u64::<1>(b01, b01));
            let sa1 = veorq_u64(e23, vextq_u64::<1>(e23, e23));
            let sb1 = veorq_u64(b23, vextq_u64::<1>(b23, b23));
            let dca = veorq_u64(e01, e23);
            let dcb = veorq_u64(b01, b23);
            let dsa = veorq_u64(sa0, sa1);
            let dsb = veorq_u64(sb0, sb1);
            // Consecutive lane pairs: [s10,sinf0], [c0,t11], [c2,d0], [dt,d2].
            self.add_product(0, vextq_u64::<1>(e01, sa0), vextq_u64::<1>(b01, sb0));
            self.add_product(1, e23, b23);
            self.add_product(2, vzip1q_u64(sa1, dca), vzip1q_u64(sb1, dcb));
            self.add_product(3, vextq_u64::<1>(dca, dsa), vextq_u64::<1>(dcb, dsb));
        }
    }

    #[inline(always)]
    unsafe fn finish(self) -> [F256Unreduced; 8] {
        unsafe {
            let mut out = [F256Unreduced::ZERO; 8];
            for i in 0..4 {
                out[2 * i] = F256Unreduced {
                    r0: vgetq_lane_u64::<0>(self.lo[i]),
                    r1: vgetq_lane_u64::<0>(self.hi[i]),
                };
                out[2 * i + 1] = F256Unreduced {
                    r0: vgetq_lane_u64::<1>(self.lo[i]),
                    r1: vgetq_lane_u64::<1>(self.hi[i]),
                };
            }
            out
        }
    }
}

/// Fold two adjacent 64-bit messages through eight byte lookup tables.
///
/// # Safety
/// `table` contains eight 256-entry tables and `src` points to 16 valid bytes.
#[inline(always)]
unsafe fn byte_dot2(table: *const F128, src: *const u8) -> uint64x2_t {
    unsafe {
        let x = u64::from_le((src as *const u64).read_unaligned());
        let y = u64::from_le((src.add(8) as *const u64).read_unaligned());
        let mut acc = vdupq_n_u64(0);
        for j in 0..8 {
            let base = table.add(j * 256);
            let vx = vld1_u64(base.add(((x >> (j * 8)) & 255) as usize) as *const u64);
            let vy = vld1_u64(base.add(((y >> (j * 8)) & 255) as usize) as *const u64);
            acc = veorq_u64(acc, vcombine_u64(vx, vy));
        }
        acc
    }
}

/// AG's live block fold; partial blocks have already been cleansed by the
/// caller, so this does not read bytes outside the declared source block.
pub(crate) fn fold_ag_block(
    a: &[u8],
    b: &[u8],
    msg_base: usize,
    table: &[[F128; 256]],
    ao: &mut [F128],
    bo: &mut [F128],
    eq: &[F128],
) -> [F256Unreduced; 8] {
    assert_eq!(ao.len(), 64);
    assert_eq!(bo.len(), 64);
    assert!(table.len() >= 8 && eq.len() >= 16);
    assert!(msg_base <= a.len() / 8 && 64 <= a.len() / 8 - msg_base);
    assert!(msg_base <= b.len() / 8 && 64 <= b.len() / 8 - msg_base);
    // SAFETY: the checks above cover every eight-byte message, lookup and
    // pair store. This module is compiled only when NEON and PMULL exist.
    unsafe {
        let mut acc = LookaheadAcc::zero();
        let tp = table.as_ptr() as *const F128;
        let ap = a.as_ptr().add(msg_base * 8);
        let bp = b.as_ptr().add(msg_base * 8);
        for q in 0..16 {
            let base = q * 4;
            let a01 = byte_dot2(tp, ap.add(base * 8));
            let a23 = byte_dot2(tp, ap.add((base + 2) * 8));
            let b01 = byte_dot2(tp, bp.add(base * 8));
            let b23 = byte_dot2(tp, bp.add((base + 2) * 8));
            vst1q_u64(ao.as_mut_ptr().add(base) as *mut u64, a01);
            vst1q_u64(ao.as_mut_ptr().add(base + 2) as *mut u64, a23);
            vst1q_u64(bo.as_mut_ptr().add(base) as *mut u64, b01);
            vst1q_u64(bo.as_mut_ptr().add(base + 2) as *mut u64, b23);
            acc.add(a01, a23, b01, b23, eq[q]);
        }
        acc.finish()
    }
}

/// Fold two adjacent outputs, with either one or two pending variables.
/// The caller supplies four input words when PER_U=8 and eight when PER_U=16.
#[inline(always)]
unsafe fn fold_two<const PER_U: usize>(
    p: *const F128,
    r0: uint64x2_t,
    r1: uint64x2_t,
) -> uint64x2_t {
    unsafe {
        let first = vld2q_u64(p as *const u64);
        let x = veorq_u64(first.0, mul2(r0, veorq_u64(first.0, first.1)));
        if PER_U == 8 {
            return x;
        }
        let second = vld2q_u64(p.add(4) as *const u64);
        let y = veorq_u64(second.0, mul2(r0, veorq_u64(second.0, second.1)));
        let even = vuzp1q_u64(x, y);
        let odd = vuzp2q_u64(x, y);
        veorq_u64(even, mul2(r1, veorq_u64(even, odd)))
    }
}

/// One eq-low chunk of the one/two-variable fold + lookahead pass.
/// Empty and partial chunks use exactly the same global weights as scalar.
pub(crate) fn fold_lookahead_chunk<const PER_U: usize>(
    a: &[F128],
    b: &[F128],
    ao: &mut [F128],
    bo: &mut [F128],
    rhos: (F128, F128),
    eq: &[F128],
) -> [F256Unreduced; 8] {
    assert!(PER_U == 8 || PER_U == 16);
    assert_eq!(ao.len() % 4, 0);
    let n = ao.len() / 4;
    assert_eq!(bo.len(), ao.len());
    assert_eq!(a.len(), n * PER_U);
    assert_eq!(b.len(), a.len());
    assert!(eq.len() >= n);
    // SAFETY: all paired reads and writes are bounded by the shape checks.
    unsafe {
        let mut acc = LookaheadAcc::zero();
        let r0 = vdupq_n_u64(rhos.0.lo);
        let r1 = vdupq_n_u64(rhos.1.lo);
        for u in 0..n {
            let ap = a.as_ptr().add(u * PER_U);
            let bp = b.as_ptr().add(u * PER_U);
            let a01 = fold_two::<PER_U>(ap, r0, r1);
            let a23 = fold_two::<PER_U>(ap.add(PER_U / 2), r0, r1);
            let b01 = fold_two::<PER_U>(bp, r0, r1);
            let b23 = fold_two::<PER_U>(bp.add(PER_U / 2), r0, r1);
            vst1q_u64(ao.as_mut_ptr().add(4 * u) as *mut u64, a01);
            vst1q_u64(ao.as_mut_ptr().add(4 * u + 2) as *mut u64, a23);
            vst1q_u64(bo.as_mut_ptr().add(4 * u) as *mut u64, b01);
            vst1q_u64(bo.as_mut_ptr().add(4 * u + 2) as *mut u64, b23);
            acc.add(a01, a23, b01, b23, eq[u]);
        }
        acc.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zerocheck::multilinear::lookahead_accum;
    use flock_field::test_rng::Rng;

    #[test]
    fn paired_lookahead_matches_scalar_unreduced() {
        let mut rng = Rng(0x64aa);
        for trial in 0..100 {
            let a = core::array::from_fn(|_| rng.f128());
            let b = core::array::from_fn(|_| rng.f128());
            let eq = match trial {
                0 => F128::ZERO,
                1 => F128::ONE,
                _ => rng.f128(),
            };
            let mut expected = [F256Unreduced::ZERO; 8];
            lookahead_accum(&a, &b, eq, &mut expected);
            let actual = unsafe {
                let mut acc = LookaheadAcc::zero();
                acc.add(
                    vld1q_u64(a.as_ptr() as *const u64),
                    vld1q_u64(a.as_ptr().add(2) as *const u64),
                    vld1q_u64(b.as_ptr() as *const u64),
                    vld1q_u64(b.as_ptr().add(2) as *const u64),
                    eq,
                );
                acc.finish()
            };
            assert_eq!(actual, expected);
        }
    }

    fn check_chunk<const PER_U: usize>(rng: &mut Rng) {
        for n in [0, 1, 2, 3, 7, 16, 31] {
            let a = rng.f128_vec(n * PER_U);
            let b = rng.f128_vec(n * PER_U);
            let eq = rng.f128_vec(n + 1);
            for rhos in [
                (F128::ZERO, F128::ZERO),
                (F128::ONE, F128::ONE),
                (rng.f128(), rng.f128()),
            ] {
                let mut ao = vec![F128::ZERO; 4 * n];
                let mut bo = ao.clone();
                let actual = fold_lookahead_chunk::<PER_U>(&a, &b, &mut ao, &mut bo, rhos, &eq);
                let mut expected = [F256Unreduced::ZERO; 8];
                let fold = |v: &[F128]| {
                    let x = v[0] + rhos.0 * (v[0] + v[1]);
                    if PER_U == 8 {
                        x
                    } else {
                        let y = v[2] + rhos.0 * (v[2] + v[3]);
                        x + rhos.1 * (x + y)
                    }
                };
                for u in 0..n {
                    let ga = core::array::from_fn(|v| fold(&a[u * PER_U + v * (PER_U / 4)..]));
                    let gb = core::array::from_fn(|v| fold(&b[u * PER_U + v * (PER_U / 4)..]));
                    assert_eq!(&ao[4 * u..4 * u + 4], &ga);
                    assert_eq!(&bo[4 * u..4 * u + 4], &gb);
                    lookahead_accum(&ga, &gb, eq[u], &mut expected);
                }
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn paired_fold_chunks_match_scalar_including_partial_chunks() {
        let mut rng = Rng(0x6408);
        check_chunk::<8>(&mut rng);
        check_chunk::<16>(&mut rng);
    }

    #[test]
    fn paired_ag_lookup_and_sums_match_scalar_at_nonzero_offset() {
        let mut rng = Rng(0x64a6);
        let table: Vec<[F128; 256]> = (0..8)
            .map(|_| core::array::from_fn(|_| rng.f128()))
            .collect();
        let mut a = vec![0u8; 1024];
        let mut b = a.clone();
        rng.fill_bytes(&mut a);
        rng.fill_bytes(&mut b);
        let eq = rng.f128_vec(16);
        for base in [0, 3, 64] {
            let mut ao = [F128::ZERO; 64];
            let mut bo = ao;
            let actual = fold_ag_block(&a, &b, base, &table, &mut ao, &mut bo, &eq);
            let dot = |src: &[u8], r: usize| {
                (0..8).fold(F128::ZERO, |sum, j| sum + table[j][src[r * 8 + j] as usize])
            };
            let mut expected = [F256Unreduced::ZERO; 8];
            for q in 0..16 {
                let ga = core::array::from_fn(|v| dot(&a, base + 4 * q + v));
                let gb = core::array::from_fn(|v| dot(&b, base + 4 * q + v));
                assert_eq!(&ao[4 * q..4 * q + 4], &ga);
                assert_eq!(&bo[4 * q..4 * q + 4], &gb);
                lookahead_accum(&ga, &gb, eq[q], &mut expected);
            }
            assert_eq!(actual, expected);
        }
    }
}
