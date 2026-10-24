//! Two adjacent GF(2^64) lanes per NEON register. These kernels retain the
//! scalar butterfly order, including an odd final lane, and do not change
//! the surrounding NTT schedule or the interleaved layout.

use core::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64, vld1q_u64, vst1q_u64};
use flock_field::f64_neon::mul2;

use crate::field::F128;

/// # Safety
/// PMULL/NEON must be available; the two slices must have equal lengths.
#[inline]
pub(super) unsafe fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    // A process-local diagnostic switch for end-to-end A/B measurements.
    // Both choices implement exactly the same ordered lane butterflies.
    static WIDTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let width = *WIDTH.get_or_init(|| {
        super::super::field64_diagnostic_choice("FLOCK_FIELD64_NTT_ROW_LANES", 8, &[2, 8])
    });
    // SAFETY: the caller's feature and slice contracts are unchanged.
    unsafe {
        if width == 8 {
            butterfly_row_pair_with_width::<8>(top, bot, twiddle);
        } else {
            butterfly_row_pair_with_width::<2>(top, bot, twiddle);
        }
    }
}

/// The two-lane baseline and eight-lane unroll share the exact same pair
/// arithmetic, so diagnostics isolate independent-chain scheduling alone.
///
/// # Safety
/// PMULL/NEON must be available; the two slices must have equal lengths.
#[inline]
unsafe fn butterfly_row_pair_with_width<const WIDTH: usize>(
    top: &mut [F128],
    bot: &mut [F128],
    twiddle: F128,
) {
    debug_assert_eq!(top.len(), bot.len());
    debug_assert!(WIDTH == 2 || WIDTH == 8);
    // SAFETY: F128 is repr(transparent) over u64, and every vector access
    // covers two existing elements. NEON loads/stores permit unaligned data.
    unsafe {
        let t = vdupq_n_u64(twiddle.lo);
        let mut lane = 0;
        if WIDTH == 8 {
            while lane + 8 <= top.len() {
                // Four independent product chains, as in leanVM's NEON8
                // NTT leaf, with our shared field reducer kept unchanged.
                let v0 = vld1q_u64(bot.as_ptr().add(lane).cast());
                let v1 = vld1q_u64(bot.as_ptr().add(lane + 2).cast());
                let v2 = vld1q_u64(bot.as_ptr().add(lane + 4).cast());
                let v3 = vld1q_u64(bot.as_ptr().add(lane + 6).cast());
                let p0 = mul2(v0, t);
                let p1 = mul2(v1, t);
                let p2 = mul2(v2, t);
                let p3 = mul2(v3, t);
                let n0 = veorq_u64(vld1q_u64(top.as_ptr().add(lane).cast()), p0);
                let n1 = veorq_u64(vld1q_u64(top.as_ptr().add(lane + 2).cast()), p1);
                let n2 = veorq_u64(vld1q_u64(top.as_ptr().add(lane + 4).cast()), p2);
                let n3 = veorq_u64(vld1q_u64(top.as_ptr().add(lane + 6).cast()), p3);
                vst1q_u64(top.as_mut_ptr().add(lane).cast(), n0);
                vst1q_u64(top.as_mut_ptr().add(lane + 2).cast(), n1);
                vst1q_u64(top.as_mut_ptr().add(lane + 4).cast(), n2);
                vst1q_u64(top.as_mut_ptr().add(lane + 6).cast(), n3);
                vst1q_u64(bot.as_mut_ptr().add(lane).cast(), veorq_u64(v0, n0));
                vst1q_u64(bot.as_mut_ptr().add(lane + 2).cast(), veorq_u64(v1, n1));
                vst1q_u64(bot.as_mut_ptr().add(lane + 4).cast(), veorq_u64(v2, n2));
                vst1q_u64(bot.as_mut_ptr().add(lane + 6).cast(), veorq_u64(v3, n3));
                lane += 8;
            }
        }
        while lane + 2 <= top.len() {
            let u = vld1q_u64(top.as_ptr().add(lane).cast());
            let v = vld1q_u64(bot.as_ptr().add(lane).cast());
            let new_u = veorq_u64(u, mul2(v, t));
            vst1q_u64(top.as_mut_ptr().add(lane).cast(), new_u);
            vst1q_u64(bot.as_mut_ptr().add(lane).cast(), veorq_u64(v, new_u));
            lane += 2;
        }
        super::portable::butterfly_row_pair(&mut top[lane..], &mut bot[lane..], twiddle);
    }
}

/// # Safety
/// PMULL/NEON must be available; all four slices must have equal lengths.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    debug_assert!(a.len() == b.len() && a.len() == c.len() && a.len() == d.len());
    // SAFETY: equal slice lengths and the loop bound make each load/store
    // valid; mutable slices are disjoint by the caller's Rust borrows.
    unsafe {
        let t0 = vdupq_n_u64(t_outer.lo);
        let t1a = vdupq_n_u64(t_inner_a.lo);
        let t1b = vdupq_n_u64(t_inner_b.lo);
        let mut lane = 0;
        while lane + 2 <= a.len() {
            let xa = vld1q_u64(a.as_ptr().add(lane).cast());
            let xb = vld1q_u64(b.as_ptr().add(lane).cast());
            let xc = vld1q_u64(c.as_ptr().add(lane).cast());
            let xd = vld1q_u64(d.as_ptr().add(lane).cast());
            let na = veorq_u64(xa, mul2(xc, t0));
            let nb = veorq_u64(xb, mul2(xd, t0));
            let nc = veorq_u64(xc, na);
            let nd = veorq_u64(xd, nb);
            let out_a = veorq_u64(na, mul2(nb, t1a));
            let out_c = veorq_u64(nc, mul2(nd, t1b));
            vst1q_u64(a.as_mut_ptr().add(lane).cast(), out_a);
            vst1q_u64(b.as_mut_ptr().add(lane).cast(), veorq_u64(nb, out_a));
            vst1q_u64(c.as_mut_ptr().add(lane).cast(), out_c);
            vst1q_u64(d.as_mut_ptr().add(lane).cast(), veorq_u64(nd, out_c));
            lane += 2;
        }
        super::portable::butterfly_fused_2layer(
            &mut a[lane..],
            &mut b[lane..],
            &mut c[lane..],
            &mut d[lane..],
            t_outer,
            t_inner_a,
            t_inner_b,
        );
    }
}

/// Fused two-layer block-zero spine: its outer and first inner twiddles
/// vanish, leaving only the lower pair's multiply.
///
/// # Safety
/// PMULL/NEON must be available; all four slices must have equal lengths.
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_zero(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_inner_b: F128,
) {
    debug_assert!(a.len() == b.len() && a.len() == c.len() && a.len() == d.len());
    // SAFETY: same two-element bounds and representation as the general kernel.
    unsafe {
        let t = vdupq_n_u64(t_inner_b.lo);
        let mut lane = 0;
        while lane + 2 <= a.len() {
            let xa = vld1q_u64(a.as_ptr().add(lane).cast());
            let xb = vld1q_u64(b.as_ptr().add(lane).cast());
            let xc = veorq_u64(vld1q_u64(c.as_ptr().add(lane).cast()), xa);
            let xd = veorq_u64(vld1q_u64(d.as_ptr().add(lane).cast()), xb);
            let out_c = veorq_u64(xc, mul2(xd, t));
            vst1q_u64(b.as_mut_ptr().add(lane).cast(), veorq_u64(xb, xa));
            vst1q_u64(c.as_mut_ptr().add(lane).cast(), out_c);
            vst1q_u64(d.as_mut_ptr().add(lane).cast(), veorq_u64(xd, out_c));
            lane += 2;
        }
        super::portable::butterfly_fused_2layer(
            &mut a[lane..],
            &mut b[lane..],
            &mut c[lane..],
            &mut d[lane..],
            F128::ZERO,
            F128::ZERO,
            t_inner_b,
        );
    }
}

#[inline(always)]
unsafe fn bf(values: &mut [uint64x2_t; 8], u: usize, v: usize, t: uint64x2_t) {
    // SAFETY: caller ensures the architecture features; indices are constants
    // below eight at every call site.
    unsafe {
        let new_u = veorq_u64(values[u], mul2(values[v], t));
        values[v] = veorq_u64(values[v], new_u);
        values[u] = new_u;
    }
}

/// # Safety
/// PMULL/NEON must be available; all eight slices must have equal lengths.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer(
    rows: [&mut [F128]; 8],
    t0: F128,
    t1: &[F128; 2],
    t2: &[F128; 4],
) {
    let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
    debug_assert!(
        [&r1, &r2, &r3, &r4, &r5, &r6, &r7]
            .iter()
            .all(|r| r.len() == r0.len())
    );
    // SAFETY: each pointer access is within its slice, and every row is
    // independently borrowed. Scalar code handles the final odd element.
    unsafe {
        let t0v = vdupq_n_u64(t0.lo);
        let t1v = [vdupq_n_u64(t1[0].lo), vdupq_n_u64(t1[1].lo)];
        let t2v = [
            vdupq_n_u64(t2[0].lo),
            vdupq_n_u64(t2[1].lo),
            vdupq_n_u64(t2[2].lo),
            vdupq_n_u64(t2[3].lo),
        ];
        let mut lane = 0;
        while lane + 2 <= r0.len() {
            let mut v = [
                vld1q_u64(r0.as_ptr().add(lane).cast()),
                vld1q_u64(r1.as_ptr().add(lane).cast()),
                vld1q_u64(r2.as_ptr().add(lane).cast()),
                vld1q_u64(r3.as_ptr().add(lane).cast()),
                vld1q_u64(r4.as_ptr().add(lane).cast()),
                vld1q_u64(r5.as_ptr().add(lane).cast()),
                vld1q_u64(r6.as_ptr().add(lane).cast()),
                vld1q_u64(r7.as_ptr().add(lane).cast()),
            ];
            bf(&mut v, 0, 4, t0v);
            bf(&mut v, 1, 5, t0v);
            bf(&mut v, 2, 6, t0v);
            bf(&mut v, 3, 7, t0v);
            bf(&mut v, 0, 2, t1v[0]);
            bf(&mut v, 1, 3, t1v[0]);
            bf(&mut v, 4, 6, t1v[1]);
            bf(&mut v, 5, 7, t1v[1]);
            bf(&mut v, 0, 1, t2v[0]);
            bf(&mut v, 2, 3, t2v[1]);
            bf(&mut v, 4, 5, t2v[2]);
            bf(&mut v, 6, 7, t2v[3]);
            vst1q_u64(r0.as_mut_ptr().add(lane).cast(), v[0]);
            vst1q_u64(r1.as_mut_ptr().add(lane).cast(), v[1]);
            vst1q_u64(r2.as_mut_ptr().add(lane).cast(), v[2]);
            vst1q_u64(r3.as_mut_ptr().add(lane).cast(), v[3]);
            vst1q_u64(r4.as_mut_ptr().add(lane).cast(), v[4]);
            vst1q_u64(r5.as_mut_ptr().add(lane).cast(), v[5]);
            vst1q_u64(r6.as_mut_ptr().add(lane).cast(), v[6]);
            vst1q_u64(r7.as_mut_ptr().add(lane).cast(), v[7]);
            lane += 2;
        }
        super::portable::butterfly_fused_3layer(
            [
                &mut r0[lane..],
                &mut r1[lane..],
                &mut r2[lane..],
                &mut r3[lane..],
                &mut r4[lane..],
                &mut r5[lane..],
                &mut r6[lane..],
                &mut r7[lane..],
            ],
            t0,
            t1,
            t2,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    fn row(rng: &mut Rng, len: usize) -> Vec<F128> {
        (0..len)
            .map(|i| match i % 7 {
                0 => F128::ZERO,
                1 => F128::ONE,
                2 => F128::from_u64(u64::MAX),
                3 => F128::from_u64(1 << 63),
                _ => rng.f128(),
            })
            .collect()
    }

    #[test]
    fn field64_neon_butterflies_match_portable_with_tails_and_guards() {
        let mut rng = Rng::new(0x4e54_5436_34);
        for len in (0..18).chain([31, 46, 63, 64, 65]) {
            for offset in [0, 1] {
                for twiddle in [F128::ZERO, F128::ONE, rng.f128()] {
                    let mut actual = [row(&mut rng, len + 3), row(&mut rng, len + 3)];
                    let original = actual.clone();
                    let mut expected = actual.clone();
                    let [a, b] = &mut expected;
                    super::super::portable::butterfly_row_pair(
                        &mut a[offset..offset + len],
                        &mut b[offset..offset + len],
                        twiddle,
                    );
                    let [a, b] = &mut actual;
                    unsafe {
                        butterfly_row_pair(
                            &mut a[offset..offset + len],
                            &mut b[offset..offset + len],
                            twiddle,
                        );
                    }
                    assert_eq!(actual, expected, "row pair len={len}, offset={offset}");
                    for width in [2, 8] {
                        let mut candidate = original.clone();
                        let [a, b] = &mut candidate;
                        // SAFETY: module cfg and guarded equal-length slices.
                        unsafe {
                            if width == 2 {
                                butterfly_row_pair_with_width::<2>(
                                    &mut a[offset..offset + len],
                                    &mut b[offset..offset + len],
                                    twiddle,
                                );
                            } else {
                                butterfly_row_pair_with_width::<8>(
                                    &mut a[offset..offset + len],
                                    &mut b[offset..offset + len],
                                    twiddle,
                                );
                            }
                        }
                        assert_eq!(
                            candidate, expected,
                            "row pair width={width}, len={len}, offset={offset}"
                        );
                    }

                    let inner_a = rng.f128();
                    let inner_b = rng.f128();
                    for zero in [false, true] {
                        let outer = if zero { F128::ZERO } else { twiddle };
                        let inner_a = if zero { F128::ZERO } else { inner_a };
                        let mut actual: [Vec<F128>; 4] =
                            std::array::from_fn(|_| row(&mut rng, len + 3));
                        let mut expected = actual.clone();
                        let [a, b, c, d] = &mut expected;
                        super::super::portable::butterfly_fused_2layer(
                            &mut a[offset..offset + len],
                            &mut b[offset..offset + len],
                            &mut c[offset..offset + len],
                            &mut d[offset..offset + len],
                            outer,
                            inner_a,
                            inner_b,
                        );
                        let [a, b, c, d] = &mut actual;
                        unsafe {
                            if zero {
                                butterfly_fused_2layer_zero(
                                    &mut a[offset..offset + len],
                                    &mut b[offset..offset + len],
                                    &mut c[offset..offset + len],
                                    &mut d[offset..offset + len],
                                    inner_b,
                                );
                            } else {
                                butterfly_fused_2layer(
                                    &mut a[offset..offset + len],
                                    &mut b[offset..offset + len],
                                    &mut c[offset..offset + len],
                                    &mut d[offset..offset + len],
                                    outer,
                                    inner_a,
                                    inner_b,
                                );
                            }
                        }
                        assert_eq!(
                            actual, expected,
                            "fused2 len={len}, offset={offset}, zero={zero}"
                        );
                    }

                    let t1 = [rng.f128(), rng.f128()];
                    let t2 = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
                    let mut actual: [Vec<F128>; 8] =
                        std::array::from_fn(|_| row(&mut rng, len + 3));
                    let mut expected = actual.clone();
                    let [a, b, c, d, e, f, g, h] = &mut expected;
                    super::super::portable::butterfly_fused_3layer(
                        [
                            &mut a[offset..offset + len],
                            &mut b[offset..offset + len],
                            &mut c[offset..offset + len],
                            &mut d[offset..offset + len],
                            &mut e[offset..offset + len],
                            &mut f[offset..offset + len],
                            &mut g[offset..offset + len],
                            &mut h[offset..offset + len],
                        ],
                        twiddle,
                        &t1,
                        &t2,
                    );
                    let [a, b, c, d, e, f, g, h] = &mut actual;
                    unsafe {
                        butterfly_fused_3layer(
                            [
                                &mut a[offset..offset + len],
                                &mut b[offset..offset + len],
                                &mut c[offset..offset + len],
                                &mut d[offset..offset + len],
                                &mut e[offset..offset + len],
                                &mut f[offset..offset + len],
                                &mut g[offset..offset + len],
                                &mut h[offset..offset + len],
                            ],
                            twiddle,
                            &t1,
                            &t2,
                        );
                    }
                    assert_eq!(actual, expected, "fused3 len={len}, offset={offset}");
                }
            }
        }
    }

    #[test]
    #[ignore = "diagnostic benchmark; run serially on an otherwise idle machine"]
    fn field64_ntt_row_unroll_costs() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = Rng::new(0x4e54_5438);
        for len in [46, 92, 4096, 65536] {
            let original = [row(&mut rng, len), row(&mut rng, len)];
            let twiddle = rng.f128();
            let repetitions = (1usize << 22) / len;
            let mut times = [Vec::new(), Vec::new()];
            for trial in 0..5 {
                for index in [trial % 2, 1 - trial % 2] {
                    let [mut top, mut bot] = original.clone();
                    let start = Instant::now();
                    for _ in 0..repetitions {
                        // SAFETY: this module is feature-gated and both
                        // allocations have the same initialized length.
                        unsafe {
                            if index == 0 {
                                butterfly_row_pair_with_width::<2>(
                                    black_box(&mut top),
                                    black_box(&mut bot),
                                    black_box(twiddle),
                                );
                            } else {
                                butterfly_row_pair_with_width::<8>(
                                    black_box(&mut top),
                                    black_box(&mut bot),
                                    black_box(twiddle),
                                );
                            }
                        }
                    }
                    times[index].push(start.elapsed().as_secs_f64());
                    black_box((top, bot));
                }
            }
            for samples in &mut times {
                samples.sort_by(f64::total_cmp);
            }
            eprintln!(
                "field64_ntt_row_unroll lanes={len} iterations={repetitions} pair_ms={:.3} unroll8_ms={:.3} speedup={:.3}",
                times[0][2] * 1000.0,
                times[1][2] * 1000.0,
                times[0][2] / times[1][2]
            );
        }
    }
}
