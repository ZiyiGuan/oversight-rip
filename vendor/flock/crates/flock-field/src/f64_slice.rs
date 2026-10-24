//! Width-correct slice kernels for the optional 64-bit base field.
use crate::F128;

/// Fold adjacent pairs with one active-field product per result.
#[inline]
pub fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    assert!(
        base <= src.len() / 2 && dst.len() <= src.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let done = {
        use core::arch::aarch64::{vdupq_n_u64, veorq_u64, vld2q_u64, vst1q_u64};
        // SAFETY: F128 is repr(transparent) over u64. Each vector iteration
        // reads four elements and writes two; the bounds above and loop bound
        // cover every access. The shared and exclusive slices cannot overlap.
        unsafe {
            let mut done = 0;
            let rv = vdupq_n_u64(r.lo);
            while done + 2 <= dst.len() {
                let pair = vld2q_u64(src.as_ptr().add(2 * (base + done)).cast());
                let folded =
                    veorq_u64(pair.0, crate::f64_neon::mul2(veorq_u64(pair.0, pair.1), rv));
                vst1q_u64(dst.as_mut_ptr().add(done).cast(), folded);
                done += 2;
            }
            done
        }
    };
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    let done = 0;
    for (t, value) in dst.iter_mut().enumerate().skip(done) {
        let i = 2 * (base + t);
        *value = src[i] + (src[i] + src[i + 1]) * r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offset_and_tail_match_two_product_formula() {
        let mut rng = crate::test_rng::Rng(71);
        let src = rng.f128_vec(30);
        let r = rng.f128();
        let mut dst = [F128::ZERO; 9];
        fold_pairs(&src, 3, &mut dst, r);
        for (j, actual) in dst.into_iter().enumerate() {
            let i = 2 * (j + 3);
            assert_eq!(actual, src[i] * (F128::ONE + r) + src[i + 1] * r);
        }
    }

    #[test]
    fn every_short_length_and_offset_matches_scalar_with_guards() {
        let mut rng = crate::test_rng::Rng(117);
        for offset in 0..4 {
            for len in 0..33 {
                let src = rng.f128_vec(2 * (offset + len) + 3);
                for r in [F128::ZERO, F128::ONE, rng.f128()] {
                    let guard = F128::from_u64(0x1234abcd);
                    let mut dst = vec![guard; len + 2];
                    fold_pairs(&src, offset, &mut dst[1..1 + len], r);
                    assert_eq!(dst[0], guard);
                    assert_eq!(dst[len + 1], guard);
                    for j in 0..len {
                        let i = 2 * (offset + j);
                        assert_eq!(dst[j + 1], src[i] * (F128::ONE + r) + src[i + 1] * r);
                    }
                }
            }
        }
    }
}
