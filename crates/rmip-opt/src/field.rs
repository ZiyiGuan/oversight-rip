use rmip_field::F128;

#[inline]
fn clmul64(a: u64, b: u64) -> u128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { core::arch::aarch64::vmull_p64(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let mut acc: u128 = 0;
        let mut i = 0;
        while i < 64 {
            if (b >> i) & 1 == 1 {
                acc ^= (a as u128) << i;
            }
            i += 1;
        }
        acc
    }
}

#[inline]
fn clmul128_karatsuba(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let lo = clmul64(a0, b0);
    let hi = clmul64(a1, b1);
    let mid = clmul64(a0 ^ a1, b0 ^ b1) ^ lo ^ hi;
    (hi ^ (mid >> 64), lo ^ (mid << 64))
}

#[inline]
fn mul_by_reduction(v: u128) -> (u128, u128) {
    let low = (v << 7) ^ (v << 2) ^ (v << 1) ^ v;
    let carry = (v >> 121) ^ (v >> 126) ^ (v >> 127);
    (carry, low)
}

#[inline]
pub fn mul(a: F128, b: F128) -> F128 {
    let (hi, lo) = clmul128_karatsuba(a.to_u128(), b.to_u128());
    let (c1, r1) = mul_by_reduction(hi);
    let (c2, r2) = mul_by_reduction(c1);
    debug_assert_eq!(c2, 0);
    F128::new(lo ^ r1 ^ r2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmip_field::{BinaryField, REDUCTION};

    #[test]
    fn agrees_with_the_reference_multiply() {
        let mut s: u64 = 0x243f_6a88_85a3_08d3;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s as u128) << 64) | s.wrapping_mul(0x9E37_79B9_7F4A_7C15) as u128
        };
        for _ in 0..20_000 {
            let (a, b) = (F128::new(next()), F128::new(next()));
            assert_eq!(mul(a, b), a.mul_schoolbook(b), "{a:?} * {b:?}");
            assert_eq!(
                mul(a, b),
                a.mul(b),
                "predecessor disagrees with the adopted field: {a:?} * {b:?}"
            );
        }
        for a in [0u128, 1, 2, 1 << 127, u128::MAX, REDUCTION] {
            for b in [0u128, 1, 2, 1 << 127, u128::MAX, REDUCTION] {
                let (x, y) = (F128::new(a), F128::new(b));
                assert_eq!(mul(x, y), x.mul_schoolbook(y), "{a:#x} * {b:#x}");
                assert_eq!(
                    mul(x, y),
                    x.mul(y),
                    "predecessor disagrees with the adopted field: {a:#x} * {b:#x}"
                );
            }
        }
    }

    #[test]
    fn same_field_axioms() {
        let x = F128::new(2);
        let mut acc = F128::ONE;
        for _ in 0..128 {
            acc = mul(acc, x);
        }
        assert_eq!(acc, F128::new(REDUCTION));

        let times_x = |v: u128| (v << 1) ^ if v >> 127 == 1 { REDUCTION } else { 0 };
        let top = F128::new(1u128 << 127);
        let mut want = 1u128 << 127;
        for i in 0..128u32 {
            assert_eq!(
                mul(top, F128::new(1u128 << i)).to_u128(),
                want,
                "x^(127+{i})"
            );
            want = times_x(want);
        }
    }
}
