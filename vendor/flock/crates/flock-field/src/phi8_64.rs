//! AES GF(2^8) subfield embedding into the optional GF(2^64) base field.
//!
//! This is a newly generated embedding, not a truncation of GHASH constants.
//! `AES_ROOT` is a root of t^8+t^4+t^3+t+1 in the 64-bit field. Exhaustive
//! multiplication and addition tests pin the entire homomorphism.
use crate::{F8, F128};

const AES_ROOT: u64 = 0xb5ed_b706_6563_2cca;

const fn multiply(mut a: u64, mut b: u64) -> u64 {
    let mut out = 0;
    let mut i = 0;
    while i < 64 {
        out ^= a & 0u64.wrapping_sub(b & 1);
        a = (a << 1) ^ (0x1b & 0u64.wrapping_sub(a >> 63));
        b >>= 1;
        i += 1;
    }
    out
}

const fn make_table() -> [F128; 256] {
    let mut table = [F128::ZERO; 256];
    let mut powers = [1u64; 8];
    let mut i = 1;
    while i < 8 {
        powers[i] = multiply(powers[i - 1], AES_ROOT);
        i += 1;
    }
    let mut byte = 0;
    while byte < 256 {
        let mut image = 0;
        i = 0;
        while i < 8 {
            if (byte >> i) & 1 != 0 {
                image ^= powers[i];
            }
            i += 1;
        }
        table[byte] = F128::from_u64(image);
        byte += 1;
    }
    table
}

pub static PHI_8_TABLE: [F128; 256] = make_table();

#[inline]
pub fn phi8(a: F8) -> F128 {
    PHI_8_TABLE[a.0 as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aes_polynomial_and_exhaustive_homomorphism() {
        let r = F128::from_u64(AES_ROOT);
        let r2 = r.square();
        let r4 = r2.square();
        assert_eq!(r4.square() + r4 + r2 * r + r + F128::ONE, F128::ZERO);
        assert_eq!(phi8(F8::ZERO), F128::ZERO);
        assert_eq!(phi8(F8::ONE), F128::ONE);
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                assert_eq!(phi8(F8(a) + F8(b)), phi8(F8(a)) + phi8(F8(b)));
                assert_eq!(phi8(F8(a) * F8(b)), phi8(F8(a)) * phi8(F8(b)));
            }
        }
        for a in 0u8..=255 {
            for b in 0u8..a {
                assert_ne!(phi8(F8(a)), phi8(F8(b)));
            }
        }
    }
}
