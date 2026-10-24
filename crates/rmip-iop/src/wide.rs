use std::cmp::Ordering;
use std::fmt;

const LIMBS: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct U1024 {
    limbs: [u64; LIMBS],
}

impl U1024 {
    pub const ZERO: U1024 = U1024 { limbs: [0; LIMBS] };
    pub const ONE: U1024 = {
        let mut l = [0u64; LIMBS];
        l[0] = 1;
        U1024 { limbs: l }
    };
    pub const MAX: U1024 = U1024 {
        limbs: [u64::MAX; LIMBS],
    };

    pub fn pow2(k: u32) -> U1024 {
        assert!(k < 1024, "2^{k} does not fit 1024 bits");
        let mut l = [0u64; LIMBS];
        l[(k / 64) as usize] = 1u64 << (k % 64);
        U1024 { limbs: l }
    }

    pub fn bits(self) -> u32 {
        for i in (0..LIMBS).rev() {
            if self.limbs[i] != 0 {
                return 64 * i as u32 + (64 - self.limbs[i].leading_zeros());
            }
        }
        0
    }

    pub fn checked_mul_u64(self, m: u64) -> Option<U1024> {
        let mut out = [0u64; LIMBS];
        let mut carry: u128 = 0;
        for (o, limb) in out.iter_mut().zip(self.limbs) {
            let t = (limb as u128) * (m as u128) + carry;
            *o = t as u64;
            carry = t >> 64;
        }
        (carry == 0).then_some(U1024 { limbs: out })
    }

    pub fn checked_add(self, other: U1024) -> Option<U1024> {
        let mut out = [0u64; LIMBS];
        let mut carry = 0u64;
        for ((o, a), b) in out.iter_mut().zip(self.limbs).zip(other.limbs) {
            let (s, c1) = a.overflowing_add(b);
            let (s, c2) = s.overflowing_add(carry);
            *o = s;
            carry = u64::from(c1) + u64::from(c2);
        }
        (carry == 0).then_some(U1024 { limbs: out })
    }

    pub fn checked_add_u64(self, a: u64) -> Option<U1024> {
        self.checked_add(U1024::from(a))
    }

    pub fn checked_sub_u64(self, a: u64) -> Option<U1024> {
        let mut out = self.limbs;
        let mut borrow = a;
        for limb in out.iter_mut() {
            if borrow == 0 {
                break;
            }
            let (d, b) = limb.overflowing_sub(borrow);
            *limb = d;
            borrow = u64::from(b);
        }
        (borrow == 0).then_some(U1024 { limbs: out })
    }

    pub fn bits_at(self, offset: u32, width: u32) -> u64 {
        debug_assert!(width <= 64);
        if width == 0 {
            return 0;
        }
        let limb = |i: usize| -> u64 {
            if i < LIMBS {
                self.limbs[i]
            } else {
                0
            }
        };
        let (i, shift) = ((offset / 64) as usize, offset % 64);
        let lo = limb(i) >> shift;
        let word = if shift == 0 {
            lo
        } else {
            lo | (limb(i + 1) << (64 - shift))
        };
        if width == 64 {
            word
        } else {
            word & ((1u64 << width) - 1)
        }
    }

    pub fn div_rem_u64(self, d: u64) -> (U1024, u64) {
        assert!(d != 0, "division by zero");
        let mut q = [0u64; LIMBS];
        let mut rem: u128 = 0;
        for i in (0..LIMBS).rev() {
            let cur = (rem << 64) | self.limbs[i] as u128;
            q[i] = (cur / d as u128) as u64;
            rem = cur % d as u128;
        }
        (U1024 { limbs: q }, rem as u64)
    }

    pub fn checked_pow_u64(base: u64, exp: u32) -> Option<U1024> {
        let mut acc = U1024::ONE;
        for _ in 0..exp {
            acc = acc.checked_mul_u64(base)?;
        }
        Some(acc)
    }

    pub fn to_be_bytes(self) -> [u8; 128] {
        let mut b = [0u8; 128];
        for (i, limb) in self.limbs.iter().enumerate() {
            let at = 128 - 8 * (i + 1);
            b[at..at + 8].copy_from_slice(&limb.to_be_bytes());
        }
        b
    }

    pub fn from_be_bytes(b: [u8; 128]) -> U1024 {
        let mut limbs = [0u64; LIMBS];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let at = 128 - 8 * (i + 1);
            *limb = u64::from_be_bytes(b[at..at + 8].try_into().unwrap());
        }
        U1024 { limbs }
    }

    pub fn fits_u128(self) -> bool {
        self.limbs[2..].iter().all(|&l| l == 0)
    }

    pub fn fits_u256(self) -> bool {
        self.limbs[4..].iter().all(|&l| l == 0)
    }
}

impl Ord for U1024 {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..LIMBS).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => continue,
                o => return o,
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for U1024 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

macro_rules! from_small {
    ($($t:ty),*) => {$(
        impl From<$t> for U1024 {
            fn from(v: $t) -> U1024 {
                let v = v as u128;
                let mut l = [0u64; LIMBS];
                l[0] = v as u64;
                l[1] = (v >> 64) as u64;
                U1024 { limbs: l }
            }
        }
    )*};
}
from_small!(u8, u16, u32, u64, usize, u128);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooWide;

impl fmt::Display for TooWide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("coordinate does not fit the requested integer width")
    }
}

impl std::error::Error for TooWide {}

macro_rules! try_into_small {
    ($($t:ty),*) => {$(
        impl TryFrom<U1024> for $t {
            type Error = TooWide;
            fn try_from(v: U1024) -> Result<$t, TooWide> {
                if !v.fits_u128() {
                    return Err(TooWide);
                }
                let lo = (v.limbs[0] as u128) | ((v.limbs[1] as u128) << 64);
                <$t>::try_from(lo).map_err(|_| TooWide)
            }
        }
    )*};
}
try_into_small!(u8, u16, u32, u64, usize, u128);

impl fmt::Display for U1024 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.fits_u128() {
            return write!(f, "{}", u128::try_from(*self).unwrap());
        }
        const CHUNK: u64 = 10_000_000_000_000_000_000;
        let mut chunks = Vec::new();
        let mut v = *self;
        while v != U1024::ZERO {
            let (q, r) = v.div_rem_u64(CHUNK);
            chunks.push(r);
            v = q;
        }
        let mut s = chunks.pop().unwrap().to_string();
        while let Some(c) = chunks.pop() {
            s.push_str(&format!("{c:019}"));
        }
        f.write_str(&s)
    }
}

impl fmt::Debug for U1024 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 11
    }

    #[test]
    fn bits_at_agrees_with_division_by_a_power_of_two() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for limbs_used in [1usize, 2, 3, 16] {
            let v = random_wide(&mut s, limbs_used);
            for width in [1u32, 7, 21, 32, 63, 64] {
                let mut rest = v;
                let mut offset = 0u32;
                while offset + width <= 1024 {
                    let (quot, rem) = if width == 64 {
                        let (q, r) = rest.div_rem_u64(u64::MAX);
                        let _ = (q, r);
                        let mut l = rest.limbs;
                        let r = l[0];
                        l.rotate_left(1);
                        l[LIMBS - 1] = 0;
                        (U1024 { limbs: l }, r)
                    } else {
                        rest.div_rem_u64(1u64 << width)
                    };
                    assert_eq!(
                        v.bits_at(offset, width),
                        rem,
                        "width {width} offset {offset}"
                    );
                    rest = quot;
                    offset += width;
                }
            }
        }
    }

    fn random_wide(s: &mut u64, limbs_used: usize) -> U1024 {
        let mut l = [0u64; LIMBS];
        for limb in l.iter_mut().take(limbs_used) {
            *limb = lcg(s) | (lcg(s) << 53);
        }
        U1024 { limbs: l }
    }

    #[test]
    fn agrees_with_u128_where_both_fit() {
        let mut s = 7u64;
        for _ in 0..2000 {
            let a = ((lcg(&mut s) as u128) << 32) ^ lcg(&mut s) as u128;
            let m = lcg(&mut s) >> 20;
            let d = (lcg(&mut s) >> 30) | 1;
            let w = U1024::from(a);
            assert_eq!(
                w.checked_mul_u64(m).map(|x| u128::try_from(x).unwrap()),
                a.checked_mul(m as u128)
            );
            assert_eq!(
                w.checked_add_u64(m).map(|x| u128::try_from(x).unwrap()),
                Some(a + m as u128)
            );
            let (q, r) = w.div_rem_u64(d);
            assert_eq!(
                (u128::try_from(q).unwrap(), r as u128),
                (a / d as u128, a % d as u128)
            );
            assert_eq!(w.to_string(), a.to_string());
            assert_eq!(U1024::from_be_bytes(w.to_be_bytes()), w);
            assert_eq!(u128::try_from(w), Ok(a));
            assert_eq!(w.bits(), 128 - a.leading_zeros());
        }
    }

    #[test]
    fn multiplication_and_division_invert_above_u128() {
        let mut s = 11u64;
        for round in 0..2000 {
            let w = random_wide(&mut s, 3 + round % 13);
            let m = (lcg(&mut s) >> 40) | 1;
            if let Some(p) = w.checked_mul_u64(m) {
                let (q, r) = p.div_rem_u64(m);
                assert_eq!((q, r), (w, 0));
                assert!(p > w || m == 1);
            }
            assert_eq!(u64::try_from(w), Err(TooWide));
            assert_eq!(u128::try_from(w), Err(TooWide));
            assert_eq!(U1024::from_be_bytes(w.to_be_bytes()), w);
            let (q, r) = w.div_rem_u64(1_000_003);
            let back = q
                .checked_mul_u64(1_000_003)
                .unwrap()
                .checked_add_u64(r)
                .unwrap();
            assert_eq!(back, w);
        }
    }

    #[test]
    fn the_tuple_domains_that_motivate_the_width() {
        let d = U1024::checked_pow_u64(1 << 18, 34).unwrap();
        assert!(!d.fits_u256());
        assert_eq!(d, U1024::pow2(612));
        assert_eq!(d.bits(), 613);
        assert!(d.checked_mul_u64(34 << 16).is_some());
        let l0 = U1024::checked_pow_u64(1 << 29, 34).unwrap();
        assert_eq!(l0, U1024::pow2(986));
        assert!(l0.checked_mul_u64(34 * 32).is_some());
        assert_eq!(U1024::checked_pow_u64(1 << 29, 36), None);
        let small = U1024::checked_pow_u64(1 << 13, 6).unwrap();
        assert!(small.fits_u128());
        assert_eq!(u128::try_from(small), Ok(1u128 << 78));
        assert_eq!(U1024::MAX.checked_add_u64(1), None);
        assert_eq!(U1024::MAX.checked_mul_u64(2), None);
        assert_eq!(U1024::MAX.bits(), 1024);
        assert_eq!(U1024::from(1u8).to_string(), "1");
        assert_eq!(
            U1024::pow2(128).to_string(),
            "340282366920938463463374607431768211456"
        );
        assert_eq!(
            U1024::pow2(256).to_string(),
            "115792089237316195423570985008687907853269984665640564039457584007913129639936"
        );
    }

    #[test]
    fn wide_addition_carries_and_subtraction_borrows() {
        let a = U1024::from(u128::MAX);
        assert_eq!(a.checked_add(U1024::ONE), Some(U1024::pow2(128)));
        assert_eq!(U1024::MAX.checked_add(U1024::ONE), None);
        assert_eq!(
            a.checked_add(a),
            Some(U1024::pow2(129).checked_sub_u64(2).unwrap())
        );
        assert_eq!(U1024::pow2(128).checked_sub_u64(1), Some(a));
        assert_eq!(U1024::ZERO.checked_sub_u64(1), None);
        assert_eq!(U1024::from(5u64).checked_sub_u64(5), Some(U1024::ZERO));
        assert_eq!(
            U1024::pow2(700)
                .checked_sub_u64(1)
                .unwrap()
                .checked_add_u64(1),
            Some(U1024::pow2(700))
        );
    }

    #[test]
    fn order_is_numeric() {
        assert!(U1024::pow2(128) > U1024::from(u128::MAX));
        assert!(U1024::pow2(1023) > U1024::pow2(1022).checked_mul_u64(1).unwrap());
        assert!(U1024::from(5u64) < U1024::from(6u64));
        assert!(U1024::pow2(64) > U1024::from(u64::MAX));
        assert_eq!(U1024::default(), U1024::ZERO);
        let mut s = 3u64;
        for _ in 0..500 {
            let a = random_wide(&mut s, 16);
            let b = random_wide(&mut s, 16);
            let (sa, sb) = (a.to_be_bytes(), b.to_be_bytes());
            assert_eq!(
                a.cmp(&b),
                sa.cmp(&sb),
                "numeric order is big-endian byte order"
            );
        }
    }

    #[test]
    fn bytes_are_big_endian() {
        let one = U1024::ONE.to_be_bytes();
        assert_eq!(one[127], 1);
        assert!(one[..127].iter().all(|&b| b == 0));
        let top = U1024::pow2(1023).to_be_bytes();
        assert_eq!(top[0], 0x80);
        assert!(top[1..].iter().all(|&b| b == 0));
        let mut thirty_two = [0u8; 128];
        thirty_two[96] = 1;
        assert_eq!(U1024::from_be_bytes(thirty_two), U1024::pow2(248));
    }
}
