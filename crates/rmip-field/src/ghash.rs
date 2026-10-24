// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Vendored from Flock (`crates/flock-core/src/field/gf2_128.rs`), whose own
// header records that its default `Mul` is a port of `mul_clmul` from binius64
// (https://github.com/binius-zk/binius64, `crates/field/src/arch/shared/ghash.rs`).
// Changes made here: `serde` derives dropped (this crate has no dependencies),
// `new(lo, hi)` renamed `from_parts` so `F128::new` can keep taking a `u128`,
// and their `inv` dropped in favour of the `Option`-returning one in the
// `BinaryField` impl. The arithmetic is unmodified.
use core::ops::{Add, AddAssign, BitXor, BitXorAssign, Mul, MulAssign};

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct F128 {
    pub lo: u64,
    pub hi: u64,
}

impl F128 {
    pub const ZERO: Self = Self { lo: 0, hi: 0 };
    pub const ONE: Self = Self { lo: 1, hi: 0 };

    #[inline]
    pub const fn from_parts(lo: u64, hi: u64) -> Self {
        Self { lo, hi }
    }

    #[inline]
    pub const fn generator() -> Self {
        Self { lo: 2, hi: 0 }
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.lo == 0 && self.hi == 0
    }

    #[inline]
    pub fn square(self) -> Self {
        ghash_square(self)
    }

    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> F256Unreduced {
        ghash_mul_unreduced(self, rhs)
    }
}

#[inline]
pub fn dot_lazy(a: &[F128], b: &[F128]) -> F128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { aarch64::ghash_dot_lazy_neon(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let mut acc = F256Unreduced::ZERO;
        for (x, y) in a.iter().zip(b) {
            acc ^= x.mul_unreduced(*y);
        }
        acc.reduce()
    }
}

#[inline]
pub fn accumulate(acc: &mut F256Unreduced, a: F128, b: F128) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { aarch64::ghash_accumulate_neon(acc, a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        *acc ^= a.mul_unreduced(b);
    }
}

#[inline]
pub fn axpy_unreduced(acc: &mut [F256Unreduced], c: F128, row: &[F128]) {
    assert_eq!(acc.len(), row.len(), "one accumulator per element");
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { aarch64::ghash_axpy_unreduced_neon(acc, c, row) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        for (a, r) in acc.iter_mut().zip(row) {
            *a ^= c.mul_unreduced(*r);
        }
    }
}

impl Add for F128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo ^ rhs.lo,
            hi: self.hi ^ rhs.hi,
        }
    }
}

impl AddAssign for F128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
        self.hi ^= rhs.hi;
    }
}

impl Mul for F128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            unsafe { aarch64::ghash_mul_binius(self, rhs) }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
        {
            unsafe { x86_64::ghash_mul_karatsuba_barrett(self, rhs) }
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        )))]
        {
            software::ghash_mul(self, rhs)
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
    let carry = z.hi >> 63;
    let mask = 0u64.wrapping_sub(carry);
    F128 {
        lo: (z.lo << 1) ^ (0x87 & mask),
        hi: (z.hi << 1) | (z.lo >> 63),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F256Unreduced {
    pub r0: u64,
    pub r1: u64,
    pub r2: u64,
    pub r3: u64,
}

impl F256Unreduced {
    pub const ZERO: Self = Self {
        r0: 0,
        r1: 0,
        r2: 0,
        r3: 0,
    };

    #[inline]
    pub fn reduce(self) -> F128 {
        ghash_reduce(self.r0, self.r1, self.r2, self.r3)
    }
}

impl BitXor for F256Unreduced {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self {
            r0: self.r0 ^ rhs.r0,
            r1: self.r1 ^ rhs.r1,
            r2: self.r2 ^ rhs.r2,
            r3: self.r3 ^ rhs.r3,
        }
    }
}

impl BitXorAssign for F256Unreduced {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.r0 ^= rhs.r0;
        self.r1 ^= rhs.r1;
        self.r2 ^= rhs.r2;
        self.r3 ^= rhs.r3;
    }
}

#[inline]
pub fn ghash_reduce(r0: u64, r1: u64, r2: u64, r3: u64) -> F128 {
    let s1_lo = r2 << 1;
    let s1_hi = (r3 << 1) | (r2 >> 63);
    let s2_lo = r2 << 2;
    let s2_hi = (r3 << 2) | (r2 >> 62);
    let s7_lo = r2 << 7;
    let s7_hi = (r3 << 7) | (r2 >> 57);

    let t_lo = r2 ^ s1_lo ^ s2_lo ^ s7_lo;
    let t_hi = r3 ^ s1_hi ^ s2_hi ^ s7_hi;

    let ov = (r3 >> 63) ^ (r3 >> 62) ^ (r3 >> 57);
    let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

    F128 {
        lo: r0 ^ t_lo ^ corr,
        hi: r1 ^ t_hi,
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub mod aarch64;

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
pub mod x86_64;

#[path = "ghash/portable.rs"]
pub mod software;

#[inline]
pub fn ghash_mul_unreduced(a: F128, b: F128) -> F256Unreduced {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { aarch64::ghash_mul_unreduced_neon(a, b) }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        unsafe { x86_64::ghash_mul_unreduced_x86(a, b) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        software::ghash_mul_unreduced(a, b)
    }
}

#[inline]
pub fn ghash_square(a: F128) -> F128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { aarch64::ghash_square(a) }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        unsafe { x86_64::ghash_square_x86(a) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        software::ghash_square(a)
    }
}

#[inline]
pub fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { aarch64::ghash_mul_schoolbook(a, b) }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        unsafe { x86_64::ghash_mul_schoolbook(a, b) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        software::ghash_mul(a, b)
    }
}
