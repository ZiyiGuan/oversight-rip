use std::ops::{Add, AddAssign, Mul, MulAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct F64(u64);

pub const MODULUS_TAIL: u64 = 0x1B;

impl F64 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const X: Self = Self(2);
    pub const KAPPA: usize = 64;

    #[inline]
    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_u64(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn from_bool(b: bool) -> Self {
        Self(u64::from(b))
    }

    #[inline]
    pub fn square(self) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            unsafe { neon::mul(self, self) }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            let (lo, hi) = clmul64(self.0, self.0);
            Self(reduce(lo, hi))
        }
    }

    pub fn inv(self) -> Option<Self> {
        if self == Self::ZERO {
            return None;
        }
        let mut result = Self::ONE;
        let mut base = self;
        for i in 0..64 {
            base = base.square();
            if i < 63 {
                result *= base;
            }
        }
        Some(result)
    }

    pub fn pow(self, mut exp: u128) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            base = base.square();
            exp >>= 1;
        }
        result
    }

    pub fn to_be_bytes(self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    pub fn from_be_bytes(b: [u8; 8]) -> Self {
        Self(u64::from_be_bytes(b))
    }
}

impl crate::BinaryField for F64 {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const KAPPA: usize = 64;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn square(self) -> Self {
        F64::square(self)
    }
    #[inline]
    fn inv(self) -> Option<Self> {
        F64::inv(self)
    }
    #[inline]
    fn pow(self, exp: u128) -> Self {
        F64::pow(self, exp)
    }
    fn to_be_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[8..].copy_from_slice(&self.0.to_be_bytes());
        out
    }
    fn from_be_bytes(b: [u8; 16]) -> Self {
        Self(u64::from_be_bytes(b[8..].try_into().expect("eight bytes")))
    }
}

impl Add for F64 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl AddAssign for F64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for F64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            unsafe { neon::mul(self, rhs) }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            let (lo, hi) = clmul64(self.0, rhs.0);
            Self(reduce(lo, hi))
        }
    }
}

impl MulAssign for F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[inline]
pub fn reduce(lo: u64, hi: u64) -> u64 {
    let c = (hi >> 60) ^ (hi >> 61) ^ (hi >> 63);
    lo ^ hi ^ (hi << 1) ^ (hi << 3) ^ (hi << 4) ^ c ^ (c << 1) ^ (c << 3) ^ (c << 4)
}

#[inline]
pub fn clmul64(a: u64, b: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { aarch64_clmul64(a, b) }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        unsafe { x86_64_clmul64(a, b) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        software_clmul64(a, b)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn aarch64_clmul64(a: u64, b: u64) -> (u64, u64) {
    let p: u128 = core::arch::aarch64::vmull_p64(a, b);
    (p as u64, (p >> 64) as u64)
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline]
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn x86_64_clmul64(a: u64, b: u64) -> (u64, u64) {
    use core::arch::x86_64::*;
    let va = _mm_set_epi64x(0, a as i64);
    let vb = _mm_set_epi64x(0, b as i64);
    let p = _mm_clmulepi64_si128::<0x00>(va, vb);
    let lo = _mm_cvtsi128_si64(p) as u64;
    let hi = _mm_cvtsi128_si64(_mm_unpackhi_epi64(p, p)) as u64;
    (lo, hi)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Product64 {
    pub lo: u64,
    pub hi: u64,
}

impl Product64 {
    pub const ZERO: Self = Self { lo: 0, hi: 0 };

    #[inline]
    pub fn reduce(self) -> F64 {
        F64(reduce(self.lo, self.hi))
    }
}

impl std::ops::BitXorAssign for Product64 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
        self.hi ^= rhs.hi;
    }
}

impl F64 {
    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> Product64 {
        let (lo, hi) = clmul64(self.0, rhs.0);
        Product64 { lo, hi }
    }
}

#[inline]
pub fn dot_lazy(a: &[F64], b: &[F64]) -> F64 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { neon::dot_lazy(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let mut acc = Product64::ZERO;
        for (x, y) in a.iter().zip(b) {
            acc ^= x.mul_unreduced(*y);
        }
        acc.reduce()
    }
}

#[inline]
pub fn axpy_unreduced(acc: &mut [Product64], c: F64, row: &[F64]) {
    assert_eq!(acc.len(), row.len(), "one accumulator per element");
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { neon::axpy_unreduced(acc, c, row) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        for (a, r) in acc.iter_mut().zip(row) {
            *a ^= c.mul_unreduced(*r);
        }
    }
}

#[inline]
pub fn accumulate(acc: &mut Product64, a: F64, b: F64) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { neon::accumulate(acc, a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        *acc ^= a.mul_unreduced(b);
    }
}

pub fn reduce_products(acc: &[Product64]) -> Vec<F64> {
    acc.iter().map(|p| p.reduce()).collect()
}

#[inline]
pub fn mul2(a: [F64; 2], b: [F64; 2]) -> [F64; 2] {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { neon::mul2(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        [a[0] * b[0], a[1] * b[1]]
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod neon {
    use super::{Product64, F64};
    use core::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn mul2_vec(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        let pa = vreinterpretq_p64_u64(a);
        let pb = vreinterpretq_p64_u64(b);
        let p0 =
            vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<0>(pa), vgetq_lane_p64::<0>(pb)));
        let p1 = vreinterpretq_u64_p128(vmull_high_p64(pa, pb));
        let lo = vzip1q_u64(p0, p1);
        let hi = vzip2q_u64(p0, p1);
        let c = veorq_u64(
            veorq_u64(vshrq_n_u64::<60>(hi), vshrq_n_u64::<61>(hi)),
            vshrq_n_u64::<63>(hi),
        );
        let fold = |v: uint64x2_t| {
            veorq_u64(
                veorq_u64(v, vshlq_n_u64::<1>(v)),
                veorq_u64(vshlq_n_u64::<3>(v), vshlq_n_u64::<4>(v)),
            )
        };
        veorq_u64(veorq_u64(lo, fold(hi)), fold(c))
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn mul(a: F64, b: F64) -> F64 {
        let tail = vreinterpretq_p64_u64(vdupq_n_u64(super::MODULUS_TAIL));
        let p = vreinterpretq_u64_p128(vmull_p64(a.0, b.0));
        let t = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p), tail));
        let u = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t), tail));
        F64(vgetq_lane_u64::<0>(veorq_u64(veorq_u64(p, t), u)))
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn dot_lazy(a: &[F64], b: &[F64]) -> F64 {
        let n = a.len().min(b.len());
        let pa = a.as_ptr().cast::<u64>();
        let pb = b.as_ptr().cast::<u64>();
        let mut acc0 = vdupq_n_u64(0);
        let mut acc1 = vdupq_n_u64(0);
        let mut acc2 = vdupq_n_u64(0);
        let mut acc3 = vdupq_n_u64(0);
        let mut i = 0usize;
        while i + 4 <= n {
            let va = vreinterpretq_p64_u64(vld1q_u64(pa.add(i)));
            let vb = vreinterpretq_p64_u64(vld1q_u64(pb.add(i)));
            let va2 = vreinterpretq_p64_u64(vld1q_u64(pa.add(i + 2)));
            let vb2 = vreinterpretq_p64_u64(vld1q_u64(pb.add(i + 2)));
            acc0 = veorq_u64(
                acc0,
                vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<0>(va), vgetq_lane_p64::<0>(vb))),
            );
            acc1 = veorq_u64(acc1, vreinterpretq_u64_p128(vmull_high_p64(va, vb)));
            acc2 = veorq_u64(
                acc2,
                vreinterpretq_u64_p128(vmull_p64(
                    vgetq_lane_p64::<0>(va2),
                    vgetq_lane_p64::<0>(vb2),
                )),
            );
            acc3 = veorq_u64(acc3, vreinterpretq_u64_p128(vmull_high_p64(va2, vb2)));
            i += 4;
        }
        let mut acc = veorq_u64(veorq_u64(acc0, acc1), veorq_u64(acc2, acc3));
        while i < n {
            acc = veorq_u64(
                acc,
                vreinterpretq_u64_p128(vmull_p64(*pa.add(i), *pb.add(i))),
            );
            i += 1;
        }
        Product64 {
            lo: vgetq_lane_u64::<0>(acc),
            hi: vgetq_lane_u64::<1>(acc),
        }
        .reduce()
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn accumulate(acc: &mut Product64, a: F64, b: F64) {
        let p = vreinterpretq_u64_p128(vmull_p64(a.0, b.0));
        let ptr = (acc as *mut Product64).cast::<u64>();
        vst1q_u64(ptr, veorq_u64(vld1q_u64(ptr), p));
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn axpy_unreduced(acc: &mut [Product64], c: F64, row: &[F64]) {
        let n = row.len();
        let pr = row.as_ptr().cast::<u64>();
        let pacc = acc.as_mut_ptr().cast::<u64>();
        for i in 0..n {
            let p = vreinterpretq_u64_p128(vmull_p64(c.0, *pr.add(i)));
            let slot = pacc.add(2 * i);
            vst1q_u64(slot, veorq_u64(vld1q_u64(slot), p));
        }
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn mul2(a: [F64; 2], b: [F64; 2]) -> [F64; 2] {
        let va = vld1q_u64(a.as_ptr().cast::<u64>());
        let vb = vld1q_u64(b.as_ptr().cast::<u64>());
        let r = mul2_vec(va, vb);
        [F64(vgetq_lane_u64::<0>(r)), F64(vgetq_lane_u64::<1>(r))]
    }

    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn load4(x: &[F64; 8]) -> [uint64x2_t; 4] {
        let p = x.as_ptr().cast::<u64>();
        [
            vld1q_u64(p),
            vld1q_u64(p.add(2)),
            vld1q_u64(p.add(4)),
            vld1q_u64(p.add(6)),
        ]
    }

    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn store4(x: &mut [F64; 8], v: [uint64x2_t; 4]) {
        let p = x.as_mut_ptr().cast::<u64>();
        vst1q_u64(p, v[0]);
        vst1q_u64(p.add(2), v[1]);
        vst1q_u64(p.add(4), v[2]);
        vst1q_u64(p.add(6), v[3]);
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn dot8(a: &[F64; 8], b: &[F64; 8]) -> F64 {
        let va = load4(a);
        let vb = load4(b);
        let s0 = veorq_u64(mul2_vec(va[0], vb[0]), mul2_vec(va[1], vb[1]));
        let s1 = veorq_u64(mul2_vec(va[2], vb[2]), mul2_vec(va[3], vb[3]));
        let s = veorq_u64(s0, s1);
        F64(vgetq_lane_u64::<0>(s) ^ vgetq_lane_u64::<1>(s))
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn square8(y: &mut [F64; 8]) {
        let v = load4(y);
        store4(
            y,
            [
                mul2_vec(v[0], v[0]),
                mul2_vec(v[1], v[1]),
                mul2_vec(v[2], v[2]),
                mul2_vec(v[3], v[3]),
            ],
        );
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn scale_add8(acc: &mut [F64; 8], c: F64, y: &[F64; 8]) {
        let vc = vdupq_n_u64(c.0);
        let vy = load4(y);
        let va = load4(acc);
        store4(
            acc,
            [
                veorq_u64(va[0], mul2_vec(vc, vy[0])),
                veorq_u64(va[1], mul2_vec(vc, vy[1])),
                veorq_u64(va[2], mul2_vec(vc, vy[2])),
                veorq_u64(va[3], mul2_vec(vc, vy[3])),
            ],
        );
    }
}

#[inline]
pub fn dot8(a: &[F64; 8], b: &[F64; 8]) -> F64 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { neon::dot8(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let mut acc = F64::ZERO;
        for (x, y) in a.iter().zip(b) {
            acc += *x * *y;
        }
        acc
    }
}

#[inline]
pub fn square8(y: &mut [F64; 8]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { neon::square8(y) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        for v in y.iter_mut() {
            *v = v.square();
        }
    }
}

#[inline]
pub fn scale_add8(acc: &mut [F64; 8], c: F64, y: &[F64; 8]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { neon::scale_add8(acc, c, y) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        for (a, v) in acc.iter_mut().zip(y) {
            *a += c * *v;
        }
    }
}

pub fn dot(a: &[F64], b: &[F64]) -> F64 {
    assert_eq!(a.len(), b.len());
    let mut acc = [[F64::ZERO; 2]; 2];
    let (quads_a, rest_a) = a.as_chunks::<4>();
    let (quads_b, rest_b) = b.as_chunks::<4>();
    for (ca, cb) in quads_a.iter().zip(quads_b) {
        let p = mul2([ca[0], ca[1]], [cb[0], cb[1]]);
        let q = mul2([ca[2], ca[3]], [cb[2], cb[3]]);
        acc[0][0] += p[0];
        acc[0][1] += p[1];
        acc[1][0] += q[0];
        acc[1][1] += q[1];
    }
    let mut tail = F64::ZERO;
    for (x, y) in rest_a.iter().zip(rest_b) {
        tail += *x * *y;
    }
    acc[0][0] + acc[0][1] + acc[1][0] + acc[1][1] + tail
}

pub fn software_clmul64(a: u64, b: u64) -> (u64, u64) {
    let mut acc: u128 = 0;
    for i in 0..64 {
        if (b >> i) & 1 == 1 {
            acc ^= (a as u128) << i;
        }
    }
    (acc as u64, (acc >> 64) as u64)
}

pub fn reference_mul(a: F64, b: F64) -> F64 {
    let (lo, hi) = software_clmul64(a.0, b.0);
    let mut acc: u128 = ((hi as u128) << 64) | lo as u128;
    let p: u128 = (1u128 << 64) | MODULUS_TAIL as u128;
    for degree in (64..128).rev() {
        if (acc >> degree) & 1 == 1 {
            acc ^= p << (degree - 64);
        }
    }
    F64(acc as u64)
}
