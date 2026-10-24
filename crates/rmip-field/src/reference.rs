pub const REDUCTION: u128 = (1 << 7) | (1 << 2) | (1 << 1) | 1;

#[inline]
pub fn clmul64(a: u64, b: u64) -> u128 {
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

#[inline]
pub fn clmul128_schoolbook(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let lo = clmul64(a0, b0);
    let hi = clmul64(a1, b1);
    let mid = clmul64(a1, b0) ^ clmul64(a0, b1);
    (hi ^ (mid >> 64), lo ^ (mid << 64))
}

#[inline]
pub fn clmul128_karatsuba(a: u128, b: u128) -> (u128, u128) {
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
pub fn reduce(hi: u128, lo: u128) -> u128 {
    let (c1, r1) = mul_by_reduction(hi);
    let (c2, r2) = mul_by_reduction(c1);
    debug_assert_eq!(c2, 0, "the second fold must not carry");
    lo ^ r1 ^ r2
}

#[inline]
pub fn mul(a: u128, b: u128) -> u128 {
    let (hi, lo) = clmul128_schoolbook(a, b);
    reduce(hi, lo)
}

#[inline]
pub fn mul_karatsuba(a: u128, b: u128) -> u128 {
    let (hi, lo) = clmul128_karatsuba(a, b);
    reduce(hi, lo)
}

#[inline]
pub fn square(a: u128) -> u128 {
    mul(a, a)
}
