use std::collections::HashMap;

use crate::{skip_domain, PackedField};

pub struct Subfield8<F: PackedField> {
    pub embed: Vec<F>,
    decode: HashMap<u128, u8>,
    log: [u8; 256],
    antilog: [u8; 255],
}

impl<F: PackedField> Default for Subfield8<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PackedField> Subfield8<F> {
    pub fn new() -> Self {
        let g = F::subfield_8_generator();
        let xi = F::subfield_8_root();
        let basis: Vec<F> = (0..8)
            .scan(F::ONE, |p, _| {
                let out = *p;
                *p *= xi;
                Some(out)
            })
            .collect();
        let embed: Vec<F> = (0..256u32)
            .map(|byte| {
                let mut e = F::ZERO;
                for (i, b) in basis.iter().enumerate() {
                    if (byte >> i) & 1 == 1 {
                        e += *b;
                    }
                }
                e
            })
            .collect();
        let decode: HashMap<u128, u8> = embed
            .iter()
            .enumerate()
            .map(|(byte, e)| (e.to_bits(), byte as u8))
            .collect();
        assert_eq!(decode.len(), 256, "the basis is independent over GF(2)");
        let mut log = [0u8; 256];
        let mut antilog = [0u8; 255];
        let mut power = F::ONE;
        for (k, slot) in antilog.iter_mut().enumerate() {
            let byte = decode[&power.to_bits()];
            *slot = byte;
            log[byte as usize] = k as u8;
            power *= g;
        }
        assert_eq!(power, F::ONE, "g has order 255");
        let table = Self {
            embed,
            decode,
            log,
            antilog,
        };
        for a in [1u8, 2, 3, 7, 19, 128, 255] {
            for b in [1u8, 2, 5, 11, 97, 200, 254] {
                assert_eq!(
                    table.embed[table.mul(a, b) as usize],
                    table.embed[a as usize] * table.embed[b as usize]
                );
            }
        }
        table
    }

    pub fn decode(&self, e: F) -> u8 {
        *self
            .decode
            .get(&e.to_bits())
            .expect("an element of GF(2^8)")
    }

    pub fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        let k = (self.log[a as usize] as usize + self.log[b as usize] as usize) % 255;
        self.antilog[k]
    }

    pub fn inv(&self, a: u8) -> u8 {
        assert_ne!(a, 0, "inverse of zero");
        self.antilog[(255 - self.log[a as usize] as usize) % 255]
    }

    #[inline]
    pub fn mul64(&self, a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon::mul64(a, b) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut out = [0u8; 64];
            for i in 0..64 {
                out[i] = self.mul(a[i], b[i]);
            }
            out
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn mul8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
        let m = vdup_n_p8(0x1B);
        let p = vreinterpretq_u16_p16(vmull_p8(vreinterpret_p8_u8(a), vreinterpret_p8_u8(b)));
        let h = vmovn_u16(vshrq_n_u16::<8>(p));
        let t = vreinterpretq_u16_p16(vmull_p8(vreinterpret_p8_u8(h), m));
        let h2 = vmovn_u16(vshrq_n_u16::<8>(t));
        let t2 = vreinterpretq_u16_p16(vmull_p8(vreinterpret_p8_u8(h2), m));
        vmovn_u16(veorq_u16(veorq_u16(p, t), t2))
    }

    #[inline(always)]
    unsafe fn gather<const BH: usize, const ODD: bool>(
        table: *const u8,
        byte: u8,
        d: &mut [uint8x16_t; 4],
    ) {
        let r = table.add(byte as usize * 64);
        for (q, dq) in d.iter_mut().enumerate() {
            let mut v = vld1q_u8(r.add((q ^ BH) * 16));
            if ODD {
                v = vextq_u8::<8>(v, v);
            }
            *dq = veorq_u8(*dq, v);
        }
    }

    #[inline(always)]
    unsafe fn on_lambda_regs(table: *const u8, word: u64) -> [uint8x16_t; 4] {
        let r = table.add((word & 0xFF) as usize * 64);
        let mut d = [
            vld1q_u8(r),
            vld1q_u8(r.add(16)),
            vld1q_u8(r.add(32)),
            vld1q_u8(r.add(48)),
        ];
        gather::<0, true>(table, (word >> 8) as u8, &mut d);
        gather::<1, false>(table, (word >> 16) as u8, &mut d);
        gather::<1, true>(table, (word >> 24) as u8, &mut d);
        gather::<2, false>(table, (word >> 32) as u8, &mut d);
        gather::<2, true>(table, (word >> 40) as u8, &mut d);
        gather::<3, false>(table, (word >> 48) as u8, &mut d);
        gather::<3, true>(table, (word >> 56) as u8, &mut d);
        d
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn on_lambda(table: &[super::Row; 256], bits: u64) -> [u8; 64] {
        let d = on_lambda_regs(table.as_ptr() as *const u8, bits);
        let mut out = [0u8; 64];
        for (q, v) in d.iter().enumerate() {
            vst1q_u8(out.as_mut_ptr().add(16 * q), *v);
        }
        out
    }

    #[inline(always)]
    unsafe fn mul_x2(v: uint8x16_t) -> uint8x16_t {
        let m7 = vreinterpretq_u8_s8(vshrq_n_s8::<7>(vreinterpretq_s8_u8(v)));
        let m6 = vreinterpretq_u8_s8(vshrq_n_s8::<7>(vreinterpretq_s8_u8(vshlq_n_u8::<1>(v))));
        veorq_u8(
            veorq_u8(vshlq_n_u8::<2>(v), vandq_u8(m7, vdupq_n_u8(0x36))),
            vandq_u8(m6, vdupq_n_u8(0x1B)),
        )
    }

    #[inline(always)]
    unsafe fn reduce8(p: uint16x8_t) -> uint8x8_t {
        let m = vdup_n_p8(0x1B);
        let h = vmovn_u16(vshrq_n_u16::<8>(p));
        let t = vreinterpretq_u16_p16(vmull_p8(vreinterpret_p8_u8(h), m));
        let h2 = vmovn_u16(vshrq_n_u16::<8>(t));
        let t2 = vreinterpretq_u16_p16(vmull_p8(vreinterpret_p8_u8(h2), m));
        vmovn_u16(veorq_u16(veorq_u16(p, t), t2))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn products8(
        base: &[super::Row; 256],
        base_x4: &[super::Row; 256],
        a: &[u64; 8],
        b: &[u64; 8],
    ) -> [u8; 64] {
        let t0 = base.as_ptr() as *const u8;
        let t4 = base_x4.as_ptr() as *const u8;
        let mut acc = [vdupq_n_u16(0); 8];
        macro_rules! block {
            ($k:literal) => {{
                let (ak, bk) = (a[$k], b[$k]);
                if ak != 0 && bk != 0 {
                    let mut da = on_lambda_regs(if $k >= 4 { t4 } else { t0 }, ak);
                    let db = on_lambda_regs(t0, bk);
                    if ($k >> 1) & 1 == 1 {
                        for v in da.iter_mut() {
                            *v = mul_x2(*v);
                        }
                    }
                    for q in 0..4 {
                        let lo = vreinterpretq_u16_p16(vmull_p8(
                            vreinterpret_p8_u8(vget_low_u8(da[q])),
                            vreinterpret_p8_u8(vget_low_u8(db[q])),
                        ));
                        let hi = vreinterpretq_u16_p16(vmull_p8(
                            vreinterpret_p8_u8(vget_high_u8(da[q])),
                            vreinterpret_p8_u8(vget_high_u8(db[q])),
                        ));
                        if $k & 1 == 1 {
                            acc[2 * q] = veorq_u16(acc[2 * q], vshlq_n_u16::<1>(lo));
                            acc[2 * q + 1] = veorq_u16(acc[2 * q + 1], vshlq_n_u16::<1>(hi));
                        } else {
                            acc[2 * q] = veorq_u16(acc[2 * q], lo);
                            acc[2 * q + 1] = veorq_u16(acc[2 * q + 1], hi);
                        }
                    }
                }
            }};
        }
        block!(0);
        block!(1);
        block!(2);
        block!(3);
        block!(4);
        block!(5);
        block!(6);
        block!(7);
        let mut out = [0u8; 64];
        for (i, p) in acc.iter().enumerate() {
            vst1_u8(out.as_mut_ptr().add(8 * i), reduce8(*p));
        }
        out
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn transpose_8x64(w: &[u64; 8]) -> [u8; 64] {
        let p = w.as_ptr() as *const u8;
        let table = uint8x16x4_t(
            vld1q_u8(p),
            vld1q_u8(p.add(16)),
            vld1q_u8(p.add(32)),
            vld1q_u8(p.add(48)),
        );
        const IDX: [[u8; 16]; 4] = [
            [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57],
            [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59],
            [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61],
            [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63],
        ];
        let mut out = [0u8; 64];
        for (j, idx) in IDX.iter().enumerate() {
            let mut y = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(idx.as_ptr())));
            let t = vandq_u64(
                veorq_u64(y, vshrq_n_u64::<7>(y)),
                vdupq_n_u64(0x00AA_00AA_00AA_00AA),
            );
            y = veorq_u64(y, veorq_u64(t, vshlq_n_u64::<7>(t)));
            let t = vandq_u64(
                veorq_u64(y, vshrq_n_u64::<14>(y)),
                vdupq_n_u64(0x0000_CCCC_0000_CCCC),
            );
            y = veorq_u64(y, veorq_u64(t, vshlq_n_u64::<14>(t)));
            let t = vandq_u64(
                veorq_u64(y, vshrq_n_u64::<28>(y)),
                vdupq_n_u64(0x0000_0000_F0F0_F0F0),
            );
            y = veorq_u64(y, veorq_u64(t, vshlq_n_u64::<28>(t)));
            vst1q_u8(out.as_mut_ptr().add(16 * j), vreinterpretq_u8_u64(y));
        }
        out
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mul64(a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
        let mut out = [0u8; 64];
        for i in 0..8 {
            let va = vld1_u8(a.as_ptr().add(8 * i));
            let vb = vld1_u8(b.as_ptr().add(8 * i));
            vst1_u8(out.as_mut_ptr().add(8 * i), mul8(va, vb));
        }
        out
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Row(pub [u8; 64]);

#[inline]
pub fn transpose_8x64(w: &[u64; 8]) -> [u8; 64] {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::transpose_8x64(w) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        transpose_8x64_portable(w)
    }
}

pub fn transpose_8x64_portable(w: &[u64; 8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (j, chunk) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut x = 0u64;
        for (k, word) in w.iter().enumerate() {
            x |= ((word >> (8 * j)) & 0xFF) << (8 * k);
        }
        let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
        x ^= t ^ (t << 7);
        let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
        x ^= t ^ (t << 14);
        let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
        x ^= t ^ (t << 28);
        chunk.copy_from_slice(&x.to_le_bytes());
    }
    out
}

pub struct SkipTables<F: PackedField> {
    pub subfield: Subfield8<F>,
    pub s: Vec<F>,
    pub lambda: Vec<F>,
    pub matrix: Vec<[u8; 64]>,
    base: Box<[Row; 256]>,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    base_x4: Box<[Row; 256]>,
    s_inv_den: Vec<F>,
    lambda_inv_den: Vec<F>,
    full_inv_den: Vec<F>,
    pub fixed: FixedEq<F>,
}

pub struct FixedEq<F: PackedField> {
    pub count: usize,
    pub r: Vec<F>,
    pub scale: Vec<F>,
    convert: Vec<[F; 256]>,
    reduce: [u8; 256],
}

impl<F: PackedField> FixedEq<F> {
    fn new(subfield: &Subfield8<F>) -> Self {
        let xi = F::subfield_8_root();
        let degree = F::BITS / 8;
        let medium_log = degree.trailing_zeros() as usize;
        let count = 3 + medium_log;
        let gamma = F::from_bits(2);
        let mut gammas = vec![F::ONE; 1 << medium_log];
        for b in 1..gammas.len() {
            gammas[b] = gammas[b - 1] * gamma;
        }
        let mut r = Vec::with_capacity(count);
        let mut scale = vec![F::ONE];
        for i in 0..count {
            let a = if i < 3 {
                xi.pow(1u128 << i)
            } else {
                gamma.pow(1u128 << (i - 3))
            };
            let ri = a * (F::ONE + a).inv().expect("a ≠ 1");
            scale.push(scale[i] * (F::ONE + ri));
            r.push(ri);
        }
        let mut rows: Vec<u128> = Vec::with_capacity(1 << count);
        for g in &gammas {
            for k in 0..8 {
                rows.push((xi.pow(k as u128) * *g).to_bits());
            }
        }
        let mut basis: Vec<u128> = Vec::new();
        for mut v in rows {
            for &b in &basis {
                if v & (1u128 << (127 - b.leading_zeros())) != 0 {
                    v ^= b;
                }
            }
            assert!(v != 0, "the fixed eq weights must be GF(2)-independent");
            basis.push(v);
            basis.sort_unstable_by(|a, b| b.cmp(a));
        }
        let convert: Vec<[F; 256]> = gammas
            .iter()
            .map(|g| {
                let mut t = [F::ZERO; 256];
                for (v, slot) in t.iter_mut().enumerate() {
                    *slot = *g * subfield.embed[v];
                }
                t
            })
            .collect();
        let mut reduce = [0u8; 256];
        for (h, slot) in reduce.iter_mut().enumerate() {
            *slot = subfield.mul(h as u8, 0x1B);
        }
        Self {
            count,
            r,
            scale,
            convert,
            reduce,
        }
    }

    #[inline]
    pub fn convert_rows(&self, s: &[[u8; 64]]) -> [F; 64] {
        let mut out = [F::ZERO; 64];
        for (t, row) in self.convert.iter().zip(s) {
            if row.as_chunks::<8>().0.iter().all(|c| *c == [0u8; 8]) {
                continue;
            }
            for l in 0..64 {
                out[l] += t[row[l] as usize];
            }
        }
        out
    }

    #[inline]
    pub fn reduce16(&self, p: u16) -> u8 {
        (p as u8) ^ self.reduce[(p >> 8) as usize]
    }
}

fn inverse_denominators<F: PackedField>(nodes: &[F]) -> Vec<F> {
    (0..nodes.len())
        .map(|j| {
            let mut den = F::ONE;
            for (i, n) in nodes.iter().enumerate() {
                if i != j {
                    den *= nodes[j] + *n;
                }
            }
            den.inv().expect("distinct nodes")
        })
        .collect()
}

fn lagrange_weights_with<F: PackedField>(nodes: &[F], inv_den: &[F], z: F) -> Vec<F> {
    let n = nodes.len();
    let mut prefix = vec![F::ONE; n + 1];
    for (i, node) in nodes.iter().enumerate() {
        prefix[i + 1] = prefix[i] * (z + *node);
    }
    let mut suffix = F::ONE;
    let mut out = vec![F::ZERO; n];
    for j in (0..n).rev() {
        out[j] = prefix[j] * suffix * inv_den[j];
        suffix *= z + nodes[j];
    }
    out
}

impl<F: PackedField> Default for SkipTables<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PackedField> SkipTables<F> {
    pub fn new() -> Self {
        let subfield = Subfield8::<F>::new();
        let full = skip_domain::<F>(7);
        let s: Vec<F> = full[..64].to_vec();
        let lambda: Vec<F> = full[64..].to_vec();
        let sb: Vec<u8> = s.iter().map(|p| subfield.decode(*p)).collect();
        let lb: Vec<u8> = lambda.iter().map(|p| subfield.decode(*p)).collect();
        let mut matrix = vec![[0u8; 64]; 64];
        for (li, &l) in lb.iter().enumerate() {
            for j in 0..64 {
                let mut num = 1u8;
                let mut den = 1u8;
                for i in 0..64 {
                    if i != j {
                        num = subfield.mul(num, l ^ sb[i]);
                        den = subfield.mul(den, sb[j] ^ sb[i]);
                    }
                }
                matrix[li][j] = subfield.mul(num, subfield.inv(den));
            }
        }
        let mut base: Box<[Row; 256]> = Box::new([Row([0u8; 64]); 256]);
        for (byte, row) in base.iter_mut().enumerate() {
            for (li, out) in row.0.iter_mut().enumerate() {
                for (bit, &w) in matrix[li][..8].iter().enumerate() {
                    if (byte >> bit) & 1 == 1 {
                        *out ^= w;
                    }
                }
            }
        }
        for (li, row) in matrix.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                assert_eq!(
                    w,
                    matrix[li ^ (j & !7)][j & 7],
                    "L_j(λ) = L_(j mod 8)(λ + s_(j − j mod 8))"
                );
            }
        }
        let mut base_x4: Box<[Row; 256]> = Box::new([Row([0u8; 64]); 256]);
        for (dst, src) in base_x4.iter_mut().zip(base.iter()) {
            for (d, &v) in dst.0.iter_mut().zip(&src.0) {
                *d = subfield.mul(v, 0x10);
            }
        }
        let full: Vec<F> = s.iter().chain(&lambda).copied().collect();
        let s_inv_den = inverse_denominators(&s);
        let lambda_inv_den = inverse_denominators(&lambda);
        let full_inv_den = inverse_denominators(&full);
        let fixed = FixedEq::new(&subfield);
        Self {
            subfield,
            s,
            lambda,
            matrix,
            base,
            base_x4,
            s_inv_den,
            lambda_inv_den,
            full_inv_den,
            fixed,
        }
    }

    #[inline]
    pub fn on_lambda(&self, bits: u64) -> [u8; 64] {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon::on_lambda(&self.base, bits) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.on_lambda_portable(bits)
        }
    }

    pub fn on_lambda_portable(&self, bits: u64) -> [u8; 64] {
        let mut out = [0u8; 64];
        for c in 0..8 {
            let row = &self.base[((bits >> (8 * c)) & 0xFF) as usize].0;
            for (l, o) in out.iter_mut().enumerate() {
                *o ^= row[l ^ (8 * c)];
            }
        }
        out
    }

    #[inline]
    pub fn products8(&self, a: &[u64; 8], b: &[u64; 8]) -> [u8; 64] {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon::products8(&self.base, &self.base_x4, a, b) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.products8_portable(a, b)
        }
    }

    pub fn products8_portable(&self, a: &[u64; 8], b: &[u64; 8]) -> [u8; 64] {
        let mut acc = [0u16; 64];
        for k in 0..8 {
            if a[k] == 0 || b[k] == 0 {
                continue;
            }
            let ab = self.subfield.mul64(
                &self.on_lambda_portable(a[k]),
                &self.on_lambda_portable(b[k]),
            );
            for (l, v) in ab.iter().enumerate() {
                acc[l] ^= u16::from(*v) << k;
            }
        }
        let mut out = [0u8; 64];
        for (o, &p) in out.iter_mut().zip(&acc) {
            *o = self.fixed.reduce16(p);
        }
        out
    }

    pub fn extend_to_lambda(&self, v: &[F; 64]) -> [F; 64] {
        let mut out = [F::ZERO; 64];
        for (o, row) in out.iter_mut().zip(&self.matrix) {
            for (s, &w) in row.iter().enumerate() {
                if w != 0 {
                    *o += self.subfield.embed[w as usize] * v[s];
                }
            }
        }
        out
    }

    pub fn fold_weights(&self, z: F) -> Vec<F> {
        lagrange_weights_with(&self.s, &self.s_inv_den, z)
    }

    pub fn lambda_weights(&self, z: F) -> Vec<F> {
        lagrange_weights_with(&self.lambda, &self.lambda_inv_den, z)
    }

    pub fn combined_lambda_weights(&self, z: F) -> Vec<F> {
        let nodes: Vec<F> = self.s.iter().chain(&self.lambda).copied().collect();
        lagrange_weights_with(&nodes, &self.full_inv_den, z)[64..].to_vec()
    }
}

pub fn lagrange_weights<F: PackedField>(nodes: &[F], z: F) -> Vec<F> {
    (0..nodes.len())
        .map(|j| {
            let mut num = F::ONE;
            let mut den = F::ONE;
            for (i, n) in nodes.iter().enumerate() {
                if i != j {
                    num *= z + *n;
                    den *= nodes[j] + *n;
                }
            }
            num * den.inv().expect("distinct nodes")
        })
        .collect()
}
