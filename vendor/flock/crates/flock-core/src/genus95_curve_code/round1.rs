//! Production round-1 AG-code univariate round message kernel — eq-folded encode·product·fold, fused
//! AB+C in a single pass over the witness.
//!
//! Lifted verbatim from `benches/urm_bitslice.rs` (auto-generated genus-95
//! by-point encode `M` (160×64) + Hasse/Leibniz product + GHASH fold), which is
//! cross-checked byte-identical to a scalar reference. `aarch64`/NEON only (the
//! production target); the packed-witness input path, the eq-from-`r` derivation,
//! the 160→222 layout bridge, the `product_code_message` cross-check, and Metal
//! land in later M1/M3 steps.
//!
//! Output is the 160 by-point FRESH product-code coordinates
//! (D¹·64 | D²·64 | D³·32, two garbage rows at the D³ points 28/31). The 64
//! systematic "value" coordinates are the raw witness product; the verifier
//! reconstructs them from the zerocheck identity, so they are not emitted here.

use std::{
    arch::aarch64::{
        uint8x16_t, uint64x2_t, vandq_u8, vandq_u64, vbslq_u8, vcombine_u8, vdupq_n_u8,
        vdupq_n_u64, veorq_u8, veorq_u64, vgetq_lane_u64, vld1_u8, vld1q_u8, vmull_p64,
        vreinterpretq_u8_u16, vreinterpretq_u8_u32, vreinterpretq_u8_u64, vreinterpretq_u16_u8,
        vreinterpretq_u32_u8, vreinterpretq_u64_p128, vreinterpretq_u64_u8, vshlq_n_u8, vshrq_n_u8,
        vst1q_u8, vtrn1q_u8, vtrn1q_u16, vtrn1q_u32, vtrn1q_u64, vtrn2q_u8, vtrn2q_u16, vtrn2q_u32,
        vtrn2q_u64,
    },
    sync::OnceLock,
};

use rayon::{
    current_num_threads,
    prelude::{IntoParallelIterator, ParallelIterator},
};

use crate::{
    field::{F128, F256Unreduced},
    genus95_curve_code::{
        messages::BaseMessage, product::extended_base_product_message,
        slp_derived::encode_straight_line_program_derived,
    },
    zerocheck::BlockCoverage,
};

// One binary AG message is 64 bits. The fixed friendly weights must be
// linearly independent, so each block has exactly the active field degree
// messages (128 natively; 64 with field64), not a reduced 128-coefficient
// polynomial over the smaller field.
const BLOCK_BYTES: usize = 8 * F128::BITS;

#[inline(always)]
unsafe fn product_bs(
    af: &[uint8x16_t; 160],
    bf: &[uint8x16_t; 160],
    ax: &[uint8x16_t; 64],
    bx: &[uint8x16_t; 64],
    out: &mut [uint8x16_t; 160],
) {
    unsafe {
        for p in 0..64 {
            out[p] = veorq_u8(vandq_u8(af[p], bx[p]), vandq_u8(ax[p], bf[p]));
        }
        for p in 0..64 {
            out[64 + p] = veorq_u8(
                veorq_u8(vandq_u8(af[64 + p], bx[p]), vandq_u8(af[p], bf[p])),
                vandq_u8(ax[p], bf[64 + p]),
            );
        }
        for p in 0..32 {
            out[128 + p] = veorq_u8(
                veorq_u8(vandq_u8(af[128 + p], bx[p]), vandq_u8(af[64 + p], bf[p])),
                veorq_u8(vandq_u8(af[p], bf[64 + p]), vandq_u8(ax[p], bf[128 + p])),
            );
        }
    }
}

#[inline(always)]
unsafe fn fold_bs(prod: &[uint8x16_t; 160], eq: F128, res: &mut [F128; 160]) {
    unsafe {
        for j in 0..160 {
            let pf = vreinterpretq_u64_u8(prod[j]);
            res[j] += eq
                * F128::from_polynomial(
                    u128::from(vgetq_lane_u64::<0>(pf))
                        | (u128::from(vgetq_lane_u64::<1>(pf)) << 64),
                );
        }
    }
}

#[inline]
unsafe fn transpose16x16(r: &mut [uint8x16_t; 16]) {
    unsafe {
        for i in 0..16 {
            if i & 1 == 0 {
                let (x, y) = (r[i], r[i + 1]);
                r[i] = vtrn1q_u8(x, y);
                r[i + 1] = vtrn2q_u8(x, y);
            }
        }
        for i in 0..16 {
            if i & 2 == 0 {
                let x = vreinterpretq_u16_u8(r[i]);
                let y = vreinterpretq_u16_u8(r[i + 2]);
                r[i] = vreinterpretq_u8_u16(vtrn1q_u16(x, y));
                r[i + 2] = vreinterpretq_u8_u16(vtrn2q_u16(x, y));
            }
        }
        for i in 0..16 {
            if i & 4 == 0 {
                let x = vreinterpretq_u32_u8(r[i]);
                let y = vreinterpretq_u32_u8(r[i + 4]);
                r[i] = vreinterpretq_u8_u32(vtrn1q_u32(x, y));
                r[i + 4] = vreinterpretq_u8_u32(vtrn2q_u32(x, y));
            }
        }
        for i in 0..16 {
            if i & 8 == 0 {
                let x = vreinterpretq_u64_u8(r[i]);
                let y = vreinterpretq_u64_u8(r[i + 8]);
                r[i] = vreinterpretq_u8_u64(vtrn1q_u64(x, y));
                r[i + 8] = vreinterpretq_u8_u64(vtrn2q_u64(x, y));
            }
        }
    }
}

/// Transpose two 8×8 byte matrices sharing vector halves. Unlike the full
/// transpose, no fourth stage mixes the independent low/high matrices.
#[cfg(feature = "field64")]
#[inline]
unsafe fn transpose8x16(r: &mut [uint8x16_t; 8]) {
    unsafe {
        for i in 0..8 {
            if i & 1 == 0 {
                let (x, y) = (r[i], r[i + 1]);
                r[i] = vtrn1q_u8(x, y);
                r[i + 1] = vtrn2q_u8(x, y);
            }
        }
        for i in 0..8 {
            if i & 2 == 0 {
                let x = vreinterpretq_u16_u8(r[i]);
                let y = vreinterpretq_u16_u8(r[i + 2]);
                r[i] = vreinterpretq_u8_u16(vtrn1q_u16(x, y));
                r[i + 2] = vreinterpretq_u8_u16(vtrn2q_u16(x, y));
            }
        }
        for i in 0..8 {
            if i & 4 == 0 {
                let x = vreinterpretq_u32_u8(r[i]);
                let y = vreinterpretq_u32_u8(r[i + 4]);
                r[i] = vreinterpretq_u8_u32(vtrn1q_u32(x, y));
                r[i + 4] = vreinterpretq_u8_u32(vtrn2q_u32(x, y));
            }
        }
    }
}

/// Like [`transpose_128x128`] but reads each 16-byte row as `lo`'s 8 bytes at
/// `base_lo + row*8` (low lanes) ‖ `hi`'s 8 bytes at `base_hi + row*8` (high
/// lanes) — straight from the packed witnesses, no intermediate interleave buf.
/// So `dst[0..64]` are `lo`'s planes and `dst[64..128]` are `hi`'s. (`lo == hi`
/// with different bases pairs two blocks of one witness.)
fn transpose_128x128_2src(
    lo: &[u8],
    base_lo: usize,
    hi: &[u8],
    base_hi: usize,
    dst: &mut [uint8x16_t; 128],
) {
    transpose_128x128_2src_impl::<false>(lo, [base_lo; 2], hi, [base_hi; 2], dst);
}

/// With `PAIRED`, the first and second 64 source rows belong to independent
/// field64 blocks. The transpose puts their friendly coefficients in the low
/// and high halves of each plane, keeping both halves of the Boolean SLP busy.
/// In the ordinary mode the inactive field64 rows remain zero.
#[inline]
fn transpose_128x128_2src_impl<const PAIRED: bool>(
    lo: &[u8],
    base_lo: [usize; 2],
    hi: &[u8],
    base_hi: [usize; 2],
    dst: &mut [uint8x16_t; 128],
) {
    debug_assert!(!PAIRED || F128::BITS == 64);
    unsafe {
        let (m4, m2, m1) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x33), vdupq_n_u8(0x55));
        let mut ws = [0u8; 8 * 16 * 16];
        for gi in 0..16usize {
            let mut q = [vdupq_n_u8(0); 8];
            for k in 0..8 {
                let row = gi * 8 + k;
                if PAIRED || row < F128::BITS {
                    let (block, row) = if PAIRED {
                        (row / 64, row % 64)
                    } else {
                        (0, row)
                    };
                    let l = vld1_u8(lo.as_ptr().add(base_lo[block] + row * 8));
                    let h = vld1_u8(hi.as_ptr().add(base_hi[block] + row * 8));
                    q[k] = vcombine_u8(l, h);
                }
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m4, t, vshlq_n_u8::<4>(q[b]));
                q[b] = vbslq_u8(m4, vshrq_n_u8::<4>(t), q[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m2, t, vshlq_n_u8::<2>(q[b]));
                q[b] = vbslq_u8(m2, vshrq_n_u8::<2>(t), q[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m1, t, vshlq_n_u8::<1>(q[b]));
                q[b] = vbslq_u8(m1, vshrq_n_u8::<1>(t), q[b]);
            }
            for k in 0..8 {
                vst1q_u8(ws.as_mut_ptr().add((k * 16 + gi) * 16), q[k]);
            }
        }
        for k in 0..8usize {
            let mut v = [vdupq_n_u8(0); 16];
            for i in 0..16 {
                v[i] = vld1q_u8(ws.as_ptr().add((k * 16 + i) * 16));
            }
            transpose16x16(&mut v);
            for c in 0..16 {
                dst[c * 8 + k] = v[c];
            }
        }
    }
}

fn transpose_128x128(src: &[u8], dst: &mut [uint8x16_t; 128]) {
    unsafe {
        let (m4, m2, m1) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x33), vdupq_n_u8(0x55));
        let mut ws = [0u8; 8 * 16 * 16];
        for gi in 0..16usize {
            let so = gi * 8 * 16;
            let mut q = [vdupq_n_u8(0); 8];
            for k in 0..8 {
                q[k] = vld1q_u8(src.as_ptr().add(so + k * 16));
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m4, t, vshlq_n_u8::<4>(q[b]));
                q[b] = vbslq_u8(m4, vshrq_n_u8::<4>(t), q[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m2, t, vshlq_n_u8::<2>(q[b]));
                q[b] = vbslq_u8(m2, vshrq_n_u8::<2>(t), q[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m1, t, vshlq_n_u8::<1>(q[b]));
                q[b] = vbslq_u8(m1, vshrq_n_u8::<1>(t), q[b]);
            }
            for k in 0..8 {
                vst1q_u8(ws.as_mut_ptr().add((k * 16 + gi) * 16), q[k]);
            }
        }
        for k in 0..8usize {
            let mut v = [vdupq_n_u8(0); 16];
            for i in 0..16 {
                v[i] = vld1q_u8(ws.as_ptr().add((k * 16 + i) * 16));
            }
            transpose16x16(&mut v);
            for c in 0..16 {
                dst[c * 8 + k] = v[c];
            }
        }
    }
}

#[inline(always)]
unsafe fn fold_c(cp: &[uint8x16_t; 64], eq: F128, wbar: &mut [F128; 64]) {
    unsafe {
        for k in 0..64 {
            let pf = vreinterpretq_u64_u8(cp[k]);
            wbar[k] += eq
                * F128::from_polynomial(
                    u128::from(vgetq_lane_u64::<0>(pf))
                        | (u128::from(vgetq_lane_u64::<1>(pf)) << 64),
                );
        }
    }
}

/// NEON-resident unreduced accumulator for one coordinate: the three Karatsuba
/// parts of `Σ eq·x` kept as vectors — `[ll, cross, hh]` where `ll = Σ lo·lo`,
/// `cross = Σ (lo·hi ^ hi·lo)`, `hh = Σ hi·hi`. Folded to (r0..r3) + reduced
/// mod p ONCE per chunk. All-vector: no per-mult lane extracts, no reduction.
#[cfg(not(feature = "field64"))]
const UNRED_PARTS: usize = 3;
#[cfg(feature = "field64")]
const UNRED_PARTS: usize = 1;
type UnredAcc = [uint64x2_t; UNRED_PARTS];

#[inline(always)]
unsafe fn pmull_u64(a: u64, b: u64) -> uint64x2_t {
    unsafe { vreinterpretq_u64_p128(vmull_p64(a, b)) }
}

/// `acc ^= eq · x` unreduced: 4 PMULL + 4 VEOR, entirely in NEON registers.
#[inline(always)]
unsafe fn mul_acc_unred(acc: &mut UnredAcc, eq: F128, x: uint64x2_t) {
    unsafe {
        let xl = vgetq_lane_u64::<0>(x);
        let xh = vgetq_lane_u64::<1>(x);
        acc[0] = veorq_u64(acc[0], pmull_u64(eq.lo, xl));
        #[cfg(not(feature = "field64"))]
        {
            acc[1] = veorq_u64(
                acc[1],
                veorq_u64(pmull_u64(eq.lo, xh), pmull_u64(eq.high_u64(), xl)),
            );
            acc[2] = veorq_u64(acc[2], pmull_u64(eq.high_u64(), xh));
        }
        #[cfg(feature = "field64")]
        debug_assert_eq!(xh, 0, "inactive friendly lanes must be zero");
    }
}

/// Both halves hold independent field64 values, not one 128-bit element.
#[cfg(feature = "field64")]
#[inline(always)]
unsafe fn mul_acc_unred_pair(acc: &mut UnredAcc, eq: [F128; 2], x: uint64x2_t) {
    unsafe {
        let low = pmull_u64(eq[0].lo, vgetq_lane_u64::<0>(x));
        let high = pmull_u64(eq[1].lo, vgetq_lane_u64::<1>(x));
        acc[0] = veorq_u64(acc[0], veorq_u64(low, high));
    }
}

/// Fold an [`UnredAcc`] to `(r0..r3)` and reduce mod p (same math as
/// [`crate::field::F256Unreduced::reduce`], so results are bit-identical).
#[inline]
fn reduce_unred(acc: &UnredAcc) -> F128 {
    unsafe {
        #[cfg(not(feature = "field64"))]
        {
            F256Unreduced {
                r0: vgetq_lane_u64::<0>(acc[0]),
                r1: vgetq_lane_u64::<1>(acc[0]) ^ vgetq_lane_u64::<0>(acc[1]),
                r2: vgetq_lane_u64::<0>(acc[2]) ^ vgetq_lane_u64::<1>(acc[1]),
                r3: vgetq_lane_u64::<1>(acc[2]),
            }
            .reduce()
        }
        #[cfg(feature = "field64")]
        {
            // Only 64 friendly coefficients are active, so the full product
            // has degree at most 126 and fits one unreduced 128-bit lane.
            F256Unreduced {
                r0: vgetq_lane_u64::<0>(acc[0]),
                r1: vgetq_lane_u64::<1>(acc[0]),
            }
            .reduce()
        }
    }
}

/// PROTOTYPE fusion of [`product_bs`] + [`fold_bs`] with NEON-resident LAZY
/// REDUCTION: each product coordinate is formed in a register and eq-multiplied
/// unreduced (4 PMULL + 4 VEOR, no mod-p fold, no lane extracts) into a vector
/// accumulator; the caller reduces each coordinate once per chunk. Kills the
/// 2.5 KB `prod` buffer AND the per-block reduction work.
#[inline(always)]
unsafe fn product_fold_bs(
    af: &[uint8x16_t; 160],
    bf: &[uint8x16_t; 160],
    ax: &[uint8x16_t; 64],
    bx: &[uint8x16_t; 64],
    eq: F128,
    res: &mut [UnredAcc; 160],
) {
    unsafe {
        #[inline(always)]
        unsafe fn acc(res: &mut UnredAcc, eq: F128, pr: uint8x16_t) {
            unsafe {
                mul_acc_unred(res, eq, vreinterpretq_u64_u8(pr));
            }
        }
        for p in 0..64 {
            let pr = veorq_u8(vandq_u8(af[p], bx[p]), vandq_u8(ax[p], bf[p]));
            acc(&mut res[p], eq, pr);
        }
        for p in 0..64 {
            let pr = veorq_u8(
                veorq_u8(vandq_u8(af[64 + p], bx[p]), vandq_u8(af[p], bf[p])),
                vandq_u8(ax[p], bf[64 + p]),
            );
            acc(&mut res[64 + p], eq, pr);
        }
        for p in 0..32 {
            let pr = veorq_u8(
                veorq_u8(vandq_u8(af[128 + p], bx[p]), vandq_u8(af[64 + p], bf[p])),
                veorq_u8(vandq_u8(af[p], bf[64 + p]), vandq_u8(ax[p], bf[128 + p])),
            );
            acc(&mut res[128 + p], eq, pr);
        }
    }
}

/// Each SIMD half is one independent field64 block, with its own equality
/// weight. Boolean encoding/products are lane-wise; only the final fold mixes
/// the halves, by adding their separately weighted field products.
#[cfg(feature = "field64")]
#[inline(always)]
unsafe fn product_fold_bs_pair(
    af: &[uint8x16_t; 160],
    bf: &[uint8x16_t; 160],
    ax: &[uint8x16_t; 64],
    bx: &[uint8x16_t; 64],
    eq: [F128; 2],
    res: &mut [UnredAcc; 160],
) {
    unsafe {
        #[inline(always)]
        unsafe fn acc(res: &mut UnredAcc, eq: [F128; 2], pr: uint8x16_t) {
            unsafe {
                mul_acc_unred_pair(res, eq, vreinterpretq_u64_u8(pr));
            }
        }
        for p in 0..64 {
            let pr = veorq_u8(vandq_u8(af[p], bx[p]), vandq_u8(ax[p], bf[p]));
            acc(&mut res[p], eq, pr);
        }
        for p in 0..64 {
            let pr = veorq_u8(
                veorq_u8(vandq_u8(af[64 + p], bx[p]), vandq_u8(af[p], bf[p])),
                vandq_u8(ax[p], bf[64 + p]),
            );
            acc(&mut res[64 + p], eq, pr);
        }
        for p in 0..32 {
            let pr = veorq_u8(
                veorq_u8(vandq_u8(af[128 + p], bx[p]), vandq_u8(af[64 + p], bf[p])),
                veorq_u8(vandq_u8(af[p], bf[64 + p]), vandq_u8(ax[p], bf[128 + p])),
            );
            acc(&mut res[128 + p], eq, pr);
        }
    }
}

// ---------------------------------------------------------------------------
// Production path: encode against `M` derived from the M2 evaluator's own
// extension, so the kernel emits coordinates in the evaluator's basis (identity
// bridge to `product_code_message`). The legacy bench SLP (`M_MASK` /
// `encode_slp`, a *different coordinate labeling* of the same code) was deleted
// in the bloat sweep; `m_derived_from_evaluator_is_identity_bridge` pins the
// bridge.
// ---------------------------------------------------------------------------

/// The by-point fresh encode `M` (160×64) for the genus-95 product code, derived
/// once from the M2 evaluator's extension. Fresh slot `s` (0..158) == evaluator
/// product coord `64 + s` (order1|order2|order3); slots 158,159 are D³ garbage
/// (only 30 D³ points exist). Built lazily from `extended_base_product_message`.
fn derived_m() -> &'static [u64; 160] {
    static M: OnceLock<[u64; 160]> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = [0u64; 160];
        for j in 0..64 {
            let ext = extended_base_product_message(BaseMessage(1u64 << j));
            for s in 0..158 {
                if ext.get_bit(64 + s) {
                    m[s] |= 1u64 << j;
                }
            }
        }
        m
    })
}

/// Direct bitsliced encode `out = M · inp`: one XOR per set bit of each row.
/// The generated `super::slp_derived` provides the optimized form.
#[inline]
unsafe fn encode_direct(m: &[u64; 160], inp: &[uint8x16_t; 64], out: &mut [uint8x16_t; 160]) {
    unsafe {
        for s in 0..160 {
            let mut acc = vdupq_n_u8(0);
            let mut mask = m[s];
            while mask != 0 {
                let j = mask.trailing_zeros() as usize;
                acc = veorq_u8(acc, inp[j]);
                mask &= mask - 1;
            }
            out[s] = acc;
        }
    }
}

/// Read a bit-packed witness into blocks of `F128::BITS` binary AG messages.
///
/// AG variable order is skip in the low 6 bits, then `F128::LOG_BITS`
/// friendly coordinates, then the remaining outer coordinates. Each binary
/// skip-message is 8 bytes, so a block occupies `BLOCK_BYTES` contiguous
/// bytes (1024 natively, 512 with field64). Message i is weighted by gamma^i;
/// these coefficients are linearly independent because i < F128::BITS.
/// The NEON transpose retains 128 lanes and zeros the inactive high lanes.
pub fn blocks_from_packed(packed: &[u8]) -> Vec<[uint8x16_t; 64]> {
    assert_eq!(
        packed.len() % BLOCK_BYTES,
        0,
        "packed witness must contain whole active-field-sized AG blocks"
    );
    let n = packed.len() / BLOCK_BYTES;
    let z = unsafe { vdupq_n_u8(0) };
    // NEON bit-transpose, pairing two 64-bit-message blocks into one 128-wide
    // `transpose_128x128` so the transpose runs at full utilization (no zero
    // padding). Halves the transpose calls vs one-block-per-pass.
    let mut out: Vec<[uint8x16_t; 64]> = Vec::with_capacity(n);
    let mut buf = [0u8; 128 * 16];
    let mut planes = [z; 128];
    let mut o = 0;
    while o + 1 < n {
        let (b0, b1) = (o * BLOCK_BYTES, (o + 1) * BLOCK_BYTES);
        for r in 0..F128::BITS {
            buf[r * 16..r * 16 + 8].copy_from_slice(&packed[b0 + r * 8..b0 + r * 8 + 8]);
            buf[r * 16 + 8..r * 16 + 16].copy_from_slice(&packed[b1 + r * 8..b1 + r * 8 + 8]);
        }
        transpose_128x128(&buf, &mut planes);
        let mut p0 = [z; 64];
        p0.copy_from_slice(&planes[0..64]);
        out.push(p0);
        let mut p1 = [z; 64];
        p1.copy_from_slice(&planes[64..128]);
        out.push(p1);
        o += 2;
    }
    if o < n {
        // Final odd block: pad the high half with zeros.
        let b0 = o * BLOCK_BYTES;
        for r in 0..F128::BITS {
            buf[r * 16..r * 16 + 8].copy_from_slice(&packed[b0 + r * 8..b0 + r * 8 + 8]);
            buf[r * 16 + 8..r * 16 + 16].fill(0);
        }
        transpose_128x128(&buf, &mut planes);
        let mut p0 = [z; 64];
        p0.copy_from_slice(&planes[0..64]);
        out.push(p0);
    }
    out
}

/// Raw fused AB+C round-1 over pre-bitsliced blocks: returns the AB product
/// fresh coords (160, D-scaled; slots 158/159 garbage) and the folded C message
/// `wbar` (64, D-scaled) WITHOUT encoding C. The AG-skip protocol sends `wbar`
/// (the c message) directly, not its codeword, so the prover wants it raw.
pub fn round1_raw(
    a_pl: &[[uint8x16_t; 64]],
    b_pl: &[[uint8x16_t; 64]],
    c_pl: &[[uint8x16_t; 64]],
    eq: &[F128],
    n: usize,
) -> ([F128; 160], [F128; 64]) {
    let m = derived_m();
    let z = unsafe { vdupq_n_u8(0) };
    let mut res = [F128::ZERO; 160];
    let mut wbar = [F128::ZERO; 64];
    let mut af = [z; 160];
    let mut bf = [z; 160];
    let mut prod = [z; 160];
    for o in 0..n {
        unsafe {
            encode_direct(m, &a_pl[o], &mut af);
            encode_direct(m, &b_pl[o], &mut bf);
            product_bs(&af, &bf, &a_pl[o], &b_pl[o], &mut prod);
            fold_bs(&prod, eq[o], &mut res);
            fold_c(&c_pl[o], eq[o], &mut wbar);
        }
    }
    (res, wbar)
}

/// [`round1_raw`] reading the three packed witnesses (one `eq` per BLOCK_BYTES-byte block).
pub fn round1_raw_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64]) {
    crate::suboptimal_path!(
        "reference round-1 (raw, non-bitsliced)",
        "round1_straight_line_program_packed_banks_fused"
    );
    let ap = blocks_from_packed(a_packed);
    let bp = blocks_from_packed(b_packed);
    let cp = blocks_from_packed(c_packed);
    let n = ap.len();
    assert_eq!(eq.len(), n, "one eq weight per block");
    round1_raw(&ap, &bp, &cp, eq, n)
}

// ---------------------------------------------------------------------------
// Fast derived-`M` encode: four-Russians LUT (row-major) + in-register product.
// This replaces `encode_direct` on the production path (~10× faster). The LUT is
// built from `derived_m`, so the kernel stays in the evaluator's basis (identity
// bridge). Same four-Russians blocking the M2 evaluator already uses.
// ---------------------------------------------------------------------------

/// Round-1 via the Paar straight-line encode (`super::slp_derived`): bit-slice
/// a/b/c, run the SLP on planes, product, fold. No *output* transpose (the SLP
/// works on planes directly; it pays an *input* transpose instead). Output
/// matches [`round1_raw_packed`] on the 158 fresh coords + `wbar`.
pub fn round1_straight_line_program_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64]) {
    crate::suboptimal_path!(
        "unfused SLP round-1",
        "round1_straight_line_program_packed_banks_fused"
    );
    let n = a_packed.len() / BLOCK_BYTES;
    assert_eq!(eq.len(), n, "one eq weight per block");
    let nthreads = current_num_threads().max(1);
    let chunk0 = n.div_ceil(8 * nthreads).max(1); // ~8 chunks/thread
    let chunk = chunk0 + (chunk0 & 1); // even, so the block-pair loop tiles each chunk exactly
    let nchunks = n.div_ceil(chunk);
    (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let mut res = [F128::ZERO; 160];
            let mut wbar = [F128::ZERO; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut prod = [z; 160];
            // Per-chunk bitslice scratch. Process blocks in PAIRS so every
            // transpose_128x128 is fully used (128 input columns, no zero pad):
            // a+b of a block pair into one transpose (a→[0..64], b→[64..128]); the
            // two blocks' c pair into one more (c_o→[0..64], c_{o+1}→[64..128]).
            // 3 transposes per 2 blocks vs 4 with c padded per-block.
            let mut pab = [z; 128];
            let mut pc = [z; 128];
            let mut o = start;
            while o + 1 < end {
                let (cb0, cb1) = (o * BLOCK_BYTES, (o + 1) * BLOCK_BYTES);
                // The two blocks' c paired straight from packed (no interleave buf):
                // pc[0..64] = block o's c, pc[64..128] = block o+1's c.
                transpose_128x128_2src(c_packed, cb0, c_packed, cb1, &mut pc);
                let cp0: &[uint8x16_t; 64] = pc[0..64].try_into().unwrap();
                let cp1: &[uint8x16_t; 64] = pc[64..128].try_into().unwrap();
                unsafe {
                    process_block(
                        a_packed, b_packed, cb0, eq[o], cp0, &mut pab, &mut af, &mut bf, &mut prod,
                        &mut res, &mut wbar,
                    );
                    process_block(
                        a_packed,
                        b_packed,
                        cb1,
                        eq[o + 1],
                        cp1,
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut prod,
                        &mut res,
                        &mut wbar,
                    );
                }
                o += 2;
            }
            if o < end {
                // Trailing odd block (only n odd, i.e. m=13): c padded via buf.
                let mut buf = [0u8; 128 * 16];
                bitslice_block_into(c_packed, o * BLOCK_BYTES, &mut buf, &mut pc);
                let cp: &[uint8x16_t; 64] = pc[0..64].try_into().unwrap();
                unsafe {
                    process_block(
                        a_packed,
                        b_packed,
                        o * BLOCK_BYTES,
                        eq[o],
                        cp,
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut prod,
                        &mut res,
                        &mut wbar,
                    );
                }
            }
            (res, wbar)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64]),
            |(mut r1, mut w1), (r2, w2)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    w1[k] += w2[k];
                }
                (r1, w1)
            },
        )
}

/// One block of the SLP round-1: pair a+b into one transpose (a→`pab[0..64]`,
/// b→`pab[64..128]`), then encode·product·fold into `res` and fold c (`cp`, the
/// caller-supplied c planes) into `wbar`. The accumulators are shared across the
/// chunk's blocks; `buf`/`pab` are reused scratch.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn process_block(
    a_packed: &[u8],
    b_packed: &[u8],
    base: usize,
    eq_o: F128,
    cp: &[uint8x16_t; 64],
    pab: &mut [uint8x16_t; 128],
    af: &mut [uint8x16_t; 160],
    bf: &mut [uint8x16_t; 160],
    prod: &mut [uint8x16_t; 160],
    res: &mut [F128; 160],
    wbar: &mut [F128; 64],
) {
    // a+b straight from the packed witnesses into one transpose (no interleave buf).
    transpose_128x128_2src(a_packed, base, b_packed, base, pab);
    let ap: &[uint8x16_t; 64] = (&pab[0..64]).try_into().unwrap();
    let bp: &[uint8x16_t; 64] = (&pab[64..128]).try_into().unwrap();
    unsafe {
        encode_straight_line_program_derived(ap, af);
        encode_straight_line_program_derived(bp, bf);
        product_bs(af, bf, ap, bp, prod);
        fold_bs(prod, eq_o, res);
        fold_c(cp, eq_o, wbar);
    }
}

/// PROTOTYPE [`process_block`] with the fused product+fold ([`product_fold_bs`])
/// — no `prod` buffer. The c-path is handled separately by the caller via
/// [`transpose_fold_c_banks_2src`], so `cp` is gone too.
#[inline]
unsafe fn process_block_fused(
    a_packed: &[u8],
    b_packed: &[u8],
    base: usize,
    eq_o: F128,
    pab: &mut [uint8x16_t; 128],
    af: &mut [uint8x16_t; 160],
    bf: &mut [uint8x16_t; 160],
    res: &mut [UnredAcc; 160],
) {
    transpose_128x128_2src(a_packed, base, b_packed, base, pab);
    let ap: &[uint8x16_t; 64] = (&pab[0..64]).try_into().unwrap();
    let bp: &[uint8x16_t; 64] = (&pab[64..128]).try_into().unwrap();
    unsafe {
        encode_straight_line_program_derived(ap, af);
        encode_straight_line_program_derived(bp, bf);
        product_fold_bs(af, bf, ap, bp, eq_o, res);
    }
}

/// Encode and multiply two independent field64 blocks in one 128-lane SLP.
/// The bases may be nonconsecutive: the padded path pairs adjacent live blocks
/// across dead runs without ever loading a dead block.
#[cfg(feature = "field64")]
#[inline]
unsafe fn process_block_pair_fused(
    a_packed: &[u8],
    b_packed: &[u8],
    bases: [usize; 2],
    eq: [F128; 2],
    pab: &mut [uint8x16_t; 128],
    af: &mut [uint8x16_t; 160],
    bf: &mut [uint8x16_t; 160],
    res: &mut [UnredAcc; 160],
) {
    transpose_128x128_2src_impl::<true>(a_packed, bases, b_packed, bases, pab);
    let ap: &[uint8x16_t; 64] = (&pab[..64]).try_into().unwrap();
    let bp: &[uint8x16_t; 64] = (&pab[64..]).try_into().unwrap();
    unsafe {
        encode_straight_line_program_derived(ap, af);
        encode_straight_line_program_derived(bp, bf);
        product_fold_bs_pair(af, bf, ap, bp, eq, res);
    }
}

/// Two-bank variant of [`fold_c`] for `s_hat_v_c` capture: split each plane's
/// `pf = Σ_i x^i·bit_i` by the parity of the friendly index `i` (= the 7th
/// packing bit / friendly bit 0) into `bank0` (even `i`) and `bank1` (odd `i`).
/// Since the even/odd bit sets partition `pf`, `bank0[k] + bank1[k]` equals
/// [`fold_c`]'s `wbar[k]` bit-for-bit (XOR partition + field-mult distributivity).
const C_EVEN_MASK: u64 = 0x5555_5555_5555_5555; // bits 0,2,4,…
const C_ODD_MASK: u64 = 0xAAAA_AAAA_AAAA_AAAA; // bits 1,3,5,…
unsafe fn fold_c_banks(
    cp: &[uint8x16_t; 64],
    eq: F128,
    bank0: &mut [F128; 64],
    bank1: &mut [F128; 64],
) {
    unsafe {
        for k in 0..64 {
            let pf = vreinterpretq_u64_u8(cp[k]);
            let lo = vgetq_lane_u64::<0>(pf);
            let hi = vgetq_lane_u64::<1>(pf);
            let even = F128::from_polynomial(
                u128::from(lo & C_EVEN_MASK) | (u128::from(hi & C_EVEN_MASK) << 64),
            );
            let odd = F128::from_polynomial(
                u128::from(lo & C_ODD_MASK) | (u128::from(hi & C_ODD_MASK) << 64),
            );
            bank0[k] += eq * even;
            bank1[k] += eq * odd;
        }
    }
}

/// [`process_block`] with the two-bank c-fold ([`fold_c_banks`]) for `s_hat_v_c`
/// capture. The AB path is identical; only the c accumulation differs.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn process_block_banks(
    a_packed: &[u8],
    b_packed: &[u8],
    base: usize,
    eq_o: F128,
    cp: &[uint8x16_t; 64],
    pab: &mut [uint8x16_t; 128],
    af: &mut [uint8x16_t; 160],
    bf: &mut [uint8x16_t; 160],
    prod: &mut [uint8x16_t; 160],
    res: &mut [F128; 160],
    bank0: &mut [F128; 64],
    bank1: &mut [F128; 64],
) {
    transpose_128x128_2src(a_packed, base, b_packed, base, pab);
    let ap: &[uint8x16_t; 64] = (&pab[0..64]).try_into().unwrap();
    let bp: &[uint8x16_t; 64] = (&pab[64..128]).try_into().unwrap();
    unsafe {
        encode_straight_line_program_derived(ap, af);
        encode_straight_line_program_derived(bp, bf);
        product_bs(af, bf, ap, bp, prod);
        fold_bs(prod, eq_o, res);
        fold_c_banks(cp, eq_o, bank0, bank1);
    }
}

/// [`round1_straight_line_program_packed`] that ALSO returns the two c-fold banks for `s_hat_v_c`
/// capture (split by the 7th packing bit). `res` and `bank0 + bank1` are
/// bit-identical to `round1_straight_line_program_packed`'s `(res, wbar)`. Kept separate from the
/// hot `round1_straight_line_program_packed` so the standalone round-1 microbench path is
/// untouched; production AG prove (which needs `s_hat_v_c`) calls this.
pub fn round1_straight_line_program_packed_banks(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64], [F128; 64]) {
    crate::suboptimal_path!(
        "unfused banks round-1",
        "round1_straight_line_program_packed_banks_fused"
    );
    let n = a_packed.len() / BLOCK_BYTES;
    assert_eq!(eq.len(), n, "one eq weight per block");
    let nthreads = current_num_threads().max(1);
    let chunk0 = n.div_ceil(8 * nthreads).max(1);
    let chunk = chunk0 + (chunk0 & 1);
    let nchunks = n.div_ceil(chunk);
    (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let mut res = [F128::ZERO; 160];
            let mut bank0 = [F128::ZERO; 64];
            let mut bank1 = [F128::ZERO; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut prod = [z; 160];
            let mut pab = [z; 128];
            let mut pc = [z; 128];
            let mut o = start;
            while o + 1 < end {
                let (cb0, cb1) = (o * BLOCK_BYTES, (o + 1) * BLOCK_BYTES);
                transpose_128x128_2src(c_packed, cb0, c_packed, cb1, &mut pc);
                let cp0: &[uint8x16_t; 64] = pc[0..64].try_into().unwrap();
                let cp1: &[uint8x16_t; 64] = pc[64..128].try_into().unwrap();
                unsafe {
                    process_block_banks(
                        a_packed, b_packed, cb0, eq[o], cp0, &mut pab, &mut af, &mut bf, &mut prod,
                        &mut res, &mut bank0, &mut bank1,
                    );
                    process_block_banks(
                        a_packed,
                        b_packed,
                        cb1,
                        eq[o + 1],
                        cp1,
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut prod,
                        &mut res,
                        &mut bank0,
                        &mut bank1,
                    );
                }
                o += 2;
            }
            if o < end {
                let mut buf = [0u8; 128 * 16];
                bitslice_block_into(c_packed, o * BLOCK_BYTES, &mut buf, &mut pc);
                let cp: &[uint8x16_t; 64] = pc[0..64].try_into().unwrap();
                unsafe {
                    process_block_banks(
                        a_packed,
                        b_packed,
                        o * BLOCK_BYTES,
                        eq[o],
                        cp,
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut prod,
                        &mut res,
                        &mut bank0,
                        &mut bank1,
                    );
                }
            }
            (res, bank0, bank1)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64], [F128::ZERO; 64]),
            |(mut r1, mut a0, mut a1), (r2, b0, b1)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    a0[k] += b0[k];
                    a1[k] += b1[k];
                }
                (r1, a0, a1)
            },
        )
}

/// Paired-c transpose (`transpose_128x128_2src` on `(c[base0], c[base1])`) with
/// the eq-fold done straight from the registers in pass B, using the two-bank
/// c-split of [`fold_c_banks`]:
/// each plane is masked into its even/odd-index halves in NEON registers and
/// each half is eq-multiplied unreduced into its bank. Same pass structure —
/// the `pc` buffer never exists.
fn transpose_fold_c_banks_2src(
    c_packed: &[u8],
    base0: usize,
    base1: usize,
    eq0: F128,
    eq1: F128,
    bank0: &mut [UnredAcc; 64],
    bank1: &mut [UnredAcc; 64],
) {
    unsafe {
        let (m4, m2, m1) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x33), vdupq_n_u8(0x55));
        let even = vdupq_n_u64(C_EVEN_MASK);
        let odd = vdupq_n_u64(C_ODD_MASK);
        const GROUPS: usize = F128::BITS / 8;
        let mut ws = [0u8; 8 * GROUPS * 16];
        for gi in 0..GROUPS {
            let mut q = [vdupq_n_u8(0); 8];
            for k in 0..8 {
                let row = gi * 8 + k;
                let l = vld1_u8(c_packed.as_ptr().add(base0 + row * 8));
                let h = vld1_u8(c_packed.as_ptr().add(base1 + row * 8));
                q[k] = vcombine_u8(l, h);
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m4, t, vshlq_n_u8::<4>(q[b]));
                q[b] = vbslq_u8(m4, vshrq_n_u8::<4>(t), q[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m2, t, vshlq_n_u8::<2>(q[b]));
                q[b] = vbslq_u8(m2, vshrq_n_u8::<2>(t), q[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m1, t, vshlq_n_u8::<1>(q[b]));
                q[b] = vbslq_u8(m1, vshrq_n_u8::<1>(t), q[b]);
            }
            for k in 0..8 {
                vst1q_u8(ws.as_mut_ptr().add((k * GROUPS + gi) * 16), q[k]);
            }
        }
        for k in 0..8usize {
            let mut v = [vdupq_n_u8(0); GROUPS];
            for i in 0..GROUPS {
                v[i] = vld1q_u8(ws.as_ptr().add((k * GROUPS + i) * 16));
            }
            #[cfg(not(feature = "field64"))]
            transpose16x16(&mut v);
            #[cfg(feature = "field64")]
            transpose8x16(&mut v);
            for c in 0..GROUPS {
                let x = vreinterpretq_u64_u8(v[c]);
                let j = c * 8 + k;
                #[cfg(not(feature = "field64"))]
                {
                    let (jj, eq) = if c < 8 { (j, eq0) } else { (j - 64, eq1) };
                    mul_acc_unred(&mut bank0[jj], eq, vandq_u64(x, even));
                    mul_acc_unred(&mut bank1[jj], eq, vandq_u64(x, odd));
                }
                #[cfg(feature = "field64")]
                {
                    mul_acc_unred_pair(&mut bank0[j], [eq0, eq1], vandq_u64(x, even));
                    mul_acc_unred_pair(&mut bank1[j], [eq0, eq1], vandq_u64(x, odd));
                }
            }
        }
    }
}

/// PROTOTYPE fused [`round1_straight_line_program_packed_banks`]: fused product+fold (no `prod`
/// buffer), banked c-fold straight out of the c-transpose registers (no `pc`
/// buffer), NEON-resident lazy reduction (reduce once per chunk). Bit-identical
/// to [`round1_straight_line_program_packed_banks`] on `(res, bank0, bank1)`.
pub fn round1_straight_line_program_packed_banks_fused(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64], [F128; 64]) {
    let n = a_packed.len() / BLOCK_BYTES;
    assert_eq!(eq.len(), n, "one eq weight per block");
    let nthreads = current_num_threads().max(1);
    let chunk0 = n.div_ceil(8 * nthreads).max(1);
    let chunk = chunk0 + (chunk0 & 1);
    let nchunks = n.div_ceil(chunk);
    (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let z64 = unsafe { vdupq_n_u64(0) };
            let mut res = [[z64; UNRED_PARTS]; 160];
            let mut bank0 = [[z64; UNRED_PARTS]; 64];
            let mut bank1 = [[z64; UNRED_PARTS]; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut pab = [z; 128];
            let mut o = start;
            while o + 1 < end {
                let (cb0, cb1) = (o * BLOCK_BYTES, (o + 1) * BLOCK_BYTES);
                transpose_fold_c_banks_2src(
                    c_packed,
                    cb0,
                    cb1,
                    eq[o],
                    eq[o + 1],
                    &mut bank0,
                    &mut bank1,
                );
                unsafe {
                    #[cfg(feature = "field64")]
                    process_block_pair_fused(
                        a_packed,
                        b_packed,
                        [cb0, cb1],
                        [eq[o], eq[o + 1]],
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut res,
                    );
                    #[cfg(not(feature = "field64"))]
                    {
                        process_block_fused(
                            a_packed, b_packed, cb0, eq[o], &mut pab, &mut af, &mut bf, &mut res,
                        );
                        process_block_fused(
                            a_packed,
                            b_packed,
                            cb1,
                            eq[o + 1],
                            &mut pab,
                            &mut af,
                            &mut bf,
                            &mut res,
                        );
                    }
                }
                o += 2;
            }
            if o < end {
                let mut pc = [z; 128];
                let mut buf = [0u8; 128 * 16];
                bitslice_block_into(c_packed, o * BLOCK_BYTES, &mut buf, &mut pc);
                unsafe {
                    process_block_fused(
                        a_packed,
                        b_packed,
                        o * BLOCK_BYTES,
                        eq[o],
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut res,
                    );
                    let even = vdupq_n_u64(C_EVEN_MASK);
                    let odd = vdupq_n_u64(C_ODD_MASK);
                    for k in 0..64 {
                        let x = vreinterpretq_u64_u8(pc[k]);
                        mul_acc_unred(&mut bank0[k], eq[o], vandq_u64(x, even));
                        mul_acc_unred(&mut bank1[k], eq[o], vandq_u64(x, odd));
                    }
                }
            }
            let mut res_r = [F128::ZERO; 160];
            let mut b0_r = [F128::ZERO; 64];
            let mut b1_r = [F128::ZERO; 64];
            for j in 0..160 {
                res_r[j] = reduce_unred(&res[j]);
            }
            for k in 0..64 {
                b0_r[k] = reduce_unred(&bank0[k]);
                b1_r[k] = reduce_unred(&bank1[k]);
            }
            (res_r, b0_r, b1_r)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64], [F128::ZERO; 64]),
            |(mut r1, mut a0, mut a1), (r2, b0, b1)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    a0[k] += b0[k];
                    a1[k] += b1[k];
                }
                (r1, a0, a1)
            },
        )
}

/// [`round1_straight_line_program_packed_banks_fused`] over a witness run-list: ONE parallel
/// pass over the LIVE blocks only — Dead blocks are skipped (their honest
/// contribution is zero), Partial blocks are cleansed into zeroed scratch
/// inline ([`crate::zerocheck::cleanse_block`], so no declared-dead bit is
/// ever read), and consecutive Full live blocks pair for the two-source
/// c-transpose exactly like the dense kernel. The accumulation is XOR
/// (char-2), so the changed visit order is value-identical; per-element
/// cost matches the dense kernel — no per-segment call barriers (the
/// segment-wrapper prototype paid ~450 rayon bridges at the envelope's
/// per-column run structure and LOST to the dense scan).
pub fn round1_straight_line_program_packed_banks_fused_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
    coverage: &[BlockCoverage],
) -> ([F128; 160], [F128; 64], [F128; 64]) {
    use crate::zerocheck::BlockCoverage;
    let n = a_packed.len() / BLOCK_BYTES;
    assert_eq!(eq.len(), n, "one eq weight per block");
    assert_eq!(coverage.len(), n, "one coverage entry per block");
    let live: Vec<u32> = (0..n)
        .filter(|&o| !matches!(coverage[o], BlockCoverage::Dead))
        .map(|o| o as u32)
        .collect();
    let nl = live.len();
    let nthreads = rayon::current_num_threads().max(1);
    // P+E scheduling: the P-cores (rayon) and two utility-QoS E-threads pull
    // chunks from one atomic counter (`run_hetero_chunks_stateful`), each
    // keeping a running sum merged at the end — the same shared-queue shape
    // the RS round 1 and the NTT top use, which AG round 1 had not adopted.
    // Chunks are 4× finer than the P-only loop's 8/thread so the slowest
    // core's LAST chunk cannot hold the join (an E-core runs ~1/3 P speed).
    // Measured at m=32: round 1 −2.7 / −2.8 / −4.6 ms MT, 3/3.
    let chunk0 = nl.div_ceil(32 * nthreads).max(1);
    let chunk = chunk0 + (chunk0 & 1);
    let nchunks = nl.div_ceil(chunk);
    let per_chunk = |ci: usize| -> ([F128; 160], [F128; 64], [F128; 64]) {
        let start = ci * chunk;
        let end = ((ci + 1) * chunk).min(nl);
        let z = unsafe { vdupq_n_u8(0) };
        let z64 = unsafe { vdupq_n_u64(0) };
        let mut res = [[z64; UNRED_PARTS]; 160];
        let mut bank0 = [[z64; UNRED_PARTS]; 64];
        let mut bank1 = [[z64; UNRED_PARTS]; 64];
        let mut af = [z; 160];
        let mut bf = [z; 160];
        let mut pab = [z; 128];
        // The SINGLE path (a cleansed partial, or an unpaired full):
        // the dense kernel's tail branch, source-parameterized so a
        // BLOCK_BYTES-byte scratch block passes with base 0.
        let single = |a_src: &[u8],
                      b_src: &[u8],
                      c_src: &[u8],
                      base: usize,
                      eq_o: F128,
                      af: &mut [uint8x16_t; 160],
                      bf: &mut [uint8x16_t; 160],
                      pab: &mut [uint8x16_t; 128],
                      res: &mut [UnredAcc; 160],
                      bank0: &mut [UnredAcc; 64],
                      bank1: &mut [UnredAcc; 64]| {
            let mut pc = [z; 128];
            let mut buf = [0u8; 128 * 16];
            bitslice_block_into(c_src, base, &mut buf, &mut pc);
            unsafe {
                process_block_fused(a_src, b_src, base, eq_o, pab, af, bf, res);
                let even = vdupq_n_u64(C_EVEN_MASK);
                let odd = vdupq_n_u64(C_ODD_MASK);
                for k in 0..64 {
                    let x = vreinterpretq_u64_u8(pc[k]);
                    mul_acc_unred(&mut bank0[k], eq_o, vandq_u64(x, even));
                    mul_acc_unred(&mut bank1[k], eq_o, vandq_u64(x, odd));
                }
            }
        };
        let mut i = start;
        while i < end {
            let o = live[i] as usize;
            match &coverage[o] {
                BlockCoverage::Full => {
                    // Pair with the NEXT live entry when it is Full too
                    // (2src transpose takes two independent offsets).
                    let o2 = if i + 1 < end { live[i + 1] as usize } else { o };
                    if i + 1 < end && matches!(coverage[o2], BlockCoverage::Full) {
                        transpose_fold_c_banks_2src(
                            c_packed,
                            o * BLOCK_BYTES,
                            o2 * BLOCK_BYTES,
                            eq[o],
                            eq[o2],
                            &mut bank0,
                            &mut bank1,
                        );
                        unsafe {
                            #[cfg(feature = "field64")]
                            process_block_pair_fused(
                                a_packed,
                                b_packed,
                                [o * BLOCK_BYTES, o2 * BLOCK_BYTES],
                                [eq[o], eq[o2]],
                                &mut pab,
                                &mut af,
                                &mut bf,
                                &mut res,
                            );
                            #[cfg(not(feature = "field64"))]
                            {
                                process_block_fused(
                                    a_packed,
                                    b_packed,
                                    o * BLOCK_BYTES,
                                    eq[o],
                                    &mut pab,
                                    &mut af,
                                    &mut bf,
                                    &mut res,
                                );
                                process_block_fused(
                                    a_packed,
                                    b_packed,
                                    o2 * BLOCK_BYTES,
                                    eq[o2],
                                    &mut pab,
                                    &mut af,
                                    &mut bf,
                                    &mut res,
                                );
                            }
                        }
                        i += 2;
                        continue;
                    }
                    single(
                        a_packed,
                        b_packed,
                        c_packed,
                        o * BLOCK_BYTES,
                        eq[o],
                        &mut af,
                        &mut bf,
                        &mut pab,
                        &mut res,
                        &mut bank0,
                        &mut bank1,
                    );
                    i += 1;
                }
                BlockCoverage::Partial(ranges) => {
                    let mut a_buf = [0u8; BLOCK_BYTES];
                    let mut b_buf = [0u8; BLOCK_BYTES];
                    let mut c_buf = [0u8; BLOCK_BYTES];
                    crate::zerocheck::cleanse_block(a_packed, o * BLOCK_BYTES, ranges, &mut a_buf);
                    crate::zerocheck::cleanse_block(b_packed, o * BLOCK_BYTES, ranges, &mut b_buf);
                    crate::zerocheck::cleanse_block(c_packed, o * BLOCK_BYTES, ranges, &mut c_buf);
                    single(
                        &a_buf, &b_buf, &c_buf, 0, eq[o], &mut af, &mut bf, &mut pab, &mut res,
                        &mut bank0, &mut bank1,
                    );
                    i += 1;
                }
                BlockCoverage::Dead => unreachable!("the live list has no dead entry"),
            }
        }
        let mut res_r = [F128::ZERO; 160];
        let mut b0_r = [F128::ZERO; 64];
        let mut b1_r = [F128::ZERO; 64];
        for j in 0..160 {
            res_r[j] = reduce_unred(&res[j]);
        }
        for k in 0..64 {
            b0_r[k] = reduce_unred(&bank0[k]);
            b1_r[k] = reduce_unred(&bank1[k]);
        }
        (res_r, b0_r, b1_r)
    };
    type Acc = ([F128; 160], [F128; 64], [F128; 64]);
    let zero = || -> Acc { ([F128::ZERO; 160], [F128::ZERO; 64], [F128::ZERO; 64]) };
    let merge = |(mut r1, mut a0, mut a1): Acc, (r2, b0, b1): Acc| -> Acc {
        for j in 0..160 {
            r1[j] += r2[j];
        }
        for k in 0..64 {
            a0[k] += b0[k];
            a1[k] += b1[k];
        }
        (r1, a0, a1)
    };
    let states = crate::run_hetero_chunks_stateful(nchunks, zero, |s: &mut Acc, ci| {
        let c = per_chunk(ci);
        let cur = std::mem::replace(s, zero());
        *s = merge(cur, c);
    });
    states.into_iter().fold(zero(), merge)
}

/// Bit-slice one BLOCK_BYTES-byte block at `base` into 64 low planes (the high 64 of the
/// 128-wide transpose are the zero pad). `buf`'s high 8 bytes/row stay zero.
#[inline]
fn bitslice_block_into(
    packed: &[u8],
    base: usize,
    buf: &mut [u8; 128 * 16],
    planes: &mut [uint8x16_t; 128],
) {
    for r in 0..F128::BITS {
        buf[r * 16..r * 16 + 8].copy_from_slice(&packed[base + r * 8..base + r * 8 + 8]);
    }
    transpose_128x128(buf, planes);
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeSet, HashMap},
        fs::write,
    };

    use crate::{
        genus95_curve_code::{
            BaseMessage,
            product::extended_base_product_message,
            product_code_message,
            round1::{
                BLOCK_BYTES, F128, derived_m, round1_raw_packed,
                round1_straight_line_program_packed, round1_straight_line_program_packed_banks,
                round1_straight_line_program_packed_banks_fused,
                round1_straight_line_program_packed_banks_fused_padded,
            },
        },
        test_rng::Rng,
    };

    /// Derive the by-point fresh encode `M` from the M2 evaluator's OWN extension
    /// (`extended_base_product_message`), so the kernel speaks the evaluator's
    /// coordinate convention by construction. Slot `s` (0..158) maps to evaluator
    /// product coord `64 + s` (order1|order2|order3); slots 158,159 are D³ garbage
    /// (only 30 D³ points exist). The per-row product through this `M` must then
    /// equal `product_code_message` *with the identity bridge* — proving the
    /// legacy bench `M_MASK` (since deleted) and the evaluator were the same code
    /// in a different coordinate labeling, and that deriving `M` from the
    /// evaluator reconciles them.
    #[test]
    fn m_derived_from_evaluator_is_identity_bridge() {
        let mut m_eval = [0u64; 160];
        for j in 0..64 {
            let ext = extended_base_product_message(BaseMessage(1u64 << j));
            for s in 0..158 {
                if ext.get_bit(64 + s) {
                    m_eval[s] |= 1u64 << j;
                }
            }
        }

        let mut rng = Rng(0xD00D_F00D_0000_0001);
        for _ in 0..4096 {
            let a = rng.next_u64();
            let b = rng.next_u64();
            let mut af = [false; 160];
            let mut bf = [false; 160];
            for s in 0..160 {
                af[s] = (m_eval[s] & a).count_ones() & 1 == 1;
                bf[s] = (m_eval[s] & b).count_ones() & 1 == 1;
            }
            let mut pr = [false; 160];
            for p in 0..64 {
                pr[p] = (af[p] & ((b >> p) & 1 == 1)) ^ (((a >> p) & 1 == 1) & bf[p]);
            }
            for p in 0..64 {
                pr[64 + p] = (af[64 + p] & ((b >> p) & 1 == 1))
                    ^ (af[p] & bf[p])
                    ^ (((a >> p) & 1 == 1) & bf[64 + p]);
            }
            for p in 0..32 {
                pr[128 + p] = (af[128 + p] & ((b >> p) & 1 == 1))
                    ^ (af[64 + p] & bf[p])
                    ^ (af[p] & bf[64 + p])
                    ^ (((a >> p) & 1 == 1) & bf[128 + p]);
            }
            let pm = product_code_message(BaseMessage(a), BaseMessage(b));
            for s in 0..158 {
                assert_eq!(pr[s], pm.get_bit(64 + s), "coord {s} (a={a:#x} b={b:#x})");
            }
        }
    }

    /// The Paar SLP path equals the trusted `encode_direct` path — validates the
    /// generated `slp_derived::encode_straight_line_program_derived` against the derived `M`.
    #[test]
    fn round1_straight_line_program_matches_raw() {
        let mut rng = Rng(0x9ABC_DEF0);
        // Cover even n (all block-pairs) and odd n (exercises the trailing
        // single-block path via odd-length chunks).
        for n in [4usize, 3, 5, 1] {
            let mk = |rng: &mut Rng| -> Vec<u8> {
                let mut p = vec![0u8; n * BLOCK_BYTES];
                for x in p.iter_mut() {
                    *x = rng.next_u64() as u8;
                }
                p
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            let c = mk(&mut rng);
            let eq: Vec<F128> = (0..n)
                .map(|_| {
                    let lo = rng.next_u64();
                    let hi = if F128::BITS == 128 { rng.next_u64() } else { 0 };
                    F128::new(lo, hi)
                })
                .collect();

            let (slp_ab, slp_w) = round1_straight_line_program_packed(&a, &b, &c, &eq);
            let (raw_ab, raw_w) = round1_raw_packed(&a, &b, &c, &eq);
            assert!(
                (0..158).all(|s| slp_ab[s] == raw_ab[s]),
                "SLP AB != raw AB (n={n})"
            );
            assert!(
                (0..64).all(|k| slp_w[k] == raw_w[k]),
                "SLP wbar != raw wbar (n={n})"
            );
        }
    }

    /// PROTOTYPE: the fused banks path ([`round1_straight_line_program_packed_banks_fused`]) is
    /// bit-identical to [`round1_straight_line_program_packed_banks`] on res + both banks.
    #[test]
    fn round1_straight_line_program_banks_fused_matches_banks() {
        let mut rng = Rng(0xBA2C_F05E);
        for n in [4usize, 3, 5, 1, 16] {
            let mk = |rng: &mut Rng| -> Vec<u8> {
                let mut p = vec![0u8; n * BLOCK_BYTES];
                for x in p.iter_mut() {
                    *x = rng.next_u64() as u8;
                }
                p
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            let c = mk(&mut rng);
            let eq: Vec<F128> = (0..n)
                .map(|_| {
                    let lo = rng.next_u64();
                    let hi = if F128::BITS == 128 { rng.next_u64() } else { 0 };
                    F128::new(lo, hi)
                })
                .collect();

            let (ab, b0, b1) = round1_straight_line_program_packed_banks(&a, &b, &c, &eq);
            let (fab, f0, f1) = round1_straight_line_program_packed_banks_fused(&a, &b, &c, &eq);
            assert!(
                (0..160).all(|s| ab[s] == fab[s]),
                "fused banks res != res (n={n})"
            );
            assert!(
                (0..64).all(|k| b0[k] == f0[k] && b1[k] == f1[k]),
                "fused banks != banks (n={n})"
            );
        }
    }

    /// Pairing full blocks across a dead run must preserve their independent
    /// equality weights. Partial blocks and an odd live tail retain the single
    /// path, and declared-dead bits must not contribute even when nonzero.
    #[test]
    fn round1_fused_padded_matches_cleansed_reference() {
        use crate::zerocheck::BlockCoverage;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let mut rng = Rng(0x64_FA12_DEAD);
            for n in [0usize, 1, 2, 3, 17, 129] {
                let inputs: [Vec<u8>; 3] = std::array::from_fn(|_| {
                    (0..n * BLOCK_BYTES).map(|_| rng.next_u64() as u8).collect()
                });
                let eq: Vec<F128> = (0..n)
                    .map(|i| match i % 4 {
                        0 => F128::ZERO,
                        1 => F128::ONE,
                        2 => F128::from_u64(u64::MAX),
                        _ => F128::from_u64(rng.next_u64()),
                    })
                    .collect();
                for pattern in 0..3 {
                    let coverage: Vec<BlockCoverage> = (0..n)
                        .map(|i| match (pattern, i % 7) {
                            (0, _) => BlockCoverage::Full,
                            (1, 1 | 6) | (2, _) => BlockCoverage::Dead,
                            (1, 3) => BlockCoverage::Partial(vec![
                                (1, 65),
                                (130, 191),
                                (BLOCK_BYTES * 8 - 71, BLOCK_BYTES * 8 - 3),
                            ]),
                            _ => BlockCoverage::Full,
                        })
                        .collect();
                    // Independent bit-by-bit masking, rather than the production
                    // cleanse helper used by the padded kernel.
                    let clean: [Vec<u8>; 3] = std::array::from_fn(|which| {
                        let mut out = vec![0u8; n * BLOCK_BYTES];
                        for (block, covered) in coverage.iter().enumerate() {
                            for bit in 0..BLOCK_BYTES * 8 {
                                let live = match covered {
                                    BlockCoverage::Full => true,
                                    BlockCoverage::Dead => false,
                                    BlockCoverage::Partial(ranges) => {
                                        ranges.iter().any(|&(s, e)| s <= bit && bit < e)
                                    }
                                };
                                if live {
                                    let byte = block * BLOCK_BYTES + bit / 8;
                                    out[byte] |= inputs[which][byte] & (1 << (bit % 8));
                                }
                            }
                        }
                        out
                    });
                    let expected = round1_straight_line_program_packed_banks(
                        &clean[0], &clean[1], &clean[2], &eq,
                    );
                    let actual = round1_straight_line_program_packed_banks_fused_padded(
                        &inputs[0], &inputs[1], &inputs[2], &eq, &coverage,
                    );
                    assert_eq!(actual, expected, "n={n}, pattern={pattern}");
                }
            }
        });
    }

    /// The two-bank c-fold ([`round1_straight_line_program_packed_banks`]) reconstitutes the same
    /// AB message and the same `wbar` as [`round1_straight_line_program_packed`]: `res` identical
    /// and `bank0[k] + bank1[k] == wbar[k]` (the even/odd bit split is a partition
    /// of `pf`, so the field-mult distributes back to the original fold).
    #[test]
    fn round1_straight_line_program_banks_sum_matches_wbar() {
        let mut rng = Rng(0x5A17_BA17);
        for n in [4usize, 3, 5, 1] {
            let mk = |rng: &mut Rng| -> Vec<u8> {
                let mut p = vec![0u8; n * BLOCK_BYTES];
                for x in p.iter_mut() {
                    *x = rng.next_u64() as u8;
                }
                p
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            let c = mk(&mut rng);
            let eq: Vec<F128> = (0..n)
                .map(|_| {
                    let lo = rng.next_u64();
                    let hi = if F128::BITS == 128 { rng.next_u64() } else { 0 };
                    F128::new(lo, hi)
                })
                .collect();

            let (ab, w) = round1_straight_line_program_packed(&a, &b, &c, &eq);
            let (ab2, bank0, bank1) = round1_straight_line_program_packed_banks(&a, &b, &c, &eq);
            assert!((0..158).all(|s| ab[s] == ab2[s]), "banks AB != AB (n={n})");
            assert!(
                (0..64).all(|k| bank0[k] + bank1[k] == w[k]),
                "bank0 + bank1 != wbar (n={n})"
            );
        }
    }

    /// Independent scalar AG-message check over exactly the active number
    /// of friendly coefficients. The field64 path must consume 64 messages
    /// per block; the native path must consume 128, with no discarded rows.
    #[test]
    fn round1_bitslice_matches_scalar_product_code() {
        for blocks in [1usize, 3] {
            let mut rng = Rng(0x64_128_222);
            let mut inputs = [Vec::new(), Vec::new(), Vec::new()];
            for input in &mut inputs {
                for _ in 0..blocks * F128::BITS {
                    input.extend_from_slice(&rng.next_u64().to_le_bytes());
                }
            }
            let eq: Vec<F128> = (0..blocks)
                .map(|_| F128::from_u64(rng.next_u64()))
                .collect();
            let mut expected_ab = [F128::ZERO; 158];
            let mut expected_c = [F128::ZERO; 64];
            for (block, &weight) in eq.iter().enumerate() {
                let mut coefficient_weight = weight;
                for row in 0..F128::BITS {
                    let offset = (block * F128::BITS + row) * 8;
                    let read = |input: &[u8]| {
                        u64::from_le_bytes(input[offset..offset + 8].try_into().unwrap())
                    };
                    let product = product_code_message(
                        BaseMessage(read(&inputs[0])),
                        BaseMessage(read(&inputs[1])),
                    );
                    for (coordinate, value) in expected_ab.iter_mut().enumerate() {
                        if product.get_bit(64 + coordinate) {
                            *value += coefficient_weight;
                        }
                    }
                    let c = read(&inputs[2]);
                    for (coordinate, value) in expected_c.iter_mut().enumerate() {
                        if (c >> coordinate) & 1 != 0 {
                            *value += coefficient_weight;
                        }
                    }
                    coefficient_weight = crate::field::mul_by_x(coefficient_weight);
                }
            }
            let (actual_ab, actual_c) =
                round1_straight_line_program_packed(&inputs[0], &inputs[1], &inputs[2], &eq);
            let (fused_ab, even, odd) = round1_straight_line_program_packed_banks_fused(
                &inputs[0], &inputs[1], &inputs[2], &eq,
            );
            assert_eq!(&actual_ab[..158], &expected_ab);
            assert_eq!(&fused_ab[..158], &expected_ab);
            assert_eq!(actual_c, expected_c);
            for coordinate in 0..64 {
                assert_eq!(even[coordinate] + odd[coordinate], expected_c[coordinate]);
            }
        }
    }

    /// CODEGEN: Paar greedy SLP for the derived `M`, emitted to
    /// `src/genus95_curve_code/slp_derived.rs`. Run:
    /// `cargo test --release --lib _generate_straight_line_program_derived -- --ignored --nocapture`.
    #[ignore]
    #[test]
    fn _generate_straight_line_program_derived() {
        let m = derived_m();
        // Rows over signals; signals 0..64 are the inputs. Paar repeatedly pulls
        // out the most-common co-occurring pair into a new signal (one XOR gate).
        let mut rows: Vec<BTreeSet<usize>> = (0..160)
            .map(|k| (0..64).filter(|&j| (m[k] >> j) & 1 == 1).collect())
            .collect();
        let mut gates: Vec<(usize, usize)> = Vec::new();
        let mut next = 64usize;
        loop {
            let mut counts: HashMap<(usize, usize), u32> = HashMap::new();
            for row in &rows {
                let v: Vec<usize> = row.iter().copied().collect();
                for i in 0..v.len() {
                    for j in (i + 1)..v.len() {
                        *counts.entry((v[i], v[j])).or_insert(0) += 1;
                    }
                }
            }
            // Deterministic pick: highest count, ties broken by smallest pair.
            let mut best: Option<((usize, usize), u32)> = None;
            for (&pair, &c) in &counts {
                best = match best {
                    Some((bp, bc)) if bc > c || (bc == c && bp < pair) => Some((bp, bc)),
                    _ => Some((pair, c)),
                };
            }
            let ((a, b), cnt) = match best {
                Some(x) => x,
                None => break,
            };
            if cnt < 2 {
                break;
            }
            let s = next;
            next += 1;
            gates.push((a, b));
            for row in &mut rows {
                if row.contains(&a) && row.contains(&b) {
                    row.remove(&a);
                    row.remove(&b);
                    row.insert(s);
                }
            }
        }
        let chain: usize = rows.iter().map(|r| r.len().saturating_sub(1)).sum();
        eprintln!(
            "SLP(derived M): {} gates + {} chain XORs = {} total ops",
            gates.len(),
            chain,
            gates.len() + chain
        );

        let sig = |i: usize| {
            if i < 64 {
                format!("inp[{i}]")
            } else {
                format!("s{i}")
            }
        };
        let mut src = String::new();
        src.push_str(
            "//! AUTO-GENERATED by `round1::tests::_generate_straight_line_program_derived` — Paar greedy\n",
        );
        src.push_str(
            "//! straight-line program for the evaluator-derived `M` (160x64). Do not edit.\n",
        );
        src.push_str("use std::arch::aarch64::*;\n\n");
        src.push_str("#[inline(never)]\n");
        src.push_str("pub(crate) unsafe fn encode_straight_line_program_derived(inp: &[uint8x16_t; 64], out: &mut [uint8x16_t; 160]) {\n    unsafe {\n");
        for (g, &(a, b)) in gates.iter().enumerate() {
            src.push_str(&format!(
                "        let s{} = veorq_u8({}, {});\n",
                64 + g,
                sig(a),
                sig(b)
            ));
        }
        for k in 0..160 {
            let v: Vec<usize> = rows[k].iter().copied().collect();
            if v.is_empty() {
                src.push_str(&format!("        out[{k}] = vdupq_n_u8(0);\n"));
            } else {
                let mut e = sig(v[0]);
                for &x in &v[1..] {
                    e = format!("veorq_u8({}, {})", e, sig(x));
                }
                src.push_str(&format!("        out[{k}] = {e};\n"));
            }
        }
        src.push_str("    }\n}\n");
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/genus95_curve_code/slp_derived.rs"
        );
        write(path, src).expect("write slp_derived.rs");
        eprintln!("wrote {path}");
    }
}
