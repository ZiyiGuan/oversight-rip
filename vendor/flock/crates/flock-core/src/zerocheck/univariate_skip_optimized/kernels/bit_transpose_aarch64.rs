//! Field-independent NEON byte transpose shared by both field backends.

use core::arch::aarch64::{
    uint8x16x4_t, vandq_u64, vdupq_n_u64, veorq_u64, vld1q_u8, vqtbl4q_u8, vreinterpretq_u8_u64,
    vreinterpretq_u64_u8, vshlq_n_u64, vshrq_n_u64, vst1q_u8,
};

/// NEON 64-byte bit-transpose. Two-stage:
///   1. `vqtbl4q_u8` reorders the 64 input bytes so each 8-byte group within
///      the output is one byte-chunk's worth of `x_small=0..8` bytes.
///   2. Three rounds of bit-swap at distances 7, 14, 28 across `uint64x2_t`
///      lanes do the actual 8×8 bit transpose.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn bit_transpose_64bytes_neon(input: &[u8; 64], output: &mut [u8; 64]) {
    unsafe {
        let in_ptr = input.as_ptr();
        let v0 = vld1q_u8(in_ptr);
        let v1 = vld1q_u8(in_ptr.add(16));
        let v2 = vld1q_u8(in_ptr.add(32));
        let v3 = vld1q_u8(in_ptr.add(48));
        let table = uint8x16x4_t(v0, v1, v2, v3);

        // vqtbl4q indexes that bring bytes belonging to byte-chunk b ∈ 0..8
        // into contiguous 8-byte runs, packed two-chunks-per-Q-reg.
        const IDX0: [u8; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57];
        const IDX1: [u8; 16] = [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59];
        const IDX2: [u8; 16] = [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61];
        const IDX3: [u8; 16] = [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63];

        let mut y0 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX0.as_ptr())));
        let mut y1 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX1.as_ptr())));
        let mut y2 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX2.as_ptr())));
        let mut y3 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX3.as_ptr())));

        let mask1 = vdupq_n_u64(0x00AA00AA00AA00AA);
        let mask2 = vdupq_n_u64(0x0000CCCC0000CCCC);
        let mask3 = vdupq_n_u64(0x00000000F0F0F0F0);

        // Round 1: distance 7.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<7>(y0)), mask1);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<7>(y1)), mask1);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<7>(y2)), mask1);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<7>(y3)), mask1);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<7>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<7>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<7>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<7>(t3)));

        // Round 2: distance 14.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<14>(y0)), mask2);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<14>(y1)), mask2);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<14>(y2)), mask2);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<14>(y3)), mask2);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<14>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<14>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<14>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<14>(t3)));

        // Round 3: distance 28.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<28>(y0)), mask3);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<28>(y1)), mask3);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<28>(y2)), mask3);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<28>(y3)), mask3);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<28>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<28>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<28>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<28>(t3)));

        let out_ptr = output.as_mut_ptr();
        vst1q_u8(out_ptr, vreinterpretq_u8_u64(y0));
        vst1q_u8(out_ptr.add(16), vreinterpretq_u8_u64(y1));
        vst1q_u8(out_ptr.add(32), vreinterpretq_u8_u64(y2));
        vst1q_u8(out_ptr.add(48), vreinterpretq_u8_u64(y3));
    }
}
