#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod portable {
    pub const LANES: usize = 8;

    type Lanes = [u32; LANES];

    #[inline(always)]
    fn add(a: Lanes, b: Lanes) -> Lanes {
        let mut r = [0u32; LANES];
        for ((r, a), b) in r.iter_mut().zip(&a).zip(&b) {
            *r = a.wrapping_add(*b);
        }
        r
    }

    #[inline(always)]
    fn xor_rotl(a: Lanes, b: Lanes, n: u32) -> Lanes {
        let mut r = [0u32; LANES];
        for ((r, a), b) in r.iter_mut().zip(&a).zip(&b) {
            *r = (a ^ b).rotate_left(n);
        }
        r
    }

    #[inline(always)]
    pub(super) fn quarter_round(x: &mut [Lanes; 16], a: usize, b: usize, c: usize, d: usize) {
        x[a] = add(x[a], x[b]);
        x[d] = xor_rotl(x[d], x[a], 16);
        x[c] = add(x[c], x[d]);
        x[b] = xor_rotl(x[b], x[c], 12);
        x[a] = add(x[a], x[b]);
        x[d] = xor_rotl(x[d], x[a], 8);
        x[c] = add(x[c], x[d]);
        x[b] = xor_rotl(x[b], x[c], 7);
    }

    pub fn blocks(state: &[u32; 16], base: u64, out: &mut [u8; 64 * LANES]) {
        let mut input: [Lanes; 16] = [[0; LANES]; 16];
        for (slot, w) in input.iter_mut().zip(state) {
            *slot = [*w; LANES];
        }
        let counters: [u64; LANES] = std::array::from_fn(|l| base.wrapping_add(l as u64));
        input[12] = counters.map(|c| c as u32);
        input[13] = counters.map(|c| (c >> 32) as u32);
        let mut x = input;
        for _ in 0..10 {
            quarter_round(&mut x, 0, 4, 8, 12);
            quarter_round(&mut x, 1, 5, 9, 13);
            quarter_round(&mut x, 2, 6, 10, 14);
            quarter_round(&mut x, 3, 7, 11, 15);
            quarter_round(&mut x, 0, 5, 10, 15);
            quarter_round(&mut x, 1, 6, 11, 12);
            quarter_round(&mut x, 2, 7, 8, 13);
            quarter_round(&mut x, 3, 4, 9, 14);
        }
        for (i, (xi, ii)) in x.iter().zip(&input).enumerate() {
            let s = add(*xi, *ii);
            for (l, w) in s.iter().enumerate() {
                let at = l * 64 + 4 * i;
                out[at..at + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct ChaCha20 {
    state: [u32; 16],
}

impl ChaCha20 {
    pub const BLOCK: usize = 64;
    const LANES: usize = portable::LANES;

    const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

    pub fn new(key: &[u8; 32], nonce: u64) -> Self {
        let mut state = [0u32; 16];
        state[..4].copy_from_slice(&Self::CONSTANTS);
        for (w, k) in state[4..12].iter_mut().zip(key.as_chunks::<4>().0) {
            *w = u32::from_le_bytes(*k);
        }
        state[14] = nonce as u32;
        state[15] = (nonce >> 32) as u32;
        Self { state }
    }

    #[cfg(test)]
    fn from_words(words: [u32; 16]) -> Self {
        Self { state: words }
    }

    fn counter(&self) -> u64 {
        (self.state[12] as u64) | ((self.state[13] as u64) << 32)
    }

    fn set_counter(&mut self, c: u64) {
        self.state[12] = c as u32;
        self.state[13] = (c >> 32) as u32;
    }

    pub fn fill(&mut self, out: &mut [u8]) {
        assert_eq!(out.len() % (Self::BLOCK * Self::LANES), 0);
        #[cfg(target_arch = "aarch64")]
        {
            self.fill_neon(out);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.fill_portable(out);
        }
    }

    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub fn fill_portable(&mut self, out: &mut [u8]) {
        let (chunks, rest) = out.as_chunks_mut::<{ Self::BLOCK * portable::LANES }>();
        assert!(rest.is_empty(), "a whole number of steps");
        for chunk in chunks {
            let base = self.counter();
            portable::blocks(&self.state, base, chunk);
            self.set_counter(base.wrapping_add(Self::LANES as u64));
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn fill_neon(&mut self, out: &mut [u8]) {
        let (chunks, rest) = out.as_chunks_mut::<{ Self::BLOCK * 4 }>();
        assert!(rest.is_empty(), "a whole number of steps");
        for chunk in chunks {
            let base = self.counter();
            unsafe { neon::blocks4(&self.state, base, chunk) };
            self.set_counter(base.wrapping_add(4));
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn rotl<const N: i32, const M: i32>(x: uint32x4_t) -> uint32x4_t {
        vsliq_n_u32::<N>(vshrq_n_u32::<M>(x), x)
    }

    #[inline(always)]
    unsafe fn quarter_round(x: &mut [uint32x4_t; 16], a: usize, b: usize, c: usize, d: usize) {
        x[a] = vaddq_u32(x[a], x[b]);
        x[d] = rotl::<16, 16>(veorq_u32(x[d], x[a]));
        x[c] = vaddq_u32(x[c], x[d]);
        x[b] = rotl::<12, 20>(veorq_u32(x[b], x[c]));
        x[a] = vaddq_u32(x[a], x[b]);
        x[d] = rotl::<8, 24>(veorq_u32(x[d], x[a]));
        x[c] = vaddq_u32(x[c], x[d]);
        x[b] = rotl::<7, 25>(veorq_u32(x[b], x[c]));
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn blocks4(state: &[u32; 16], base: u64, out: &mut [u8; 256]) {
        let mut input: [uint32x4_t; 16] = [vdupq_n_u32(0); 16];
        for (slot, w) in input.iter_mut().zip(state) {
            *slot = vdupq_n_u32(*w);
        }
        let counters: [u64; 4] = std::array::from_fn(|l| base.wrapping_add(l as u64));
        let lo: [u32; 4] = std::array::from_fn(|l| counters[l] as u32);
        let hi: [u32; 4] = std::array::from_fn(|l| (counters[l] >> 32) as u32);
        input[12] = vld1q_u32(lo.as_ptr());
        input[13] = vld1q_u32(hi.as_ptr());
        let mut x = input;
        for _ in 0..10 {
            quarter_round(&mut x, 0, 4, 8, 12);
            quarter_round(&mut x, 1, 5, 9, 13);
            quarter_round(&mut x, 2, 6, 10, 14);
            quarter_round(&mut x, 3, 7, 11, 15);
            quarter_round(&mut x, 0, 5, 10, 15);
            quarter_round(&mut x, 1, 6, 11, 12);
            quarter_round(&mut x, 2, 7, 8, 13);
            quarter_round(&mut x, 3, 4, 9, 14);
        }
        for (i, (xi, ii)) in x.iter().zip(&input).enumerate() {
            let s = vaddq_u32(*xi, *ii);
            let mut words = [0u32; 4];
            vst1q_u32(words.as_mut_ptr(), s);
            for (l, w) in words.iter().enumerate() {
                let at = l * 64 + 4 * i;
                out[at..at + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_neon_kernel_matches_the_portable_lanes() {
        let key: [u8; 32] = std::array::from_fn(|i| (i * 13 + 5) as u8);
        let mut a = ChaCha20::new(&key, 0xfedc_ba98_7654_3210);
        let mut b = a.clone();
        a.set_counter(0xffff_fffe);
        b.set_counter(0xffff_fffe);
        let mut out_a = vec![0u8; 4096];
        let mut out_b = vec![0u8; 4096];
        a.fill_neon(&mut out_a);
        b.fill_portable(&mut out_b);
        assert_eq!(out_a, out_b);
        assert_eq!(a.counter(), b.counter());
    }

    #[test]
    fn the_quarter_round_matches_the_rfc() {
        const L: usize = portable::LANES;
        let mut x: [[u32; L]; 16] = [[0; L]; 16];
        x[0] = [0x1111_1111; L];
        x[1] = [0x0102_0304; L];
        x[2] = [0x9b8d_6f43; L];
        x[3] = [0x0123_4567; L];
        portable::quarter_round(&mut x, 0, 1, 2, 3);
        assert_eq!(x[0], [0xea2a_92f4; L]);
        assert_eq!(x[1], [0xcb1c_f8ce; L]);
        assert_eq!(x[2], [0x4581_472e; L]);
        assert_eq!(x[3], [0x5881_c4bb; L]);
    }

    #[test]
    fn the_block_function_matches_the_rfc() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        let mut s = ChaCha20::new(&key, 0);
        s.state[12] = 1;
        s.state[13] = 0x0900_0000;
        s.state[14] = 0x4a00_0000;
        s.state[15] = 0;
        let s = ChaCha20::from_words(s.state);
        let mut out = vec![0u8; ChaCha20::BLOCK * ChaCha20::LANES];
        s.clone().fill_portable(&mut out);
        let mut out_fill = vec![0u8; ChaCha20::BLOCK * ChaCha20::LANES];
        s.clone().fill(&mut out_fill);
        assert_eq!(
            out, out_fill,
            "the active kernel agrees with the portable one"
        );
        let expected: [u8; 64] = [
            0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20,
            0x71, 0xc4, 0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a,
            0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2,
            0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9,
            0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
        ];
        assert_eq!(&out[..64], &expected[..]);
    }

    #[test]
    fn the_lanes_are_consecutive_blocks() {
        let key: [u8; 32] = std::array::from_fn(|i| (i * 7 + 3) as u8);
        const L: usize = ChaCha20::LANES;
        let mut whole = ChaCha20::new(&key, 0x0123_4567_89ab_cdef);
        let mut out = vec![0u8; ChaCha20::BLOCK * L * 2];
        whole.fill(&mut out);
        for l in 0..(2 * L as u64) {
            let mut one = ChaCha20::new(&key, 0x0123_4567_89ab_cdef);
            one.set_counter(l);
            let mut step = vec![0u8; ChaCha20::BLOCK * L];
            one.fill_portable(&mut step);
            assert_eq!(&step[..64], &out[l as usize * 64..(l as usize + 1) * 64]);
        }
        assert_eq!(whole.counter(), 2 * L as u64);
    }

    #[test]
    #[ignore = "measurement"]
    fn keystream_throughput() {
        let key = [7u8; 32];
        let mut s = ChaCha20::new(&key, 1);
        let mut buf = vec![0u8; 1 << 16];
        s.fill(&mut buf);
        let t = std::time::Instant::now();
        let passes = 1000;
        for _ in 0..passes {
            s.fill(&mut buf);
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "ChaCha20 keystream: {:.0} MB/s ({:.2} us per 64 KiB)",
            (passes * buf.len()) as f64 / secs / 1e6,
            secs / passes as f64 * 1e6
        );
        let t = std::time::Instant::now();
        for _ in 0..passes {
            s.fill_portable(&mut buf);
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "ChaCha20 keystream, portable lanes: {:.0} MB/s ({:.2} us per 64 KiB)",
            (passes * buf.len()) as f64 / secs / 1e6,
            secs / passes as f64 * 1e6
        );
    }
}
