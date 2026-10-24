use rmip_field::{F128, F64};
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use rmip_packed::subfield::SkipTables;
use rmip_packed::zerocheck::{eq_table, eval_bits, prove, verify, PackedBits, ZerocheckError};
use rmip_packed::PackedField;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn witness<F: PackedField>(m: usize, seed: u64) -> (PackedBits<F>, PackedBits<F>, PackedBits<F>) {
    let mut s = seed;
    let a: Vec<bool> = (0..1usize << m).map(|_| lcg(&mut s) & 1 == 1).collect();
    let b: Vec<bool> = (0..1usize << m).map(|_| lcg(&mut s) & 1 == 1).collect();
    let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| x & y).collect();
    (
        PackedBits::from_bits(m, &a),
        PackedBits::from_bits(m, &b),
        PackedBits::from_bits(m, &c),
    )
}

fn honest<F: PackedField>(m: usize, seed: u64) {
    let tables = SkipTables::<F>::new();
    let (a, b, c) = witness::<F>(m, seed);
    let mut pch = SeededChallenger::new(seed ^ 0xF00D);
    let (proof, claim) = prove(&tables, &a, &b, &c, &mut pch);
    let mut vch = SeededChallenger::new(seed ^ 0xF00D);
    let verified = verify(&tables, m, &proof, &mut vch).expect("honest proof accepted");
    assert_eq!(verified, claim, "the two sides agree on the claims");
    assert_eq!(
        pch.transcript().canonical(),
        vch.transcript().canonical(),
        "one transcript"
    );
    assert_eq!(
        eval_bits(&tables, &a, claim.z, &claim.rho),
        claim.a_eval,
        "â(z, ρ)"
    );
    assert_eq!(
        eval_bits(&tables, &b, claim.z, &claim.rho),
        claim.b_eval,
        "b̂(z, ρ)"
    );
    assert_eq!(
        claim.c_claim.evaluate(&c),
        claim.c_claim.value,
        "ĉ(z, r) as the bit-claim states it"
    );
    assert_eq!(claim.c_s_hat.len(), F::BITS);
    let _ = eq_table::<F>(&claim.r);
}

fn flipped_bit_rejected<F: PackedField>(m: usize, seed: u64) {
    let tables = SkipTables::<F>::new();
    let (a, b, mut c) = witness::<F>(m, seed);
    let i = (lcg(&mut { seed }) as usize) % (1 << m);
    let word = i / F::BITS;
    c.words[word] = F::from_bits(c.words[word].to_bits() ^ (1u128 << (i % F::BITS)));
    let mut pch = SeededChallenger::new(seed);
    let (proof, _) = prove(&tables, &a, &b, &c, &mut pch);
    let mut vch = SeededChallenger::new(seed);
    assert_eq!(
        verify(&tables, m, &proof, &mut vch),
        Err(ZerocheckError::FinalProductMismatch),
        "a dishonest witness is caught at the final check"
    );
}

#[test]
fn an_honest_zerocheck_is_accepted_and_its_claims_are_true() {
    for (m, seed) in [(7usize, 1u64), (9, 2), (13, 3), (16, 4)] {
        honest::<F64>(m, seed);
        honest::<F128>(m, seed);
    }
}

#[test]
fn a_flipped_bit_is_rejected() {
    for (m, seed) in [(7usize, 11u64), (10, 12), (13, 13)] {
        flipped_bit_rejected::<F64>(m, seed);
        flipped_bit_rejected::<F128>(m, seed);
    }
}

#[test]
fn the_eq_table_is_indexed_lowest_bit_first() {
    let r = [F64::new(3), F64::new(5)];
    let t = eq_table(&r);
    let one = F64::ONE;
    assert_eq!(t[0], (one + r[0]) * (one + r[1]));
    assert_eq!(t[1], r[0] * (one + r[1]));
    assert_eq!(t[2], (one + r[0]) * r[1]);
    assert_eq!(t[3], r[0] * r[1]);
}

#[test]
#[ignore = "measurement"]
fn zerocheck_throughput() {
    use std::time::Instant;
    fn run<F: PackedField>(name: &str, m: usize) {
        let tables = SkipTables::<F>::new();
        let (a, b, c) = witness::<F>(m, 77);
        let mut best_p = f64::MAX;
        let mut best_v = f64::MAX;
        for _ in 0..3 {
            let mut pch = SeededChallenger::new(5);
            let t = Instant::now();
            let (proof, _) = prove(&tables, &a, &b, &c, &mut pch);
            best_p = best_p.min(t.elapsed().as_secs_f64());
            let mut vch = SeededChallenger::new(5);
            let t = Instant::now();
            verify(&tables, m, &proof, &mut vch).unwrap();
            best_v = best_v.min(t.elapsed().as_secs_f64());
        }
        let bits = (1usize << m) as f64;
        println!(
            "{name}: m = {m}: prover {:.3} s ({:.2} ns per bit), verifier {:.2} ms",
            best_p,
            best_p * 1e9 / bits,
            best_v * 1e3
        );
    }
    run::<F64>("GF(2^64) ", 22);
    run::<F128>("GF(2^128)", 22);
}

#[test]
#[ignore = "measurement"]
fn round_one_parts() {
    use rmip_packed::phases::Phases;
    use rmip_packed::subfield::transpose_8x64;
    use rmip_packed::zerocheck::prove_with_phases;
    use std::hint::black_box;
    use std::time::Instant;
    let m: usize = std::env::var("RMIP_BITS_LOG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(26);
    fn run<F: PackedField>(name: &str, m: usize) {
        let tables = SkipTables::<F>::new();
        let (a, b, c) = witness::<F>(m, 91);
        let blocks = a.blocks();
        let bits = (1usize << m) as f64;
        let ns = |t: std::time::Duration| t.as_secs_f64() * 1e9 / bits;
        let mut sink = 0u8;
        let t = Instant::now();
        for x in 0..blocks {
            let av = tables.on_lambda(a.block(x));
            let bv = tables.on_lambda(b.block(x));
            let ab = tables.subfield.mul64(&av, &bv);
            sink ^= ab[x & 63];
        }
        let t_lambda = t.elapsed();
        let t = Instant::now();
        for x0 in (0..blocks).step_by(8) {
            let aw: [u64; 8] = std::array::from_fn(|k| a.block(x0 + k));
            let bw: [u64; 8] = std::array::from_fn(|k| b.block(x0 + k));
            let r = tables.products8(&aw, &bw);
            sink ^= r[x0 & 63];
        }
        let t_products = t.elapsed();
        let group = 1usize << tables.fixed.count;
        let mut s = 0x5151u64;
        let rows: Vec<[u8; 64]> = (0..group / 8)
            .map(|_| std::array::from_fn(|_| (lcg(&mut s) & 0xFF) as u8))
            .collect();
        let groups = blocks / group;
        let mut facc = F::ZERO;
        let t = Instant::now();
        for g in 0..groups {
            let out = tables.fixed.convert_rows(black_box(&rows));
            facc += out[g & 63];
        }
        let t_convert = t.elapsed();
        let t = Instant::now();
        for x0 in (0..blocks).step_by(8) {
            let cw: [u64; 8] = std::array::from_fn(|k| c.block(x0 + k));
            let tr = transpose_8x64(&cw);
            sink ^= tr[x0 & 63];
        }
        let t_transpose = t.elapsed();
        let mut phases = Phases::new();
        let mut pch = SeededChallenger::new(5);
        let t = Instant::now();
        let (proof, _) = prove_with_phases(&tables, &a, &b, &c, &mut pch, Some(&mut phases));
        let t_prove = t.elapsed();
        black_box((sink, facc, proof));
        let converts = if F::BITS == 64 { 2.0 } else { 3.0 };
        println!(
            "{name}: m = {m}: on_lambda x2 + mul64 {:.3} ns/bit; products8 {:.3}; convert_rows {:.3} per call x{converts} per group = {:.3}; transpose_8x64 {:.3}; round one {:.3} ({:.1} ms), zerocheck {:.3}, prover {:.3} ns/bit",
            ns(t_lambda),
            ns(t_products),
            ns(t_convert),
            ns(t_convert) * converts,
            ns(t_transpose),
            ns(phases.total("zerocheck round one")),
            phases.total("zerocheck round one").as_secs_f64() * 1e3,
            ns(phases.total("zerocheck")),
            ns(t_prove)
        );
    }
    run::<F64>("GF(2^64) ", m);
    run::<F128>("GF(2^128)", m);
}

struct OnesThen {
    ones: usize,
    state: u64,
    t: rmip_iop::transcript::Transcript,
}

impl OnesThen {
    fn new(ones: usize, seed: u64) -> Self {
        Self {
            ones,
            state: seed,
            t: rmip_iop::transcript::Transcript::default(),
        }
    }
}

impl RawChallenger for OnesThen {
    fn sample(&mut self) -> F128 {
        let c = if self.ones > 0 {
            self.ones -= 1;
            F128::ONE
        } else {
            let lo = lcg(&mut self.state) as u128;
            let hi = lcg(&mut self.state) as u128;
            F128::new(lo | (hi << 64))
        };
        self.t.record_coin(c);
        c
    }
    fn record_ordinary(&mut self, b: &[u8]) {
        self.t.record_ordinary(b.to_vec());
    }
    fn post_oracle(&mut self, i: rmip_iop::transcript::OracleId) {
        self.t.post_oracle(i);
    }
    fn transcript(&self) -> &rmip_iop::transcript::Transcript {
        &self.t
    }
}

#[test]
fn an_eq_coordinate_equal_to_one_is_redrawn() {
    use rmip_packed::zerocheck::{ZerocheckError, EQ_COIN_REDRAWS};
    let tables = SkipTables::<rmip_field::F64>::new();
    let (a, b, c) = witness::<rmip_field::F64>(13, 9);
    for ones in [1, EQ_COIN_REDRAWS] {
        let (proof, claim) = prove(&tables, &a, &b, &c, &mut OnesThen::new(ones, 5));
        let verified = verify(&tables, 13, &proof, &mut OnesThen::new(ones, 5))
            .expect("accepted after the redraw");
        assert_eq!(verified, claim);
        assert!(claim.r.iter().all(|&x| x != rmip_field::F64::ONE));
    }
    let ones = EQ_COIN_REDRAWS + 1;
    let (proof, _) = prove(&tables, &a, &b, &c, &mut OnesThen::new(ones, 5));
    assert!(matches!(
        verify(&tables, 13, &proof, &mut OnesThen::new(ones, 5)),
        Err(ZerocheckError::DegenerateEqChallenge { round: 6 })
    ));
}
