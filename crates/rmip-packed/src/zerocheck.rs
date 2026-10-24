use std::time::Instant;

use rmip_iop::transcript::RawChallenger;

use crate::phases::Phases;
use crate::ringswitch::BitClaim;
use crate::subfield::{transpose_8x64, SkipTables};
use crate::PackedField;

fn eq_point<F: PackedField>(
    tables: &SkipTables<F>,
    n: usize,
    ch: &mut dyn RawChallenger,
) -> Vec<F> {
    let fixed = tables.fixed.count.min(n);
    let mut r: Vec<F> = tables.fixed.r[..fixed].to_vec();
    r.extend((fixed..n).map(|_| eq_coin::<F>(ch)));
    r
}

pub const EQ_COIN_REDRAWS: usize = 4;

fn eq_coin<F: PackedField>(ch: &mut dyn RawChallenger) -> F {
    let mut c = coin::<F>(ch);
    for _ in 0..EQ_COIN_REDRAWS {
        if c != F::ONE {
            break;
        }
        c = coin::<F>(ch);
    }
    c
}

#[derive(Clone, Debug)]
pub struct PackedBits<F: PackedField> {
    pub m: usize,
    pub words: Vec<F>,
}

impl<F: PackedField> PackedBits<F> {
    pub fn new(m: usize, words: Vec<F>) -> Self {
        assert!(m >= 7, "the skip takes six bits and a round needs one more");
        assert_eq!(words.len() * F::BITS, 1 << m);
        Self { m, words }
    }

    pub fn from_bits(m: usize, bits: &[bool]) -> Self {
        assert_eq!(bits.len(), 1 << m);
        Self::new(m, crate::pack(bits))
    }

    #[inline]
    pub fn block(&self, x: usize) -> u64 {
        let bit = 64 * x;
        let word = self.words[bit / F::BITS].to_bits();
        (word >> (bit % F::BITS)) as u64
    }

    pub fn blocks(&self) -> usize {
        1 << (self.m - 6)
    }

    pub fn bit(&self, i: usize) -> bool {
        (self.words[i / F::BITS].to_bits() >> (i % F::BITS)) & 1 == 1
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckProof<F> {
    pub round1: Vec<F>,
    pub s_hat_c: Vec<F>,
    pub rounds: Vec<(F, F)>,
    pub final_a: F,
    pub final_b: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckClaim<F> {
    pub r: Vec<F>,
    pub z: F,
    pub rho: Vec<F>,
    pub a_eval: F,
    pub b_eval: F,
    pub c_claim: BitClaim<F>,
    pub c_s_hat: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZerocheckError {
    BadLength {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    DegenerateEqChallenge {
        round: usize,
    },
    FinalProductMismatch,
}

pub fn coin<F: PackedField>(ch: &mut dyn RawChallenger) -> F {
    F::from_bits(ch.sample().to_u128())
}

pub fn record<F: PackedField>(ch: &mut dyn RawChallenger, elements: &[F]) {
    let mut bytes = Vec::with_capacity(16 * elements.len());
    for e in elements {
        bytes.extend_from_slice(&e.to_bits().to_be_bytes());
    }
    ch.record_ordinary(&bytes);
}

pub fn eq_table<F: PackedField>(r: &[F]) -> Vec<F> {
    let mut table = vec![F::ONE];
    for &ri in r {
        let len = table.len();
        let mut next = Vec::with_capacity(2 * len);
        next.extend(table.iter().map(|&t| t * (F::ONE + ri)));
        next.extend(table.iter().map(|&t| t * ri));
        table = next;
    }
    table
}

pub fn eval_bits<F: PackedField>(
    tables: &SkipTables<F>,
    bits: &PackedBits<F>,
    z: F,
    rho: &[F],
) -> F {
    assert_eq!(rho.len(), bits.m - 6);
    let w = tables.fold_weights(z);
    let eq = eq_table(rho);
    let mut acc = F::ZERO;
    for (x, &weight) in eq.iter().enumerate().take(bits.blocks()) {
        let block = bits.block(x);
        let mut folded = F::ZERO;
        for (j, wj) in w.iter().enumerate() {
            if (block >> j) & 1 == 1 {
                folded += *wj;
            }
        }
        acc += weight * folded;
    }
    acc
}

fn fold_tables<F: PackedField>(weights: &[F]) -> Vec<[F; 256]> {
    assert_eq!(weights.len(), 64);
    weights
        .chunks(8)
        .map(|chunk| {
            let mut t = [F::ZERO; 256];
            for (byte, slot) in t.iter_mut().enumerate() {
                let mut acc = F::ZERO;
                for (i, w) in chunk.iter().enumerate() {
                    if (byte >> i) & 1 == 1 {
                        acc += *w;
                    }
                }
                *slot = acc;
            }
            t
        })
        .collect()
}

#[inline]
fn fold_two<F: PackedField>(tables: &[[F; 256]], wa: u64, wb: u64) -> (F, F) {
    let (mut a0, mut a1, mut b0, mut b1) = (F::ZERO, F::ZERO, F::ZERO, F::ZERO);
    for k in 0..4 {
        let (t0, t1) = (&tables[2 * k], &tables[2 * k + 1]);
        let (s0, s1) = (16 * k, 16 * k + 8);
        a0 += t0[((wa >> s0) & 0xFF) as usize];
        b0 += t0[((wb >> s0) & 0xFF) as usize];
        a1 += t1[((wa >> s1) & 0xFF) as usize];
        b1 += t1[((wb >> s1) & 0xFF) as usize];
    }
    (a0 + a1, b0 + b1)
}

fn fold_block<F: PackedField>(tables: &[[F; 256]], block: u64) -> F {
    if block == 0 {
        return F::ZERO;
    }
    let mut acc = F::ZERO;
    for (c, t) in tables.iter().enumerate() {
        acc += t[((block >> (8 * c)) & 0xFF) as usize];
    }
    acc
}

pub fn prove<F: PackedField>(
    tables: &SkipTables<F>,
    a: &PackedBits<F>,
    b: &PackedBits<F>,
    c: &PackedBits<F>,
    ch: &mut dyn RawChallenger,
) -> (ZerocheckProof<F>, ZerocheckClaim<F>) {
    prove_with_phases(tables, a, b, c, ch, None)
}

pub fn prove_with_phases<F: PackedField>(
    tables: &SkipTables<F>,
    a: &PackedBits<F>,
    b: &PackedBits<F>,
    c: &PackedBits<F>,
    ch: &mut dyn RawChallenger,
    mut phases: Option<&mut Phases>,
) -> (ZerocheckProof<F>, ZerocheckClaim<F>) {
    let m = a.m;
    assert!(b.m == m && c.m == m);
    let n = m - 6;
    let blocks = a.blocks();

    let r = eq_point(tables, n, ch);
    let started = Instant::now();

    let fixed = tables.fixed.count.min(n);
    let per_word = F::BITS / 64;
    let in_word = per_word.trailing_zeros() as usize;
    let mut round1 = vec![F::ZERO; 64];
    let mut s_hat_c = vec![F::ZERO; F::BITS];
    if fixed < 3 {
        let eq = eq_table(&r);
        let eq_words = eq_table(&r[in_word..]);
        for (x, &w) in eq.iter().enumerate() {
            let av = tables.on_lambda(a.block(x));
            let bv = tables.on_lambda(b.block(x));
            let cv = tables.on_lambda(c.block(x));
            let ab = tables.subfield.mul64(&av, &bv);
            for l in 0..64 {
                round1[l] += w * tables.subfield.embed[(ab[l] ^ cv[l]) as usize];
            }
            let (word, p) = (x / per_word, x % per_word);
            let mut bits = c.block(x);
            while bits != 0 {
                let l = bits.trailing_zeros() as usize;
                s_hat_c[64 * p + l] += eq_words[word];
                bits &= bits - 1;
            }
        }
    } else {
        let group = 1usize << fixed;
        let eq_hi = eq_table(&r[fixed..]);
        let mut reduced_ab = vec![[0u8; 64]; group / 8];
        let bank_masks: Vec<u8> = if per_word == 1 {
            vec![0xFF]
        } else {
            vec![0x55, 0xAA]
        };
        let mut reduced_c: Vec<Vec<[u8; 64]>> = vec![vec![[0u8; 64]; group / 8]; per_word];
        let mut banks = vec![[F::ZERO; 64]; per_word];
        for (g, &w) in eq_hi.iter().enumerate() {
            let (mut any_ab, mut any_c) = (false, false);
            for medium in 0..group / 8 {
                let x0 = g * group + medium * 8;
                let a_words: [u64; 8] = std::array::from_fn(|k| a.block(x0 + k));
                let b_words: [u64; 8] = std::array::from_fn(|k| b.block(x0 + k));
                let c_words: [u64; 8] = std::array::from_fn(|k| c.block(x0 + k));
                if (0..8).all(|k| a_words[k] == 0 || b_words[k] == 0) {
                    reduced_ab[medium] = [0u8; 64];
                } else {
                    any_ab = true;
                    reduced_ab[medium] = tables.products8(&a_words, &b_words);
                }
                if c_words.iter().all(|&cw| cw == 0) {
                    for bank in reduced_c.iter_mut() {
                        bank[medium] = [0u8; 64];
                    }
                    continue;
                }
                any_c = true;
                let t = transpose_8x64(&c_words);
                if per_word == 1 {
                    reduced_c[0][medium] = t;
                } else {
                    for (bank, &mask) in reduced_c.iter_mut().zip(&bank_masks) {
                        let row = &mut bank[medium];
                        for (dst, &src) in row.iter_mut().zip(&t) {
                            *dst = src & mask;
                        }
                    }
                }
            }
            if any_ab {
                let converted_ab = tables.fixed.convert_rows(&reduced_ab);
                for l in 0..64 {
                    round1[l] += w * converted_ab[l];
                }
            }
            if any_c {
                for (bank, rows) in banks.iter_mut().zip(&reduced_c) {
                    let converted = tables.fixed.convert_rows(rows);
                    for l in 0..64 {
                        bank[l] += w * converted[l];
                    }
                }
            }
        }
        let scale = tables.fixed.scale[fixed];
        let mut c_on_s = [F::ZERO; 64];
        for bank in banks.iter_mut() {
            for l in 0..64 {
                bank[l] *= scale;
                c_on_s[l] += bank[l];
            }
        }
        let c_on_lambda = tables.extend_to_lambda(&c_on_s);
        for (v, c_l) in round1.iter_mut().zip(&c_on_lambda) {
            *v = *v * scale + *c_l;
        }
        let eq_in = eq_table(&r[..in_word]);
        for (p, bank) in banks.iter().enumerate() {
            let inv = eq_in[p].inv().expect("a fixed coordinate is not 0 or 1");
            for l in 0..64 {
                s_hat_c[64 * p + l] = bank[l] * inv;
            }
        }
    }
    if let Some(p) = phases.as_deref_mut() {
        p.add("zerocheck round one", started.elapsed());
    }
    record(ch, &round1);
    record(ch, &s_hat_c);
    let started = Instant::now();

    let z: F = coin(ch);
    let c_claim = {
        let coeffs = BitClaim::from_skip_point(tables, z, &r, F::ZERO).coeffs;
        let v_c = coeffs
            .iter()
            .zip(&s_hat_c)
            .fold(F::ZERO, |acc, (c, s)| acc + *c * *s);
        BitClaim::from_skip_point(tables, z, &r, v_c)
    };
    let fold = fold_tables(&tables.fold_weights(z));
    let mut rounds = Vec::with_capacity(n);
    let mut rho = Vec::with_capacity(n);
    let mut t: Vec<(F, F)> = Vec::with_capacity(blocks);
    let mut g1 = F::Unreduced::default();
    let mut ginf = F::Unreduced::default();
    if n == 0 {
        t.push((fold_block(&fold, a.block(0)), fold_block(&fold, b.block(0))));
    } else {
        let rest = eq_table(&r[1..]);
        for (x, &w) in rest.iter().enumerate() {
            let (a0, b0) = fold_two(&fold, a.block(2 * x), b.block(2 * x));
            let (a1, b1) = fold_two(&fold, a.block(2 * x + 1), b.block(2 * x + 1));
            t.extend_from_slice(&[(a0, b0), (a1, b1)]);
            F::accumulate(&mut g1, w, a1 * b1);
            F::accumulate(&mut ginf, w, (a0 + a1) * (b0 + b1));
        }
    }
    for i in 0..n {
        let (m1, minf) = (F::reduce(g1), F::reduce(ginf));
        record(ch, &[m1, minf]);
        rounds.push((m1, minf));
        let rho_i: F = coin(ch);
        rho.push(rho_i);
        let half = t.len() / 2;
        g1 = F::Unreduced::default();
        ginf = F::Unreduced::default();
        if i + 1 < n {
            let rest = eq_table(&r[i + 2..]);
            for (y, &w) in rest.iter().enumerate() {
                let j = 4 * y;
                let ((a0, b0), (a1, b1), (a2, b2), (a3, b3)) = (t[j], t[j + 1], t[j + 2], t[j + 3]);
                let na0 = a0 + rho_i * (a0 + a1);
                let na1 = a2 + rho_i * (a2 + a3);
                let nb0 = b0 + rho_i * (b0 + b1);
                let nb1 = b2 + rho_i * (b2 + b3);
                t[2 * y] = (na0, nb0);
                t[2 * y + 1] = (na1, nb1);
                F::accumulate(&mut g1, w, na1 * nb1);
                F::accumulate(&mut ginf, w, (na0 + na1) * (nb0 + nb1));
            }
        } else {
            let ((a0, b0), (a1, b1)) = (t[0], t[1]);
            t[0] = (a0 + rho_i * (a0 + a1), b0 + rho_i * (b0 + b1));
        }
        t.truncate(half);
    }

    let (final_a, final_b) = t[0];
    if let Some(p) = phases {
        p.add("zerocheck skip fold and rounds", started.elapsed());
    }
    record(ch, &[final_a, final_b]);
    (
        ZerocheckProof {
            round1,
            s_hat_c: s_hat_c.clone(),
            rounds,
            final_a,
            final_b,
        },
        ZerocheckClaim {
            r,
            z,
            rho,
            a_eval: final_a,
            b_eval: final_b,
            c_claim,
            c_s_hat: s_hat_c,
        },
    )
}

pub fn verify<F: PackedField>(
    tables: &SkipTables<F>,
    m: usize,
    proof: &ZerocheckProof<F>,
    ch: &mut dyn RawChallenger,
) -> Result<ZerocheckClaim<F>, ZerocheckError> {
    let n = m - 6;
    let check = |what: &'static str, expected: usize, got: usize| {
        if expected == got {
            Ok(())
        } else {
            Err(ZerocheckError::BadLength {
                what,
                expected,
                got,
            })
        }
    };
    check("round1", 64, proof.round1.len())?;
    check("s_hat_c", F::BITS, proof.s_hat_c.len())?;
    check("rounds", n, proof.rounds.len())?;

    let r = eq_point(tables, n, ch);
    record(ch, &proof.round1);
    record(ch, &proof.s_hat_c);
    let z: F = coin(ch);

    let c_claim = {
        let coeffs = BitClaim::from_skip_point(tables, z, &r, F::ZERO).coeffs;
        let v_c = coeffs
            .iter()
            .zip(&proof.s_hat_c)
            .fold(F::ZERO, |acc, (c, s)| acc + *c * *s);
        BitClaim::from_skip_point(tables, z, &r, v_c)
    };

    let cw = tables.combined_lambda_weights(z);
    let mut claim = proof
        .round1
        .iter()
        .zip(&cw)
        .fold(F::ZERO, |acc, (v, w)| acc + *v * *w)
        + c_claim.value;

    let mut rho = Vec::with_capacity(n);
    for (i, &(g1, ginf)) in proof.rounds.iter().enumerate() {
        record(ch, &[g1, ginf]);
        let ri = r[i];
        let denom = (F::ONE + ri)
            .inv()
            .ok_or(ZerocheckError::DegenerateEqChallenge { round: i })?;
        let g0 = (claim + ri * g1) * denom;
        let rho_i: F = coin(ch);
        rho.push(rho_i);
        let c1 = g0 + g1 + ginf;
        claim = g0 + c1 * rho_i + ginf * rho_i * rho_i;
    }
    record(ch, &[proof.final_a, proof.final_b]);
    if proof.final_a * proof.final_b != claim {
        return Err(ZerocheckError::FinalProductMismatch);
    }
    Ok(ZerocheckClaim {
        r,
        z,
        rho,
        a_eval: proof.final_a,
        b_eval: proof.final_b,
        c_claim,
        c_s_hat: proof.s_hat_c.clone(),
    })
}
