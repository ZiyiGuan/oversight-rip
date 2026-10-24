use rmip_iop::transcript::RawChallenger;

use crate::ringswitch::{conjugate_points, WordClaim};
use crate::zerocheck::{coin, eq_table, record};
use crate::PackedField;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConjugateFamily<F> {
    pub base: Vec<F>,
    pub values: Vec<F>,
}

impl<F: PackedField> ConjugateFamily<F> {
    pub fn from_word_claims(claims: &[WordClaim<F>]) -> Self {
        assert_eq!(claims.len(), F::BITS);
        Self {
            base: claims[0].point.clone(),
            values: claims.iter().map(|c| c.value).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchProof<F> {
    pub rounds: Vec<(F, F, F)>,
    pub final_value: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchError {
    BadLength { expected: usize, got: usize },
    RoundSumMismatch { round: usize },
    FinalMismatch,
}

fn family_tables<F: PackedField>(gammas: &[F]) -> Vec<[F; 256]> {
    let on_basis = F::frobenius_basis().inverse_twists(gammas);
    on_basis
        .chunks(8)
        .map(|chunk| {
            let mut t = [F::ZERO; 256];
            for (b, slot) in t.iter_mut().enumerate() {
                let mut acc = F::ZERO;
                for (i, v) in chunk.iter().enumerate() {
                    if (b >> i) & 1 == 1 {
                        acc += *v;
                    }
                }
                *slot = acc;
            }
            t
        })
        .collect()
}

#[inline]
fn linearized_lookup<F: PackedField>(tables: &[[F; 256]], y: F) -> F {
    let bits = y.to_bits();
    let mut acc = F::ZERO;
    for (k, t) in tables.iter().enumerate() {
        acc += t[((bits >> (8 * k)) & 0xFF) as usize];
    }
    acc
}

fn build_e<F: PackedField>(
    n_words: usize,
    families: &[ConjugateFamily<F>],
    fam_gammas: &[Vec<F>],
    plain: &[WordClaim<F>],
    plain_gammas: &[F],
) -> Vec<F> {
    let mut e = vec![F::ZERO; n_words];
    for (fam, gammas) in families.iter().zip(fam_gammas) {
        let tables = family_tables(gammas);
        let n = fam.base.len();
        assert!(1usize << n >= n_words, "a family's point spans the words");
        let n_low = n.min(12);
        let eq_low = eq_table(&fam.base[..n_low]);
        let eq_high = eq_table(&fam.base[n_low..]);
        for (h, &wh) in eq_high.iter().enumerate() {
            let start = h << n_low;
            if start >= n_words {
                break;
            }
            if wh == F::ZERO {
                continue;
            }
            let end = (start + (1 << n_low)).min(n_words);
            for (ex, &wl) in e[start..end].iter_mut().zip(&eq_low) {
                let y = wh * wl;
                if y != F::ZERO {
                    *ex += linearized_lookup(&tables, y);
                }
            }
        }
    }
    for (p, g) in plain.iter().zip(plain_gammas) {
        let (base, n_low) = plain_support(p);
        let eq = eq_table(&p.point[..n_low]);
        let end = (base + eq.len()).min(n_words);
        if base < end {
            for (ex, w) in e[base..end].iter_mut().zip(&eq) {
                *ex += *g * *w;
            }
        }
    }
    e
}

fn plain_support<F: PackedField>(p: &WordClaim<F>) -> (usize, usize) {
    let n_low = p
        .point
        .iter()
        .rposition(|c| *c != F::ZERO && *c != F::ONE)
        .map_or(0, |i| i + 1);
    let base: usize = p.point[n_low..]
        .iter()
        .enumerate()
        .filter(|(_, c)| **c == F::ONE)
        .map(|(b, _)| 1usize << (n_low + b))
        .sum();
    (base, n_low)
}

fn support_end<F: PackedField>(
    n_words: usize,
    families: &[ConjugateFamily<F>],
    plain: &[WordClaim<F>],
) -> usize {
    if !families.is_empty() {
        return n_words;
    }
    plain
        .iter()
        .map(|p| {
            let (base, n_low) = plain_support(p);
            base + (1 << n_low)
        })
        .max()
        .unwrap_or(0)
        .min(n_words)
}

fn draw_gammas<F: PackedField>(
    families: &[ConjugateFamily<F>],
    plain: &[WordClaim<F>],
    ch: &mut dyn RawChallenger,
) -> (Vec<Vec<F>>, Vec<F>) {
    let fam: Vec<Vec<F>> = families
        .iter()
        .map(|f| {
            ch.sample_many(f.values.len())
                .into_iter()
                .map(|c| F::from_bits(c.to_u128()))
                .collect()
        })
        .collect();
    let pl: Vec<F> = ch
        .sample_many(plain.len())
        .into_iter()
        .map(|c| F::from_bits(c.to_u128()))
        .collect();
    (fam, pl)
}

fn target<F: PackedField>(
    families: &[ConjugateFamily<F>],
    fam_gammas: &[Vec<F>],
    plain: &[WordClaim<F>],
    plain_gammas: &[F],
) -> F {
    let mut t = F::ZERO;
    for (f, g) in families.iter().zip(fam_gammas) {
        for (v, gj) in f.values.iter().zip(g) {
            t += *v * *gj;
        }
    }
    for (p, g) in plain.iter().zip(plain_gammas) {
        t += p.value * *g;
    }
    t
}

pub fn prove<F: PackedField>(
    words: &[F],
    families: &[ConjugateFamily<F>],
    plain: &[WordClaim<F>],
    ch: &mut dyn RawChallenger,
) -> (BatchProof<F>, WordClaim<F>) {
    prove_with_phases(words, families, plain, ch, None)
}

pub fn prove_with_phases<F: PackedField>(
    words: &[F],
    families: &[ConjugateFamily<F>],
    plain: &[WordClaim<F>],
    ch: &mut dyn RawChallenger,
    mut phases: Option<&mut crate::phases::Phases>,
) -> (BatchProof<F>, WordClaim<F>) {
    let started = std::time::Instant::now();
    let n = words.len().trailing_zeros() as usize;
    assert_eq!(words.len(), 1 << n);
    let (fam_gammas, plain_gammas) = draw_gammas(families, plain, ch);
    let live_words = words.len() - words.iter().rev().take_while(|w| **w == F::ZERO).count();
    let live = live_words
        .max(support_end(words.len(), families, plain))
        .max(1)
        .next_multiple_of(2)
        .min(words.len());
    let e = build_e(live, families, &fam_gammas, plain, &plain_gammas);
    let mut t: Vec<(F, F)> = e.into_iter().zip(words[..live].iter().copied()).collect();
    if let Some(p) = phases.as_mut() {
        p.add("batch E table", started.elapsed());
    }
    let rounds_started = std::time::Instant::now();
    let mut rounds = Vec::with_capacity(n);
    let mut rho = Vec::with_capacity(n);
    for round in 0..n {
        let half = t.len() / 2;
        let (mut p0, mut p1, mut pinf) = (
            F::Unreduced::default(),
            F::Unreduced::default(),
            F::Unreduced::default(),
        );
        for pair in t.as_chunks::<2>().0 {
            let [(e0, w0), (e1, w1)] = *pair;
            F::accumulate(&mut p0, e0, w0);
            F::accumulate(&mut p1, e1, w1);
            F::accumulate(&mut pinf, e0 + e1, w0 + w1);
        }
        let (p0, p1, pinf) = (F::reduce(p0), F::reduce(p1), F::reduce(pinf));
        record(ch, &[p0, p1, pinf]);
        rounds.push((p0, p1, pinf));
        let r: F = coin(ch);
        rho.push(r);
        for x in 0..half {
            let ((e0, w0), (e1, w1)) = (t[2 * x], t[2 * x + 1]);
            t[x] = (e0 + r * (e0 + e1), w0 + r * (w0 + w1));
        }
        t.truncate(half);
        if t.len() % 2 == 1 && round + 1 < n {
            t.push((F::ZERO, F::ZERO));
        }
    }
    let final_value = t[0].1;
    record(ch, &[final_value]);
    if let Some(p) = phases.as_mut() {
        p.add("batch rounds", rounds_started.elapsed());
    }
    (
        BatchProof {
            rounds,
            final_value,
        },
        WordClaim {
            point: rho,
            value: final_value,
        },
    )
}

fn e_at<F: PackedField>(
    rho: &[F],
    families: &[ConjugateFamily<F>],
    fam_gammas: &[Vec<F>],
    plain: &[WordClaim<F>],
    plain_gammas: &[F],
) -> F {
    let eq_point = |p: &[F]| -> F {
        p.iter().zip(rho).fold(F::ONE, |acc, (a, b)| {
            acc * (*a * *b + (F::ONE + *a) * (F::ONE + *b))
        })
    };
    let mut e = F::ZERO;
    for (f, g) in families.iter().zip(fam_gammas) {
        for (point, gj) in conjugate_points(&f.base).iter().zip(g) {
            e += *gj * eq_point(point);
        }
    }
    for (p, g) in plain.iter().zip(plain_gammas) {
        e += *g * eq_point(&p.point);
    }
    e
}

pub fn verify<F: PackedField>(
    n: usize,
    families: &[ConjugateFamily<F>],
    plain: &[WordClaim<F>],
    proof: &BatchProof<F>,
    ch: &mut dyn RawChallenger,
) -> Result<WordClaim<F>, BatchError> {
    if proof.rounds.len() != n {
        return Err(BatchError::BadLength {
            expected: n,
            got: proof.rounds.len(),
        });
    }
    let (fam_gammas, plain_gammas) = draw_gammas(families, plain, ch);
    let mut claim = target(families, &fam_gammas, plain, &plain_gammas);
    let mut rho = Vec::with_capacity(n);
    for (round, &(p0, p1, pinf)) in proof.rounds.iter().enumerate() {
        record(ch, &[p0, p1, pinf]);
        if p0 + p1 != claim {
            return Err(BatchError::RoundSumMismatch { round });
        }
        let r: F = coin(ch);
        rho.push(r);
        let c1 = p0 + p1 + pinf;
        claim = p0 + c1 * r + pinf * r * r;
    }
    record(ch, &[proof.final_value]);
    if e_at(&rho, families, &fam_gammas, plain, &plain_gammas) * proof.final_value != claim {
        return Err(BatchError::FinalMismatch);
    }
    Ok(WordClaim {
        point: rho,
        value: proof.final_value,
    })
}
