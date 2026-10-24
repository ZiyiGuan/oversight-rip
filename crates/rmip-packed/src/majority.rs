use rmip_field::F128;
use rmip_iop::backend::{Claim, CommitmentBackend, Coord, Symbol};
use rmip_iop::transcript::{OracleId, RawChallenger};

use crate::batch::{self, BatchProof, ConjugateFamily};
use crate::lincheck::{self, LincheckProof};
use crate::phases::Phases;
use crate::ringswitch::{self, BitClaim, RingSwitchProof, WordClaim};
use crate::subfield::SkipTables;
use crate::tree::{bit_claim_at, LevelWitness, Tree};
use crate::zerocheck::{self, coin, eq_table, record, ZerocheckProof};
use crate::PackedField;

pub struct LevelProof<F> {
    pub zerocheck: ZerocheckProof<F>,
    pub lincheck: LincheckProof<F>,
    pub switches: Vec<RingSwitchProof<F>>,
    pub batch: BatchProof<F>,
}

pub struct PiopProof<F> {
    pub sampled_words: Vec<F>,
    pub levels: Vec<LevelProof<F>>,
    pub cross: BatchProof<F>,
    pub batched: Claim<F128>,
}

pub struct MajorityProof<F> {
    pub piop: PiopProof<F>,
    pub answers: Vec<(OracleId, Coord, Symbol<F128>)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MajorityError {
    BelowThreshold {
        count: u64,
        threshold: u64,
    },
    CountNotBoolean,
    OneWire {
        level: usize,
    },
    BadLength(&'static str),
    Zerocheck {
        level: usize,
        err: zerocheck::ZerocheckError,
    },
    Lincheck {
        level: usize,
        err: lincheck::LincheckError,
    },
    RingSwitch {
        level: usize,
        err: ringswitch::RingSwitchError,
    },
    Batch {
        level: usize,
        err: batch::BatchError,
    },
}

fn extend_to_vector<F: PackedField>(
    w: &WordClaim<F>,
    offset: usize,
    total_log: usize,
) -> WordClaim<F> {
    let mut point = w.point.clone();
    let n = point.len();
    assert_eq!(offset % (1 << n), 0);
    point.extend(
        (n..total_log).map(|b| <F as rmip_field::BinaryField>::from_bool((offset >> b) & 1 == 1)),
    );
    WordClaim {
        point,
        value: w.value,
    }
}

fn to_backend_claim<F: PackedField>(w: &WordClaim<F>) -> Claim<F128> {
    Claim {
        point: w.point.iter().rev().map(|x| x.to_word()).collect(),
        value: w.value.to_word(),
    }
}

fn singleton_batch<F: PackedField>(claim: &WordClaim<F>) -> BatchProof<F> {
    BatchProof {
        rounds: Vec::new(),
        final_value: claim.value,
    }
}

pub fn combined_words<F: PackedField>(tree: &Tree, witness: &[LevelWitness<F>]) -> Vec<F> {
    let kappa_log = F::BITS.trailing_zeros() as usize;
    let (offsets, total_log) = tree.layout(kappa_log);
    let mut words = vec![F::ZERO; 1 << total_log];
    for (w, &off) in witness.iter().zip(&offsets) {
        words[off..off + w.z.words.len()].copy_from_slice(&w.z.words);
    }
    words
}

fn boolean_point<F: PackedField>(index: usize, n: usize) -> Vec<F> {
    (0..n)
        .map(|b| F::from_bool((index >> b) & 1 == 1))
        .collect()
}

fn sampled_claims<F: PackedField>(
    tree: &Tree,
    sampled: &[usize],
    sampled_words: &[F],
) -> Vec<WordClaim<F>> {
    let kappa = F::BITS;
    let n = tree.levels[0].words_log(kappa.trailing_zeros() as usize);
    sampled
        .iter()
        .zip(sampled_words)
        .map(|(&view, &w)| WordClaim {
            point: boolean_point(tree.answer_bit(view) / kappa, n),
            value: w,
        })
        .collect()
}

pub fn output_fold<F: PackedField>(tree: &Tree, level: usize, z_vec: &[F], rho_t: &[F]) -> F {
    let b = &tree.levels[level].block;
    assert_eq!(1 << rho_t.len(), b.out_stride);
    eq_table(rho_t)
        .iter()
        .zip(&z_vec[b.out_base..b.out_base + b.out_stride])
        .fold(F::ZERO, |acc, (e, z)| acc + *e * *z)
}

pub fn count_from_top<F: PackedField>(tree: &Tree, z_vec: &[F]) -> Option<u64> {
    let b = &tree.levels.last().unwrap().block;
    let mut count = 0u64;
    for t in 0..b.out_bits {
        let v = z_vec[b.out_base + t];
        if v == F::ONE {
            count |= 1 << t;
        } else if v != F::ZERO {
            return None;
        }
    }
    Some(count)
}

struct Link<F> {
    coins: Vec<F>,
    value: F,
}

fn link_claims<F: PackedField>(
    tree: &Tree,
    level: usize,
    link: Option<&Link<F>>,
) -> Vec<BitClaim<F>> {
    let mut claims = Vec::new();
    if let Some(link) = link {
        let (input, _) = tree.copy_points::<F>(level, &link.coins);
        claims.push(bit_claim_at(&input, link.value));
    }
    claims
}

fn draw_link<F: PackedField>(
    tree: &Tree,
    level: usize,
    z_vec: &[F],
    rho_out: &[F],
    ch: &mut dyn RawChallenger,
) -> Link<F> {
    let t_log = tree.levels[level + 1].block.in_stride.trailing_zeros() as usize;
    let rho_t: Vec<F> = (0..t_log).map(|_| coin(ch)).collect();
    let value = output_fold(tree, level, z_vec, &rho_t);
    let mut coins = rho_t;
    coins.extend_from_slice(rho_out);
    Link { coins, value }
}

pub fn sampled_bit<F: PackedField>(tree: &Tree, view: usize, word: F) -> bool {
    (word.to_bits() >> (tree.answer_bit(view) % F::BITS)) & 1 == 1
}

pub fn prove_piop<F: PackedField>(
    tables: &SkipTables<F>,
    tree: &Tree,
    witness: &[LevelWitness<F>],
    words: &[F],
    sampled: &[usize],
    ch: &mut dyn RawChallenger,
) -> PiopProof<F> {
    prove_piop_with_phases(tables, tree, witness, words, sampled, ch, None)
}

pub fn prove_piop_with_phases<F: PackedField>(
    tables: &SkipTables<F>,
    tree: &Tree,
    witness: &[LevelWitness<F>],
    words: &[F],
    sampled: &[usize],
    ch: &mut dyn RawChallenger,
    mut phases: Option<&mut Phases>,
) -> PiopProof<F> {
    use std::time::Instant;
    macro_rules! timed {
        ($name:literal, $e:expr) => {{
            let started = Instant::now();
            let out = $e;
            if let Some(p) = phases.as_deref_mut() {
                p.add($name, started.elapsed());
            }
            out
        }};
    }
    let kappa = F::BITS;
    let kappa_log = kappa.trailing_zeros() as usize;
    let (offsets, total_log) = tree.layout(kappa_log);
    let depth = tree.levels.len();
    let sampled_words: Vec<F> = sampled
        .iter()
        .map(|&x| witness[0].z.words[tree.answer_bit(x) / kappa])
        .collect();
    record(ch, &sampled_words);
    let mut levels = Vec::with_capacity(depth);
    let mut per_level = Vec::with_capacity(depth);
    let mut link: Option<Link<F>> = None;
    for (level, w) in witness.iter().enumerate() {
        if let Some(p) = phases.as_deref_mut() {
            p.level = level;
        }
        let r1cs = tree.levels[level].r1cs();
        let (zerocheck, zc) = timed!(
            "zerocheck",
            zerocheck::prove_with_phases(tables, &w.a, &w.b, &w.z, ch, phases.as_deref_mut())
        );
        let (lincheck, lc) = timed!(
            "lincheck",
            lincheck::prove_with_phases(tables, r1cs, &w.z, &zc.rho, ch, phases.as_deref_mut())
        );
        let next_link =
            (level + 1 < depth).then(|| draw_link(tree, level, &lincheck.z_vec, &lc.outer, ch));
        let mut word_claims = vec![
            ringswitch::word_claims(
                &lc.bit_claim(tables),
                &lc.partial_evaluations(&lincheck.z_vec),
            ),
            ringswitch::word_claims(&zc.c_claim, &zc.c_s_hat),
        ];
        let claims = link_claims(tree, level, link.as_ref());
        let (switches, link_words) = timed!(
            "ring switch",
            ringswitch::prove_many(&w.z.words, &claims, ch)
        );
        word_claims.extend(link_words);
        let families: Vec<ConjugateFamily<F>> = word_claims
            .iter()
            .map(|c| ConjugateFamily::from_word_claims(c))
            .collect();
        let plain = if level == 0 {
            sampled_claims(tree, sampled, &sampled_words)
        } else {
            Vec::new()
        };
        let (batch, word_claim) = timed!(
            "batch",
            batch::prove_with_phases(&w.z.words, &families, &plain, ch, phases.as_deref_mut())
        );
        per_level.push(extend_to_vector(&word_claim, offsets[level], total_log));
        levels.push(LevelProof {
            zerocheck,
            lincheck,
            switches,
            batch,
        });
        link = next_link;
    }
    if let Some(p) = phases.as_deref_mut() {
        p.level = 0;
    }
    let (cross, final_claim) = if per_level.len() == 1 {
        let claim = per_level.pop().expect("one claim");
        (singleton_batch(&claim), claim)
    } else {
        timed!(
            "cross-level batch",
            batch::prove(words, &[], &per_level, ch)
        )
    };
    let batched = to_backend_claim(&final_claim);
    PiopProof {
        sampled_words,
        levels,
        cross,
        batched,
    }
}

pub fn prove<F, B>(
    backend: &mut B,
    tables: &SkipTables<F>,
    tree: &Tree,
    witness: &[LevelWitness<F>],
    sampled: &[usize],
    ch: &mut dyn RawChallenger,
) -> MajorityProof<F>
where
    F: PackedField,
    B: CommitmentBackend<Field = F128>,
{
    let words = combined_words(tree, witness);
    backend.commit(&F::to_words(&words), ch);
    let piop = prove_piop(tables, tree, witness, &words, sampled, ch);
    backend.open(std::slice::from_ref(&piop.batched), ch);
    let mut plan = backend.query_plan(ch);
    let mut answers = Vec::new();
    let mut prev: Option<Symbol<F128>> = None;
    while let Some((id, coord)) = plan.next(prev.as_ref()) {
        let prefix = ch.transcript().prefix_at(id);
        let symbol = backend.reconstruct(id, &prefix, coord);
        answers.push((id, coord, symbol.clone()));
        prev = Some(symbol);
    }
    MajorityProof { piop, answers }
}

pub fn verify_piop<F: PackedField>(
    tables: &SkipTables<F>,
    tree: &Tree,
    threshold: u64,
    sampled: &[usize],
    proof: &PiopProof<F>,
    ch: &mut dyn RawChallenger,
) -> Result<Claim<F128>, MajorityError> {
    verify_piop_with_phases(tables, tree, threshold, sampled, proof, ch, None)
}

pub fn verify_piop_with_phases<F: PackedField>(
    tables: &SkipTables<F>,
    tree: &Tree,
    threshold: u64,
    sampled: &[usize],
    proof: &PiopProof<F>,
    ch: &mut dyn RawChallenger,
    mut phases: Option<&mut Phases>,
) -> Result<Claim<F128>, MajorityError> {
    use std::time::Instant;
    macro_rules! timed {
        ($name:literal, $e:expr) => {{
            let started = Instant::now();
            let out = $e;
            if let Some(p) = phases.as_deref_mut() {
                p.add($name, started.elapsed());
            }
            out
        }};
    }
    let kappa = F::BITS;
    let kappa_log = kappa.trailing_zeros() as usize;
    let (offsets, total_log) = tree.layout(kappa_log);
    let depth = tree.levels.len();
    if proof.sampled_words.len() != sampled.len() {
        return Err(MajorityError::BadLength("sampled words"));
    }
    if proof.levels.len() != depth {
        return Err(MajorityError::BadLength("levels"));
    }
    record(ch, &proof.sampled_words);
    let mut per_level = Vec::with_capacity(depth);
    let mut link: Option<Link<F>> = None;
    for (level, lp) in proof.levels.iter().enumerate() {
        if let Some(p) = phases.as_deref_mut() {
            p.level = level;
        }
        let r1cs = timed!("verifier r1cs", tree.levels[level].r1cs());
        let zc = timed!(
            "verifier zerocheck",
            zerocheck::verify(tables, r1cs.m, &lp.zerocheck, ch)
                .map_err(|err| MajorityError::Zerocheck { level, err })
        )?;
        let lc = timed!(
            "verifier lincheck",
            lincheck::verify(tables, r1cs, &zc, &lp.lincheck, ch)
                .map_err(|err| MajorityError::Lincheck { level, err })
        )?;
        let z_vec = &lp.lincheck.z_vec;
        if z_vec[0] != F::ONE {
            return Err(MajorityError::OneWire { level });
        }
        if level + 1 == depth {
            let count = count_from_top(tree, z_vec).ok_or(MajorityError::CountNotBoolean)?;
            if count < threshold {
                return Err(MajorityError::BelowThreshold { count, threshold });
            }
        }
        let next_link = (level + 1 < depth).then(|| draw_link(tree, level, z_vec, &lc.outer, ch));
        let mut word_claims = timed!(
            "verifier ring switch",
            vec![
                ringswitch::word_claims(&lc.bit_claim(tables), &lc.partial_evaluations(z_vec)),
                ringswitch::word_claims(&zc.c_claim, &zc.c_s_hat),
            ]
        );
        let claims = link_claims(tree, level, link.as_ref());
        let link_words = timed!(
            "verifier ring switch",
            ringswitch::verify_many(&claims, &lp.switches, ch)
                .map_err(|err| MajorityError::RingSwitch { level, err })
        )?;
        word_claims.extend(link_words);
        let families: Vec<ConjugateFamily<F>> = word_claims
            .iter()
            .map(|c| ConjugateFamily::from_word_claims(c))
            .collect();
        let plain = if level == 0 {
            sampled_claims(tree, sampled, &proof.sampled_words)
        } else {
            Vec::new()
        };
        let n = tree.levels[level].words_log(kappa_log);
        let word_claim = timed!(
            "verifier batch",
            batch::verify(n, &families, &plain, &lp.batch, ch)
                .map_err(|err| MajorityError::Batch { level, err })
        )?;
        per_level.push(extend_to_vector(&word_claim, offsets[level], total_log));
        link = next_link;
    }
    if let Some(p) = phases.as_deref_mut() {
        p.level = 0;
    }
    let final_claim = if per_level.len() == 1 {
        let claim = per_level.pop().expect("one claim");
        if proof.cross != singleton_batch(&claim) {
            return Err(MajorityError::Batch {
                level: depth,
                err: batch::BatchError::FinalMismatch,
            });
        }
        claim
    } else {
        timed!(
            "verifier cross-level batch",
            batch::verify(total_log, &[], &per_level, &proof.cross, ch)
                .map_err(|err| MajorityError::Batch { level: depth, err })
        )?
    };
    Ok(to_backend_claim(&final_claim))
}
