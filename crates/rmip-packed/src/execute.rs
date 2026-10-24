use rmip_field::F128;
use rmip_iop::backend::{Claim, CommitmentBackend, Coord, OpeningVerifier, QueryPlan, Symbol};
use rmip_iop::transcript::{OracleId, RawChallenger};

use crate::batch::{self, BatchProof, ConjugateFamily};
use crate::lincheck::{self, BlockR1cs, LincheckProof};
use crate::ringswitch::{self, RingSwitchProof, WordClaim};
use crate::subfield::SkipTables;
use crate::zerocheck::{self, PackedBits, ZerocheckProof};

pub struct PackedProof {
    pub zerocheck: ZerocheckProof<F128>,
    pub lincheck: LincheckProof<F128>,
    pub switches: Vec<RingSwitchProof<F128>>,
    pub batch: BatchProof<F128>,
    pub batched: Claim<F128>,
    pub answers: Vec<(OracleId, Coord, Symbol<F128>)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedError {
    Zerocheck(zerocheck::ZerocheckError),
    Lincheck(lincheck::LincheckError),
    RingSwitch(ringswitch::RingSwitchError),
    Batch(batch::BatchError),
    PlanMismatch,
    Opening,
}

fn to_backend_claim(w: &WordClaim<F128>) -> Claim<F128> {
    Claim {
        point: w.point.iter().rev().copied().collect(),
        value: w.value,
    }
}

pub fn prove<B: CommitmentBackend<Field = F128>>(
    backend: &mut B,
    tables: &SkipTables<F128>,
    r1cs: &BlockR1cs,
    z: &PackedBits<F128>,
    ch: &mut dyn RawChallenger,
) -> PackedProof {
    backend.commit(&z.words, ch);
    let a = r1cs.apply(&r1cs.a0, z);
    let b = r1cs.apply(&r1cs.b0, z);
    let (zerocheck, zc) = zerocheck::prove(tables, &a, &b, z, ch);
    let (lincheck, lc) = lincheck::prove(tables, r1cs, z, &zc.rho, ch);
    let claims = vec![lc.bit_claim(tables)];
    let (switches, mut word_claims) = ringswitch::prove_many(&z.words, &claims, ch);
    word_claims.push(ringswitch::word_claims(&zc.c_claim, &zc.c_s_hat));
    let families: Vec<ConjugateFamily<F128>> = word_claims
        .iter()
        .map(|c| ConjugateFamily::from_word_claims(c))
        .collect();
    let (batch, batched_word) = batch::prove(&z.words, &families, &[], ch);
    let batched = to_backend_claim(&batched_word);
    backend.open(std::slice::from_ref(&batched), ch);
    let mut plan = backend.query_plan(ch);
    let mut answers = Vec::new();
    let mut prev: Option<Symbol<F128>> = None;
    while let Some((id, coord)) = plan.next(prev.as_ref()) {
        let prefix = ch.transcript().prefix_at(id);
        let symbol = backend.reconstruct(id, &prefix, coord);
        answers.push((id, coord, symbol.clone()));
        prev = Some(symbol);
    }
    PackedProof {
        zerocheck,
        lincheck,
        switches,
        batch,
        batched,
        answers,
    }
}

pub fn verify_piop(
    tables: &SkipTables<F128>,
    r1cs: &BlockR1cs,
    proof: &PackedProof,
    ch: &mut dyn RawChallenger,
) -> Result<Claim<F128>, PackedError> {
    let zc =
        zerocheck::verify(tables, r1cs.m, &proof.zerocheck, ch).map_err(PackedError::Zerocheck)?;
    let lc =
        lincheck::verify(tables, r1cs, &zc, &proof.lincheck, ch).map_err(PackedError::Lincheck)?;
    let claims = vec![lc.bit_claim(tables)];
    let mut word_claims =
        ringswitch::verify_many(&claims, &proof.switches, ch).map_err(PackedError::RingSwitch)?;
    word_claims.push(ringswitch::word_claims(&zc.c_claim, &zc.c_s_hat));
    let families: Vec<ConjugateFamily<F128>> = word_claims
        .iter()
        .map(|c| ConjugateFamily::from_word_claims(c))
        .collect();
    let n = r1cs.m - 7;
    let batched = batch::verify(n, &families, &[], &proof.batch, ch).map_err(PackedError::Batch)?;
    Ok(to_backend_claim(&batched))
}

pub fn check_opening<V: OpeningVerifier<F128>>(
    verifier: &V,
    plan: &mut dyn QueryPlan<F128>,
    batched: &Claim<F128>,
    answers: &[(OracleId, Coord, Symbol<F128>)],
) -> Result<(), PackedError> {
    let mut prev: Option<Symbol<F128>> = None;
    for (id, coord, symbol) in answers {
        match plan.next(prev.as_ref()) {
            Some((pid, pcoord)) if pid == *id && pcoord == *coord => {}
            _ => return Err(PackedError::PlanMismatch),
        }
        prev = Some(symbol.clone());
    }
    if plan.next(prev.as_ref()).is_some() {
        return Err(PackedError::PlanMismatch);
    }
    if !verifier.check(std::slice::from_ref(batched), answers) {
        return Err(PackedError::Opening);
    }
    Ok(())
}
