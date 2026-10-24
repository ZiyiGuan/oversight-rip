use crate::backend::{Claim, Coord, OpeningVerifier, Symbol};
use crate::batch::{self, BatchMessageProver, BatchProof, BatchProofError};
use crate::driver::{Config, Statement};
use crate::lincheck::Coefficients;
use crate::local_protocol::{self, LocalError, LocalMessageProver, LocalProof};
use crate::relation_protocol::{self, RelationError, RelationMessageProver, RelationProof};
use crate::sumcheck::RoundMessageProver;
use crate::transcript::{OracleId, RawChallenger};
use rmip_field::F128;
use rmip_frontend::oracle::Oracle;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommonProof {
    pub relation: RelationProof,
    pub local: LocalProof,
    pub batching: BatchProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommonError {
    BatchingDisabled,
    InvalidRelationClaimCount { expected: usize, got: usize },
    Relation(RelationError),
    Local(LocalError),
    Batching(BatchProofError),
}

pub struct CommonProverOutput {
    pub proof: CommonProof,
    pub reduced_claim: Claim<F128>,
}

pub struct CommonProverRecordingOutput {
    pub output: CommonProverOutput,
    pub verdict: Result<(), CommonError>,
}

pub struct CommonVerification {
    pub reduced_claim: Claim<F128>,
    pub claims_submitted: usize,
    pub pad_claims: usize,
    pub oracle_calls: usize,
    pub t_mat: Duration,
    pub verdict: Result<(), CommonError>,
}

pub struct CommonInteraction {
    pub proof: CommonProof,
    pub verification: CommonVerification,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinalError {
    Common(CommonError),
    OpeningMismatch,
}

pub struct FinalVerification {
    pub opening_ok: bool,
    pub verdict: Result<(), FinalError>,
}

pub struct FinalInteraction {
    pub proof: CommonProof,
    pub verification: FinalVerification,
}

pub trait CommonMessageProver:
    RelationMessageProver + LocalMessageProver + BatchMessageProver
{
}

impl<T> CommonMessageProver for T where
    T: RelationMessageProver + LocalMessageProver + BatchMessageProver
{
}

pub struct HonestCommonProver<'a> {
    relation: relation_protocol::HonestRelationProver<'a>,
    local: local_protocol::HonestLocalProver<'a>,
    batching: batch::HonestBatchProver,
}

impl<'a> HonestCommonProver<'a> {
    pub fn new(st: &'a Statement<'_>, table: &'a [F128]) -> Self {
        Self {
            relation: relation_protocol::HonestRelationProver::new(st, table),
            local: local_protocol::HonestLocalProver::new(st.inst, table),
            batching: batch::HonestBatchProver::for_table(table),
        }
    }
}

impl RelationMessageProver for HonestCommonProver<'_> {
    fn holo_frame(&mut self) -> Vec<F128> {
        unreachable!(
            "live holographic proving is not implemented; the honest \
             interactive provers run scan-shaped statements and the SNARG \
             replays a tape"
        )
    }

    fn observe_public_input_point(&mut self, r_pub: &[F128]) {
        self.relation.observe_public_input_point(r_pub);
    }

    fn begin_zerocheck(&mut self, tau: &[F128]) {
        self.relation.begin_zerocheck(tau);
    }

    fn zerocheck_round_message(&mut self, round: usize) -> Vec<F128> {
        self.relation.zerocheck_round_message(round)
    }

    fn receive_zerocheck_challenge(&mut self, round: usize, challenge: F128) {
        self.relation.receive_zerocheck_challenge(round, challenge);
    }

    fn zerocheck_endpoint(&mut self) -> Vec<F128> {
        self.relation.zerocheck_endpoint()
    }

    fn begin_lincheck(&mut self, row_point: &[F128], coefficients: Coefficients) {
        self.relation.begin_lincheck(row_point, coefficients);
    }

    fn lincheck_round_message(&mut self, round: usize) -> Vec<F128> {
        self.relation.lincheck_round_message(round)
    }

    fn receive_lincheck_challenge(&mut self, round: usize, challenge: F128) {
        self.relation.receive_lincheck_challenge(round, challenge);
    }

    fn lincheck_endpoint(&mut self) -> Vec<F128> {
        self.relation.lincheck_endpoint()
    }
}

impl LocalMessageProver for HonestCommonProver<'_> {
    fn answer_blocks(&mut self, sampling_coins: &[F128], sampled: &[usize]) -> Vec<Vec<F128>> {
        self.local.answer_blocks(sampling_coins, sampled)
    }
}

impl BatchMessageProver for HonestCommonProver<'_> {
    fn begin_batch(&mut self, claims: &[Claim<F128>], gammas: &[F128]) {
        self.batching.begin_batch(claims, gammas);
    }
}

impl RoundMessageProver for HonestCommonProver<'_> {
    fn round_message(&mut self, round: usize) -> Vec<F128> {
        self.batching.round_message(round)
    }

    fn receive_challenge(&mut self, round: usize, challenge: F128) {
        self.batching.receive_challenge(round, challenge);
    }
}

fn set_first(slot: &mut Option<CommonError>, error: CommonError) {
    if slot.is_none() {
        *slot = Some(error);
    }
}

fn padded_claims(
    st: &Statement<'_>,
    cfg: &Config,
    relation: &[Claim<F128>],
    public: &Claim<F128>,
    local: &[Claim<F128>],
) -> Result<(Vec<Claim<F128>>, usize), CommonError> {
    if relation.len() != cfg.k_rel {
        return Err(CommonError::InvalidRelationClaimCount {
            expected: cfg.k_rel,
            got: relation.len(),
        });
    }
    let total = cfg.k_rel + cfg.t * (st.inst.b_max() + 1);
    let mut claims = relation.to_vec();
    claims.extend_from_slice(local);
    assert!(
        claims.len() <= total,
        "public claim bound is smaller than the claims emitted by loc"
    );
    let before = claims.len();
    while claims.len() < total {
        claims.push(public.clone());
    }
    Ok((claims, total - before))
}

pub fn prove(
    st: &Statement<'_>,
    table: &[F128],
    cfg: &Config,
    ch: &mut dyn RawChallenger,
) -> Result<CommonProverOutput, CommonError> {
    let recorded = prove_recording(st, table, cfg, ch)?;
    recorded.verdict?;
    Ok(recorded.output)
}

pub fn prove_recording(
    st: &Statement<'_>,
    table: &[F128],
    cfg: &Config,
    ch: &mut dyn RawChallenger,
) -> Result<CommonProverRecordingOutput, CommonError> {
    if !cfg.batch_claims {
        return Err(CommonError::BatchingDisabled);
    }
    let relation = relation_protocol::prove_with_claims(st, table, ch);
    let local = local_protocol::prove_with_claims(st.inst, table, cfg.t, ch);
    let (claims, _) = padded_claims(
        st,
        cfg,
        &relation.claims,
        &relation.public_input_claim,
        &local.claims,
    )?;
    let batching = batch::prove_recording(&claims, table, ch);
    let verdict = batching.verdict.map_err(CommonError::Batching);
    Ok(CommonProverRecordingOutput {
        output: CommonProverOutput {
            proof: CommonProof {
                relation: relation.proof,
                local: local.proof,
                batching: batching.proof,
            },
            reduced_claim: batching.claim,
        },
        verdict,
    })
}

pub fn verify<O: Oracle>(
    st: &Statement<'_>,
    oracle: &O,
    cfg: &Config,
    proof: &CommonProof,
    ch: &mut dyn RawChallenger,
) -> CommonVerification {
    let relation = relation_protocol::verify(st, &proof.relation, ch);
    let local = local_protocol::verify(st.inst, oracle, cfg.t, &proof.local, ch);
    let mut first = None;
    if !cfg.batch_claims {
        set_first(&mut first, CommonError::BatchingDisabled);
    }
    if let Err(error) = relation.verdict {
        set_first(&mut first, CommonError::Relation(error));
    }
    if let Err(error) = local.verdict {
        set_first(&mut first, CommonError::Local(error));
    }

    let (claims, pad_claims) = match padded_claims(
        st,
        cfg,
        &relation.claims,
        &relation.public_input_claim,
        &local.claims,
    ) {
        Ok(result) => result,
        Err(error) => {
            set_first(&mut first, error);
            let fallback = Config {
                t: cfg.t,
                batch_claims: cfg.batch_claims,
                pad_queries_to: cfg.pad_queries_to,
                k_rel: relation.claims.len(),
            };
            padded_claims(
                st,
                &fallback,
                &relation.claims,
                &relation.public_input_claim,
                &local.claims,
            )
            .expect("the fallback relation count is exact")
        }
    };
    let batching = batch::verify_proof(&claims, st.layout.nu, &proof.batching, ch);
    if let Err(error) = batching.verdict {
        set_first(&mut first, CommonError::Batching(error));
    }

    CommonVerification {
        reduced_claim: batching.claim,
        claims_submitted: claims.len(),
        pad_claims,
        oracle_calls: local.oracle_calls,
        t_mat: relation.t_mat,
        verdict: first.map_or(Ok(()), Err),
    }
}

pub fn verify_interactive<O, P>(
    st: &Statement<'_>,
    oracle: &O,
    cfg: &Config,
    prover: &mut P,
    ch: &mut dyn RawChallenger,
) -> CommonInteraction
where
    O: Oracle,
    P: CommonMessageProver + ?Sized,
{
    let (relation_proof, relation) = relation_protocol::verify_interactive(st, prover, ch);
    let (local_proof, local) =
        local_protocol::verify_interactive(st.inst, oracle, cfg.t, prover, ch);

    let mut first = None;
    if !cfg.batch_claims {
        set_first(&mut first, CommonError::BatchingDisabled);
    }
    if let Err(error) = &relation.verdict {
        set_first(&mut first, CommonError::Relation(error.clone()));
    }
    if let Err(error) = &local.verdict {
        set_first(&mut first, CommonError::Local(error.clone()));
    }

    let (claims, pad_claims) = match padded_claims(
        st,
        cfg,
        &relation.claims,
        &relation.public_input_claim,
        &local.claims,
    ) {
        Ok(result) => result,
        Err(error) => {
            set_first(&mut first, error);
            let fallback = Config {
                t: cfg.t,
                batch_claims: cfg.batch_claims,
                pad_queries_to: cfg.pad_queries_to,
                k_rel: relation.claims.len(),
            };
            padded_claims(
                st,
                &fallback,
                &relation.claims,
                &relation.public_input_claim,
                &local.claims,
            )
            .expect("the fallback relation count is exact")
        }
    };
    let batching = batch::verify_interactive(&claims, st.layout.nu, prover, ch);
    if let Err(error) = &batching.verification.verdict {
        set_first(&mut first, CommonError::Batching(error.clone()));
    }

    let verification = CommonVerification {
        reduced_claim: batching.verification.claim,
        claims_submitted: claims.len(),
        pad_claims,
        oracle_calls: local.oracle_calls,
        t_mat: relation.t_mat,
        verdict: first.map_or(Ok(()), Err),
    };
    let proof = CommonProof {
        relation: relation_proof,
        local: local_proof,
        batching: batching.proof,
    };
    CommonInteraction {
        proof,
        verification,
    }
}

pub fn verify_with_opening<O, V>(
    st: &Statement<'_>,
    oracle: &O,
    cfg: &Config,
    proof: &CommonProof,
    opening_verifier: &V,
    answers: &[(OracleId, Coord, Symbol<F128>)],
    ch: &mut dyn RawChallenger,
) -> FinalVerification
where
    O: Oracle,
    V: OpeningVerifier<F128>,
{
    let common = verify(st, oracle, cfg, proof, ch);
    let opening_ok = opening_verifier.check(std::slice::from_ref(&common.reduced_claim), answers);
    let verdict = match &common.verdict {
        Err(error) => Err(FinalError::Common(error.clone())),
        Ok(()) if !opening_ok => Err(FinalError::OpeningMismatch),
        Ok(()) => Ok(()),
    };
    FinalVerification {
        opening_ok,
        verdict,
    }
}

pub fn verify_interactive_with_opening<O, P, V, F>(
    st: &Statement<'_>,
    oracle: &O,
    cfg: &Config,
    prover: &mut P,
    finish_opening: F,
    ch: &mut dyn RawChallenger,
) -> FinalInteraction
where
    O: Oracle,
    P: CommonMessageProver + ?Sized,
    V: OpeningVerifier<F128>,
    F: FnOnce(
        &Claim<F128>,
        &mut P,
        &mut dyn RawChallenger,
    ) -> (V, Vec<(OracleId, Coord, Symbol<F128>)>),
{
    let interaction = verify_interactive(st, oracle, cfg, prover, ch);
    let (opening_verifier, answers) =
        finish_opening(&interaction.verification.reduced_claim, prover, ch);
    let opening_ok = opening_verifier.check(
        std::slice::from_ref(&interaction.verification.reduced_claim),
        &answers,
    );
    let verdict = match &interaction.verification.verdict {
        Err(error) => Err(FinalError::Common(error.clone())),
        Ok(()) if !opening_ok => Err(FinalError::OpeningMismatch),
        Ok(()) => Ok(()),
    };
    let final_verification = FinalVerification {
        opening_ok,
        verdict,
    };
    FinalInteraction {
        proof: interaction.proof,
        verification: final_verification,
    }
}
