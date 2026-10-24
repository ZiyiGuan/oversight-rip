use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, trace, Instance};
use rmip_iop::backend::{Coord, Symbol};
use rmip_iop::batch::BatchMessageProver;
use rmip_iop::common_protocol::{self, CommonError, FinalError, HonestCommonProver};
use rmip_iop::driver::{Config, Statement};
use rmip_iop::lincheck::Coefficients;
use rmip_iop::local_protocol::LocalMessageProver;
use rmip_iop::mle::{eval_mle, lift};
use rmip_iop::relation_protocol::RelationMessageProver;
use rmip_iop::sumcheck::RoundMessageProver;
use rmip_iop::transcript::{Entry, OsChallenger, RawChallenger, SeededChallenger};
use std::cell::Cell;

const VIEWS: usize = 16;

struct Fixture {
    inst: Instance,
    sys: r1cs::R1cs,
    layout: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    ordinary: deoracle::OrdinaryCircuit,
    oracle: SparseTableOracle,
    table: Vec<F128>,
}

fn fixture() -> Fixture {
    let mnist = MnistInstance::new(VIEWS);
    let circuit = mnist.build();
    let oracle = mnist.oracle();
    let (ordinary, sites) = deoracle::deoraclize(&circuit);
    let layout = layout::TraceLayout::new(&ordinary);
    let sys = r1cs::emit(&ordinary, &layout, &R1csConfig::default());
    let locations = loc::build(&sites, &layout);
    let x = mnist.yes();
    assert!(
        circuit.eval(x.as_slice(), &oracle),
        "the true label must be accepted"
    );
    let (_, answers) = circuit.eval_with_answers(x.as_slice(), &oracle);
    let table =
        lift(trace::witness_trace(&ordinary, &layout, x.as_slice(), answers.as_slice()).as_slice());
    let inst = Instance {
        layout: layout.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(locations),
        ell: mnist.ell(),
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: mnist.eps_rob,
    };
    Fixture {
        inst,
        sys,
        layout,
        circuit,
        ordinary,
        oracle,
        table,
    }
}

fn statement(f: &Fixture) -> Statement<'_> {
    Statement {
        inst: &f.inst,
        sys: &f.sys,
        layout: &f.layout,
        circuit: &f.circuit,
        dc: &f.ordinary,
        holo: None,
    }
}

fn coin_count(ch: &SeededChallenger) -> usize {
    ch.transcript()
        .entries()
        .iter()
        .filter(|entry| matches!(entry, Entry::Coin(_)))
        .count()
}

#[test]
fn common_verifier_derives_the_opening_claim_without_the_trace() {
    let f = fixture();
    let cfg = Config::default();
    let mut prover = SeededChallenger::new(0xC011);
    let made = common_protocol::prove(&statement(&f), &f.table, &cfg, &mut prover).unwrap();

    let mut verifier = SeededChallenger::new(0xC011);
    let checked =
        common_protocol::verify(&statement(&f), &f.oracle, &cfg, &made.proof, &mut verifier);
    assert_eq!(checked.verdict, Ok(()));
    assert_eq!(checked.reduced_claim, made.reduced_claim);
    assert_eq!(
        checked.reduced_claim.value,
        eval_mle(&f.table, &checked.reduced_claim.point)
    );
    assert_eq!(f.inst.b_max(), 64);
    assert_eq!(
        checked.claims_submitted,
        cfg.k_rel + cfg.t * (f.inst.b_max() + 1)
    );
    assert_eq!(checked.oracle_calls, cfg.t);
    assert_eq!(
        prover.transcript().entries(),
        verifier.transcript().entries()
    );
}

#[test]
fn final_verifier_checks_the_public_opening_without_witness_access() {
    let f = fixture();
    let cfg = Config::default();
    let mut prover = SeededChallenger::new(0x0A11);
    let made = common_protocol::prove(&statement(&f), &f.table, &cfg, &mut prover).unwrap();
    let verifier = rmip_backend_trace::RawTraceVerifier::from_public_length(f.table.len());
    let answers: Vec<_> = f
        .table
        .iter()
        .enumerate()
        .map(|(coord, &value)| (0, Coord::from(coord), Symbol(vec![value])))
        .collect();

    let mut challenger = SeededChallenger::new(0x0A11);
    let checked = common_protocol::verify_with_opening(
        &statement(&f),
        &f.oracle,
        &cfg,
        &made.proof,
        &verifier,
        &answers,
        &mut challenger,
    );
    assert_eq!(checked.verdict, Ok(()));
    assert!(checked.opening_ok);

    let mut bad_answers = answers;
    bad_answers[0].2 .0[0] = bad_answers[0].2 .0[0].add(F128::ONE);
    let mut challenger = SeededChallenger::new(0x0A11);
    let checked = common_protocol::verify_with_opening(
        &statement(&f),
        &f.oracle,
        &cfg,
        &made.proof,
        &verifier,
        &bad_answers,
        &mut challenger,
    );
    assert_eq!(checked.verdict, Err(FinalError::OpeningMismatch));
    assert!(!checked.opening_ok);
}

#[test]
fn every_common_phase_accepts_arbitrary_messages_and_rejects_them_in_place() {
    let f = fixture();
    let cfg = Config::default();
    let mut prover = SeededChallenger::new(41);
    let made = common_protocol::prove(&statement(&f), &f.table, &cfg, &mut prover).unwrap();
    let mut honest_ch = SeededChallenger::new(41);
    let honest =
        common_protocol::verify(&statement(&f), &f.oracle, &cfg, &made.proof, &mut honest_ch);
    assert_eq!(honest.verdict, Ok(()));

    let mut bad_relation = made.proof.clone();
    bad_relation.relation.zerocheck.round_polys[0][0] =
        bad_relation.relation.zerocheck.round_polys[0][0].add(F128::ONE);
    let mut ch = SeededChallenger::new(41);
    assert!(matches!(
        common_protocol::verify(&statement(&f), &f.oracle, &cfg, &bad_relation, &mut ch).verdict,
        Err(CommonError::Relation(_))
    ));
    assert_eq!(coin_count(&ch), coin_count(&honest_ch));

    let mut bad_local = made.proof.clone();
    let last = bad_local.local.blocks[0].len() - 1;
    bad_local.local.blocks[0][last] = bad_local.local.blocks[0][last].add(F128::ONE);
    let mut ch = SeededChallenger::new(41);
    assert!(matches!(
        common_protocol::verify(&statement(&f), &f.oracle, &cfg, &bad_local, &mut ch).verdict,
        Err(CommonError::Local(_))
    ));
    assert_eq!(coin_count(&ch), coin_count(&honest_ch));

    let honest_reduced = {
        let mut ch = SeededChallenger::new(41);
        common_protocol::verify(&statement(&f), &f.oracle, &cfg, &made.proof, &mut ch).reduced_claim
    };
    let mut bad_batch = made.proof;
    bad_batch.batching.sumcheck.round_polys[0][0] =
        bad_batch.batching.sumcheck.round_polys[0][0].add(F128::ONE);
    let mut ch = SeededChallenger::new(41);
    let diverted = common_protocol::verify(&statement(&f), &f.oracle, &cfg, &bad_batch, &mut ch);
    assert_eq!(
        diverted.verdict,
        Ok(()),
        "recovery accepts any well-formed stream"
    );
    assert_ne!(
        diverted.reduced_claim.value, honest_reduced.value,
        "the corruption must divert the claim, or the opening could never catch it"
    );
    assert_eq!(coin_count(&ch), coin_count(&honest_ch));
}

#[test]
fn live_common_core_uses_one_verifier_owned_coin_stream_end_to_end() {
    let f = fixture();
    let cfg = Config::default();
    let st = statement(&f);
    let mut prover = HonestCommonProver::new(&st, &f.table);
    let mut verifier = OsChallenger::new();
    let interaction =
        common_protocol::verify_interactive(&st, &f.oracle, &cfg, &mut prover, &mut verifier);
    assert_eq!(interaction.verification.verdict, Ok(()));
    assert_eq!(interaction.verification.oracle_calls, cfg.t);
    assert_eq!(
        interaction.verification.reduced_claim.value,
        eval_mle(&f.table, &interaction.verification.reduced_claim.point)
    );
}

#[test]
fn live_common_core_hands_the_reduced_claim_to_an_opening_callback() {
    let f = fixture();
    let cfg = Config::default();
    let st = statement(&f);
    let mut prover = HonestCommonProver::new(&st, &f.table);
    let opening = rmip_backend_trace::RawTraceVerifier::from_public_length(f.table.len());
    let answers: Vec<_> = f
        .table
        .iter()
        .enumerate()
        .map(|(coord, &value)| (0, Coord::from(coord), Symbol(vec![value])))
        .collect();
    let mut verifier = OsChallenger::new();
    let interaction = common_protocol::verify_interactive_with_opening(
        &st,
        &f.oracle,
        &cfg,
        &mut prover,
        move |_claim, _prover, _challenger| (opening, answers),
        &mut verifier,
    );
    assert_eq!(interaction.verification.verdict, Ok(()));
    assert!(interaction.verification.opening_ok);
}

struct CorruptLocalCommon<'a> {
    inner: HonestCommonProver<'a>,
}

impl RelationMessageProver for CorruptLocalCommon<'_> {
    fn holo_frame(&mut self) -> Vec<F128> {
        unreachable!("these test provers run scan-shaped statements")
    }

    fn observe_public_input_point(&mut self, r_pub: &[F128]) {
        self.inner.observe_public_input_point(r_pub);
    }

    fn begin_zerocheck(&mut self, tau: &[F128]) {
        self.inner.begin_zerocheck(tau);
    }

    fn zerocheck_round_message(&mut self, round: usize) -> Vec<F128> {
        self.inner.zerocheck_round_message(round)
    }

    fn receive_zerocheck_challenge(&mut self, round: usize, challenge: F128) {
        self.inner.receive_zerocheck_challenge(round, challenge);
    }

    fn zerocheck_endpoint(&mut self) -> Vec<F128> {
        self.inner.zerocheck_endpoint()
    }

    fn begin_lincheck(&mut self, row_point: &[F128], coefficients: Coefficients) {
        self.inner.begin_lincheck(row_point, coefficients);
    }

    fn lincheck_round_message(&mut self, round: usize) -> Vec<F128> {
        self.inner.lincheck_round_message(round)
    }

    fn receive_lincheck_challenge(&mut self, round: usize, challenge: F128) {
        self.inner.receive_lincheck_challenge(round, challenge);
    }

    fn lincheck_endpoint(&mut self) -> Vec<F128> {
        self.inner.lincheck_endpoint()
    }
}

impl LocalMessageProver for CorruptLocalCommon<'_> {
    fn answer_blocks(&mut self, sampling_coins: &[F128], sampled: &[usize]) -> Vec<Vec<F128>> {
        let mut blocks = self.inner.answer_blocks(sampling_coins, sampled);
        blocks[0][0] = F128::new(2);
        blocks
    }
}

impl BatchMessageProver for CorruptLocalCommon<'_> {
    fn begin_batch(&mut self, claims: &[rmip_iop::backend::Claim<F128>], gammas: &[F128]) {
        self.inner.begin_batch(claims, gammas);
    }
}

impl RoundMessageProver for CorruptLocalCommon<'_> {
    fn round_message(&mut self, round: usize) -> Vec<F128> {
        self.inner.round_message(round)
    }

    fn receive_challenge(&mut self, round: usize, challenge: F128) {
        self.inner.receive_challenge(round, challenge);
    }
}

#[test]
fn malformed_live_common_messages_keep_common_coins_and_still_invoke_the_opening_hook() {
    let f = fixture();
    let cfg = Config::default();
    let st = statement(&f);
    let seed = 0xBAD5;

    let mut honest_prover = HonestCommonProver::new(&st, &f.table);
    let mut honest_challenger = SeededChallenger::new(seed);
    let honest = common_protocol::verify_interactive(
        &st,
        &f.oracle,
        &cfg,
        &mut honest_prover,
        &mut honest_challenger,
    );
    assert_eq!(honest.verification.verdict, Ok(()));
    let honest_coins = coin_count(&honest_challenger);

    let opening = rmip_backend_trace::RawTraceVerifier::from_public_length(f.table.len());
    let answers: Vec<_> = f
        .table
        .iter()
        .enumerate()
        .map(|(coord, &value)| (0, Coord::from(coord), Symbol(vec![value])))
        .collect();
    let opening_called = Cell::new(false);
    let mut bad_prover = CorruptLocalCommon {
        inner: HonestCommonProver::new(&st, &f.table),
    };
    let mut bad_challenger = SeededChallenger::new(seed);
    let checked = common_protocol::verify_interactive_with_opening(
        &st,
        &f.oracle,
        &cfg,
        &mut bad_prover,
        |_claim, _prover, _challenger| {
            opening_called.set(true);
            (opening, answers)
        },
        &mut bad_challenger,
    );
    assert!(matches!(
        checked.verification.verdict,
        Err(FinalError::Common(CommonError::Local(_)))
    ));
    assert!(opening_called.get());
    assert_eq!(coin_count(&bad_challenger), honest_coins);
}
