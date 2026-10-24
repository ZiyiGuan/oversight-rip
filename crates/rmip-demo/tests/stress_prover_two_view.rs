use std::cell::RefCell;

use rmip_backend_interleaved::{Ligero, Shape};
use rmip_demo::harness::{mnist_case, Case};
use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::SparseTableOracle;
use rmip_iop::backend::{
    BackendParams, Claim, CommitmentBackend, Coord, LongOracle, QueryPlan, Symbol,
};
use rmip_iop::driver::{committed_trace, execute_recording, Cheat, Config, Execution, Statement};
use rmip_iop::transcript::{
    OracleId, OsChallenger, RawChallenger, SeededChallenger, Transcript, TranscriptPrefix,
};
use rmip_mip::checker::audit_with_answer;
use rmip_mip::{Audit, PCheck, SharedAdvice};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Question {
    oracle: OracleId,
    prefix: TranscriptPrefix,
    coord: Coord,
}

impl Question {
    fn checker_binary_request(&self) -> Vec<u8> {
        let prefix = self.prefix.as_bytes();
        let mut out = Vec::with_capacity(24 + prefix.len());
        out.extend_from_slice(&(self.oracle as u64).to_le_bytes());
        out.extend_from_slice(&(prefix.len() as u64).to_le_bytes());
        out.extend_from_slice(prefix);
        out.extend_from_slice(
            &u64::try_from(self.coord)
                .expect("checker wire coordinate fits u64")
                .to_le_bytes(),
        );
        out
    }
}

struct Tapped<B> {
    inner: B,
    seen: RefCell<Vec<Question>>,
}

impl<B> Tapped<B> {
    fn new(inner: B) -> Self {
        Self {
            inner,
            seen: RefCell::new(Vec::new()),
        }
    }

    fn take_only_question(&self) -> Question {
        let mut seen = self.seen.borrow_mut();
        assert_eq!(
            seen.len(),
            1,
            "the audit must put exactly one question to the checker"
        );
        seen.pop().expect("one question")
    }
}

impl<B: CommitmentBackend<Field = F128>> CommitmentBackend for Tapped<B> {
    type Field = F128;
    type Commitment = B::Commitment;
    type Verifier = B::Verifier;

    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> Self::Commitment {
        self.inner.commit(z, ch)
    }

    fn open(&mut self, claims: &[Claim<F128>], ch: &mut dyn RawChallenger) {
        self.inner.open(claims, ch);
    }

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
        self.inner.query_plan(ch)
    }

    fn oracles(&self) -> &[LongOracle<F128>] {
        self.inner.oracles()
    }

    fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
        self.seen.borrow_mut().push(Question {
            oracle: id,
            prefix: h.clone(),
            coord,
        });
        self.inner.reconstruct(id, h, coord)
    }

    fn verifier(&self) -> Self::Verifier {
        self.inner.verifier()
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        self.inner.public_params(trace_len, claims)
    }

    fn params(&self, claims: usize) -> BackendParams {
        self.inner.params(claims)
    }
}

struct FixedCoin {
    coin: u128,
    transcript: Transcript,
}

impl FixedCoin {
    fn new(coin: u128) -> Self {
        Self {
            coin,
            transcript: Transcript::new(),
        }
    }
}

impl RawChallenger for FixedCoin {
    fn sample(&mut self) -> F128 {
        let c = F128::new(self.coin);
        self.transcript.record_coin(c);
        c
    }

    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.transcript.record_ordinary(bytes);
    }

    fn post_oracle(&mut self, id: OracleId) {
        self.transcript.post_oracle(id);
    }

    fn transcript(&self) -> &Transcript {
        &self.transcript
    }
}

fn coin_for(index: usize, q: usize, nonce: u32) -> u128 {
    let stride = (1u128 << 100) + u128::from(nonce);
    index as u128 + q as u128 * stride
}

struct Fx {
    case: Case,
    oracle: SparseTableOracle,
}

fn build() -> Fx {
    let (case, oracle) = mnist_case(16);
    assert!(
        case.holo.is_none(),
        "16 views is scan-shaped; the checker here commits a plain trace"
    );
    Fx { case, oracle }
}

fn st(f: &Fx) -> Statement<'_> {
    Statement {
        inst: &f.case.inst,
        sys: &f.case.sys,
        layout: &f.case.layout,
        circuit: &f.case.circuit,
        dc: &f.case.dc,
        holo: None,
    }
}

fn config(f: &Fx) -> Config {
    Config {
        t: f.case.t,
        ..Config::default()
    }
}

fn checker(f: &Fx, shape: Shape) -> PCheck<Tapped<Ligero>> {
    let advice = SharedAdvice { seed: 0xC4EC };
    let z = committed_trace(&st(f), &f.oracle, &Cheat::default());
    let mut b = Ligero::new(shape);
    b.commit(&z, &mut SeededChallenger::new(advice.seed));
    PCheck::new(Tapped::new(b))
}

fn run(f: &Fx, shape: Shape) -> (Execution, Transcript) {
    let mut backend = Ligero::new(shape);
    let mut ch = OsChallenger::new();
    let (execution, verdict) = execute_recording(
        &st(f),
        &mut backend,
        &f.oracle,
        &config(f),
        &Cheat::default(),
        &mut ch,
    );
    verdict.expect("the MNIST yes-instance must prove");
    let transcript = ch.transcript().clone();
    (execution, transcript)
}

fn audit_at(
    execution: &Execution,
    checker: &PCheck<Tapped<Ligero>>,
    index: usize,
    nonce: u32,
) -> (Audit, Symbol<F128>, Question) {
    let q = execution.q_long();
    let mut coins = FixedCoin::new(coin_for(index, q, nonce));
    let (verdict, reply) = audit_with_answer(execution, checker, &mut coins);
    let audited = match verdict {
        Audit::Accepted { audited }
        | Audit::UnderlyingRejected { audited }
        | Audit::Inconsistent { audited } => audited,
    };
    assert_eq!(audited, index, "the forced coin must select occurrence J");
    (verdict, reply, checker.backend().take_only_question())
}

fn wire_checker_query(
    transcript: &Transcript,
    execution: &Execution,
    index: usize,
    reply: &Symbol<F128>,
) -> (Vec<u8>, Vec<u8>) {
    let mut stream = Vec::new();
    rmip_demo::wire::write_execution(
        &mut stream,
        transcript,
        execution,
        &execution.occurrences[index],
        reply,
    )
    .expect("a completed honest execution serializes");
    let frames = rmip_demo::wire::decode(&stream).expect("the serializer emits canonical frames");
    let query = frames
        .iter()
        .find(|frame| frame.kind == rmip_demo::wire::Kind::CheckerQuery)
        .expect("one checker question per stream")
        .payload
        .to_vec();
    (query, stream)
}

#[test]
fn only_the_triple_crosses_the_boundary_between_two_unrelated_executions() {
    let f = build();
    let shape = Shape::balanced(f.case.layout.nu, 4, 4.0);
    let checker = checker(&f, shape);

    let (first, first_transcript) = run(&f, shape);
    let mut found = None;
    for _ in 0..400 {
        let (second, second_transcript) = run(&f, shape);
        let hit = first.occurrences.iter().enumerate().find_map(|(i, a)| {
            second
                .occurrences
                .iter()
                .position(|b| b.coord == a.coord && b.oracle == a.oracle)
                .map(|j| (i, j))
        });
        if let Some((i, j)) = hit {
            found = Some((second, second_transcript, i, j));
            break;
        }
    }
    let (second, second_transcript, i, j) =
        found.expect("two executions sharing an opened column within 400 tries");

    assert_ne!(
        first_transcript.entries(),
        second_transcript.entries(),
        "two OS-coin executions must not produce the same transcript"
    );

    assert_eq!(
        first.occurrences[i].prefix, second.occurrences[j].prefix,
        "the codeword oracle's posting prefix precedes every coin"
    );

    let (first_verdict, first_reply, first_question) = audit_at(&first, &checker, i, 1);
    let (second_verdict, second_reply, second_question) = audit_at(&second, &checker, j, 2);
    assert!(first_verdict.accepted() && second_verdict.accepted());

    assert_eq!(
        first_question, second_question,
        "the checker's question leaked something beyond (oracle, prefix, coordinate)"
    );
    assert_eq!(
        first_question.checker_binary_request(),
        second_question.checker_binary_request(),
        "the out-of-process request frame differs between the two executions"
    );
    assert_eq!(
        first_reply, second_reply,
        "F*_{{i,h}}(theta) is not a function of the triple alone"
    );

    let (first_query, first_stream) =
        wire_checker_query(&first_transcript, &first, i, &first_reply);
    let (second_query, second_stream) =
        wire_checker_query(&second_transcript, &second, j, &second_reply);
    assert_ne!(
        first_stream, second_stream,
        "the two executions must serialize differently overall"
    );
    assert_eq!(
        first_query, second_query,
        "the rmip-wire-v1 checker question differs between the two executions"
    );

    println!(
        "q_long = {}, matched occurrence {i} against {j} at column {}",
        first.q_long(),
        first.occurrences[i].coord
    );
}

#[test]
fn two_different_audit_coins_that_select_the_same_index_ask_the_same_question() {
    let f = build();
    let shape = Shape::balanced(f.case.layout.nu, 4, 4.0);
    let checker = checker(&f, shape);
    let (execution, _) = run(&f, shape);
    let index = execution.q_long() - 1;

    let (_, reply_a, question_a) = audit_at(&execution, &checker, index, 3);
    let (_, reply_b, question_b) = audit_at(&execution, &checker, index, 4);
    assert_ne!(
        coin_for(index, execution.q_long(), 3),
        coin_for(index, execution.q_long(), 4),
        "the two audits must draw different coins"
    );
    assert_eq!(question_a, question_b);
    assert_eq!(reply_a, reply_b);
}

#[test]
fn nothing_prover_one_produced_after_the_posting_reaches_the_checker() {
    let f = build();
    let shape = Shape::balanced(f.case.layout.nu, 4, 4.0);
    let checker = checker(&f, shape);
    let (mut execution, _) = run(&f, shape);
    let index = 0;

    let (before, reply_before, question_before) = audit_at(&execution, &checker, index, 5);
    assert!(before.accepted());

    let q = execution.q_long();
    for other in 0..q {
        if other == index {
            continue;
        }
        let occurrence = &mut execution.occurrences[other];
        occurrence.coord = Coord::from((usize::try_from(occurrence.coord).unwrap() + 1) % q.max(2));
        let mut answer = occurrence.answer.0.clone();
        answer[0] = answer[0].add(F128::ONE);
        occurrence.answer = Symbol(answer);
    }
    execution.iop_accepted = false;

    let (after, reply_after, question_after) = audit_at(&execution, &checker, index, 6);
    assert_eq!(
        question_before, question_after,
        "a change to another occurrence or to the IOP verdict reached the checker"
    );
    assert_eq!(reply_before, reply_after);
    assert!(
        matches!(after, Audit::UnderlyingRejected { .. }),
        "a consistent audit over a rejecting IOP must not be an acceptance: {after:?}"
    );
    assert!(!after.accepted());
}

#[test]
fn the_verifiers_own_answer_for_the_audited_occurrence_is_not_shown_to_the_checker() {
    let f = build();
    let shape = Shape::balanced(f.case.layout.nu, 4, 4.0);
    let checker = checker(&f, shape);
    let (mut execution, _) = run(&f, shape);
    let index = 1;

    let (before, reply_before, question_before) = audit_at(&execution, &checker, index, 7);
    assert!(before.accepted());

    let mut answer = execution.occurrences[index].answer.0.clone();
    answer[0] = answer[0].add(F128::ONE);
    execution.occurrences[index].answer = Symbol(answer);

    let (after, reply_after, question_after) = audit_at(&execution, &checker, index, 8);
    assert_eq!(
        question_before, question_after,
        "the verifier's held answer leaked into the checker's question"
    );
    assert_eq!(
        reply_before, reply_after,
        "the checker's reply followed the verifier's corrupted answer"
    );
    assert!(
        matches!(after, Audit::Inconsistent { .. }),
        "a corrupted audited answer must be caught: {after:?}"
    );
}

#[test]
fn the_checker_reply_does_not_depend_on_how_many_questions_it_has_answered() {
    let f = build();
    let shape = Shape::balanced(f.case.layout.nu, 4, 4.0);
    let checker = checker(&f, shape);

    let (first, _) = run(&f, shape);
    let (_, first_reply, first_question) = audit_at(&first, &checker, 0, 9);
    assert_eq!(checker.questions_asked(), 1);

    for repetition in 0..11u32 {
        let (execution, _) = run(&f, shape);
        let index = (repetition as usize) % execution.q_long();
        let (verdict, _, _) = audit_at(&execution, &checker, index, 100 + repetition);
        assert!(verdict.accepted(), "repetition {repetition} must accept");
    }
    assert_eq!(checker.questions_asked(), 12);

    let (_, last_reply, last_question) = audit_at(&first, &checker, 0, 10);
    assert_eq!(checker.questions_asked(), 13);
    assert_eq!(
        first_question, last_question,
        "the same occurrence produced a different question in a later repetition"
    );
    assert_eq!(
        first_reply, last_reply,
        "the checker's reply changed between repetitions on the same triple"
    );

    println!(
        "one checker, {} questions, identical reply at repetitions 1 and 13",
        checker.questions_asked()
    );
}
