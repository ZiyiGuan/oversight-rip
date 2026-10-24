use rmip_backend_trace::RawTrace;
use rmip_frontend::bits::BitSlice;
use rmip_frontend::mnist::{label_bits, MnistInstance, LABEL_BITS};
use rmip_frontend::oracle::{CountingOracle, Oracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance, Ratio};

use rmip_field::BinaryField;
use rmip_iop::backend::Coord;
use rmip_iop::driver::{execute, Cheat, Config, Execution, Reject, Statement};
use rmip_iop::transcript::{RawChallenger, SeededChallenger};

const VIEWS: usize = 16;

struct Fixture {
    inst: Instance,
    sys: r1cs::R1cs,
    layout: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    oracle: Box<dyn Oracle>,
    honest_answers: Vec<bool>,
}

impl Fixture {
    fn oracle(&self) -> &dyn Oracle {
        &*self.oracle
    }

    fn statement(&self) -> Statement<'_> {
        Statement {
            inst: &self.inst,
            sys: &self.sys,
            layout: &self.layout,
            circuit: &self.circuit,
            dc: &self.dc,
            holo: None,
        }
    }
}

fn fixture() -> Fixture {
    let mnist = MnistInstance::new(VIEWS);
    let circuit = mnist.build();
    let oracle = mnist.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);

    let x = mnist.yes();
    let (accepted, answers) = circuit.eval_with_answers(x.as_slice(), &oracle);
    assert!(accepted, "the true label must be accepted");

    let inst = Instance {
        layout: lay.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(l),
        ell: mnist.ell(),
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: mnist.eps_rob,
    };
    Fixture {
        inst,
        sys,
        layout: lay,
        circuit,
        dc,
        oracle: Box::new(oracle),
        honest_answers: answers.as_slice().to_vec(),
    }
}

fn run_with(f: &Fixture, cfg: &Config, cheat: &Cheat, seed: u64) -> Result<Execution, Reject> {
    let st = f.statement();
    let mut backend = RawTrace::new();
    let mut ch = SeededChallenger::new(seed);
    execute(&st, &mut backend, &f.oracle(), cfg, cheat, &mut ch)
}

fn run(f: &Fixture, cheat: &Cheat, seed: u64) -> Result<Execution, Reject> {
    run_with(f, &Config::default(), cheat, seed)
}

#[test]
fn honest_prover_is_accepted_every_time() {
    let f = fixture();
    let honest = Cheat::default();
    let cfg = Config::default();
    assert_eq!(f.inst.b_max(), 64);
    let k = cfg.k_rel + cfg.t * (f.inst.b_max() + 1);
    for seed in 1..=1000u64 {
        let e = run(&f, &honest, seed)
            .unwrap_or_else(|r| panic!("honest execution rejected at seed {seed}: {r:?}"));
        assert_eq!(e.oracle_calls, cfg.t);
        assert_eq!(e.claims_submitted, k);
        assert_eq!(e.occurrences.len(), f.layout.n_trace);
    }
}

#[test]
fn corrupted_trace_is_rejected_at_zerocheck() {
    let f = fixture();
    let touched: Vec<usize> = f
        .sys
        .a
        .nz
        .iter()
        .chain(f.sys.b.nz.iter())
        .chain(f.sys.c.nz.iter())
        .map(|(_, c)| *c)
        .filter(|c| *c != 0)
        .collect();

    let mut caught = 0;
    for &coord in touched.iter().take(30) {
        let cheat = Cheat {
            corrupt_trace_coord: Some(coord),
            ..Default::default()
        };
        match run(&f, &cheat, 42) {
            Err(Reject::ZerocheckEndpoint) => caught += 1,
            Err(Reject::OracleMismatch { .. }) => {}
            Err(other) => panic!("coord {coord} rejected at the WRONG check: {other:?}"),
            Ok(_) => panic!("coord {coord} corruption was ACCEPTED"),
        }
    }
    assert!(caught > 0, "no corruption reached zerocheck");
}

#[test]
fn wrong_public_input_is_rejected_at_the_public_input_check() {
    let f = fixture();
    let swap_to = |label: usize| -> Cheat {
        let target = label_bits(label);
        let bits: Vec<usize> = (0..LABEL_BITS)
            .filter(|&k| f.inst.x.get(k) != target.get(k))
            .collect();
        assert!(!bits.is_empty(), "label {label} is the true label");
        Cheat {
            public_input_bits: bits,
            local_cheat_answers: Some(f.honest_answers.clone()),
            ..Default::default()
        }
    };
    let other_labels: Vec<usize> = (0..10).filter(|&y| label_bits(y) != f.inst.x).collect();
    assert_eq!(other_labels.len(), 9);

    let no_samples = Config {
        t: 0,
        ..Config::default()
    };
    for &label in &other_labels {
        let cheat = swap_to(label);
        for seed in [1u64, 7, 99] {
            match run_with(&f, &no_samples, &cheat, seed) {
                Err(e) if e.is_joint_opening() => {}
                Err(other) => panic!("label {label}: rejected at the wrong check: {other:?}"),
                Ok(_) => panic!("label {label}: public-input swap was ACCEPTED at seed {seed}"),
            }
        }
    }

    let cheat = swap_to(other_labels[0]);
    for seed in 1..=24u64 {
        match run(&f, &cheat, seed) {
            Err(Reject::OracleMismatch { .. }) => {}
            Err(e) if e.is_joint_opening() => {}
            Err(other) => panic!("seed {seed}: rejected at the wrong check: {other:?}"),
            Ok(_) => panic!("seed {seed}: public-input swap was ACCEPTED"),
        }
    }
}

#[test]
fn single_bit_input_flip_is_caught_by_zerocheck_instead() {
    let f = fixture();
    for bit in 0..LABEL_BITS {
        let cheat = Cheat {
            public_input_bits: vec![bit],
            ..Default::default()
        };
        match run(&f, &cheat, 5) {
            Err(Reject::ZerocheckEndpoint) => {}
            Err(other) => panic!("bit {bit}: expected zerocheck, got {other:?}"),
            Ok(_) => panic!("bit {bit}: accepted a no-instance"),
        }
    }
}

#[test]
fn trace_swap_is_caught_by_the_joint_opening() {
    let f = fixture();
    assert!(
        f.honest_answers.iter().all(|&a| a),
        "every honest answer must be true for the flipped answer to match the broken tag"
    );
    let cheat = Cheat {
        trace_swap: true,
        bad_oracle_answer: Some(0),
        ..Default::default()
    };
    for seed in [2u64, 13, 77] {
        match run(&f, &cheat, seed) {
            Err(e) if e.is_joint_opening() => {}
            Err(other) => panic!("rejected at the wrong check: {other:?}"),
            Ok(_) => panic!("trace swap was ACCEPTED at seed {seed}"),
        }
    }
    let swap_only = Cheat {
        trace_swap: true,
        ..Default::default()
    };
    assert!(
        matches!(
            run(&f, &swap_only, 2),
            Err(Reject::OracleMismatch { occurrence: 0 })
        ),
        "the unpaired swap must be caught by the oracle check on occurrence 0"
    );
}

#[test]
fn bad_oracle_answer_is_rejected_at_the_validity_check() {
    let f = fixture();
    for h in 0..3 {
        let cheat = Cheat {
            bad_oracle_answer: Some(h),
            ..Default::default()
        };
        match run(&f, &cheat, 5) {
            Err(Reject::OracleMismatch { occurrence }) => assert_eq!(occurrence, h),
            Err(other) => panic!("rejected at the wrong check: {other:?}"),
            Ok(_) => panic!("bad oracle answer at occurrence {h} was ACCEPTED"),
        }
    }
}

#[test]
fn padded_counts_are_execution_independent() {
    let f = fixture();
    let honest = Cheat::default();
    let mut ks = std::collections::HashSet::new();
    let mut qs = std::collections::HashSet::new();
    for seed in 1..=200u64 {
        let e = run(&f, &honest, seed).unwrap();
        ks.insert(e.claims_submitted);
        qs.insert(e.occurrences.len());
    }
    assert_eq!(ks.len(), 1, "claim count varied across executions: {ks:?}");
    assert_eq!(qs.len(), 1, "query count varied across executions: {qs:?}");
}

#[test]
fn occurrences_are_tagged_for_the_audit() {
    let f = fixture();
    let e = run(&f, &Cheat::default(), 11).unwrap();
    let first = e.occurrences[0].prefix.clone();
    for o in &e.occurrences {
        assert_eq!(
            o.oracle, 0,
            "the raw trace's unpadded real plan queries only its backend-owned trace oracle"
        );
        assert_eq!(o.prefix, first);
        assert_eq!(o.answer.0.len(), 1, "alphabet is Field");
    }
}

#[test]
fn oracle_calls_are_counted_both_ways() {
    let f = fixture();
    let counting = CountingOracle::new(f.oracle());
    let st = f.statement();
    let mut backend = RawTrace::new();
    let mut ch = SeededChallenger::new(3);
    let e = execute(
        &st,
        &mut backend,
        &counting,
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    )
    .unwrap();
    assert_eq!(counting.occurrences(), f.inst.ell + e.oracle_calls);
    assert_eq!(counting.distinct(), f.inst.ell);
    assert!(counting.distinct() < counting.occurrences());
}

fn least_repetitions(eps: Ratio, sigma_inv: u128) -> usize {
    let (num, den) = (u128::from(eps.num), u128::from(eps.den));
    (1u32..)
        .find(|&t| sigma_inv * (den - num).pow(t) <= den.pow(t))
        .expect("the miss base is below one") as usize
}

#[test]
fn t_is_derived_from_eps_rob() {
    let f = fixture();
    let c = Config::for_instance(&f.inst, 0.125);
    assert_eq!(c.t, least_repetitions(f.inst.eps_rob, 8));
    assert_eq!(c.t, 4, "16 views at sigma_loc = 1/8");
    assert_ne!(Config::default().t, c.t);

    let mut half = f.inst.clone();
    half.eps_rob = Ratio::new(1, 2);
    assert_eq!(Config::for_instance(&half, 0.125).t, 3);
    assert_eq!(
        Config::for_instance(&half, 0.125).t,
        least_repetitions(half.eps_rob, 8)
    );
    let below_one_sixteenth = f64::from_bits(0x3faf_ffff_ffff_ffff);
    assert_eq!(Config::for_instance(&half, below_one_sixteenth).t, 5);
    let mut perfect_gap = f.inst.clone();
    perfect_gap.eps_rob = Ratio::new(1, 1);
    assert_eq!(Config::for_instance(&perfect_gap, f64::MIN_POSITIVE).t, 1);

    let mut narrow = f.inst.clone();
    narrow.eps_rob = Ratio::new(1, 8);
    let n = Config::for_instance(&narrow, 0.125);
    assert_eq!(n.t, least_repetitions(narrow.eps_rob, 8));
    assert!(n.t > c.t, "a smaller gap must need more samples");
}

#[test]
fn query_phase_pads_with_a_public_dummy_oracle() {
    use rmip_iop::driver::DUMMY_ORACLE;
    let f = fixture();
    let real = f.layout.n_trace;
    let target = real + 6;
    let st = f.statement();
    let mut backend = RawTrace::new();
    let mut ch = SeededChallenger::new(3);
    let cfg = Config {
        pad_queries_to: Some(target),
        ..Default::default()
    };
    let e = execute(
        &st,
        &mut backend,
        &f.oracle(),
        &cfg,
        &Cheat::default(),
        &mut ch,
    )
    .unwrap();

    assert_eq!(e.occurrences.len(), target, "padded to a public constant");
    let dummies = e
        .occurrences
        .iter()
        .filter(|o| o.oracle == DUMMY_ORACLE)
        .count();
    assert_eq!(dummies, target - real);
    for o in e.occurrences.iter().filter(|o| o.oracle == DUMMY_ORACLE) {
        assert!(
            o.prefix.is_empty(),
            "the padding lemma posts the dummy at the empty prefix"
        );
        assert_eq!(o.answer, rmip_iop::driver::dummy_symbol());
    }
}

#[test]
fn non_boolean_supplied_block_is_rejected_before_the_predicate() {
    let f = fixture();
    for bad in [2u128, 3, 0xdead_beef, u128::MAX] {
        let cheat = Cheat {
            supply_non_boolean_block: Some(bad),
            ..Default::default()
        };
        match run(&f, &cheat, 5) {
            Err(Reject::NonBooleanBlock { occurrence }) => assert_eq!(occurrence, 0),
            Err(other) => panic!("value {bad:#x} rejected at the wrong check: {other:?}"),
            Ok(_) => panic!("non-Boolean block value {bad:#x} was ACCEPTED"),
        }
    }
    for ok in [0u128, 1] {
        let cheat = Cheat {
            supply_non_boolean_block: Some(ok),
            ..Default::default()
        };
        assert!(
            !matches!(run(&f, &cheat, 5), Err(Reject::NonBooleanBlock { .. })),
            "value {ok} is Boolean and must not trip the Booleanity check"
        );
    }
}

#[test]
fn trace_commitment_precedes_the_first_coin_and_all_oracles_precede_queries() {
    use rmip_iop::driver::DUMMY_ORACLE;
    use rmip_iop::transcript::Entry;

    let f = fixture();
    let st = f.statement();
    let cfg = Config {
        pad_queries_to: Some(f.layout.n_trace + 16),
        ..Default::default()
    };
    let mut backend = RawTrace::new();
    let mut ch = SeededChallenger::new(9);
    let e = execute(
        &st,
        &mut backend,
        &f.oracle(),
        &cfg,
        &Cheat::default(),
        &mut ch,
    )
    .unwrap();

    let entries = ch.transcript().entries();
    let first_coin = entries
        .iter()
        .position(|x| matches!(x, Entry::Coin(_)))
        .unwrap();
    let trace_post = entries
        .iter()
        .position(|x| matches!(x, Entry::OraclePost(0)))
        .expect("the trace oracle is posted");
    assert!(
        trace_post < first_coin,
        "the commitment must precede every coin: post at {trace_post}, first coin at {first_coin}"
    );

    let dummy_post = entries
        .iter()
        .position(|x| matches!(x, Entry::OraclePost(DUMMY_ORACLE)))
        .expect("the dummy is posted when padding is needed");
    assert_eq!(dummy_post, 0);
    assert!(dummy_post < trace_post);

    let full = ch.transcript().canonical().len();
    for o in &e.occurrences {
        assert_eq!(
            o.prefix,
            ch.transcript().prefix_at(o.oracle),
            "occurrence on oracle {} carries a prefix that is not the one \
             recorded at its posting event",
            o.oracle
        );
        assert!(
            o.prefix.len() < full,
            "prefix not strictly shorter than the transcript"
        );
    }
    let dummies: Vec<_> = e
        .occurrences
        .iter()
        .filter(|o| o.oracle == DUMMY_ORACLE)
        .collect();
    assert!(!dummies.is_empty());
    for d in &dummies[1..] {
        assert_eq!(d.prefix, dummies[0].prefix);
    }
}

#[test]
fn lincheck_coefficients_are_three_separate_draws() {
    use rmip_iop::lincheck::Coefficients;

    for seed in 1..=200u64 {
        let mut a = SeededChallenger::new(seed);
        let expect: Vec<_> = (0..3).map(|_| a.sample()).collect();

        let mut b = SeededChallenger::new(seed);
        let c = Coefficients::sample(&mut b);
        assert_eq!(c.lambda_a(), expect[0], "seed {seed}");
        assert_eq!(c.lambda_b(), expect[1], "lambda_b must be the NEXT coin");
        assert_eq!(
            c.lambda_c(),
            expect[2],
            "lambda_c must be the coin after that"
        );

        assert_ne!(c.lambda_b(), c.lambda_a().mul(c.lambda_a()));
        assert_ne!(
            c.lambda_c(),
            c.lambda_a().mul(c.lambda_a()).mul(c.lambda_a())
        );
    }
}

#[test]
fn reported_oracle_calls_match_an_instrumented_count() {
    let f = fixture();
    let st = f.statement();
    for t in [1usize, 3, 7] {
        let cfg = Config {
            t,
            ..Default::default()
        };
        let counting = CountingOracle::new(f.oracle());
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(4);
        let e = execute(
            &st,
            &mut backend,
            &counting,
            &cfg,
            &Cheat::default(),
            &mut ch,
        )
        .unwrap();

        assert_eq!(
            e.oracle_calls, t,
            "the reported count is the VERIFIER's t calls"
        );
        assert_eq!(
            counting.occurrences(),
            f.inst.ell + t,
            "instrumented total must be ell (witness) + t (validity) at t = {t}"
        );
        assert_eq!(counting.distinct(), f.inst.ell);
    }
}

#[test]
fn the_transcript_is_complete_before_any_coordinate_is_revealed() {
    let f = fixture();
    let st = f.statement();
    let n = f.layout.n_trace;
    for pad in [None, Some(n + 16), Some(2 * n)] {
        let cfg = Config {
            pad_queries_to: pad,
            ..Default::default()
        };
        for seed in [1u64, 17, 99] {
            let mut backend = RawTrace::new();
            let mut ch = SeededChallenger::new(seed);
            let e = execute(
                &st,
                &mut backend,
                &f.oracle(),
                &cfg,
                &Cheat::default(),
                &mut ch,
            )
            .unwrap();
            assert_eq!(
                e.transcript_at_query_start, e.transcript_at_query_end,
                "the transcript grew during the query phase (pad = {pad:?}, seed = {seed})"
            );
            assert_eq!(e.occurrences.len(), pad.unwrap_or(n).max(n));
        }
    }
}

#[test]
fn claim_padding_is_vacuous_for_uniform_query_widths() {
    let f = fixture();
    let widths = f.circuit.b_widths();
    assert!(
        widths.windows(2).all(|w| w[0] == w[1]),
        "this test's premise is that the circuit has uniform b_j"
    );
    assert_eq!(f.inst.b_max(), 64, "the sparse-judge query is 64 bits");

    for t in [1usize, 3, 7] {
        let cfg = Config {
            t,
            ..Default::default()
        };
        let st = f.statement();
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(3);
        let e = execute(
            &st,
            &mut backend,
            &f.oracle(),
            &cfg,
            &Cheat::default(),
            &mut ch,
        )
        .unwrap();

        assert_eq!(
            e.pad_claims, 0,
            "padding fired at t = {t}, so the premise changed"
        );
        assert_eq!(e.claims_submitted, cfg.k_rel + t * (f.inst.b_max() + 1));
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlanMode {
    Honest,
    UnpostedOracle,
    TooMany,
    TooFew,
    OutOfRange,
    AdaptiveOnAnswers,
}

struct Misbehaving {
    inner: RawTrace,
    post_late_oracle: bool,
    post_duplicate_late_oracle: bool,
    speak_late: bool,
    expose_unregistered_oracle: bool,
    replace_oracle_after_later_commit_coin: bool,
    replace_oracle_during_open: bool,
    declare_zero_queries: bool,
    post_wrong_width_symbol: bool,
    replacement: Option<Vec<rmip_iop::backend::LongOracle<rmip_field::F128>>>,
    plan: PlanMode,
}

struct Adaptive {
    n: usize,
    visited: Vec<bool>,
    cursor: usize,
    emitted: usize,
}

impl rmip_iop::backend::QueryPlan<rmip_field::F128> for Adaptive {
    fn next(
        &mut self,
        prev: Option<&rmip_iop::backend::Symbol<rmip_field::F128>>,
    ) -> Option<(rmip_iop::transcript::OracleId, rmip_iop::backend::Coord)> {
        if self.emitted >= self.n {
            return None;
        }
        if let Some(p) = prev {
            let step = if p.0[0] == rmip_field::F128::ONE {
                7
            } else {
                1
            };
            self.cursor = (self.cursor + step) % self.n;
        }
        while self.visited[self.cursor] {
            self.cursor = (self.cursor + 1) % self.n;
        }
        self.visited[self.cursor] = true;
        self.emitted += 1;
        Some((0, Coord::from(self.cursor)))
    }
}

impl rmip_iop::backend::CommitmentBackend for Misbehaving {
    type Field = rmip_field::F128;
    type Commitment = usize;
    type Verifier = rmip_backend_trace::RawTraceVerifier;

    fn commit(&mut self, z: &[Self::Field], ch: &mut dyn RawChallenger) -> usize {
        self.replacement = None;
        let commitment = self.inner.commit(z, ch);
        if self.replace_oracle_after_later_commit_coin {
            let _ = ch.sample();
            let mut replacement = self.inner.oracles().to_vec();
            let symbols = std::sync::Arc::make_mut(&mut replacement[0].symbols);
            symbols[0].0[0] = symbols[0].0[0].add(rmip_field::F128::ONE);
            self.replacement = Some(replacement);
        }
        if self.post_wrong_width_symbol {
            let bad = rmip_iop::backend::LongOracle::new(
                7777,
                rmip_iop::backend::Alphabet::Column { elements: 2 },
                vec![rmip_iop::backend::Symbol(vec![rmip_field::F128::ZERO])],
            );
            ch.post_long_oracle(&bad);
        }
        if self.expose_unregistered_oracle {
            let mut replacement = self
                .replacement
                .take()
                .unwrap_or_else(|| self.inner.oracles().to_vec());
            replacement.push(rmip_iop::backend::LongOracle::new(
                4242,
                rmip_iop::backend::Alphabet::Field,
                vec![rmip_iop::backend::Symbol(vec![rmip_field::F128::ZERO])],
            ));
            self.replacement = Some(replacement);
        }
        commitment
    }
    fn open(&mut self, c: &[rmip_iop::backend::Claim<Self::Field>], ch: &mut dyn RawChallenger) {
        self.inner.open(c, ch);
        if self.replace_oracle_during_open {
            let mut replacement = self.inner.oracles().to_vec();
            let symbols = std::sync::Arc::make_mut(&mut replacement[0].symbols);
            symbols[0].0[0] = symbols[0].0[0].add(rmip_field::F128::ONE);
            self.replacement = Some(replacement);
        }
    }
    fn query_plan(
        &self,
        ch: &mut dyn RawChallenger,
    ) -> Box<dyn rmip_iop::backend::QueryPlan<Self::Field>> {
        if self.post_late_oracle {
            ch.post_oracle(1234);
        }
        if self.post_duplicate_late_oracle {
            ch.post_oracle(0);
        }
        if self.speak_late {
            ch.record_ordinary(b"a binding message after the cut");
        }
        let n = self.inner.n();
        match self.plan {
            PlanMode::Honest => self.inner.query_plan(ch),
            PlanMode::UnpostedOracle => Box::new(Fixed {
                picked: vec![(4242, Coord::ZERO)],
                next: 0,
            }),
            PlanMode::TooMany => Box::new(Fixed {
                picked: (0..=n).map(|k| (0, Coord::from(k % n))).collect(),
                next: 0,
            }),
            PlanMode::TooFew => Box::new(Fixed {
                picked: (0..n - 5).map(|k| (0, Coord::from(k))).collect(),
                next: 0,
            }),
            PlanMode::OutOfRange => Box::new(Fixed {
                picked: vec![(0, Coord::from(n + 3))],
                next: 0,
            }),
            PlanMode::AdaptiveOnAnswers => Box::new(Adaptive {
                n,
                visited: vec![false; n],
                cursor: 0,
                emitted: 0,
            }),
        }
    }
    fn oracles(&self) -> &[rmip_iop::backend::LongOracle<Self::Field>] {
        self.replacement
            .as_deref()
            .unwrap_or_else(|| self.inner.oracles())
    }
    fn reconstruct(
        &self,
        id: rmip_iop::transcript::OracleId,
        h: &rmip_iop::transcript::TranscriptPrefix,
        coord: rmip_iop::backend::Coord,
    ) -> rmip_iop::backend::Symbol<Self::Field> {
        self.inner.reconstruct(id, h, coord)
    }
    fn verifier(&self) -> Self::Verifier {
        self.inner.verifier()
    }
    fn public_params(&self, trace_len: usize, claims: usize) -> rmip_iop::backend::BackendParams {
        let mut params = self.inner.public_params(trace_len, claims);
        if self.declare_zero_queries {
            params.q_long = 0;
            params.a_long = 0;
        }
        params
    }
    fn params(&self, k: usize) -> rmip_iop::backend::BackendParams {
        let mut params = self.inner.params(k);
        if self.declare_zero_queries {
            params.q_long = 0;
            params.a_long = 0;
        }
        params
    }
}

#[derive(Clone, Copy, Debug)]
enum ParamDriftAt {
    DuringCommitCoin,
    AfterOpen,
    DuringQueries,
}

struct DriftingParams {
    inner: RawTrace,
    changed: std::rc::Rc<std::cell::Cell<bool>>,
    at: ParamDriftAt,
}

struct DriftingPlan {
    inner: Box<dyn rmip_iop::backend::QueryPlan<rmip_field::F128>>,
    changed: std::rc::Rc<std::cell::Cell<bool>>,
}

impl rmip_iop::backend::QueryPlan<rmip_field::F128> for DriftingPlan {
    fn next(
        &mut self,
        prev: Option<&rmip_iop::backend::Symbol<rmip_field::F128>>,
    ) -> Option<(rmip_iop::transcript::OracleId, rmip_iop::backend::Coord)> {
        self.changed.set(true);
        self.inner.next(prev)
    }
}

impl rmip_iop::backend::CommitmentBackend for DriftingParams {
    type Field = rmip_field::F128;
    type Commitment = usize;
    type Verifier = rmip_backend_trace::RawTraceVerifier;

    fn commit(&mut self, z: &[Self::Field], ch: &mut dyn RawChallenger) -> usize {
        self.changed.set(
            matches!(self.at, ParamDriftAt::DuringCommitCoin) && ch.sample().to_u128() & 1 == 1,
        );
        self.inner.commit(z, ch)
    }

    fn open(
        &mut self,
        claims: &[rmip_iop::backend::Claim<Self::Field>],
        ch: &mut dyn RawChallenger,
    ) {
        self.inner.open(claims, ch);
        if matches!(self.at, ParamDriftAt::AfterOpen) {
            self.changed.set(true);
        }
    }

    fn query_plan(
        &self,
        ch: &mut dyn RawChallenger,
    ) -> Box<dyn rmip_iop::backend::QueryPlan<Self::Field>> {
        let plan = self.inner.query_plan(ch);
        if matches!(self.at, ParamDriftAt::DuringQueries) {
            Box::new(DriftingPlan {
                inner: plan,
                changed: self.changed.clone(),
            })
        } else {
            plan
        }
    }

    fn oracles(&self) -> &[rmip_iop::backend::LongOracle<Self::Field>] {
        self.inner.oracles()
    }

    fn reconstruct(
        &self,
        id: rmip_iop::transcript::OracleId,
        h: &rmip_iop::transcript::TranscriptPrefix,
        coord: rmip_iop::backend::Coord,
    ) -> rmip_iop::backend::Symbol<Self::Field> {
        self.inner.reconstruct(id, h, coord)
    }

    fn verifier(&self) -> Self::Verifier {
        self.inner.verifier()
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> rmip_iop::backend::BackendParams {
        self.inner.public_params(trace_len, claims)
    }

    fn params(&self, claims: usize) -> rmip_iop::backend::BackendParams {
        let mut params = self.inner.params(claims);
        if self.changed.get() {
            params.q_long = 0;
            params.a_long = 0;
        }
        params
    }
}

#[test]
fn backend_schedule_is_public_before_commitment_coins() {
    let f = fixture();
    let st = f.statement();
    let mut saw_unchanged = false;
    let mut saw_changed = false;

    for seed in 0..64 {
        let changed = std::rc::Rc::new(std::cell::Cell::new(false));
        let mut backend = DriftingParams {
            inner: RawTrace::new(),
            changed: changed.clone(),
            at: ParamDriftAt::DuringCommitCoin,
        };
        let (execution, verdict) = rmip_iop::driver::execute_recording(
            &st,
            &mut backend,
            &f.oracle(),
            &Config::default(),
            &Cheat::default(),
            &mut SeededChallenger::new(seed),
        );
        assert_eq!(
            execution.occurrences.len(),
            f.layout.n_trace,
            "seed {seed}: the pre-commit q must drive every execution"
        );
        if changed.get() {
            saw_changed = true;
            assert_eq!(
                verdict,
                Err(Reject::BackendParamsChanged),
                "seed {seed}: a commitment coin changed q/A"
            );
        } else {
            saw_unchanged = true;
            assert!(
                verdict.is_ok(),
                "seed {seed}: honest public schedule rejected: {verdict:?}"
            );
        }
        if saw_changed && saw_unchanged {
            break;
        }
    }
    assert!(
        saw_changed && saw_unchanged,
        "test coins did not exercise both branches"
    );
}

#[test]
fn backend_schedule_is_frozen_through_opening_and_the_query_phase() {
    let f = fixture();
    let st = f.statement();

    for at in [ParamDriftAt::AfterOpen, ParamDriftAt::DuringQueries] {
        let mut backend = DriftingParams {
            inner: RawTrace::new(),
            changed: std::rc::Rc::new(std::cell::Cell::new(false)),
            at,
        };
        let (execution, verdict) = rmip_iop::driver::execute_recording(
            &st,
            &mut backend,
            &f.oracle(),
            &Config::default(),
            &Cheat::default(),
            &mut SeededChallenger::new(41),
        );
        assert_eq!(verdict, Err(Reject::BackendParamsChanged), "{at:?}");
        assert_eq!(
            execution.occurrences.len(),
            f.layout.n_trace,
            "{at:?}: the frozen q must drive the rejected execution"
        );
    }
}

struct Fixed {
    picked: Vec<(rmip_iop::transcript::OracleId, rmip_iop::backend::Coord)>,
    next: usize,
}

impl rmip_iop::backend::QueryPlan<rmip_field::F128> for Fixed {
    fn next(
        &mut self,
        _prev: Option<&rmip_iop::backend::Symbol<rmip_field::F128>>,
    ) -> Option<(rmip_iop::transcript::OracleId, rmip_iop::backend::Coord)> {
        let x = *self.picked.get(self.next)?;
        self.next += 1;
        Some(x)
    }
}

fn misbehaving_plan(plan: PlanMode) -> Misbehaving {
    Misbehaving {
        inner: RawTrace::new(),
        post_late_oracle: false,
        post_duplicate_late_oracle: false,
        speak_late: false,
        expose_unregistered_oracle: false,
        replace_oracle_after_later_commit_coin: false,
        replace_oracle_during_open: false,
        declare_zero_queries: false,
        post_wrong_width_symbol: false,
        replacement: None,
        plan,
    }
}

fn run_plan_recording(f: &Fixture, plan: PlanMode) -> (Execution, Result<(), Reject>) {
    let st = f.statement();
    let mut backend = misbehaving_plan(plan);
    let mut ch = SeededChallenger::new(23);
    rmip_iop::driver::execute_recording(
        &st,
        &mut backend,
        &f.oracle(),
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    )
}

fn run_plan(f: &Fixture, plan: PlanMode) -> Result<(), Reject> {
    run_plan_recording(f, plan).1
}

#[test]
fn the_query_phase_guards_fire() {
    let f = fixture();
    let n = f.layout.n_trace;
    assert_eq!(run_plan(&f, PlanMode::Honest), Ok(()));
    assert_eq!(
        run_plan(&f, PlanMode::UnpostedOracle),
        Err(Reject::OracleInconsistent {
            oracle: 4242,
            coord: Box::new(Coord::ZERO)
        })
    );
    match run_plan(&f, PlanMode::TooMany) {
        Err(Reject::QueryCount { expected, got }) => assert!(got > expected),
        other => panic!("expected QueryCount, got {other:?}"),
    }
    match run_plan(&f, PlanMode::TooFew) {
        Err(Reject::QueryCount { expected, got }) => {
            assert!(
                got < expected,
                "expected a SHORT phase, got {got} of {expected}"
            )
        }
        other => panic!("expected QueryCount for a short plan, got {other:?}"),
    }
    assert_eq!(
        run_plan(&f, PlanMode::OutOfRange),
        Err(Reject::OracleInconsistent {
            oracle: 0,
            coord: Box::new(Coord::from(n + 3))
        })
    );

    for mode in [
        PlanMode::UnpostedOracle,
        PlanMode::TooMany,
        PlanMode::TooFew,
        PlanMode::OutOfRange,
    ] {
        let (execution, verdict) = run_plan_recording(&f, mode);
        assert!(verdict.is_err());
        assert_eq!(
            execution.occurrences.len(),
            n,
            "malformed plan changed Step 3's public occurrence count"
        );
    }
}

#[test]
fn a_zero_query_backend_is_rejected_but_still_yields_an_auditable_execution() {
    let f = fixture();
    let st = f.statement();
    let mut backend = Misbehaving {
        declare_zero_queries: true,
        ..misbehaving_plan(PlanMode::Honest)
    };
    let mut challenger = SeededChallenger::new(29);
    let (execution, verdict) = rmip_iop::driver::execute_recording(
        &st,
        &mut backend,
        &f.oracle(),
        &Config::default(),
        &Cheat::default(),
        &mut challenger,
    );
    assert!(
        matches!(verdict, Err(Reject::InadmissibleBackend(message)) if message.contains("q_long = 0"))
    );
    assert_eq!(
        execution.occurrences.len(),
        1,
        "Step 3 must never receive q = 0"
    );
    assert!(!execution.iop_accepted());
}

#[test]
fn replacing_a_posted_oracle_during_open_is_rejected() {
    let f = fixture();
    let st = f.statement();
    let mut backend = Misbehaving {
        replace_oracle_during_open: true,
        ..misbehaving_plan(PlanMode::Honest)
    };
    let mut ch = SeededChallenger::new(23);

    let (execution, verdict) = rmip_iop::driver::execute_recording(
        &st,
        &mut backend,
        &f.oracle(),
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    );

    assert_eq!(
        execution.occurrences.len(),
        f.layout.n_trace,
        "the fixed query phase still runs"
    );
    assert_eq!(
        verdict,
        Err(Reject::OracleMutation(
            rmip_iop::backend::FreezeError::Replaced(0)
        ))
    );
}

#[test]
fn posting_time_snapshot_excludes_later_coins_in_the_same_method() {
    let f = fixture();
    let st = f.statement();
    let mut backend = Misbehaving {
        replace_oracle_after_later_commit_coin: true,
        ..misbehaving_plan(PlanMode::Honest)
    };
    let mut ch = SeededChallenger::new(31);

    let (_, verdict) = rmip_iop::driver::execute_recording(
        &st,
        &mut backend,
        &f.oracle(),
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    );
    assert_eq!(
        verdict,
        Err(Reject::OracleMutation(
            rmip_iop::backend::FreezeError::Replaced(0)
        ))
    );
}

#[test]
fn backend_content_without_a_registered_post_is_rejected_not_panicked() {
    let f = fixture();
    let st = f.statement();
    let mut backend = Misbehaving {
        expose_unregistered_oracle: true,
        ..misbehaving_plan(PlanMode::Honest)
    };
    let mut ch = SeededChallenger::new(37);

    let (_, verdict) = rmip_iop::driver::execute_recording(
        &st,
        &mut backend,
        &f.oracle(),
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    );
    assert_eq!(
        verdict,
        Err(Reject::OracleMutation(
            rmip_iop::backend::FreezeError::Unregistered(4242)
        ))
    );
}

#[test]
fn dummy_padding_does_not_hide_a_short_real_query_plan() {
    let f = fixture();
    let st = f.statement();
    let n = f.layout.n_trace;
    let mut backend = misbehaving_plan(PlanMode::TooFew);
    let mut ch = SeededChallenger::new(23);
    let cfg = Config {
        pad_queries_to: Some(n + 32),
        ..Default::default()
    };
    let (execution, verdict) = rmip_iop::driver::execute_recording(
        &st,
        &mut backend,
        &f.oracle(),
        &cfg,
        &Cheat::default(),
        &mut ch,
    );

    assert_eq!(
        execution.occurrences.len(),
        n + 32,
        "the public audit length still gets padded"
    );
    assert_eq!(
        verdict,
        Err(Reject::QueryCount {
            expected: n,
            got: n - 5
        }),
        "the verdict must compare the real plan against q_long before padding"
    );
}

#[test]
fn an_adaptive_query_plan_is_supported_end_to_end() {
    let f = fixture();
    assert_eq!(run_plan(&f, PlanMode::AdaptiveOnAnswers), Ok(()));
}

fn run_misbehaving(
    f: &Fixture,
    post_late_oracle: bool,
    post_duplicate_late_oracle: bool,
    speak_late: bool,
) -> Result<(), Reject> {
    let st = f.statement();
    let mut backend = Misbehaving {
        post_late_oracle,
        post_duplicate_late_oracle,
        speak_late,
        ..misbehaving_plan(PlanMode::Honest)
    };
    let mut ch = SeededChallenger::new(17);
    execute(
        &st,
        &mut backend,
        &f.oracle(),
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    )
    .map(|_| ())
}

#[test]
fn an_oracle_posted_during_sampling_is_rejected() {
    let f = fixture();
    run_misbehaving(&f, false, false, false).expect("the delegating wrapper is honest");
    assert_eq!(
        run_misbehaving(&f, true, false, false),
        Err(Reject::LateOraclePost { oracle: 1234 })
    );
}

#[test]
fn a_duplicate_oracle_post_during_sampling_is_rejected() {
    let f = fixture();
    assert_eq!(
        run_misbehaving(&f, false, true, false),
        Err(Reject::OracleMutation(
            rmip_iop::backend::FreezeError::DuplicateId(0)
        ))
    );
}

#[test]
fn a_binding_message_during_sampling_is_rejected() {
    let f = fixture();
    assert_eq!(
        run_misbehaving(&f, false, false, true),
        Err(Reject::LateBindingMessage)
    );
}

fn sweep_cheats(f: &Fixture) -> Vec<(&'static str, Cheat)> {
    vec![
        (
            "corrupt_trace_coord",
            Cheat {
                corrupt_trace_coord: Some(f.layout.output_coord()),
                ..Default::default()
            },
        ),
        (
            "wrong_public_input",
            Cheat {
                public_input_bits: vec![0],
                ..Default::default()
            },
        ),
        (
            "trace_swap",
            Cheat {
                trace_swap: true,
                bad_oracle_answer: Some(0),
                ..Default::default()
            },
        ),
        (
            "bad_oracle_answer",
            Cheat {
                bad_oracle_answer: Some(0),
                ..Default::default()
            },
        ),
        (
            "supply_non_boolean_block",
            Cheat {
                supply_non_boolean_block: Some(0x1234_5678),
                ..Default::default()
            },
        ),
    ]
}

#[test]
fn a_rejecting_execution_makes_the_same_occurrences_as_an_accepting_one() {
    use rmip_iop::driver::execute_recording;
    let f = fixture();

    let run = |cheat: &Cheat, seed: u64| {
        let st = f.statement();
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(seed);
        execute_recording(
            &st,
            &mut backend,
            &f.oracle(),
            &Config::default(),
            cheat,
            &mut ch,
        )
    };

    let (honest, verdict) = run(&Cheat::default(), 7);
    verdict.expect("the honest baseline must accept");
    let q = honest.occurrences.len();
    let calls = honest.oracle_calls;
    assert!(q > 0);

    for (name, cheat) in &sweep_cheats(&f) {
        let (e, verdict) = run(cheat, 7);
        assert!(verdict.is_err(), "{name}: expected a rejection");
        assert_eq!(
            e.occurrences.len(),
            q,
            "{name}: rejected with {} occurrences, honest makes {q} -- hypothesis (6.2) \
             requires every execution to make the same number",
            e.occurrences.len()
        );
        assert_eq!(
            e.oracle_calls, calls,
            "{name}: Q_O = R*t must not depend on the verdict"
        );
        assert_eq!(
            e.transcript_at_query_start, e.transcript_at_query_end,
            "{name}"
        );
    }
}

#[test]
fn the_transcript_length_does_not_depend_on_the_verdict() {
    use rmip_iop::driver::execute_recording;
    let f = fixture();
    let run = |cheat: &Cheat| {
        let st = f.statement();
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(11);
        let (e, v) = execute_recording(
            &st,
            &mut backend,
            &f.oracle(),
            &Config::default(),
            cheat,
            &mut ch,
        );
        (e.rounds, e.coins, e.short_bytes, v.is_ok())
    };

    let (r0, c0, s0, ok) = run(&Cheat::default());
    assert!(ok);

    for (name, cheat) in &sweep_cheats(&f) {
        let (r1, c1, s1, ok1) = run(cheat);
        assert!(!ok1, "{name}: must be rejected");
        assert_eq!(
            (r1, c1, s1),
            (r0, c0, s0),
            "{name}: the transcript changed shape on rejection"
        );
    }
}

struct AnyBitSet;

impl Oracle for AnyBitSet {
    fn query(&self, u: &BitSlice) -> bool {
        u.iter().any(|&b| b)
    }
}

fn mixed_width_fixture() -> Fixture {
    use rmip_frontend::circuit::CircuitBuilder;

    let n = 4;
    let mut b = CircuitBuilder::new(n);
    let x: Vec<_> = (0..n).map(|i| b.input(i)).collect();
    let a0 = b.oracle(vec![x[0], x[1]]);
    let a1 = b.oracle(vec![x[1], x[2], x[3]]);
    let a2 = b.oracle(vec![x[0], x[1], x[2]]);
    let t01 = b.xor(a0, a1);
    let out = b.xor(t01, a2);
    let circuit = b.finish(out);

    assert_eq!(
        circuit.b_widths(),
        vec![2, 3, 3],
        "the point of this fixture"
    );

    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);

    let mut xb = rmip_frontend::bits::BitVec::zeros(n);
    xb.set(1, true);
    let (accepted, answers) = circuit.eval_with_answers(xb.as_slice(), &AnyBitSet);
    assert!(accepted, "must be a yes-instance");

    let inst = rmip_frontend::Instance {
        layout: lay.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(l),
        ell: 3,
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x: xb,
        eps_rob: Ratio::new(1, 3),
    };
    Fixture {
        inst,
        sys,
        layout: lay,
        circuit,
        dc,
        oracle: Box::new(AnyBitSet),
        honest_answers: answers.as_slice().to_vec(),
    }
}

#[test]
fn the_claim_padding_absorbs_non_uniform_query_widths() {
    use rmip_iop::driver::execute_recording;
    let f = mixed_width_fixture();
    let cfg = Config {
        t: 4,
        ..Default::default()
    };

    let mut ks = std::collections::HashSet::new();
    let mut pads = std::collections::HashSet::new();
    for seed in 1..=120u64 {
        let st = f.statement();
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(seed);
        let (e, verdict) = execute_recording(
            &st,
            &mut backend,
            &f.oracle(),
            &cfg,
            &Cheat::default(),
            &mut ch,
        );
        verdict.expect("honest execution on a yes-instance");
        ks.insert(e.claims_submitted);
        pads.insert(e.pad_claims);
    }

    assert_eq!(ks.len(), 1, "K varied across executions: {ks:?}");
    assert_eq!(
        *ks.iter().next().unwrap(),
        cfg.k_rel + cfg.t * (f.inst.b_max() + 1)
    );

    assert!(
        pads.len() > 1,
        "the padding did not vary, so this fixture does not exercise it: {pads:?}"
    );
    assert!(*pads.iter().max().unwrap() > 0);
}

#[test]
fn a_wrong_zerocheck_terminal_is_caught_at_the_endpoint() {
    let f = fixture();
    let r = run(
        &f,
        &Cheat {
            wrong_zerocheck_endpoint: true,
            ..Default::default()
        },
        5,
    );
    match r {
        Err(Reject::ZerocheckEndpoint) => {}
        Ok(_) => panic!("accepted a wrong terminal"),
        Err(other) => panic!("expected ZerocheckEndpoint, got {other:?}"),
    }
}

#[test]
fn a_wrong_lincheck_terminal_is_caught_at_the_endpoint() {
    let f = fixture();
    let r = run(
        &f,
        &Cheat {
            wrong_lincheck_endpoint: true,
            ..Default::default()
        },
        5,
    );
    match r {
        Err(Reject::LincheckEndpoint) => {}
        Ok(_) => panic!("accepted a wrong terminal"),
        Err(other) => panic!("expected LincheckEndpoint, got {other:?}"),
    }
}

#[test]
fn a_corrupted_lincheck_round_is_caught_by_the_sumcheck() {
    let f = fixture();
    let r = run(
        &f,
        &Cheat {
            corrupt_lincheck_round: true,
            ..Default::default()
        },
        5,
    );
    match r {
        Err(Reject::Lincheck(_)) => {}
        Ok(_) => panic!("accepted a corrupted lincheck round"),
        Err(other) => panic!("expected Reject::Lincheck, got {other:?}"),
    }
}

#[test]
fn a_non_boolean_answer_coordinate_is_caught() {
    let f = fixture();
    let r = run(
        &f,
        &Cheat {
            non_boolean_answer: Some(0),
            ..Default::default()
        },
        5,
    );
    match r {
        Err(Reject::NonBooleanBlock { occurrence: 0 }) => {}
        Ok(_) => panic!("accepted a non-Boolean answer coordinate"),
        Err(other) => panic!("expected NonBooleanBlock at 0, got {other:?}"),
    }
}

#[test]
fn a_boolean_point_selects_the_table_coordinate_it_names() {
    use rmip_iop::driver::boolean_point;
    use rmip_iop::mle::eval_mle;

    let nu = 4usize;
    let table: Vec<rmip_field::F128> = (0..1u128 << nu).map(rmip_field::F128::new).collect();
    for s in 0..table.len() {
        assert_eq!(
            eval_mle(&table, &boolean_point(s, nu)),
            table[s],
            "boolean_point({s}, {nu}) must select coordinate {s}"
        );
    }
    assert_eq!(boolean_point(1, nu)[0], rmip_field::F128::ZERO);
    assert_eq!(boolean_point(1, nu)[nu - 1], rmip_field::F128::ONE);
}

#[test]
fn a_backend_posting_a_symbol_of_the_wrong_width_is_rejected() {
    let f = fixture();
    let st = f.statement();
    let build = |wrong: bool| Misbehaving {
        post_wrong_width_symbol: wrong,
        ..misbehaving_plan(PlanMode::Honest)
    };

    let mut control = build(false);
    let mut ch = SeededChallenger::new(31);
    execute(
        &st,
        &mut control,
        &f.oracle(),
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    )
    .expect("the control wrapper must be accepted");

    let mut backend = build(true);
    let mut ch = SeededChallenger::new(31);
    let verdict = execute(
        &st,
        &mut backend,
        &f.oracle(),
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    );
    assert_eq!(
        verdict.err(),
        Some(Reject::OracleMutation(
            rmip_iop::backend::FreezeError::SymbolWidth {
                oracle: 7777,
                declared: 2,
                found: 1
            }
        ))
    );
    assert!(
        ch.transcript().try_prefix_at(7777).is_none(),
        "a refused posting must not reach the transcript"
    );
}
