use rmip_backend_trace::RawTrace;
use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance};
use rmip_iop::backend::{CommitmentBackend, Coord, Symbol};
use rmip_iop::driver::{execute, Cheat, Config, Statement};
use rmip_iop::transcript::{Entry, OsChallenger, RawChallenger, SeededChallenger};
use rmip_mip::{
    audit_one, one_shot_bound, repeated_bound, repetitions_for, Audit, PCheck, SharedAdvice,
};

const VIEWS: usize = 16;

struct Fx {
    inst: Instance,
    sys: r1cs::R1cs,
    layout: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    z: Vec<rmip_field::F128>,
    oracle: SparseTableOracle,
}

fn fx() -> Fx {
    let m = MnistInstance::new(VIEWS);
    let circuit = m.build();
    let oracle = m.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);
    let x = m.yes();
    assert!(
        circuit.eval(x.as_slice(), &oracle),
        "the true label must satisfy C^O"
    );
    let (_, a) = circuit.eval_with_answers(x.as_slice(), &oracle);
    let z = rmip_frontend::trace::witness_trace(&dc, &lay, x.as_slice(), a.as_slice());
    let inst = Instance {
        layout: lay.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(l),
        ell: m.ell(),
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: m.eps_rob,
    };
    Fx {
        inst,
        sys,
        layout: lay,
        circuit,
        dc,
        z: rmip_iop::mle::lift(z.as_slice()),
        oracle,
    }
}

fn idealized_s_iop(f: &Fx, cfg: &Config) -> f64 {
    let eps = f.inst.eps_rob.num as f64 / f.inst.eps_rob.den as f64;
    (1.0 - eps).powi(i32::try_from(cfg.t).expect("t fits i32"))
}

fn make_checker(f: &Fx, advice: SharedAdvice) -> PCheck<RawTrace> {
    let mut b = RawTrace::new();
    let mut own_ch = SeededChallenger::new(advice.seed);
    b.commit(&f.z, &mut own_ch);
    PCheck::new(b)
}

fn st(f: &Fx) -> Statement<'_> {
    Statement {
        inst: &f.inst,
        sys: &f.sys,
        layout: &f.layout,
        circuit: &f.circuit,
        dc: &f.dc,
        holo: None,
    }
}

#[test]
fn honest_two_prover_execution_accepts() {
    let f = fx();
    let advice = SharedAdvice { seed: 0xA11CE };
    let checker = make_checker(&f, advice);

    for seed in 1..=200u64 {
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(seed);
        let e = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch,
        )
        .expect("IOP accepts");
        let a = audit_one(&e, &checker, &mut ch);
        assert!(a.accepted(), "audit failed at seed {seed}: {a:?}");
    }
}

#[test]
fn a_consistent_audit_cannot_reverse_the_underlying_iop_verdict() {
    let f = fx();
    let checker = make_checker(&f, SharedAdvice { seed: 0xA11CE });
    let mut backend = RawTrace::new();
    let mut ch = SeededChallenger::new(77);
    let (execution, verdict) = rmip_iop::driver::execute_recording(
        &st(&f),
        &mut backend,
        &f.oracle,
        &Config::default(),
        &Cheat {
            bad_oracle_answer: Some(0),
            ..Default::default()
        },
        &mut ch,
    );
    assert!(
        verdict.is_err(),
        "the local oracle mismatch must reject the IOP"
    );
    assert!(!execution.iop_accepted());

    let audit = audit_one(&execution, &checker, &mut ch);
    assert!(
        audit.consistent(),
        "the oracle answers should agree: {audit:?}"
    );
    assert!(
        !audit.accepted(),
        "consistency alone must not accept the MIP"
    );
    assert!(matches!(audit, Audit::UnderlyingRejected { .. }));
}

#[test]
fn os_csprng_control_catches_an_off_oracle_answer_at_the_ideal_rate() {
    let f = fx();
    let checker = make_checker(&f, SharedAdvice { seed: 7 });

    let trials = 4000;
    let mut caught = 0;
    let mut q_long = None;
    for trial in 0..trials {
        let mut backend = RawTrace::new();
        let mut ch = OsChallenger::new();
        let mut e = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch,
        )
        .unwrap();
        let q = e.occurrences.len();
        assert_eq!(*q_long.get_or_insert(q), q, "q_long moved between trials");
        let victim = trial % q;
        let old = e.occurrences[victim].answer.0[0];
        e.occurrences[victim].answer = Symbol(vec![old.add(F128::ONE)]);

        if let Audit::Inconsistent { .. } = audit_one(&e, &checker, &mut ch) {
            caught += 1;
        }
    }
    let rate = caught as f64 / trials as f64;
    let q = q_long.expect("at least one trial ran") as f64;
    let expected = 1.0 / q;
    let sigma = (expected * (1.0 - expected) / trials as f64).sqrt();
    assert!(
        (rate - expected).abs() < 5.0 * sigma,
        "OS-CSPRNG catch rate {rate:.4}, ideal {expected:.4} = 1/q_long \
         (5 sigma = {:.4})",
        5.0 * sigma
    );
}

#[test]
fn shipped_raw_trace_checker_is_stable_on_the_same_triple_across_runs() {
    let f = fx();
    let checker = make_checker(&f, SharedAdvice { seed: 3 });
    let mut seen: std::collections::HashMap<(usize, Vec<u8>, Coord), Symbol<F128>> =
        std::collections::HashMap::new();

    for seed in 1..=50u64 {
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(seed);
        let e = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch,
        )
        .unwrap();
        for o in &e.occurrences {
            let key = (o.oracle, o.prefix.as_bytes().to_vec(), o.coord);
            let ans = checker.answer(o.oracle, &o.prefix, o.coord);
            if let Some(prev) = seen.get(&key) {
                assert_eq!(
                    prev, &ans,
                    "same triple, different answer -- F* is not a function"
                );
            } else {
                seen.insert(key, ans);
            }
        }
    }
    assert!(seen.len() > 1);
}

#[test]
fn honest_checker_answer_is_stable_on_the_same_triple() {
    let f = fx();
    let checker = make_checker(&f, SharedAdvice { seed: 9 });
    let mut backend = RawTrace::new();
    let mut ch = SeededChallenger::new(4);
    let e = execute(
        &st(&f),
        &mut backend,
        &f.oracle,
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    )
    .unwrap();

    let o = &e.occurrences[0];
    let baseline = checker.answer(o.oracle, &o.prefix, o.coord);
    for k in [7usize, 3, 63, 0, 31] {
        let other = &e.occurrences[k % e.occurrences.len()];
        let _ = checker.answer(other.oracle, &other.prefix, other.coord);
        assert_eq!(checker.answer(o.oracle, &o.prefix, o.coord), baseline);
    }
}

#[test]
fn checker_behind_a_channel() {
    use std::sync::mpsc;
    use std::thread;

    let f = fx();
    let checker = make_checker(&f, SharedAdvice { seed: 11 });
    let (to_checker, rx) =
        mpsc::channel::<(usize, rmip_iop::transcript::TranscriptPrefix, Coord)>();
    let (tx_back, from_checker) = mpsc::channel::<Symbol<F128>>();

    let handle = thread::spawn(move || {
        while let Ok((i, h, theta)) = rx.recv() {
            let a = checker.answer(i, &h, theta);
            if tx_back.send(a).is_err() {
                break;
            }
        }
    });

    let mut backend = RawTrace::new();
    let mut ch = SeededChallenger::new(21);
    let e = execute(
        &st(&f),
        &mut backend,
        &f.oracle,
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    )
    .unwrap();
    let j = rmip_iop::sampling::uniform_index(&mut ch, e.occurrences.len());
    let o = &e.occurrences[j];

    to_checker
        .send((o.oracle, o.prefix.clone(), o.coord))
        .unwrap();
    let b_j = from_checker.recv().unwrap();
    assert_eq!(b_j, o.answer, "consistency check across the channel");

    drop(to_checker);
    handle.join().unwrap();
}

#[test]
fn idealized_amplification_schedule_runs() {
    let f = fx();
    let checker = make_checker(&f, SharedAdvice { seed: 13 });
    let cfg = Config::default();

    let s_iop = idealized_s_iop(&f, &cfg);
    let q = f.layout.n_trace;
    assert!((one_shot_bound(s_iop, q) - (1.0 - (1.0 - s_iop) / q as f64)).abs() < 1e-12);
    let sigma = 2f64.powi(-20);
    let r_needed = repetitions_for(sigma, s_iop, q);
    assert!(
        repeated_bound(s_iop, q, r_needed) <= sigma,
        "R = {r_needed} must drive (D) to sigma"
    );
    assert!(
        r_needed == 1 || repeated_bound(s_iop, q, r_needed - 1) > sigma,
        "R = {r_needed} must be the least such count"
    );

    let r = 50;
    let mut accepted = true;
    let mut oracle_calls = 0;
    for i in 0..r {
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(1000 + i as u64);
        match execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &cfg,
            &Cheat::default(),
            &mut ch,
        ) {
            Ok(e) => {
                assert_eq!(
                    e.q_long(),
                    q,
                    "the raw-trace q_long is N on every repetition"
                );
                oracle_calls += e.oracle_calls;
                if !audit_one(&e, &checker, &mut ch).accepted() {
                    accepted = false;
                    break;
                }
            }
            Err(_) => {
                accepted = false;
                break;
            }
        }
    }
    assert!(accepted, "honest provers must survive every repetition");
    assert_eq!(oracle_calls, r * cfg.t);
}

#[test]
fn the_audit_accepts_when_it_lands_on_a_padding_dummy() {
    use rmip_iop::driver::DUMMY_ORACLE;
    let f = fx();
    let checker = make_checker(&f, SharedAdvice { seed: 17 });
    let n = f.layout.n_trace;
    let pad = 4 * n;
    let cfg = Config {
        pad_queries_to: Some(pad),
        ..Default::default()
    };
    let s_iop = idealized_s_iop(&f, &cfg);

    let mut landed_on_dummy = 0;
    let mut landed_on_real = 0;
    for trial in 0..400usize {
        let mut backend = RawTrace::new();
        let mut ch = OsChallenger::new();
        let e = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &cfg,
            &Cheat::default(),
            &mut ch,
        )
        .unwrap();
        assert_eq!(e.occurrences.len(), pad);
        assert_eq!(e.q_long(), pad);
        assert_eq!(
            e.a_long(),
            pad as u64,
            "{n} real Field answers plus {} dummy Field answers",
            pad - n
        );
        assert_eq!(cfg.audit_query_count(n), pad);
        assert_eq!(cfg.audit_answer_volume(n, n as u64), pad as u64);
        assert!(
            one_shot_bound(s_iop, e.q_long()) > one_shot_bound(s_iop, n),
            "padding changes the Step-3 denominator and must worsen its bound"
        );

        let before = ch.transcript().entries().len();
        let audit = audit_one(&e, &checker, &mut ch);
        let drawn = &ch.transcript().entries()[before..];
        assert_eq!(
            drawn.len(),
            1,
            "the audit must draw exactly one coin at q_long = 256, not {}",
            drawn.len()
        );
        assert!(
            matches!(drawn[0], Entry::Coin(_)),
            "the audit's only transcript entry must be the index coin"
        );
        assert!(
            audit.accepted(),
            "honest execution must survive the audit at trial {trial}"
        );
        match audit {
            Audit::Accepted { audited } => {
                if e.occurrences[audited].oracle == DUMMY_ORACLE {
                    landed_on_dummy += 1;
                } else {
                    landed_on_real += 1;
                }
            }
            Audit::UnderlyingRejected { .. } | Audit::Inconsistent { .. } => unreachable!(),
        }
    }
    assert!(
        landed_on_dummy > 100,
        "only {landed_on_dummy} audits hit a dummy"
    );
    assert!(
        landed_on_real > 30,
        "only {landed_on_real} audits hit a real oracle"
    );
}

use rmip_backend_recursive::{FoldLadder, Shape as LadderShape};

fn make_ladder_checker(f: &Fx, advice: SharedAdvice) -> PCheck<FoldLadder> {
    let mut b = FoldLadder::new(LadderShape::balanced(f.layout.nu, 4, 4.0));
    let mut own_ch = SeededChallenger::new(advice.seed);
    b.commit(&f.z, &mut own_ch);
    PCheck::new(b)
}

#[test]
fn the_checker_reconstructs_folding_oracles_in_a_full_two_prover_run() {
    let f = fx();
    let advice = SharedAdvice { seed: 0xF01D };
    let checker = make_ladder_checker(&f, advice);

    let mut seen_level1 = 0usize;
    for seed in 1..=120u64 {
        let mut backend = FoldLadder::new(LadderShape::balanced(f.layout.nu, 4, 4.0));
        let mut ch = SeededChallenger::new(seed);
        let e = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch,
        )
        .expect("IOP accepts an honest ladder execution");
        assert_eq!(backend.oracles().len(), 2);
        seen_level1 += e.occurrences.iter().filter(|o| o.oracle == 1).count();

        let a = audit_one(&e, &checker, &mut ch);
        assert!(
            a.accepted(),
            "seed {seed}: the audit rejected an honest run -- the checker could not \
             reproduce a folding oracle from its legal prefix"
        );
    }
    assert!(seen_level1 > 0, "no level-1 occurrence was ever produced");
}

#[test]
fn a_ladder_checker_on_the_wrong_trace_is_caught() {
    let f = fx();
    let advice = SharedAdvice { seed: 0xF01D };
    let wrong = {
        let mut z = f.z.clone();
        z[3] = z[3].add(F128::ONE);
        let mut b = FoldLadder::new(LadderShape::balanced(f.layout.nu, 4, 4.0));
        let mut own = SeededChallenger::new(advice.seed);
        b.commit(&z, &mut own);
        PCheck::new(b)
    };

    let mut disagreed = 0;
    for seed in 1..=60u64 {
        let mut backend = FoldLadder::new(LadderShape::balanced(f.layout.nu, 4, 4.0));
        let mut ch = SeededChallenger::new(seed);
        let Ok(e) = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch,
        ) else {
            continue;
        };
        if !audit_one(&e, &wrong, &mut ch).accepted() {
            disagreed += 1;
        }
    }
    assert!(
        disagreed > 0,
        "a checker on the wrong trace was never caught"
    );
}

fn ladder_execution(seed: u64) -> (rmip_iop::driver::Execution, PCheck<FoldLadder>) {
    let f = fx();
    let checker = make_ladder_checker(&f, SharedAdvice { seed: 0xF01D });
    let mut backend = FoldLadder::new(LadderShape::balanced(f.layout.nu, 4, 4.0));
    let mut ch = SeededChallenger::new(seed);
    let e = execute(
        &st(&f),
        &mut backend,
        &f.oracle,
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    )
    .expect("IOP accepts an honest ladder execution");
    (e, checker)
}

#[test]
fn the_drivers_recorded_prefix_passes_the_checkers_legality_check() {
    let (e, checker) = ladder_execution(7);
    let mut seen_level1 = 0usize;
    for (k, o) in e.occurrences.iter().enumerate() {
        let got = checker
            .try_answer(o.oracle, &o.prefix, o.coord)
            .unwrap_or_else(|err| panic!("occurrence {k} was refused: {err}"));
        assert_eq!(&got, &o.answer, "occurrence {k}");
        if o.oracle == 1 {
            seen_level1 += 1;
        }
    }
    assert!(seen_level1 > 0, "no level-1 occurrence in this execution");
}

#[test]
fn a_prefix_taken_after_the_post_is_refused() {
    let (e, checker) = ladder_execution(7);
    let o0 = e
        .occurrences
        .iter()
        .find(|o| o.oracle == 0)
        .expect("the ladder posts a level-0 oracle");
    let h1 = e
        .occurrences
        .iter()
        .find(|o| o.oracle == 1)
        .map(|o| o.prefix.clone())
        .expect("the ladder posts a level-1 oracle");
    assert_ne!(
        o0.prefix, h1,
        "the two prefixes must differ or this test forges nothing"
    );

    assert!(
        checker.try_answer(0, &o0.prefix, o0.coord).is_ok(),
        "the recorded prefix must still be answered"
    );
    assert_eq!(
        checker.try_answer(0, &h1, o0.coord),
        Err(rmip_mip::IllegalPrefix { oracle: 0 }),
        "a prefix taken after the post must be refused"
    );
}

#[test]
fn a_checker_that_panics_is_a_reject_not_a_crash() {
    let (honest, checker) = ladder_execution(11);
    let mut ch = OsChallenger::new();
    assert!(audit_one(&honest, &checker, &mut ch).accepted());

    let (mut foreign, _) = ladder_execution(11);
    for o in &mut foreign.occurrences {
        o.oracle = 999;
    }
    let verdict = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        audit_one(&foreign, &checker, &mut OsChallenger::new())
    }))
    .expect("a panicking prover 2 must not unwind out of the audit");
    assert!(matches!(verdict, Audit::Inconsistent { .. }), "{verdict:?}");
    assert!(!verdict.accepted());

    let late = honest
        .occurrences
        .iter()
        .find(|o| o.oracle == 1)
        .map(|o| o.prefix.clone())
        .expect("the ladder posts a level-1 oracle");
    let (mut refused, _) = ladder_execution(11);
    for o in &mut refused.occurrences {
        o.oracle = 0;
        o.prefix = late.clone();
    }
    let verdict = audit_one(&refused, &checker, &mut OsChallenger::new());
    assert!(matches!(verdict, Audit::Inconsistent { .. }), "{verdict:?}");
}

#[test]
#[should_panic(expected = "not oracle 0's posting prefix")]
fn answer_panics_on_a_prefix_taken_after_the_post() {
    let (e, checker) = ladder_execution(7);
    let h1 = e
        .occurrences
        .iter()
        .find(|o| o.oracle == 1)
        .map(|o| o.prefix.clone())
        .expect("the ladder posts a level-1 oracle");
    let _ = checker.answer(0, &h1, Coord::from(0u64));
}

fn xs_step(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn next_coin_after(c: F128) -> F128 {
    let lo = (c.to_u128() & u64::MAX as u128) as u64;
    let hi2 = xs_step(lo);
    let lo2 = xs_step(hi2);
    F128::new(((hi2 as u128) << 64) | lo2 as u128)
}

fn predicted_uniform_index(mut coin: F128, upper: usize) -> usize {
    let upper = upper as u128;
    let threshold = upper.wrapping_neg() % upper;
    loop {
        if coin.to_u128() >= threshold {
            return (coin.to_u128() % upper) as usize;
        }
        coin = next_coin_after(coin);
    }
}

fn coin_count(ch: &dyn RawChallenger) -> usize {
    ch.transcript()
        .entries()
        .iter()
        .filter(|e| matches!(e, Entry::Coin(_)))
        .count()
}

fn collusion_success<C: RawChallenger, F: Fn(u64) -> C>(
    trials: u64,
    make_ch: F,
) -> (usize, usize, usize) {
    let f = fx();
    let shape = LadderShape::balanced(f.layout.nu, 4, 4.0);
    let advice = SharedAdvice { seed: 0xF01D };
    let checker = {
        let mut b = FoldLadder::new(shape);
        let mut own = SeededChallenger::new(advice.seed);
        b.commit(&f.z, &mut own);
        PCheck::new(b)
    };

    let mut evaded = 0usize;
    let mut ran = 0usize;
    let mut q_long = None;
    for seed in 1..=trials {
        let mut backend = FoldLadder::new(shape);
        let mut ch = make_ch(seed);
        let Ok(mut e) = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch,
        ) else {
            continue;
        };
        ran += 1;

        let h = e
            .occurrences
            .iter()
            .find(|o| o.oracle == 1)
            .map(|o| o.prefix.clone())
            .expect("the ladder posts a level-1 oracle");
        let h_coins = h.coins().expect("a legal prefix decodes");
        let total = coin_count(&ch);

        let mut c = *h_coins.last().unwrap();
        for _ in 0..(total - (h_coins.len() - 1)) {
            c = next_coin_after(c);
        }
        let q = e.occurrences.len();
        assert_eq!(*q_long.get_or_insert(q), q, "q_long moved between runs");
        let predicted_j = predicted_uniform_index(c, q);

        let victim = (predicted_j + 1) % q;
        let mut bad = e.occurrences[victim].answer.0.clone();
        bad[0] = bad[0].add(F128::ONE);
        e.occurrences[victim].answer = Symbol(bad);

        if audit_one(&e, &checker, &mut ch).accepted() {
            evaded += 1;
        }
    }
    (evaded, ran, q_long.unwrap_or(0))
}

#[test]
fn a_forward_predictable_challenger_lets_the_provers_defeat_the_audit() {
    let (evaded, ran, _q) = collusion_success(300, SeededChallenger::new);
    assert!(ran > 0);
    let rate = evaded as f64 / ran as f64;
    assert!(
        rate > 0.98,
        "predicting the audit index should make a different corrupted occurrence \
         evade essentially always; got evasion {rate:.4} over {ran} runs"
    );
}

#[test]
fn os_csprng_control_matches_the_unpredictable_audit_rate() {
    let (evaded, ran, q) = collusion_success(600, |_| OsChallenger::new());
    assert!(ran > 0);
    let rate = evaded as f64 / ran as f64;
    let expected = 1.0 - 1.0 / q as f64;
    let sigma = (expected * (1.0 - expected) / ran as f64).sqrt();
    assert!(
        (rate - expected).abs() < 4.0 * sigma,
        "OS-CSPRNG control differs from the ideal unpredictable-coin rate: evasion {rate:.4}, \
         expected 1 - 1/q_long = {expected:.4} (4 sigma = {:.4}) over {ran} runs",
        4.0 * sigma
    );
}

use std::cell::Cell;

struct OccurrenceAware {
    inner: RawTrace,
    calls: Cell<usize>,
}

impl rmip_iop::backend::CommitmentBackend for OccurrenceAware {
    type Field = F128;
    type Commitment = usize;
    type Verifier = rmip_backend_trace::RawTraceVerifier;
    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
        self.inner.commit(z, ch)
    }
    fn open(&mut self, c: &[rmip_iop::backend::Claim<F128>], ch: &mut dyn RawChallenger) {
        self.inner.open(c, ch)
    }
    fn query_plan(
        &self,
        ch: &mut dyn RawChallenger,
    ) -> Box<dyn rmip_iop::backend::QueryPlan<F128>> {
        self.inner.query_plan(ch)
    }
    fn oracles(&self) -> &[rmip_iop::backend::LongOracle<F128>] {
        self.inner.oracles()
    }
    fn reconstruct(
        &self,
        id: rmip_iop::transcript::OracleId,
        h: &rmip_iop::transcript::TranscriptPrefix,
        coord: rmip_iop::backend::Coord,
    ) -> Symbol<F128> {
        let n = self.calls.get();
        self.calls.set(n + 1);
        let mut s = self.inner.reconstruct(id, h, coord);
        if n % 2 == 1 {
            s.0[0] = s.0[0].add(F128::ONE);
        }
        s
    }
    fn verifier(&self) -> Self::Verifier {
        self.inner.verifier()
    }
    fn public_params(&self, trace_len: usize, claims: usize) -> rmip_iop::backend::BackendParams {
        self.inner.public_params(trace_len, claims)
    }
    fn params(&self, k: usize) -> rmip_iop::backend::BackendParams {
        self.inner.params(k)
    }
}

#[test]
fn an_occurrence_aware_checker_is_constructible_after_all() {
    let f = fx();
    let mut b = OccurrenceAware {
        inner: RawTrace::new(),
        calls: Cell::new(0),
    };
    let mut own = SeededChallenger::new(1);
    b.commit(&f.z, &mut own);
    let h = own.transcript().prefix_at(0);

    let a = b.reconstruct(0, &h, Coord::from(3u64));
    let c = b.reconstruct(0, &h, Coord::from(3u64));
    assert_ne!(
        a, c,
        "the same (i, h, theta) must differ, or this fixture is inert"
    );
}

#[test]
fn the_audit_asks_the_checker_exactly_one_question() {
    use rmip_iop::driver::DUMMY_ORACLE;
    let f = fx();
    let advice = SharedAdvice { seed: 0xA11CE };
    let inner = {
        let mut r = RawTrace::new();
        let mut c = SeededChallenger::new(advice.seed);
        r.commit(&f.z, &mut c);
        r
    };
    let checker = PCheck::new(OccurrenceAware {
        inner,
        calls: Cell::new(0),
    });

    let cfg = Config {
        pad_queries_to: Some(4 * f.layout.n_trace),
        ..Default::default()
    };
    const RUNS: u64 = 25;
    let mut landed_on_dummy = 0usize;
    for seed in 1..=RUNS {
        let mut backend = RawTrace::new();
        let mut ch = SeededChallenger::new(seed);
        let e = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &cfg,
            &Cheat::default(),
            &mut ch,
        )
        .expect("honest execution");
        let (Audit::Accepted { audited }
        | Audit::UnderlyingRejected { audited }
        | Audit::Inconsistent { audited }) = audit_one(&e, &checker, &mut ch);
        if e.occurrences[audited].oracle == DUMMY_ORACLE {
            landed_on_dummy += 1;
        }
    }
    assert_eq!(
        checker.questions_asked(),
        RUNS as usize,
        "audit_one must ask the checker exactly once per execution; a stateful \
         checker is harmless only because of that"
    );
    assert!(
        landed_on_dummy > 0,
        "no audit landed on a padding dummy, so the dummy branch was not counted"
    );
    assert_eq!(
        checker.backend().calls.get(),
        RUNS as usize - landed_on_dummy,
        "a counter inside the backend misses exactly the dummy questions"
    );
}
