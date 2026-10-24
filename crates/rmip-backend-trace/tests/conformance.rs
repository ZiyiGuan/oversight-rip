use rmip_backend_trace::{RawTrace, TRACE_ORACLE};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{
    Alphabet, BackendParams, BooleanBinding, Claim, CommitmentBackend, Coord, Regime,
};
use rmip_iop::mle::eval_mle;
use rmip_iop::transcript::{RawChallenger, SeededChallenger};

fn trace(n: usize) -> Vec<F128> {
    (0..n)
        .map(|i| F128::from_bool((i * 7 + 3) % 5 < 2))
        .collect()
}

fn claims(z: &[F128]) -> Vec<Claim<F128>> {
    let m = z.len().trailing_zeros() as usize;
    let point: Vec<F128> = (0..m).map(|k| F128::new(k as u128 * 31 + 5)).collect();
    let value = eval_mle(z, &point);
    vec![Claim { point, value }]
}

#[test]
fn the_raw_trace_conforms() {
    let z = trace(64);
    let mut b = RawTrace::new();
    let v = rmip_backend_conformance::check(&mut b, &z, &claims(&z), false);
    assert!(v.is_empty(), "conformance violations: {v:#?}");
}

#[test]
fn the_raw_trace_is_deterministic() {
    let z = trace(64);
    let v = rmip_backend_conformance::check_deterministic(RawTrace::new, &z, &claims(&z));
    assert!(v.is_empty(), "{v:#?}");
}

#[test]
fn raw_backend_parameters_are_assumption_free() {
    let z = trace(64);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(1);
    b.commit(&z, &mut ch);
    let p: BackendParams = b.params(20);
    assert_eq!(p.q_long, 64, "q_long = N");
    assert_eq!(p.eps_bind, 0.0, "exactly zero, not negligible");
    assert!(p.eps_open >= rmip_iop::batch::error(20, 6));
    assert_eq!(p.regime, Regime::UniqueDecoding);
    assert_eq!(p.boolean_binding, BooleanBinding::Native);
    assert_eq!(p.alphabet, vec![Alphabet::Field]);
    assert_eq!(p.l_long, Coord::from(64u64));
    assert_eq!(p.a_long, 64);
    assert_eq!(
        p.ordinary_bytes, 0,
        "no opening argument, so no ordinary message"
    );
    assert!(b.params(1).eps_open < b.params(1000).eps_open);
}

#[test]
fn reading_every_coordinate_recovers_the_trace() {
    let z = trace(32);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(2);
    b.commit(&z, &mut ch);
    let h = ch.transcript().prefix_at(TRACE_ORACLE);

    let mut plan = b.query_plan(&mut ch);
    let mut read = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        read.push(b.reconstruct(id, &h, coord).0[0]);
    }
    assert_eq!(read, z);

    let c = &claims(&z)[0];
    assert_eq!(
        eval_mle(&read, &c.point),
        c.value,
        "the claim checks out against what was read"
    );
}

#[test]
fn reconstruct_is_a_function_of_the_triple() {
    let z = trace(16);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(3);
    b.commit(&z, &mut ch);
    let h = ch.transcript().prefix_at(TRACE_ORACLE);

    for coord in 0..z.len() {
        let first = b.reconstruct(TRACE_ORACLE, &h, Coord::from(coord));
        for _ in 0..4 {
            assert_eq!(b.reconstruct(TRACE_ORACLE, &h, Coord::from(coord)), first);
        }
    }
    let mut ch2 = SeededChallenger::new(99);
    ch2.record_ordinary(b"unrelated");
    ch2.post_oracle(TRACE_ORACLE);
    let h2 = ch2.transcript().prefix_at(TRACE_ORACLE);
    assert_ne!(h.as_bytes(), h2.as_bytes(), "the prefixes really do differ");
    for coord in 0..z.len() {
        assert_eq!(
            b.reconstruct(TRACE_ORACLE, &h, Coord::from(coord)),
            b.reconstruct(TRACE_ORACLE, &h2, Coord::from(coord))
        );
    }
}

#[test]
fn admissibility_gate_rejects_the_bad_combinations() {
    let z = trace(16);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(4);
    b.commit(&z, &mut ch);
    let base = b.params(4);
    assert!(base.check_admissible(false).is_ok());

    let mut needs_rows = base.clone();
    needs_rows.boolean_binding = BooleanBinding::RequiresExplicitRows;
    assert!(
        needs_rows.check_admissible(false).is_err(),
        "must demand the rows"
    );
    assert!(
        needs_rows.check_admissible(true).is_ok(),
        "satisfied once the rows are there"
    );

    let mut unresolved = base.clone();
    unresolved.regime = Regime::JohnsonOod { ood_samples: 0 };
    assert!(
        unresolved.check_admissible(true).is_err(),
        "list decoding with no selection"
    );
}

#[test]
fn opening_verifier_uses_coordinates_not_positions() {
    use rmip_iop::backend::{OpeningVerifier, Symbol};
    let z = trace(16);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(8);
    b.commit(&z, &mut ch);
    let v = b.verifier();
    let cs = claims(&z);

    let lifted = z.clone();
    let in_order: Vec<_> = (0..z.len())
        .map(|i| (0usize, Coord::from(i), Symbol(vec![lifted[i]])))
        .collect();
    assert!(v.check(&cs, &in_order), "in-order answers must verify");

    let mut shuffled = in_order.clone();
    shuffled.reverse();
    assert!(
        v.check(&cs, &shuffled),
        "order must not matter once coords are carried"
    );

    let mut dup = in_order.clone();
    dup[3] = dup[2].clone();
    assert!(!v.check(&cs, &dup), "duplicate coordinate must be refused");

    let mut permuted_values: Vec<_> = in_order.clone();
    permuted_values.swap(0, 1);
    let relabelled: Vec<_> = permuted_values
        .iter()
        .enumerate()
        .map(|(i, (id, _, s))| (*id, Coord::from(i), s.clone()))
        .collect();
    assert!(!v.check(&cs, &relabelled), "mislabelled values must fail");
}

#[test]
fn a_commit_only_instance_reconstructs_every_posted_oracle() {
    let z = trace(64);
    let v =
        rmip_backend_conformance::check_prefix_only_reconstruction(RawTrace::new, &z, &claims(&z));
    assert!(v.is_empty(), "prefix-only reconstruction failed: {v:#?}");
}

#[test]
fn reconstruct_agrees_with_the_recorded_posting_prefix() {
    let z = trace(64);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(7);
    b.commit(&z, &mut ch);
    b.open(&claims(&z), &mut ch);

    let h = b
        .posting_prefix()
        .expect("commit must record the posting prefix")
        .clone();

    let mut och = SeededChallenger::new(999);
    och.record_ordinary(b"a message that no RawTrace execution sends");
    och.sample();
    och.post_oracle(TRACE_ORACLE);
    let h_other = och.transcript().prefix_at(TRACE_ORACLE);
    assert_ne!(
        h.as_bytes(),
        h_other.as_bytes(),
        "the two prefixes really do differ, or the assertion below is vacuous"
    );

    let posted = &b.oracles()[0];
    assert_eq!(posted.symbols.len(), z.len());
    for coord in 0..z.len() {
        let at_real = b.reconstruct(TRACE_ORACLE, &h, Coord::from(coord));
        assert_eq!(at_real, posted.symbols[coord], "coord {coord}");
        assert_eq!(
            b.reconstruct(TRACE_ORACLE, &h_other, Coord::from(coord)),
            at_real,
            "this oracle depends on no coin, so reconstruction must be constant in h"
        );
    }
}

#[test]
#[should_panic(expected = "non-empty power-of-two trace")]
fn public_params_rejects_a_zero_length() {
    let b = RawTrace::new();
    let _ = b.public_params(0, 1);
}

#[test]
#[should_panic(expected = "non-empty power-of-two trace")]
fn public_params_rejects_a_non_power_of_two_length() {
    let b = RawTrace::new();
    let _ = b.public_params(12, 1);
}

#[test]
fn the_two_nu_derivations_agree() {
    use rmip_iop::backend::{composed_opening_error, OpeningVerifier, Symbol};

    let z = trace(64);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(11);
    b.commit(&z, &mut ch);

    for k in [0usize, 1, 20] {
        let expected = composed_opening_error(0.0, k, 6);
        assert_eq!(b.params(k).eps_open, expected, "params at {k} claims");
        assert_eq!(
            b.public_params(z.len(), k).eps_open,
            expected,
            "public_params at {k} claims"
        );
    }

    let verifier = b.verifier();
    let answers: Vec<_> = (0..z.len())
        .map(|c| (TRACE_ORACLE, Coord::from(c), Symbol(vec![z[c]])))
        .collect();
    let good = claims(&z);
    assert_eq!(good[0].point.len(), 6, "the fixture claim has arity nu = 6");
    assert!(verifier.check(&good, &answers));
    let short = vec![Claim {
        point: good[0].point[..5].to_vec(),
        value: good[0].value,
    }];
    assert!(
        !verifier.check(&short, &answers),
        "an arity-5 claim must fail the nu filter at N = 64"
    );
}

#[test]
fn malformed_public_opening_messages_are_rejected_without_panicking() {
    use rmip_iop::backend::{OpeningVerifier, Symbol};

    let verifier = rmip_backend_trace::RawTraceVerifier::from_public_length(1);
    let claim = Claim {
        point: Vec::new(),
        value: F128::ZERO,
    };
    assert!(!verifier.check(&[claim], &[(TRACE_ORACLE, Coord::ZERO, Symbol(Vec::new()))]));
}
