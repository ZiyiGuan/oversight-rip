use rmip_demo::party_metrics::TrackingAllocator;
use rmip_iop::metrics::{enter, Role, Session};

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn real_vectors_keep_their_allocating_owner_until_freed() {
    let session = Session::start();
    let mut payload = {
        let _prover = enter(Role::MainProver);
        vec![0_u8; 1024]
    };
    assert_eq!(
        session.report().main_prover.live_heap_bytes,
        payload.capacity()
    );
    {
        let _verifier = enter(Role::Verifier);
        payload.reserve_exact(1024);
    }
    let grown = payload.capacity();
    assert_eq!(session.report().main_prover.live_heap_bytes, grown);
    assert_eq!(session.report().main_prover.peak_heap_bytes, grown);
    assert_eq!(session.report().verifier.peak_heap_bytes, 0);
    drop(payload);
    let report = session.finish();
    assert!(report.heap_tracking);
    assert_eq!(report.main_prover.live_heap_bytes, 0);
}

#[test]
fn independent_sessions_do_not_count_each_others_retained_vectors() {
    let first = Session::start();
    let retained = {
        let _prover = enter(Role::MainProver);
        vec![0_u8; 1024]
    };
    let first = first.finish();
    let second = Session::start();
    {
        let _checker = enter(Role::CheckerProver);
        drop(retained);
        let payload = vec![0_u8; 512];
        std::hint::black_box(&payload);
    }
    let second = second.finish();
    assert_eq!(first.main_prover.peak_heap_bytes, 1024);
    assert_eq!(second.main_prover.peak_heap_bytes, 0);
    assert_eq!(second.checker_prover.peak_heap_bytes, 512);
    assert_eq!(second.checker_prover.live_heap_bytes, 0);
}

#[test]
fn cross_thread_free_releases_the_original_party_allocation() {
    let session = Session::start();
    let payload = {
        let _prover = enter(Role::MainProver);
        vec![0_u8; 1024]
    };
    std::thread::spawn(move || drop(payload)).join().unwrap();
    let report = session.finish();
    assert_eq!(report.main_prover.peak_heap_bytes, 1024);
    assert_eq!(report.main_prover.live_heap_bytes, 0);
}

#[test]
fn instrumentation_preserves_actual_protocol_messages_and_verdicts() {
    use rmip_backend_interleaved::{Ligero, Shape};
    use rmip_demo::measured_execution::statement;
    use rmip_demo::mnist_bench::sparse_mnist_case;
    use rmip_demo::party_metrics::MeasuredChallenger;
    use rmip_iop::driver::{Cheat, Config};
    use rmip_iop::transcript::{RawChallenger, SeededChallenger};

    let (case, oracle) = sparse_mnist_case(256, "sparse-mnist");
    let st = statement(&case);
    let cfg = Config::for_instance(&case.inst, 0.125);
    let shape = Shape::balanced(case.layout.nu, 4, 4.0);
    for cheat in [
        Cheat::default(),
        Cheat {
            bad_oracle_answer: Some(0),
            ..Cheat::default()
        },
    ] {
        let mut plain_coins = SeededChallenger::new(57);
        let (plain, plain_verdict) = rmip_mip_prover::execute_recording(
            &st,
            &mut Ligero::lazy(shape),
            &oracle,
            &cfg,
            &cheat,
            &mut plain_coins,
        );
        let mut measured_coins = MeasuredChallenger::new(SeededChallenger::new(57));
        let session = Session::start();
        let (measured, measured_verdict) = {
            let _role = enter(Role::Verifier);
            rmip_mip_prover::execute_recording(
                &st,
                &mut Ligero::lazy(shape),
                &oracle,
                &cfg,
                &cheat,
                &mut measured_coins,
            )
        };
        let report = session.finish();
        assert_eq!(plain_verdict, measured_verdict);
        assert_eq!(
            plain_coins.transcript().entries(),
            measured_coins.transcript().entries()
        );
        assert_eq!(plain.oracle_calls, measured.oracle_calls);
        assert_eq!(plain.short_bytes, measured.short_bytes);
        assert_eq!(plain.q_long(), measured.q_long());
        assert_eq!(plain.a_long(), measured.a_long());
        assert_eq!(plain.reduced_claim, measured.reduced_claim);
        for (plain, measured) in plain.occurrences.iter().zip(&measured.occurrences) {
            assert_eq!(plain.oracle, measured.oracle);
            assert_eq!(plain.coord, measured.coord);
            assert_eq!(plain.prefix, measured.prefix);
            assert_eq!(plain.answer, measured.answer);
        }
        assert!(report.heap_tracking);
        assert!(report.main_prover.elapsed > std::time::Duration::ZERO);
        assert!(report.verifier.elapsed > std::time::Duration::ZERO);
        assert!(report.main_prover.peak_heap_bytes > 0);
    }
}

#[test]
fn actual_two_prover_run_emits_party_measurements_and_complete_wire_costs() {
    use rmip_demo::measured_execution::run_protocol;
    use rmip_demo::mnist_bench::sparse_mnist_case;

    let (case, oracle) = sparse_mnist_case(256, "sparse-mnist");
    let execution = run_protocol(&case, &oracle, false).unwrap();
    let measured = &execution.measurements;
    assert_eq!(measured["status"], "accepted");
    assert_eq!(measured["heap_tracking"], true);
    assert_eq!(
        measured["checker_questions"],
        rmip_backend_ladder::LadderShape::for_trace(case.layout.nu, 4.0)
            .schedule()
            .tables
            .len()
    );
    assert_eq!(measured["executed_repetitions"], 1);
    assert_eq!(measured["actual_total_Q_O"], case.t);
    assert_eq!(execution.verifier_calls.len(), case.t);
    assert_eq!(execution.main_prover_calls.len(), case.inst.ell);
    let parties = &measured["party_measurements"];
    for party in [
        "main_prover",
        "verifier",
        "checker_prover_including_preparation",
    ] {
        assert!(
            parties[party]["runtime_ms"].as_f64().unwrap() > 0.0,
            "{party}"
        );
        assert!(
            parties[party]["peak_live_heap_bytes"].as_u64().unwrap() > 0,
            "{party}"
        );
        assert!(
            parties[party]["allocated_heap_bytes"].as_u64().unwrap()
                >= parties[party]["peak_live_heap_bytes"].as_u64().unwrap()
        );
    }
    assert_eq!(
        measured["prover_runtime_ms"],
        parties["main_prover"]["runtime_ms"]
    );
    assert_eq!(
        measured["verifier_runtime_ms"],
        parties["verifier"]["runtime_ms"]
    );
    assert_eq!(
        measured["prover_peak_memory_bytes"],
        parties["main_prover"]["peak_live_heap_bytes"]
    );
    assert!(
        measured["total_serialized_communication_bytes"]
            .as_u64()
            .unwrap()
            > measured["accounted_partial_two_way_payload_bytes"]
                .as_u64()
                .unwrap()
    );
    let raw = &measured["raw_base_executions"][0];
    assert_eq!(raw["accepted"], true);
    assert!(raw["main_prover_runtime_ms"].as_f64().unwrap() > 0.0);
    assert!(raw["verifier_runtime_ms"].as_f64().unwrap() > 0.0);
    assert!(raw["checker_prover_runtime_ms"].as_f64().unwrap() > 0.0);
}

#[test]
fn audit_answer_capture_preserves_single_question_and_private_coin_schedule() {
    use rmip_backend_interleaved::{Ligero, Shape};
    use rmip_demo::measured_execution::statement;
    use rmip_demo::mnist_bench::sparse_mnist_case;
    use rmip_demo::party_metrics::MeasuredChallenger;
    use rmip_field::{BinaryField, F128};
    use rmip_iop::backend::CommitmentBackend;
    use rmip_iop::driver::{Cheat, Config};
    use rmip_iop::transcript::{RawChallenger, SeededChallenger};
    use rmip_mip::{Audit, PCheck};

    let (case, oracle) = sparse_mnist_case(256, "sparse-mnist");
    let st = statement(&case);
    let cfg = Config::for_instance(&case.inst, 0.125);
    let shape = Shape::balanced(case.layout.nu, 4, 4.0);
    let (mut execution, verdict) = rmip_mip_prover::execute_recording(
        &st,
        &mut Ligero::lazy(shape),
        &oracle,
        &cfg,
        &Cheat::default(),
        &mut SeededChallenger::new(67),
    );
    assert!(verdict.is_ok());
    let correct_answers: Vec<_> = execution
        .occurrences
        .iter()
        .map(|o| o.answer.clone())
        .collect();
    let make_checker = || {
        let mut backend = Ligero::lazy(shape);
        backend.commit(&case.honest_trace, &mut SeededChallenger::new(73));
        PCheck::new(backend)
    };
    let classify = |audit: &Audit| match audit {
        Audit::Accepted { audited } => ("accepted", *audited),
        Audit::UnderlyingRejected { audited } => ("underlying_rejected", *audited),
        Audit::Inconsistent { audited } => ("inconsistent", *audited),
    };
    for expected in ["accepted", "underlying_rejected", "inconsistent"] {
        if expected != "accepted" {
            execution.iop_accepted = false;
        }
        if expected == "inconsistent" {
            for occurrence in &mut execution.occurrences {
                occurrence.answer.0[0] = occurrence.answer.0[0].add(F128::ONE);
            }
        }
        let plain_checker = make_checker();
        let measured_checker = make_checker();
        let mut plain_coins = SeededChallenger::new(79);
        let mut measured_coins = MeasuredChallenger::new(SeededChallenger::new(79));
        let plain = rmip_mip::audit_one(&execution, &plain_checker, &mut plain_coins);
        let session = Session::start();
        let (captured, answer) = {
            let _role = enter(Role::Verifier);
            rmip_mip::checker::audit_with_answer(&execution, &measured_checker, &mut measured_coins)
        };
        let report = session.finish();
        assert_eq!(classify(&plain), classify(&captured));
        let (kind, audited) = classify(&captured);
        assert_eq!(kind, expected);
        assert_eq!(answer, correct_answers[audited]);
        assert_eq!(plain_checker.questions_asked(), 1);
        assert_eq!(measured_checker.questions_asked(), 1);
        assert_eq!(
            plain_coins.transcript().entries(),
            measured_coins.transcript().entries()
        );
        assert!(report.checker_prover.elapsed > std::time::Duration::ZERO);
        assert!(report.verifier.elapsed > std::time::Duration::ZERO);
    }
}

#[test]
fn oracle_call_split_matches_the_query_plan_by_content() {
    use rmip_demo::measured_execution::run_protocol;
    use rmip_demo::mnist_bench::{sparse_instance, sparse_mnist_case};
    use rmip_frontend::mnist::sparse_query_bytes;
    use std::collections::HashSet;

    let original = sparse_instance(256);
    let (case, oracle) = sparse_mnist_case(256, "sparse-mnist");
    let run = run_protocol(&case, &oracle, false).unwrap();
    let plan: Vec<Vec<u8>> = (0..256u32)
        .map(|j| sparse_query_bytes(original.image_id, j, original.true_label as u8).to_vec())
        .collect();
    assert_eq!(run.main_prover_calls.len(), case.inst.ell);
    let main: Vec<Vec<u8>> = run
        .main_prover_calls
        .iter()
        .map(|call| call.query.clone())
        .collect();
    assert_eq!(
        main, plan,
        "the first ell calls are the witness plan, in order"
    );
    let members: HashSet<&Vec<u8>> = plan.iter().collect();
    assert_eq!(run.verifier_calls.len(), case.t);
    for call in &run.verifier_calls {
        assert!(
            members.contains(&call.query),
            "verifier re-queries a plan entry, not a fresh query"
        );
    }
}
