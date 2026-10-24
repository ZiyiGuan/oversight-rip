use rmip_demo::harness::mnist_case;
use rmip_demo::measured_execution::{
    parameters_body, parameters_for, parameters_opts, run_protocol_opts, IopBody, MipBackend,
    RunOptions, Soundness, AMPLIFIED_TARGET,
};
use rmip_frontend::report::local_sampling_repetitions;

#[test]
fn single_shot_meets_the_target_in_one_execution_on_the_ladder() {
    let (case, oracle) = mnist_case(256);
    let p = parameters_opts(&case, MipBackend::Ladder, Soundness::SingleShot);
    assert_eq!(p["soundness"], "single-shot");
    assert_eq!(p["amplification_repetitions"], 1);
    assert!(
        p["q_long"].as_u64().unwrap() > 1,
        "one occurrence per table"
    );
    assert_eq!(p["opening_columns"], 34);
    assert_eq!(p["lambda_bits"], 23.0);
    assert!(p["pins"].as_u64().unwrap() >= 2, "pins = {}", p["pins"]);
    let s_iop = p["s_iop"].as_f64().unwrap();
    assert!(s_iop <= AMPLIFIED_TARGET, "s_iop = {s_iop:e}");
    assert!(p["amplified_bound"].as_f64().unwrap() <= AMPLIFIED_TARGET);
    assert!(p["eps_open"].as_f64().unwrap() <= 2f64.powi(-22));
    assert!(p["local_miss_bound"].as_f64().unwrap() <= 2f64.powi(-21));
    let t = p["base_t"].as_u64().unwrap() as usize;
    assert_eq!(t, local_sampling_repetitions(case.eps_rob, 2f64.powi(-21)));
    assert!(
        t > case.t,
        "more samples than the amplified mode's {}",
        case.t
    );
    assert_eq!(p["configured_total_Q_O"], t as u64);
    assert_eq!(p["local_error_budget"], 2f64.powi(-21));

    let run = run_protocol_opts(
        &case,
        &oracle,
        RunOptions {
            amplified: true,
            record_lookups: true,
            soundness: Soundness::SingleShot,
            body: IopBody::Flat,
            ladder_depth: None,
        },
        MipBackend::Ladder,
    )
    .unwrap();
    let m = &run.measurements;
    assert_eq!(m["status"], "accepted");
    assert_eq!(m["executed_repetitions"], 1);
    assert_eq!(m["actual_total_Q_O"], t as u64);
    assert_eq!(run.verifier_calls.len(), t);
    assert!(
        rmip_backend_ladder::Ladder::lazy(rmip_backend_ladder::LadderShape::for_trace(
            case.layout.nu,
            23.0
        ))
        .domain_of(0)
        .fits_u256()
    );
    assert_eq!(m["communication"]["serialization"], "rmip-wire-v3");
    assert_eq!(m["raw_base_executions"][0]["q_long"], p["q_long"]);
}

#[test]
fn single_shot_at_1024_views_serializes_128_byte_coordinates() {
    let (case, oracle) = mnist_case(1024);
    let p = parameters_opts(&case, MipBackend::Ladder, Soundness::SingleShot);
    assert_eq!(p["amplification_repetitions"], 1);
    assert!(
        !rmip_backend_ladder::Ladder::lazy(rmip_backend_ladder::LadderShape::for_trace(
            case.layout.nu,
            23.0
        ))
        .domain_of(0)
        .fits_u256()
    );
    let run = run_protocol_opts(
        &case,
        &oracle,
        RunOptions {
            amplified: true,
            record_lookups: false,
            soundness: Soundness::SingleShot,
            body: IopBody::Flat,
            ladder_depth: None,
        },
        MipBackend::Ladder,
    )
    .unwrap();
    let m = &run.measurements;
    assert_eq!(m["status"], "accepted");
    assert_eq!(m["executed_repetitions"], 1);
    assert_eq!(m["communication"]["serialization"], "rmip-wire-v4");
    assert_eq!(m["communication"]["version"], 4);
}

#[test]
fn the_amplified_parameters_are_the_registry_ones() {
    let (case, _oracle) = mnist_case(256);
    let a = parameters_for(&case, MipBackend::Ladder);
    let b = parameters_opts(&case, MipBackend::Ladder, Soundness::Amplified);
    assert_eq!(a, b);
    assert_eq!(a["soundness"], "amplified");
    assert_eq!(a["opening_columns"], 6);
    assert_eq!(a["lambda_bits"], 4.0);
    assert_eq!(a["pins"], 2);
    assert_eq!(a["local_error_budget"], 0.125);
    assert_eq!(a["base_t"], case.t as u64);
    let r = a["amplification_repetitions"].as_u64().unwrap();
    assert!((9..=10).contains(&r), "R = {r}");
}

#[test]
fn single_shot_on_the_ladder_also_takes_one_execution() {
    let (case, oracle) = mnist_case(256);
    let p = parameters_opts(&case, MipBackend::Ladder, Soundness::SingleShot);
    assert_eq!(p["amplification_repetitions"], 1);
    assert_eq!(p["opening_columns"], 34);
    assert!(p["pins"].as_u64().unwrap() >= 2);
    assert!(p["s_iop"].as_f64().unwrap() <= AMPLIFIED_TARGET);
    let run = run_protocol_opts(
        &case,
        &oracle,
        RunOptions {
            amplified: true,
            record_lookups: false,
            soundness: Soundness::SingleShot,
            body: IopBody::Flat,
            ladder_depth: None,
        },
        MipBackend::Ladder,
    )
    .unwrap();
    assert_eq!(run.measurements["status"], "accepted");
    assert_eq!(run.measurements["executed_repetitions"], 1);
    assert_eq!(run.measurements["actual_total_Q_O"], p["base_t"]);
}

#[test]
#[should_panic(expected = "not reachable at this shape")]
fn a_single_shot_target_beyond_the_field_terms_is_refused() {
    let (case, _oracle) = mnist_case(256);
    let _ = parameters_body(
        &case,
        MipBackend::Ladder,
        Soundness::SingleShotAt(60),
        IopBody::Packed64,
    );
}

#[test]
fn a_forty_bit_single_shot_meets_forty_bits() {
    let (case, _oracle) = mnist_case(256);
    for body in [IopBody::Packed64, IopBody::Packed128] {
        let p = parameters_body(&case, MipBackend::Ladder, Soundness::SingleShotAt(40), body);
        assert_eq!(p["amplification_repetitions"], 1);
        assert_eq!(p["soundness"], "single-shot-40");
        assert_eq!(
            p["amplified_error_budget"].as_f64().unwrap(),
            2f64.powi(-40)
        );
        assert!(p["s_two_prover_one"].as_f64().unwrap() <= 2f64.powi(-40));
        assert!(p["opening_columns"].as_u64().unwrap() >= 64);
    }
}

#[test]
#[should_panic(expected = "one-of audit")]
fn the_flat_backend_refuses_single_shot() {
    let (case, _oracle) = mnist_case(256);
    let _ = parameters_opts(&case, MipBackend::Flat, Soundness::SingleShot);
}

#[test]
fn single_shot_runs_the_packed_body_on_the_ladder() {
    let (case, oracle) = mnist_case(256);
    for body in [IopBody::Packed64, IopBody::Packed128] {
        {
            let backend = MipBackend::Ladder;
            let p = parameters_body(&case, backend, Soundness::SingleShot, body);
            assert_eq!(p["body"], body.name());
            assert_eq!(p["claim_count"], 1);
            assert_eq!(p["amplification_repetitions"], 1);
            assert!(p["eps_relation"].as_f64().unwrap() < 2f64.powi(-40));
            let run = run_protocol_opts(
                &case,
                &oracle,
                RunOptions {
                    amplified: true,
                    record_lookups: true,
                    soundness: Soundness::SingleShot,
                    body,
                    ladder_depth: None,
                },
                backend,
            )
            .expect("the packed run");
            let m = &run.measurements;
            assert_eq!(m["status"], "accepted", "{body:?} on {backend:?}");
            assert_eq!(m["body"], body.name());
            assert_eq!(m["executed_repetitions"], 1);
            assert_eq!(m["actual_total_Q_O"], p["base_t"]);
            assert_eq!(m["checker_preparation_oracle_calls"], case.inst.ell as u64);
            assert_eq!(
                run.verifier_calls.len(),
                p["base_t"].as_u64().unwrap() as usize
            );
            assert_eq!(run.main_prover_calls.len(), case.inst.ell);
        }
    }
}

#[test]
fn single_shot_on_the_depth_zero_ladder_takes_one_execution() {
    use rmip_demo::measured_execution::parameters_body_at_depth;
    let (case, oracle) = mnist_case(256);
    let p = parameters_body_at_depth(
        &case,
        MipBackend::Ladder,
        Soundness::SingleShot,
        IopBody::Flat,
        Some(0),
    );
    assert_eq!(p["amplification_repetitions"], 1);
    assert_eq!(p["q_long"], 1, "one table");
    assert_eq!(p["ladder"]["depth"], 0);
    assert_eq!(p["ladder"]["tables"], 1);
    assert_eq!(p["opening_columns"], 34);
    assert!(p["s_iop"].as_f64().unwrap() <= AMPLIFIED_TARGET);
    let run = run_protocol_opts(
        &case,
        &oracle,
        RunOptions {
            amplified: true,
            record_lookups: false,
            soundness: Soundness::SingleShot,
            body: IopBody::Flat,
            ladder_depth: Some(0),
        },
        MipBackend::Ladder,
    )
    .unwrap();
    assert_eq!(run.measurements["status"], "accepted");
    assert_eq!(run.measurements["executed_repetitions"], 1);
    assert_eq!(run.measurements["raw_base_executions"][0]["q_long"], 1);
}

#[test]
fn the_flat_backend_opens_each_body_at_its_field() {
    use rmip_demo::measured_execution::opening_kappa;
    let (case, oracle) = mnist_case(256);
    for body in [IopBody::Flat, IopBody::Packed64, IopBody::Packed128] {
        let run = run_protocol_opts(
            &case,
            &oracle,
            RunOptions {
                amplified: false,
                record_lookups: false,
                soundness: Soundness::Amplified,
                body,
                ladder_depth: None,
            },
            MipBackend::Flat,
        )
        .expect("the run");
        let m = &run.measurements;
        assert_eq!(
            m["status"], "accepted",
            "{body:?} on the flat backend: {:?}",
            m["raw_base_executions"][0]["driver_rejection"]
        );
        assert_eq!(m["executed_repetitions"], 1);
        let raw: u64 = m["raw_base_executions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["long_answer_bytes"].as_u64().unwrap())
            .sum();
        assert!(raw > 0);
        assert_eq!(
            m["main_long_answer_bytes"], raw,
            "{body:?}: serialized width"
        );
        let native = m["main_long_answer_native_bytes"].as_u64().unwrap();
        assert_eq!(
            native * 128 / opening_kappa(body) as u64,
            raw,
            "{body:?}: native width"
        );
    }
}
