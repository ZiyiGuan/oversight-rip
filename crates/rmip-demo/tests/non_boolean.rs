use rmip_demo::weakened::{non_boolean, WeakenedTrace};
use rmip_field::F128;
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance};
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::driver::{committed_trace, execute, Cheat, Config, Reject, Statement};
use rmip_iop::transcript::SeededChallenger;

const VIEWS: usize = 16;

struct Fx {
    inst: Instance,
    sys: r1cs::R1cs,
    layout: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    oracle: SparseTableOracle,
}

fn fx(views: usize) -> Fx {
    let m = MnistInstance::new(views);
    let circuit = m.build();
    let oracle = m.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);
    let x = m.yes();
    assert!(
        circuit.eval(x.as_slice(), &oracle),
        "the true label accepts"
    );
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
        oracle,
    }
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

fn cfg(f: &Fx) -> Config {
    Config::for_instance(&f.inst, 0.125)
}

#[test]
fn weakened_backend_accepts_an_honest_trace() {
    let f = fx(VIEWS);
    for seed in 1..=50u64 {
        let mut b = WeakenedTrace::new();
        let mut ch = SeededChallenger::new(seed);
        execute(
            &st(&f),
            &mut b,
            &f.oracle,
            &cfg(&f),
            &Cheat::default(),
            &mut ch,
        )
        .unwrap_or_else(|r| panic!("honest run rejected at seed {seed}: {r:?}"));
    }
}

#[test]
fn a_non_boolean_pad_coordinate_is_caught_only_by_the_booleanity_check() {
    let f = fx(VIEWS);
    let live_end = f.layout.used_end();
    let n = f.layout.n_trace;
    assert!(live_end < n, "this instance must have a trailing pad");

    for coord in [live_end, live_end + 1, n - 1] {
        let cheat = Cheat {
            non_boolean_coord: Some((coord, non_boolean().to_u128())),
            ..Default::default()
        };
        let table = committed_trace(&st(&f), &f.oracle, &cheat);
        assert_eq!(table[coord], non_boolean());

        for seed in 1..=15u64 {
            let mut b = WeakenedTrace::new();
            let mut ch = SeededChallenger::new(seed);
            let r = execute(&st(&f), &mut b, &f.oracle, &cfg(&f), &cheat, &mut ch);
            assert!(
                matches!(
                    r,
                    Err(Reject::Batching(_)) | Err(Reject::OpeningMismatch { .. })
                ),
                "pad coordinate {coord} at seed {seed}: expected Batching or OpeningMismatch, \
                 got {:?} -- no row references it, so nothing but Booleanity should see it",
                r.as_ref().err()
            );
        }
    }
}

#[test]
fn the_booleanity_claim_is_caught_on_both_opening_paths() {
    let f = fx(VIEWS);
    let coord = f.layout.used_end();
    assert!(
        coord < f.layout.n_trace,
        "this instance must have a trailing pad"
    );
    let cheat = Cheat {
        non_boolean_coord: Some((coord, non_boolean().to_u128())),
        ..Default::default()
    };

    for (batching, label) in [(true, "batched"), (false, "direct")] {
        let config = Config {
            batch_claims: batching,
            ..cfg(&f)
        };
        let mut b = WeakenedTrace::new();
        let mut ch = SeededChallenger::new(11);
        let r = execute(&st(&f), &mut b, &f.oracle, &config, &cheat, &mut ch);
        let expected = if batching {
            matches!(r, Err(Reject::Batching(_)))
        } else {
            matches!(r, Err(Reject::OpeningMismatch { .. }))
        };
        assert!(
            expected,
            "{label}: expected the {label} catch, got {:?}",
            r.as_ref().err()
        );
    }
}

#[test]
fn a_non_boolean_gate_output_is_caught_by_zerocheck() {
    let f = fx(VIEWS);
    let coord = f.layout.p + f.inst.ell + 3;
    assert!(coord < f.layout.used_end());
    let cheat = Cheat {
        non_boolean_coord: Some((coord, non_boolean().to_u128())),
        ..Default::default()
    };

    let mut b = WeakenedTrace::new();
    let mut ch = SeededChallenger::new(3);
    let r = execute(&st(&f), &mut b, &f.oracle, &cfg(&f), &cheat, &mut ch);
    assert!(
        matches!(r, Err(Reject::ZerocheckEndpoint)),
        "expected Reject::ZerocheckEndpoint for a non-Boolean gate output, got {:?}",
        r.as_ref().err()
    );
}

#[test]
fn the_seven_view_case_has_a_two_coordinate_tail() {
    let f = fx(7);
    assert_eq!(f.layout.n_trace, 64);
    assert_eq!(f.layout.used_end(), 62, "7 views: P + g = 62");
    assert_eq!(f.layout.tail_pad_range().len(), 2);

    let g = fx(VIEWS);
    assert_eq!(g.layout.n_trace, 128);
    assert_eq!(
        g.layout.tail_pad_range().len(),
        40,
        "16 views: forty pad coordinates, which the pad tests above corrupt"
    );
}

#[test]
#[should_panic(expected = "different backend")]
fn the_honest_backend_refuses_a_field_valued_table() {
    use rmip_backend_trace::RawTrace;
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(1);
    let mut z = vec![F128::ZERO; 16];
    z[3] = non_boolean();
    b.commit(&z, &mut ch);
}

#[test]
fn an_honestly_declared_weak_backend_is_refused_by_the_gate() {
    let f = fx(VIEWS);
    let mut b = WeakenedTrace::honest_about_itself();
    let mut ch = SeededChallenger::new(1);
    let r = execute(
        &st(&f),
        &mut b,
        &f.oracle,
        &cfg(&f),
        &Cheat::default(),
        &mut ch,
    );
    assert!(
        matches!(r, Err(Reject::InadmissibleBackend(_))),
        "expected the admissibility gate, got {:?}",
        r.as_ref().err()
    );
    if let Err(Reject::InadmissibleBackend(msg)) = &r {
        assert!(
            msg.contains("Booleanity"),
            "gate should name the reason: {msg}"
        );
    }
}

#[test]
fn booleanity_costs_exactly_one_claim() {
    let f = fx(VIEWS);
    let config = cfg(&f);
    let mut b = WeakenedTrace::new();
    let mut ch = SeededChallenger::new(7);
    let e = execute(
        &st(&f),
        &mut b,
        &f.oracle,
        &config,
        &Cheat::default(),
        &mut ch,
    )
    .unwrap();
    assert_eq!(config.t, 4, "four local-validity samples at eps_rob = 7/16");
    assert_eq!(f.inst.b_max(), 64, "every sparse-judge query is 64 wires");
    assert_eq!(e.claims_submitted, 3 + 4 * 65);
    assert_eq!(e.claims_submitted, 263);
    assert_eq!(config.k_rel, 3);
}
