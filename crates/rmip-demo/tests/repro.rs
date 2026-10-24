use rmip_backend_trace::RawTrace;
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance};
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::driver::{committed_trace, execute, Cheat, Config, Statement};
use rmip_iop::transcript::SeededChallenger;

const VIEWS: usize = 16;

struct Fx {
    inst: Instance,
    sys: r1cs::R1cs,
    lay: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    oracle: SparseTableOracle,
}

fn build(x_is_yes: bool) -> Fx {
    let m = MnistInstance::new(VIEWS);
    let circuit = m.build();
    let oracle = m.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);
    let x = if x_is_yes { m.yes() } else { m.no() };
    assert_eq!(circuit.eval(x.as_slice(), &oracle), x_is_yes);
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
        lay,
        circuit,
        dc,
        oracle,
    }
}

fn st(f: &Fx) -> Statement<'_> {
    Statement {
        inst: &f.inst,
        sys: &f.sys,
        layout: &f.lay,
        circuit: &f.circuit,
        dc: &f.dc,
        holo: None,
    }
}

fn cfg(f: &Fx) -> Config {
    Config::for_instance(&f.inst, 0.125)
}

#[test]
fn honest_execution_on_a_no_instance_is_rejected() {
    let f = build(false);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(1);
    let r = execute(
        &st(&f),
        &mut b,
        &f.oracle,
        &cfg(&f),
        &Cheat::default(),
        &mut ch,
    );
    assert!(r.is_err(), "a no-instance must not be provable");

    let f = build(true);
    let mut b = RawTrace::new();
    let mut ch = SeededChallenger::new(1);
    assert!(execute(
        &st(&f),
        &mut b,
        &f.oracle,
        &cfg(&f),
        &Cheat::default(),
        &mut ch
    )
    .is_ok());
}

#[test]
fn committed_trace_matches_what_execute_commits() {
    let f = build(true);

    for cheat in [
        Cheat::default(),
        Cheat {
            public_input_bits: vec![0, 1],
            ..Default::default()
        },
        Cheat {
            corrupt_trace_coord: Some(f.lay.answer_coord(1)),
            ..Default::default()
        },
        Cheat {
            trace_swap: true,
            ..Default::default()
        },
    ] {
        let predicted = committed_trace(&st(&f), &f.oracle, &cheat);
        let mut b = RawTrace::new();
        let mut ch = SeededChallenger::new(5);
        let _ = execute(&st(&f), &mut b, &f.oracle, &cfg(&f), &cheat, &mut ch);
        let posted: Vec<rmip_field::F128> = b.oracles()[0].symbols.iter().map(|s| s.0[0]).collect();
        assert_eq!(predicted, posted, "committed_trace diverged for {cheat:?}");
    }
}
