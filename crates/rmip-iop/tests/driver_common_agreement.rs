use rmip_backend_trace::RawTrace;
use rmip_field::F128;
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, trace, Instance};
use rmip_iop::common_protocol;
use rmip_iop::driver::{committed_trace, execute_recording, Cheat, Config, Statement};
use rmip_iop::mle::lift;
use rmip_iop::transcript::{Entry, RawChallenger, SeededChallenger};

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

fn ordinary_and_coins(entries: &[Entry]) -> Vec<Entry> {
    entries
        .iter()
        .filter(|e| !matches!(e, Entry::OraclePost(_)))
        .cloned()
        .collect()
}

#[test]
fn the_driver_and_the_offline_core_emit_the_same_transcript() {
    let f = fixture();
    let st = statement(&f);
    let cfg = Config {
        t: 2,
        ..Config::default()
    };

    for seed in [1u64, 7, 42] {
        let mut backend = RawTrace::new();
        let mut driver_ch = SeededChallenger::new(seed);
        let (_exec, _) = execute_recording(
            &st,
            &mut backend,
            &f.oracle,
            &cfg,
            &Cheat::default(),
            &mut driver_ch,
        );

        let table = committed_trace(&st, &f.oracle, &Cheat::default());
        assert_eq!(
            table, f.table,
            "the driver's committed trace is the fixture's"
        );

        let mut core_ch = SeededChallenger::new(seed);
        let out = common_protocol::prove_recording(&st, &table, &cfg, &mut core_ch)
            .expect("the offline core must produce a transcript");
        assert!(
            out.verdict.is_ok(),
            "seed {seed}: honest batching must not be deferred-rejected"
        );

        let a = ordinary_and_coins(driver_ch.transcript().entries());
        let b = ordinary_and_coins(core_ch.transcript().entries());

        assert!(
            a.len() > 20,
            "seed {seed}: only {} comparable entries -- too few",
            a.len()
        );
        assert!(
            a.iter().any(|e| matches!(e, Entry::Ordinary(_))),
            "seed {seed}: no prover messages compared"
        );
        assert!(
            a.iter().any(|e| matches!(e, Entry::Coin(_))),
            "seed {seed}: no coins compared"
        );
        let posts = driver_ch
            .transcript()
            .entries()
            .iter()
            .filter(|e| matches!(e, Entry::OraclePost(_)))
            .count();
        assert!(posts > 0, "the driver must announce its long oracles");
        assert_eq!(
            core_ch
                .transcript()
                .entries()
                .iter()
                .filter(|e| matches!(e, Entry::OraclePost(_)))
                .count(),
            0,
            "the offline core has no backend and must post nothing"
        );
        assert_eq!(
            a.len(),
            b.len(),
            "seed {seed}: entry counts differ ({} driver vs {} core)",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x, y,
                "seed {seed}: entry {i} differs between the two implementations"
            );
        }
    }
}

#[test]
fn the_two_implementations_reduce_to_the_same_claim() {
    let f = fixture();
    let st = statement(&f);
    let cfg = Config {
        t: 2,
        ..Config::default()
    };

    let mut first = None;
    for seed in [1u64, 7, 42] {
        let table = committed_trace(&st, &f.oracle, &Cheat::default());

        let mut driver_ch = SeededChallenger::new(seed);
        let mut backend = RawTrace::new();
        let (exec, verdict) = execute_recording(
            &st,
            &mut backend,
            &f.oracle,
            &cfg,
            &Cheat::default(),
            &mut driver_ch,
        );
        verdict.unwrap_or_else(|r| panic!("seed {seed}: the honest driver must accept: {r:?}"));
        let driven = exec
            .reduced_claim
            .clone()
            .expect("batching is enabled, so the driver opened one reduced claim");

        let mut core_ch = SeededChallenger::new(seed);
        let out = common_protocol::prove_recording(&st, &table, &cfg, &mut core_ch).unwrap();
        let claim = out.output.reduced_claim.clone();

        assert_eq!(
            driven, claim,
            "seed {seed}: the two implementations reduced differently"
        );

        assert_eq!(
            claim.point.len(),
            f.layout.nu,
            "the reduced claim is at nu variables"
        );
        assert_eq!(
            rmip_iop::mle::eval_mle(&f.table, &claim.point),
            claim.value,
            "seed {seed}: the reduced claim must hold of the committed trace"
        );
        if let Some(prev) = first.replace(claim.clone()) {
            assert_ne!(
                prev.point, claim.point,
                "distinct seeds must give distinct claim points"
            );
        }
    }
}
