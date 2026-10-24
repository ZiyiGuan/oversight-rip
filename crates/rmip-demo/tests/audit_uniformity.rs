use rmip_backend_recursive::{FoldLadder, Shape as LadderShape};
use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance};
use rmip_iop::backend::{CommitmentBackend, Symbol};
use rmip_iop::driver::{committed_trace, execute, Cheat, Config, Statement};
use rmip_iop::transcript::{OsChallenger, SeededChallenger};
use rmip_mip::{audit_one, PCheck, SharedAdvice};

const VIEWS: usize = 16;

struct Fx {
    inst: Instance,
    sys: r1cs::R1cs,
    lay: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    oracle: SparseTableOracle,
}

fn build() -> Fx {
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

fn catch_rate<F: Fn(usize, usize) -> usize>(
    trials: usize,
    victim_rule: F,
) -> (usize, usize, usize) {
    let f = build();
    let cfg = Config::for_instance(&f.inst, 0.125);
    let shape = LadderShape::balanced(f.lay.nu, 4, 4.0);
    let advice = SharedAdvice { seed: 0xC4EC };
    let checker = {
        let z = committed_trace(&st(&f), &f.oracle, &Cheat::default());
        let mut b = FoldLadder::new(shape);
        let mut own = SeededChallenger::new(advice.seed);
        b.commit(&z, &mut own);
        PCheck::new(b)
    };

    let mut caught = 0;
    let mut ran = 0;
    let mut q_long = 0;
    for i in 0..trials {
        let mut backend = FoldLadder::new(shape);
        let mut ch = OsChallenger::new();
        let Ok(mut e) = execute(
            &st(&f),
            &mut backend,
            &f.oracle,
            &cfg,
            &Cheat::default(),
            &mut ch,
        ) else {
            continue;
        };
        ran += 1;
        q_long = e.occurrences.len();
        let victim = victim_rule(i, q_long);
        let old = e.occurrences[victim].answer.clone();
        let mut bad = old.0.clone();
        bad[0] = bad[0].add(F128::ONE);
        e.occurrences[victim].answer = Symbol(bad);
        if !audit_one(&e, &checker, &mut ch).accepted() {
            caught += 1;
        }
    }
    (caught, ran, q_long)
}

fn mixed(i: usize, q: usize) -> usize {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) % q as u64) as usize
}

#[test]
fn os_csprng_control_catches_at_one_over_q_long_under_both_victim_rules() {
    let trials = 6_000;

    let (c_lin, n_lin, q) = catch_rate(trials, |i, q| (i * 7 + 1) % q);
    let (c_mix, n_mix, q2) = catch_rate(trials, mixed);
    assert_eq!(q, q2);
    assert_eq!(q, 12, "the 16-view fold ladder should yield 12 occurrences");

    let p = 1.0 / q as f64;
    let rate_lin = c_lin as f64 / n_lin as f64;
    let rate_mix = c_mix as f64 / n_mix as f64;
    let sigma = (p * (1.0 - p) / n_lin as f64).sqrt();

    println!(
        "q_long = {q}, ideal {p:.5}\n  linear victim: {rate_lin:.5} ({c_lin}/{n_lin})\n  mixed  victim: {rate_mix:.5} ({c_mix}/{n_mix})\n  5 sigma = {:.5}",
        5.0 * sigma
    );

    assert!(
        (rate_mix - p).abs() < 5.0 * sigma,
        "OS-CSPRNG control with a mixed victim differs from 1/q_long: {rate_mix:.5} vs {p:.5}"
    );
    assert!(
        (rate_lin - p).abs() < 5.0 * sigma,
        "OS-CSPRNG control with a linear victim differs from 1/q_long: {rate_lin:.5} vs {p:.5}"
    );
}
