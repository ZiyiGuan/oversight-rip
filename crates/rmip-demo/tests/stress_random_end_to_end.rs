use rmip_backend_interleaved::{Ligero, Shape};
use rmip_frontend::bits::{BitSlice, BitVec};
use rmip_frontend::circuit::{Gate, OracleCircuit, WireId};
use rmip_frontend::oracle::{encode, Oracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance, Ratio};
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::driver::{committed_trace, execute_recording, Cheat, Config, Reject, Statement};
use rmip_iop::transcript::SeededChallenger;
use rmip_mip::{audit_one, PCheck, SharedAdvice};
use sha2::{Digest, Sha256};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        (self.next_u64() % n as u64) as usize
    }
    fn flip(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

#[derive(Clone, Copy, Debug)]
enum Ora {
    Zero,
    One,
    Length,
    Parity,
    Sha,
    Keyed(u64),
}

impl Oracle for Ora {
    fn query(&self, u: &BitSlice) -> bool {
        match self {
            Ora::Zero => false,
            Ora::One => true,
            Ora::Length => u.len() % 3 == 1,
            Ora::Parity => u.iter().filter(|b| **b).count() % 2 == 1,
            Ora::Sha => Sha256::digest(encode(u))[0] & 0x80 != 0,
            Ora::Keyed(key) => {
                let mut message = key.to_be_bytes().to_vec();
                message.extend_from_slice(&encode(u));
                Sha256::digest(message)[0] & 0x80 != 0
            }
        }
    }
}

fn oracle_zoo() -> Vec<(&'static str, Ora)> {
    vec![
        ("const-0", Ora::Zero),
        ("const-1", Ora::One),
        ("length", Ora::Length),
        ("parity", Ora::Parity),
        ("sha256", Ora::Sha),
        ("keyed-A", Ora::Keyed(0x5EED_0000_0000_00A1)),
    ]
}

#[derive(Clone, Copy, Debug)]
struct Gen {
    max_n: usize,
    max_extra: usize,
    max_fanin: usize,
}

impl Default for Gen {
    fn default() -> Self {
        Self {
            max_n: 4,
            max_extra: 12,
            max_fanin: 4,
        }
    }
}

fn random_circuit(rng: &mut Rng, cfg: Gen) -> OracleCircuit {
    let n = rng.below(cfg.max_n + 1);
    let mut gates: Vec<Gate> = (0..n).map(Gate::Input).collect();
    gates.push(Gate::Const(rng.flip()));
    let mut occurrence = 0usize;
    let mut oracle_wires: Vec<WireId> = Vec::new();
    let mut past_queries: Vec<Vec<WireId>> = Vec::new();
    let extra = 2 + rng.below(cfg.max_extra);
    for _ in 0..extra {
        let here = gates.len();
        let gate = match rng.below(12) {
            0 => Gate::Const(rng.flip()),
            1 => Gate::Not(rng.below(here)),
            2 | 3 => Gate::And(rng.below(here), rng.below(here)),
            4 | 5 => Gate::Or(rng.below(here), rng.below(here)),
            6 | 7 => Gate::Xor(rng.below(here), rng.below(here)),
            _ => {
                let query = if !past_queries.is_empty() && rng.below(4) == 0 {
                    past_queries[rng.below(past_queries.len())].clone()
                } else {
                    let width = 1 + rng.below(cfg.max_fanin);
                    (0..width)
                        .map(|_| {
                            if !oracle_wires.is_empty() && rng.below(3) == 0 {
                                oracle_wires[rng.below(oracle_wires.len())]
                            } else {
                                rng.below(here)
                            }
                        })
                        .collect()
                };
                past_queries.push(query.clone());
                oracle_wires.push(here);
                let g = Gate::Oracle { occurrence, query };
                occurrence += 1;
                g
            }
        };
        gates.push(gate);
    }
    if occurrence == 0 {
        let here = gates.len();
        gates.push(Gate::Oracle {
            occurrence: 0,
            query: vec![rng.below(here)],
        });
    }
    let output = rng.below(gates.len());
    let c = OracleCircuit { n, gates, output };
    c.validate()
        .expect("the generator must emit valid circuits");
    c
}

struct Fixture {
    inst: Instance,
    sys: r1cs::R1cs,
    lay: layout::TraceLayout,
    circuit: OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    yes: Option<BitVec>,
    no: Option<BitVec>,
}

fn fixture<O: Oracle>(circuit: OracleCircuit, o: &O, rng: &mut Rng) -> Fixture {
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);

    let mut yes = None;
    let mut no = None;
    for attempt in 0..48 {
        let x = if attempt < (1usize << circuit.n.min(5)) && circuit.n <= 5 {
            BitVec::from_bools((0..circuit.n).map(|i| (attempt >> i) & 1 == 1).collect())
        } else {
            BitVec::from_bools((0..circuit.n).map(|_| rng.flip()).collect())
        };
        if circuit.eval(x.as_slice(), o) {
            yes.get_or_insert(x);
        } else {
            no.get_or_insert(x);
        }
        if yes.is_some() && no.is_some() {
            break;
        }
    }

    let ell = circuit.ell();
    let inst = Instance {
        layout: lay.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(l),
        ell,
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x: BitVec::zeros(circuit.n),
        eps_rob: Ratio::new(1, 2),
    };
    Fixture {
        inst,
        sys,
        lay,
        circuit,
        dc,
        yes,
        no,
    }
}

fn statement(f: &Fixture) -> Statement<'_> {
    Statement {
        inst: &f.inst,
        sys: &f.sys,
        layout: &f.lay,
        circuit: &f.circuit,
        dc: &f.dc,
        holo: None,
    }
}

struct Run {
    verdict: Result<(), Reject>,
    audited: bool,
    q_long: usize,
    calls: usize,
    claims: usize,
    pad: usize,
}

fn run_once<O: Oracle>(f: &Fixture, x: &BitVec, o: &O, seed: u64) -> Run {
    run_with(f, x, o, &Cheat::default(), seed)
}

fn run_with<O: Oracle>(f: &Fixture, x: &BitVec, o: &O, cheat: &Cheat, seed: u64) -> Run {
    let mut inst = f.inst.clone();
    inst.x = x.clone();
    let fx = Fixture {
        inst,
        sys: f.sys.clone(),
        lay: f.lay.clone(),
        circuit: f.circuit.clone(),
        dc: f.dc.clone(),
        yes: None,
        no: None,
    };
    let st = statement(&fx);
    let cfg = Config::for_instance(&fx.inst, 0.125);
    let shape = Shape::balanced(fx.lay.nu, 4, 4.0);

    let advice = SharedAdvice {
        seed: seed ^ 0xC4EC_C4EC,
    };
    let checker = {
        let z = committed_trace(&st, o, cheat);
        let mut b = Ligero::lazy(shape);
        b.commit(&z, &mut SeededChallenger::new(advice.seed));
        PCheck::new(b)
    };

    let mut backend = Ligero::new(shape);
    let mut ch = SeededChallenger::new(seed);
    let (execution, verdict) = execute_recording(&st, &mut backend, o, &cfg, cheat, &mut ch);
    let q_long = execution.occurrences.len();
    let calls = execution.oracle_calls;
    let claims = execution.claims_submitted;
    let pad = execution.pad_claims;
    let mut audit_coins = SeededChallenger::new(seed ^ 0xA0D1_7000_0000_0001);
    let audited = audit_one(&execution, &checker, &mut audit_coins).accepted();
    Run {
        verdict,
        audited,
        q_long,
        calls,
        claims,
        pad,
    }
}

fn trials(default: usize) -> usize {
    std::env::var("RMIP_STRESS_TRIALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Default, Debug)]
struct Tally {
    yes_runs: usize,
    no_runs: usize,
    no_yes_instance: usize,
    no_no_instance: usize,
    padded_runs: usize,
    max_pad: usize,
    rejections: std::collections::BTreeMap<String, usize>,
}

fn sweep(circuits: usize, base: u64, cfg: Gen) -> Tally {
    let zoo = oracle_zoo();
    let mut tally = Tally::default();
    for k in 0..circuits {
        let seed = base.wrapping_add((k as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
        let mut rng = Rng::new(seed);
        let circuit = random_circuit(&mut rng, cfg);
        for (name, o) in &zoo {
            let f = fixture(circuit.clone(), o, &mut rng);
            match &f.yes {
                Some(x) => {
                    let r = run_once(&f, x, o, seed);
                    assert_eq!(
                        r.verdict,
                        Ok(()),
                        "seed {seed} / {name}: honest yes-instance rejected"
                    );
                    assert!(
                        r.audited,
                        "seed {seed} / {name}: the audit refused an honest execution"
                    );
                    assert!(r.q_long >= 1, "seed {seed} / {name}: q_long = 0");
                    let t = Config::for_instance(&f.inst, 0.125).t;
                    assert_eq!(
                        r.calls, t,
                        "seed {seed} / {name}: one oracle call per sample"
                    );
                    assert_eq!(
                        r.claims,
                        3 + t * (f.inst.b_max() + 1),
                        "seed {seed} / {name}: K is not k_rel + t*(b_max+1)"
                    );
                    tally.max_pad = tally.max_pad.max(r.pad);
                    if r.pad > 0 {
                        tally.padded_runs += 1;
                    }
                    tally.yes_runs += 1;
                }
                None => tally.no_yes_instance += 1,
            }
            match &f.no {
                Some(x) => {
                    let r = run_once(&f, x, o, seed ^ 0xF001);
                    let e = r
                        .verdict
                        .expect_err("a false output claim must be rejected");
                    assert!(
                        !r.audited,
                        "seed {seed} / {name}: the audit accepted a false claim"
                    );
                    *tally.rejections.entry(format!("{e:?}")).or_default() += 1;
                    tally.no_runs += 1;
                }
                None => tally.no_no_instance += 1,
            }
        }
    }
    tally
}

#[test]
fn random_end_to_end_executions_accept_iff_the_claim_is_true() {
    let tally = sweep(trials(8), 0x5A17_0000_0000_0001, Gen::default());
    println!("{tally:?}");
    assert!(tally.yes_runs > 0, "no yes-instance was reached");
    assert!(tally.no_runs > 0, "no false claim was reached");
    assert!(
        tally.padded_runs > 0 && tally.max_pad > 0,
        "the claim-padding path was never exercised: {tally:?}"
    );
    assert_eq!(
        tally.rejections.keys().collect::<Vec<_>>(),
        vec!["ZerocheckEndpoint"],
        "a false output claim should be caught at the zerocheck, deterministically"
    );
}

#[test]
fn edge_t_greater_than_ell() {
    let circuit = OracleCircuit {
        n: 2,
        gates: vec![
            Gate::Input(0),
            Gate::Input(1),
            Gate::Const(true),
            Gate::Oracle {
                occurrence: 0,
                query: vec![0, 1, 2],
            },
            Gate::Or(3, 2),
        ],
        output: 4,
    };
    circuit.validate().expect("valid");
    assert_eq!(circuit.ell(), 1);
    for (name, o) in oracle_zoo() {
        let mut rng = Rng::new(0x7E57_0000_0000_0001);
        let f = fixture(circuit.clone(), &o, &mut rng);
        let cfg = Config::for_instance(&f.inst, 0.125);
        assert!(cfg.t > f.inst.ell, "{name}: this edge needs t > ell");
        let x = f.yes.clone().expect("Or(_, true) is always a yes-instance");
        let r = run_once(&f, &x, &o, 0x7E57);
        assert_eq!(r.verdict, Ok(()), "{name}");
        assert!(r.audited, "{name}");
        assert_eq!(r.calls, cfg.t, "{name}: t calls even at t > ell");
    }
}

#[test]
fn edge_smallest_instance_end_to_end() {
    let circuit = OracleCircuit {
        n: 0,
        gates: vec![
            Gate::Const(true),
            Gate::Oracle {
                occurrence: 0,
                query: vec![0],
            },
        ],
        output: 1,
    };
    circuit.validate().expect("valid");
    let x = BitVec::zeros(0);
    for (name, o) in oracle_zoo() {
        let mut rng = Rng::new(0x5A11_0000_0000_0001);
        let f = fixture(circuit.clone(), &o, &mut rng);
        let true_claim = circuit.eval(x.as_slice(), &o);
        let r = run_once(&f, &x, &o, 0x5A11);
        assert!(r.q_long >= 1, "{name}");
        if true_claim {
            assert_eq!(r.verdict, Ok(()), "{name}");
            assert!(r.audited, "{name}");
        } else {
            assert!(r.verdict.is_err(), "{name}");
            assert!(!r.audited, "{name}");
        }
    }
}

#[test]
fn a_cheating_answer_vector_is_caught_exactly_by_the_local_oracle_check() {
    let zoo = oracle_zoo();
    let mut configurations = 0usize;
    let mut k = 0u64;
    while configurations < 6 && k < 400 {
        let seed = 0xC4EA_7000_0000_0001u64.wrapping_add(k.wrapping_mul(0x2545_F491_4F6C_DD1D));
        k += 1;
        let mut rng = Rng::new(seed);
        let circuit = random_circuit(
            &mut rng,
            Gen {
                max_n: 3,
                max_extra: 10,
                max_fanin: 4,
            },
        );
        let ell = circuit.ell();
        if !(2..=8).contains(&ell) {
            continue;
        }
        let (name, o) = zoo[(k as usize) % zoo.len()];
        let f = fixture(circuit.clone(), &o, &mut rng);
        let Some(x) = f.no.clone() else { continue };

        let mut best: Option<(usize, Vec<bool>)> = None;
        for mask in 0..(1usize << ell) {
            let a: Vec<bool> = (0..ell).map(|j| (mask >> j) & 1 == 1).collect();
            if !f.dc.eval(x.as_slice(), &a) {
                continue;
            }
            let (_, sites) = deoracle::deoraclize(&circuit);
            let bad = deoracle::bad_set(&f.dc, &sites, x.as_slice(), &a, &o).len();
            if best.as_ref().is_none_or(|(b, _)| bad < *b) {
                best = Some((bad, a));
            }
        }
        let Some((bad, a)) = best else { continue };
        assert!(bad >= 1, "a no-instance cannot have an empty Bad set");
        configurations += 1;

        let cheat = Cheat {
            local_cheat_answers: Some(a.clone()),
            ..Cheat::default()
        };
        let t = Config::for_instance(&f.inst, 0.125).t;
        let runs = 2000usize;
        let mut accepted = 0usize;
        for r in 0..runs {
            let run = run_with(
                &f,
                &x,
                &o,
                &cheat,
                seed ^ (r as u64).wrapping_mul(0x9E37_79B9),
            );
            match &run.verdict {
                Ok(()) => {
                    assert!(run.audited, "{name}: IOP accepted but the audit did not");
                    accepted += 1;
                }
                Err(Reject::OracleMismatch { .. }) => assert!(!run.audited),
                Err(other) => panic!(
                    "seed {seed} / {name}: a consistent cheating trace was rejected by {other:?}, \
                     not by the local oracle check"
                ),
            }
        }
        let predicted = (1.0 - bad as f64 / ell as f64).powi(t as i32);
        let observed = accepted as f64 / runs as f64;
        let sigma = (predicted * (1.0 - predicted) / runs as f64).sqrt();
        println!(
            "{name}: ell={ell} |Bad|={bad} t={t} predicted={predicted:.4} observed={observed:.4} \
             (5 sigma = {:.4})",
            5.0 * sigma
        );
        assert!(
            (observed - predicted).abs() <= 5.0 * sigma + 1.0 / runs as f64,
            "seed {seed} / {name}: acceptance {observed} is not (1 - |Bad|/ell)^t = {predicted}"
        );
    }
    assert!(
        configurations >= 3,
        "only {configurations} cheating configurations were reachable"
    );
}

#[test]
#[ignore = "measurement"]
fn the_big_end_to_end_sweep() {
    let tally = sweep(trials(60), 0x60D0_0000_0000_0001, Gen::default());
    println!("{tally:?}");
    assert!(tally.yes_runs > 0 && tally.no_runs > 0);
}
