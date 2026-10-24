use rmip_backend_interleaved::{Ligero, LigeroReplay, Shape};
use rmip_demo::harness::mnist_case;
use rmip_field::{BinaryField, F128};
use rmip_frontend::bits::BitSlice;
use rmip_frontend::circuit::{CircuitBuilder, OracleCircuit};
use rmip_frontend::mnist::SparseTableOracle;
use rmip_frontend::oracle::Oracle;
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance, Ratio};
use rmip_iop::backend::{Coord, OpeningVerifier, Symbol};
use rmip_iop::common_protocol::{self, CommonError};
use rmip_iop::driver::{
    committed_trace, dummy_symbol, execute_recording, Cheat, Config, Occurrence, Reject, Statement,
    DUMMY_ORACLE,
};
use rmip_iop::matmle;
use rmip_iop::replay::{CoinDeriver, Replay, ReplayError, Tape, TapeProver, TranscriptVerifier};
use rmip_iop::transcript::{Entry, OracleId, RawChallenger, SeededChallenger};

const DUMMY_COORD: Coord = Coord::ZERO;

const SEEDS: [u64; 5] = [1, 7, 42, 1729, 2026];

struct SeededDeriver {
    ch: SeededChallenger,
}

impl SeededDeriver {
    fn new(seed: u64) -> Self {
        Self {
            ch: SeededChallenger::new(seed),
        }
    }
}

impl CoinDeriver for SeededDeriver {
    fn absorb_ordinary(&mut self, _bytes: &[u8]) {}
    fn absorb_post(&mut self, _id: OracleId, _root: Option<&[u8; 32]>) {}
    fn derive(&mut self) -> F128 {
        self.ch.sample()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum WireReject {
    Replay(ReplayError),
    PlanMismatch { index: usize },
    QueryCount { expected: usize, got: usize },
    Common(CommonError),
    OpeningMismatch,
}

fn wire_only_verify<O: Oracle>(
    st: &Statement<'_>,
    oracle: &O,
    cfg: &Config,
    shape: Shape,
    seed: u64,
    entries: &[Entry],
    occurrences: &[Occurrence],
) -> Result<(), WireReject> {
    let mut backend = LigeroReplay::new(shape);
    let mut tape = Tape::new(entries, Vec::new(), SeededDeriver::new(seed));

    tape.post_oracle(DUMMY_ORACLE);
    backend.replay_commit(&mut tape);

    let mut prover = TapeProver::starting_at(entries, tape.cursor());
    let common = common_protocol::verify_interactive(st, oracle, cfg, &mut prover, &mut tape);
    let reduced = common.verification.reduced_claim.clone();

    backend.replay_open(&reduced.point, &mut tape);

    let mut pending: Option<WireReject> = None;
    let defer = |p: &mut Option<WireReject>, e: WireReject| {
        if p.is_none() {
            *p = Some(e);
        }
    };

    let q_long = backend.q_long();
    let expected_q = cfg.audit_query_count(q_long);
    if occurrences.len() != expected_q {
        defer(
            &mut pending,
            WireReject::QueryCount {
                expected: expected_q,
                got: occurrences.len(),
            },
        );
    }

    let mut plan = backend.query_plan(&mut tape);
    let mut answered: Vec<(OracleId, Coord, Symbol<F128>)> = Vec::with_capacity(q_long);
    let mut prev: Option<Symbol<F128>> = None;
    for (i, o) in occurrences.iter().enumerate().take(q_long) {
        let Some((id, coord)) = plan.next(prev.as_ref()) else {
            defer(
                &mut pending,
                WireReject::QueryCount {
                    expected: q_long,
                    got: i,
                },
            );
            break;
        };
        if o.oracle != id || o.coord != coord {
            defer(&mut pending, WireReject::PlanMismatch { index: i });
        }
        prev = Some(o.answer.clone());
        answered.push((o.oracle, o.coord, o.answer.clone()));
    }
    if pending.is_none() && plan.next(prev.as_ref()).is_some() {
        defer(
            &mut pending,
            WireReject::QueryCount {
                expected: q_long,
                got: q_long + 1,
            },
        );
    }
    for (i, o) in occurrences
        .iter()
        .enumerate()
        .take(expected_q.min(occurrences.len()))
        .skip(q_long)
    {
        if o.oracle != DUMMY_ORACLE || o.coord != DUMMY_COORD || o.answer != dummy_symbol() {
            defer(&mut pending, WireReject::PlanMismatch { index: i });
        }
    }

    let opening_ok = backend
        .opening_verifier()
        .check(std::slice::from_ref(&reduced), &answered);
    if let Some(fault) = tape.into_fault() {
        return Err(WireReject::Replay(fault));
    }
    if let Some(e) = pending {
        return Err(e);
    }
    if let Err(e) = common.verification.verdict {
        return Err(WireReject::Common(e));
    }
    if !opening_ok {
        return Err(WireReject::OpeningMismatch);
    }
    Ok(())
}

enum Fo {
    Mnist(SparseTableOracle),
    Xor,
}

impl Oracle for Fo {
    fn query(&self, u: &BitSlice) -> bool {
        match self {
            Fo::Mnist(table) => table.query(u),
            Fo::Xor => u.iter().fold(false, |acc, &b| acc ^ b),
        }
    }
}

struct Fx {
    name: &'static str,
    inst: Instance,
    sys: r1cs::R1cs,
    lay: layout::TraceLayout,
    circuit: OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    oracle: Fo,
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

fn embedded_pair(f: &Fx) -> (rmip_holo::Embedded, rmip_holo::Embedded) {
    let (index, wit) = rmip_holo::preprocess(&f.sys.a, &f.sys.b, &f.sys.c, f.sys.mu, f.lay.nu);
    let prover_side = rmip_holo::Embedded { index, wit };
    let (index, _discarded) =
        rmip_holo::preprocess(&f.sys.a, &f.sys.b, &f.sys.c, f.sys.mu, f.lay.nu);
    let verifier_side = rmip_holo::Embedded {
        index,
        wit: rmip_holo::index::ProverIndex {
            tables: Vec::new(),
            flat: Vec::new(),
        },
    };
    (prover_side, verifier_side)
}

fn st_embedded<'a>(f: &'a Fx, h: &'a rmip_holo::Embedded) -> Statement<'a> {
    Statement {
        holo: Some(h),
        ..st(f)
    }
}

fn prepare(name: &'static str, circuit: OracleCircuit) -> Fx {
    circuit
        .validate()
        .expect("the fixture circuit must validate");
    let oracle = Fo::Xor;
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let locations = loc::build(&sites, &lay);
    let n = circuit.n;
    assert!(n <= 16, "the exhaustive search below needs a small input");
    let mut chosen = None;
    for bits in 0u32..(1u32 << n) {
        let mut x = rmip_frontend::bits::BitVec::zeros(n);
        for i in 0..n {
            x.set(i, (bits >> i) & 1 == 1);
        }
        if circuit.eval(x.as_slice(), &oracle) {
            chosen = Some(x);
            break;
        }
    }
    let x = chosen.unwrap_or_else(|| panic!("{name}: no accepting input"));
    let ell = circuit.ell();
    let inst = Instance {
        layout: lay.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(locations),
        ell,
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: Ratio::new(1, 2),
    };
    Fx {
        name,
        inst,
        sys,
        lay,
        circuit,
        dc,
        oracle,
    }
}

fn mnist_views(r: usize) -> Fx {
    let (case, oracle) = mnist_case(r);
    assert!(
        case.holo.is_none(),
        "{r} views sits below the shipped embedding threshold"
    );
    Fx {
        name: case.family,
        inst: case.inst,
        sys: case.sys,
        lay: case.layout,
        circuit: case.circuit,
        dc: case.dc,
        oracle: Fo::Mnist(oracle),
    }
}

fn mnist_16() -> Fx {
    let f = mnist_views(16);
    assert_eq!(f.lay.nu % 2, 1, "16 views must give an odd nu");
    f
}

fn mnist_32() -> Fx {
    let f = mnist_views(32);
    assert_eq!(f.lay.nu % 2, 0, "32 views must give an even nu");
    f
}

fn mixed_widths() -> Fx {
    let mut b = CircuitBuilder::new(4);
    let i0 = b.input(0);
    let i1 = b.input(1);
    let i2 = b.input(2);
    let i3 = b.input(3);
    let a1 = b.oracle(vec![i0, i1]);
    let a2 = b.oracle(vec![i0, i1, i2]);
    let a3 = b.oracle(vec![i1, i2, i3]);
    let t = b.xor(a1, a2);
    let u = b.xor(t, a3);
    let out = b.and(u, u);
    prepare("mixed-widths", b.finish(out))
}

const EXPECTED_KNOBS: usize = 11;

fn cheat_catalogue(f: &Fx) -> Vec<(&'static str, Cheat)> {
    let ordinary = f.lay.used_end() - 1;
    let mut rows = vec![
        ("honest", Cheat::default()),
        (
            "corrupt_trace_coord(3)",
            Cheat {
                corrupt_trace_coord: Some(3),
                ..Cheat::default()
            },
        ),
        (
            "corrupt_trace_coord(7)",
            Cheat {
                corrupt_trace_coord: Some(7),
                ..Cheat::default()
            },
        ),
        (
            "corrupt_trace_coord(used_end-1)",
            Cheat {
                corrupt_trace_coord: Some(ordinary),
                ..Cheat::default()
            },
        ),
        (
            "public_input_bits([0,1])",
            Cheat {
                public_input_bits: vec![0, 1],
                ..Cheat::default()
            },
        ),
        (
            "public_input_bits([0])",
            Cheat {
                public_input_bits: vec![0],
                ..Cheat::default()
            },
        ),
        (
            "bad_oracle_answer(0)",
            Cheat {
                bad_oracle_answer: Some(0),
                ..Cheat::default()
            },
        ),
        (
            "local_cheat_answers(all false)",
            Cheat {
                local_cheat_answers: Some(vec![false; f.inst.ell]),
                ..Cheat::default()
            },
        ),
        (
            "trace_swap",
            Cheat {
                trace_swap: true,
                ..Cheat::default()
            },
        ),
        (
            "wrong_zerocheck_endpoint",
            Cheat {
                wrong_zerocheck_endpoint: true,
                ..Cheat::default()
            },
        ),
        (
            "wrong_lincheck_endpoint",
            Cheat {
                wrong_lincheck_endpoint: true,
                ..Cheat::default()
            },
        ),
        (
            "non_boolean_coord(used_end-1)",
            Cheat {
                non_boolean_coord: Some((ordinary, 0x9999)),
                ..Cheat::default()
            },
        ),
    ];
    let pad = f.lay.tail_pad_range();
    if !pad.is_empty() {
        rows.push((
            "non_boolean_coord(tail pad)",
            Cheat {
                non_boolean_coord: Some((pad.start, 0x9999)),
                ..Cheat::default()
            },
        ));
    }
    rows
}

fn wire_inexpressible_cheats() -> Vec<(&'static str, Cheat)> {
    vec![
        (
            "supply_non_boolean_block(0x5)",
            Cheat {
                supply_non_boolean_block: Some(0x5),
                ..Cheat::default()
            },
        ),
        (
            "non_boolean_answer(0)",
            Cheat {
                non_boolean_answer: Some(0),
                ..Cheat::default()
            },
        ),
        (
            "corrupt_lincheck_round",
            Cheat {
                corrupt_lincheck_round: true,
                ..Cheat::default()
            },
        ),
    ]
}

fn shape_of(f: &Fx) -> Shape {
    Shape::balanced(f.lay.nu, 4, 4.0)
}

struct Row {
    driver: Result<(), Reject>,
    wire: Result<(), WireReject>,
    pad_claims: usize,
}

fn differential_with(
    driver_st: &Statement<'_>,
    wire_st: &Statement<'_>,
    oracle: &Fo,
    shape: Shape,
    cheat: &Cheat,
    seed: u64,
) -> Row {
    let cfg = Config {
        t: 2,
        ..Config::default()
    };
    let mut backend = Ligero::new(shape);
    let mut ch = SeededChallenger::new(seed);
    let (exec, driver) = execute_recording(driver_st, &mut backend, oracle, &cfg, cheat, &mut ch);
    let entries: Vec<Entry> = ch.transcript().entries().to_vec();
    let wire = wire_only_verify(
        wire_st,
        oracle,
        &cfg,
        shape,
        seed,
        &entries,
        &exec.occurrences,
    );
    Row {
        driver,
        wire,
        pad_claims: exec.pad_claims,
    }
}

fn differential(f: &Fx, cheat: &Cheat, seed: u64) -> Row {
    differential_with(&st(f), &st(f), &f.oracle, shape_of(f), cheat, seed)
}

#[test]
fn the_catalogue_covers_every_cheat_knob() {
    let f = mixed_widths();
    let names: Vec<&str> = cheat_catalogue(&f)
        .into_iter()
        .map(|(n, _)| n)
        .chain(wire_inexpressible_cheats().into_iter().map(|(n, _)| n))
        .collect();
    let knobs = [
        "corrupt_trace_coord",
        "public_input_bits",
        "bad_oracle_answer",
        "local_cheat_answers",
        "trace_swap",
        "supply_non_boolean_block",
        "non_boolean_answer",
        "wrong_zerocheck_endpoint",
        "wrong_lincheck_endpoint",
        "corrupt_lincheck_round",
        "non_boolean_coord",
    ];
    assert_eq!(
        knobs.len(),
        EXPECTED_KNOBS,
        "the knob list no longer names every field of Cheat"
    );
    for knob in knobs {
        assert!(
            names.iter().any(|n| n.starts_with(knob)),
            "no catalogue row for Cheat::{knob}"
        );
    }
}

#[test]
fn both_verifiers_accept_the_honest_prover() {
    for f in [mnist_16(), mnist_32(), mixed_widths()] {
        for seed in SEEDS {
            let row = differential(&f, &Cheat::default(), seed);
            assert!(
                row.driver.is_ok(),
                "{} seed {seed}: the honest driver must accept: {:?}",
                f.name,
                row.driver
            );
            assert_eq!(
                row.wire,
                Ok(()),
                "{} seed {seed}: the honest wire-only verifier must accept",
                f.name
            );
        }
    }
}

#[test]
fn the_mixed_width_fixture_exercises_a_padded_claim_list() {
    let f = mixed_widths();
    assert!(
        f.inst.b_widths.iter().any(|w| *w != f.inst.b_widths[0]),
        "the fixture must have non-uniform query widths, got {:?}",
        f.inst.b_widths
    );
    let mut seen_padded = false;
    for seed in 0u64..12 {
        let row = differential(&f, &Cheat::default(), seed);
        println!("mixed-widths seed {seed}: pad_claims = {}", row.pad_claims);
        seen_padded |= row.pad_claims > 0;
    }
    assert!(
        seen_padded,
        "no seed produced a padded claim list; the padding branch is untested"
    );
}

#[test]
fn the_wire_only_verifier_agrees_with_the_joint_driver_on_every_cheat() {
    let mut disagreements: Vec<String> = Vec::new();
    for f in [mnist_16(), mnist_32(), mixed_widths()] {
        let mut accepts = 0usize;
        let mut rejects = 0usize;
        for (name, cheat) in cheat_catalogue(&f) {
            for seed in SEEDS {
                let row = differential(&f, &cheat, seed);
                println!(
                    "{:<12} {:<32} seed {seed:>2}: driver = {:?}  wire = {:?}",
                    f.name, name, row.driver, row.wire
                );
                if row.driver.is_ok() != row.wire.is_ok() {
                    disagreements.push(format!(
                        "{} / {name} / seed {seed}: driver {:?}, wire {:?}",
                        f.name, row.driver, row.wire
                    ));
                }
                if let Err(r) = &row.driver {
                    if r.is_joint_opening() {
                        assert_eq!(
                            row.wire,
                            Err(WireReject::OpeningMismatch),
                            "{} / {name} / seed {seed}: the driver rejected at the joint \
                             opening, so the wire-only path must reject at ITS opening check",
                            f.name
                        );
                    }
                }
                if row.driver.is_ok() {
                    accepts += 1;
                } else {
                    rejects += 1;
                }
            }
        }
        assert!(
            accepts > 0 && rejects > 0,
            "{}: {accepts} accepts and {rejects} rejects -- the sweep is one-sided",
            f.name
        );
    }
    assert!(
        disagreements.is_empty(),
        "the wire-only verifier disagreed with the joint driver:\n  {}",
        disagreements.join("\n  ")
    );
}

#[test]
fn the_wire_only_verifier_agrees_with_the_joint_driver_on_an_embedded_statement() {
    let mut disagreements: Vec<String> = Vec::new();
    for f in [mnist_16(), mnist_32()] {
        let (prover_side, verifier_side) = embedded_pair(&f);
        let driver_st = st_embedded(&f, &prover_side);
        let wire_st = st_embedded(&f, &verifier_side);
        let shape = shape_of(&f);
        let mut accepts = 0usize;
        let mut rejects = 0usize;
        for (name, cheat) in cheat_catalogue(&f) {
            for seed in SEEDS {
                let row = differential_with(&driver_st, &wire_st, &f.oracle, shape, &cheat, seed);
                println!(
                    "embedded {:<12} {:<32} seed {seed:>2}: driver = {:?}  wire = {:?}",
                    f.name, name, row.driver, row.wire
                );
                if row.driver.is_ok() != row.wire.is_ok() {
                    disagreements.push(format!(
                        "{} / {name} / seed {seed}: driver {:?}, wire {:?}",
                        f.name, row.driver, row.wire
                    ));
                }
                if name == "honest" {
                    assert_eq!(
                        row.driver,
                        Ok(()),
                        "{} seed {seed}: the honest embedded driver must accept",
                        f.name
                    );
                    assert_eq!(
                        row.wire,
                        Ok(()),
                        "{} seed {seed}: the honest embedded wire-only verifier must accept",
                        f.name
                    );
                }
                if let Err(r) = &row.driver {
                    if r.is_joint_opening() {
                        assert_eq!(
                            row.wire,
                            Err(WireReject::OpeningMismatch),
                            "{} / {name} / seed {seed}: the driver rejected at the joint \
                             opening, so the wire-only path must reject at ITS opening check",
                            f.name
                        );
                    }
                    assert_ne!(
                        *r,
                        Reject::HolographicFrames,
                        "{} / {name} / seed {seed}: no knob here touches the holographic \
                         frames, so the honest sub-protocol must verify",
                        f.name
                    );
                }
                if row.driver.is_ok() {
                    accepts += 1;
                } else {
                    rejects += 1;
                }
            }
        }
        assert!(
            accepts > 0 && rejects > 0,
            "{}: {accepts} accepts and {rejects} rejects -- the sweep is one-sided",
            f.name
        );
    }
    assert!(
        disagreements.is_empty(),
        "on an embedded statement the wire-only verifier disagreed with the joint driver:\n  {}",
        disagreements.join("\n  ")
    );
}

#[test]
#[ignore = "demonstrates that the driver-only cheats are not expressible on the wire; run with --ignored --nocapture"]
fn the_driver_only_cheat_knobs_are_not_expressible_on_the_wire() {
    let mut divergences = 0usize;
    for f in [mnist_16(), mnist_32(), mixed_widths()] {
        for (name, cheat) in wire_inexpressible_cheats() {
            for seed in SEEDS {
                let row = differential(&f, &cheat, seed);
                println!(
                    "{:<12} {:<32} seed {seed:>2}: driver = {:?}  wire = {:?}",
                    f.name, name, row.driver, row.wire
                );
                assert!(
                    row.driver.is_err(),
                    "{} / {name} / seed {seed}: the driver must reject this knob",
                    f.name
                );
                if row.wire.is_ok() {
                    divergences += 1;
                }
                if name == "corrupt_lincheck_round" {
                    assert_eq!(
                        row.wire,
                        Ok(()),
                        "{} / seed {seed}: perturbing the verifier's own initial claim \
                         leaves an honest transcript, which must replay clean",
                        f.name
                    );
                }
            }
        }
    }
    println!("wire-accepting rows where the driver rejected: {divergences}");
    assert!(
        divergences > 0,
        "the divergence this test records has disappeared -- if the driver now \
         records the SUPPLIED block value, or `corrupt_lincheck_round` now \
         corrupts a message, delete this test and move those knobs into the \
         agreement sweep"
    );
}

fn and_only_circuit() -> OracleCircuit {
    let mut b = CircuitBuilder::new(2);
    let i0 = b.input(0);
    let i1 = b.input(1);
    let a = b.oracle(vec![i0, i1]);
    let out = b.and(a, a);
    b.finish(out)
}

fn r1cs_residual(sys: &r1cs::R1cs, z: &[F128]) -> Vec<F128> {
    let az = matmle::mul_vec(&sys.a, z);
    let bz = matmle::mul_vec(&sys.b, z);
    let cz = matmle::mul_vec(&sys.c, z);
    az.iter()
        .zip(&bz)
        .zip(&cz)
        .map(|((a, b), c)| a.mul(*b).add(*c))
        .collect()
}

#[test]
fn the_flipped_constant_slot_still_satisfies_every_r1cs_row() {
    let f = prepare("and-only", and_only_circuit());
    let statement = st(&f);

    let honest = committed_trace(&statement, &f.oracle, &Cheat::default());
    let flipped = committed_trace(
        &statement,
        &f.oracle,
        &Cheat {
            corrupt_trace_coord: Some(0),
            ..Cheat::default()
        },
    );

    assert_eq!(honest[0], F128::ONE, "the honest constant slot is 1");
    assert_eq!(
        flipped[0],
        F128::ZERO,
        "the flip must clear the constant slot"
    );
    for (i, (h, g)) in honest.iter().zip(&flipped).enumerate() {
        if i != 0 {
            assert_eq!(h, g, "the cheat must move coordinate 0 and nothing else");
        }
    }

    let honest_residual = r1cs_residual(&f.sys, &honest);
    assert!(
        !honest_residual.is_empty(),
        "the R1CS must have rows, or satisfaction is vacuous"
    );
    assert!(
        honest_residual.iter().all(|r| *r == F128::ZERO),
        "the honest trace must satisfy every row"
    );

    let residual = r1cs_residual(&f.sys, &flipped);
    let broken: Vec<usize> = residual
        .iter()
        .enumerate()
        .filter(|(_, r)| **r != F128::ZERO)
        .map(|(i, _)| i)
        .collect();
    println!(
        "and-only: {} rows, flipped z_0 -> 0, rows broken = {:?}",
        residual.len(),
        broken
    );
    assert!(
        broken.is_empty(),
        "every row must survive the flip; these did not: {broken:?}"
    );

    for (j, entry) in f.inst.loc_rows().enumerate() {
        for &coord in &entry {
            assert_eq!(
                flipped[coord], honest[coord],
                "loc[{j}] coordinate {coord} moved; the local check would reject \
                 for the wrong reason"
            );
        }
    }
}

#[test]
fn flipping_the_constant_slot_is_rejected_by_both_verifiers() {
    let f = prepare("and-only", and_only_circuit());
    let cheat = Cheat {
        corrupt_trace_coord: Some(0),
        ..Cheat::default()
    };
    for seed in [1u64, 7, 42, 1729] {
        let honest = differential(&f, &Cheat::default(), seed);
        assert!(
            honest.driver.is_ok() && honest.wire.is_ok(),
            "seed {seed}: the honest control must be accepted by both \
             (driver {:?}, wire {:?})",
            honest.driver,
            honest.wire
        );

        let row = differential(&f, &cheat, seed);
        println!(
            "and-only  z_0 := 0  seed {seed:>4}: driver = {:?}  wire = {:?}",
            row.driver, row.wire
        );
        let driver = row
            .driver
            .expect_err("seed {seed}: the joint driver must reject a flipped constant slot");
        assert!(
            driver.is_joint_opening(),
            "seed {seed}: the false public-input claim must surface at the joint \
             opening layer, got {driver:?}"
        );
        assert_eq!(
            row.wire,
            Err(WireReject::OpeningMismatch),
            "seed {seed}: the wire-only verifier must refuse the reduced claim"
        );
    }
}
