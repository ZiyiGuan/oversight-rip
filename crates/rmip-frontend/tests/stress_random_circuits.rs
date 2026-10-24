use rmip_frontend::bits::{BitSlice, BitVec};
use rmip_frontend::circuit::{Gate, OracleCircuit, WireId};
use rmip_frontend::deoracle::{self, OracleSite, OrdinaryCircuit};
use rmip_frontend::layout::TraceLayout;
use rmip_frontend::oracle::{encode, Oracle};
use rmip_frontend::r1cs::{self, R1cs, R1csConfig, SparseF2};
use rmip_frontend::{loc, trace};
use sha2::{Digest, Sha256};
use std::cell::RefCell;

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
        ("keyed-B", Ora::Keyed(0x0BAD_C0FF_EE00_1234)),
    ]
}

struct Recording<'a> {
    inner: &'a Ora,
    seen: RefCell<Vec<Vec<bool>>>,
}

impl<'a> Recording<'a> {
    fn new(inner: &'a Ora) -> Self {
        Self {
            inner,
            seen: RefCell::new(Vec::new()),
        }
    }

    fn into_queries(self) -> Vec<Vec<bool>> {
        self.seen.into_inner()
    }
}

impl Oracle for Recording<'_> {
    fn query(&self, u: &BitSlice) -> bool {
        self.seen.borrow_mut().push(u.to_vec());
        self.inner.query(u)
    }
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
            max_n: 6,
            max_extra: 26,
            max_fanin: 7,
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
            0 if n > 0 && rng.flip() => Gate::Input(rng.below(n)),
            0 => Gate::Const(rng.flip()),
            1 => Gate::Not(rng.below(here)),
            2 | 3 => Gate::And(rng.below(here), rng.below(here)),
            4 | 5 => Gate::Or(rng.below(here), rng.below(here)),
            6 | 7 => Gate::Xor(rng.below(here), rng.below(here)),
            _ => {
                let query = if !past_queries.is_empty() && rng.below(4) == 0 {
                    let which = rng.below(past_queries.len());
                    past_queries[which].clone()
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

struct Built {
    dc: OrdinaryCircuit,
    sites: Vec<OracleSite>,
    layout: TraceLayout,
    loc: Vec<Vec<usize>>,
    sys: R1cs,
}

fn build(c: &OracleCircuit) -> Built {
    let (dc, sites) = deoracle::deoraclize(c);
    let layout = TraceLayout::new(&dc);
    let loc = loc::build(&sites, &layout);
    let sys = r1cs::emit(&dc, &layout, &R1csConfig::default());
    Built {
        dc,
        sites,
        layout,
        loc,
        sys,
    }
}

fn random_input(rng: &mut Rng, n: usize) -> BitVec {
    BitVec::from_bools((0..n).map(|_| rng.flip()).collect())
}

fn r1cs_satisfied(sys: &R1cs, z: &BitSlice) -> bool {
    let mul = |m: &SparseF2| {
        let mut v = vec![false; m.rows];
        for &(r, col) in &m.nz {
            if z[col] {
                v[r] ^= true;
            }
        }
        v
    };
    let (a, b, c) = (mul(&sys.a), mul(&sys.b), mul(&sys.c));
    (0..a.len()).all(|i| (a[i] && b[i]) == c[i])
}

fn relation_holds(b: &Built, x: &BitSlice, z: &BitVec) -> bool {
    let p = trace::public_block(&b.layout, x);
    (0..b.layout.p).all(|i| z.get(i) == p.get(i)) && r1cs_satisfied(&b.sys, z.as_slice())
}

fn loc_agrees_with_recomputation(b: &Built, x: &BitSlice, z: &BitVec) -> Result<(), String> {
    let a = BitVec::from_bools(
        (0..b.layout.ell)
            .map(|j| z.get(b.layout.answer_coord(j)))
            .collect(),
    );
    for j in 0..b.layout.ell {
        let (read, answer) = trace::read_block(z.as_slice(), &b.loc[j]);
        let recomputed = b.dc.query_bundle(&b.sites, j, x, a.as_slice());
        if read.as_slice() != recomputed.as_slice() {
            return Err(format!(
                "occurrence {j}: loc reads {read:?} but the circuit asks {recomputed:?}"
            ));
        }
        if answer != a.get(j) {
            return Err(format!("occurrence {j}: loc's answer slot is not a_j"));
        }
    }
    Ok(())
}

fn check_exactness_and_loc(
    c: &OracleCircuit,
    b: &Built,
    o: &Ora,
    x: &BitVec,
    seed: u64,
) -> Vec<Vec<bool>> {
    let rec = Recording::new(o);
    let (y, a) = c.eval_with_answers(x.as_slice(), &rec);
    let asked = rec.into_queries();
    let ell = c.ell();

    assert_eq!(
        asked.len(),
        ell,
        "seed {seed}: one oracle call per occurrence"
    );

    assert_eq!(
        b.dc.eval(x.as_slice(), a.as_slice()),
        y,
        "seed {seed}: C^O(x) != Ĉ(x, a)"
    );

    for (j, truth) in asked.iter().enumerate() {
        let bundle = b.dc.query_bundle(&b.sites, j, x.as_slice(), a.as_slice());
        assert_eq!(
            bundle.as_slice(),
            &truth[..],
            "seed {seed}: occurrence {j} query bits differ"
        );
        assert_eq!(
            encode(bundle.as_slice()),
            encode(truth),
            "seed {seed}: occurrence {j} encodes differently"
        );
    }

    assert!(
        deoracle::bad_set(&b.dc, &b.sites, x.as_slice(), a.as_slice(), o).is_empty(),
        "seed {seed}: the true execution has a non-empty Bad set"
    );

    let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
    let widths = c.b_widths();
    for (j, truth) in asked.iter().enumerate() {
        assert_eq!(
            b.loc[j].len(),
            widths[j] + 1,
            "seed {seed}: loc row {j} is not b_j + 1 long"
        );
        let (read, answer) = trace::read_block(z.as_slice(), &b.loc[j]);
        assert_eq!(
            read.as_slice(),
            &truth[..],
            "seed {seed}: loc row {j} assembles the wrong query"
        );
        assert_eq!(answer, a.get(j), "seed {seed}: loc row {j} answer slot");
    }
    asked
}

#[derive(Default, Debug)]
struct FlipReport {
    ordinary: usize,
    public_caught_by_rows: usize,
    public_caught_by_public_check_only: usize,
    answer_broken: usize,
    answer_free: usize,
    tail: usize,
}

fn check_single_flips(
    b: &Built,
    x: &BitVec,
    a: &BitVec,
    seed: u64,
    report: &mut FlipReport,
) -> bool {
    let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
    let honest_ok = relation_holds(b, x.as_slice(), &z);
    if !honest_ok {
        return false;
    }
    loc_agrees_with_recomputation(b, x.as_slice(), &z)
        .unwrap_or_else(|e| panic!("seed {seed}: honest trace: {e}"));

    let (p, ell, used_end, n_trace) = (
        b.layout.p,
        b.layout.ell,
        b.layout.used_end(),
        b.layout.n_trace,
    );
    for i in 0..n_trace {
        let mut bad = z.clone();
        bad.set(i, !bad.get(i));
        let rows_ok = r1cs_satisfied(&b.sys, bad.as_slice());
        let rel_ok = relation_holds(b, x.as_slice(), &bad);
        if i < p {
            assert!(
                !rel_ok,
                "seed {seed}: flipping public-block coordinate {i} kept the relation"
            );
            if rows_ok {
                report.public_caught_by_public_check_only += 1;
            } else {
                report.public_caught_by_rows += 1;
            }
        } else if i < p + ell {
            if rel_ok {
                report.answer_free += 1;
                loc_agrees_with_recomputation(b, x.as_slice(), &bad).unwrap_or_else(|e| {
                    panic!("seed {seed}: answer flip at {i} desynchronised loc: {e}")
                });
            } else {
                report.answer_broken += 1;
            }
        } else if i < used_end {
            assert!(
                !rows_ok,
                "seed {seed}: flipping ordinary-wire coordinate {i} \
                 (P={p}, ell={ell}, used_end={used_end}) kept the R1CS satisfied"
            );
            report.ordinary += 1;
        } else {
            assert!(
                rel_ok,
                "seed {seed}: flipping trailing-pad coordinate {i} broke the relation"
            );
            loc_agrees_with_recomputation(b, x.as_slice(), &bad)
                .unwrap_or_else(|e| panic!("seed {seed}: tail flip at {i} moved a loc block: {e}"));
            report.tail += 1;
        }
    }
    true
}

fn independent_levels(c: &OracleCircuit) -> Vec<usize> {
    let ell = c.ell();
    let mut reach: Vec<Vec<bool>> = vec![Vec::new(); c.gates.len()];
    let mut level = vec![0usize; ell];
    for (i, g) in c.gates.iter().enumerate() {
        let ops: Vec<WireId> = match g {
            Gate::Input(_) | Gate::Const(_) => vec![],
            Gate::Answer(_) => unreachable!("Answer in an OracleCircuit"),
            Gate::Not(p) => vec![*p],
            Gate::And(p, q) | Gate::Or(p, q) | Gate::Xor(p, q) => vec![*p, *q],
            Gate::Xor3(p, q, s) | Gate::Maj3(p, q, s) => vec![*p, *q, *s],
            Gate::Oracle { query, .. } => query.clone(),
        };
        let mut r = vec![false; ell];
        for p in ops {
            for j in 0..ell {
                r[j] |= reach[p][j];
            }
        }
        if let Gate::Oracle { occurrence, .. } = g {
            level[*occurrence] = (0..ell)
                .filter(|&j| r[j])
                .map(|j| level[j] + 1)
                .max()
                .unwrap_or(0);
            r[*occurrence] = true;
        }
        reach[i] = r;
    }
    level
}

#[test]
fn the_level_decomposition_matches_its_definition_at_every_depth() {
    let mut depths = std::collections::BTreeMap::<usize, usize>::new();
    let zoo = oracle_zoo();
    for k in 0..trials(200) {
        let seed = 0x1E7E_1500_0000_0001u64.wrapping_add((k as u64).wrapping_mul(0x9E37_79B9));
        let mut rng = Rng::new(seed);
        let c = random_circuit(&mut rng, Gen::default());
        let level = independent_levels(&c);
        let d = level.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        assert_eq!(
            rmip_frontend::levels::decompose(&c).depth(),
            d,
            "seed {seed}: depth disagrees with the reachability computation"
        );
        *depths.entry(d).or_default() += 1;

        let (dc, sites) = deoracle::deoraclize(&c);
        for (_, o) in &zoo {
            let x = random_input(&mut rng, c.n);
            let (_, a) = c.eval_with_answers(x.as_slice(), o);
            for cut in 0..d {
                let mut a2 = a.clone();
                for (j, &l) in level.iter().enumerate() {
                    if l >= cut && rng.flip() {
                        a2.set(j, !a2.get(j));
                    }
                }
                for (j, &l) in level.iter().enumerate() {
                    if l < cut {
                        assert_eq!(
                            dc.query_bundle(&sites, j, x.as_slice(), a.as_slice())
                                .as_slice(),
                            dc.query_bundle(&sites, j, x.as_slice(), a2.as_slice())
                                .as_slice(),
                            "seed {seed}: occurrence {j} at level {l} moved when only \
                             levels >= {cut} were perturbed"
                        );
                    }
                }
            }
        }
    }
    println!("adaptivity depths reached: {depths:?}");
    assert!(
        depths.keys().any(|&d| d >= 3),
        "the sweep never reached d >= 3: {depths:?}"
    );
}

fn trials(default: usize) -> usize {
    std::env::var("RMIP_STRESS_TRIALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn sweep(circuits: usize, base_seed: u64, cfg: Gen) -> (FlipReport, usize, usize) {
    let zoo = oracle_zoo();
    let mut report = FlipReport::default();
    let mut triples = 0usize;
    let mut satisfying = 0usize;
    for k in 0..circuits {
        let seed = base_seed
            .wrapping_add(k as u64)
            .wrapping_mul(0x2545_F491_4F6C_DD1D);
        let mut rng = Rng::new(seed);
        let c = random_circuit(&mut rng, cfg);
        let b = build(&c);
        assert!(b.layout.nu > b.layout.k, "seed {seed}: nu <= k");
        for (_, o) in &zoo {
            for _ in 0..2 {
                let x = random_input(&mut rng, c.n);
                check_exactness_and_loc(&c, &b, o, &x, seed);
                let (_, a) = c.eval_with_answers(x.as_slice(), o);
                if check_single_flips(&b, &x, &a, seed, &mut report) {
                    satisfying += 1;
                }
                triples += 1;
            }
        }
    }
    (report, triples, satisfying)
}

#[test]
fn random_circuits_satisfy_exactness_tightness_and_loc() {
    let circuits = trials(60);
    let (report, triples, satisfying) = sweep(circuits, 0xD15E_A5E0_0000_0001, Gen::default());
    println!("{circuits} circuits, {triples} (circuit, oracle, input) triples, {satisfying} with a true claim");
    println!("{report:?}");
    assert!(
        report.ordinary > 0 && report.answer_free > 0 && report.tail > 0,
        "the sweep must actually reach all three coordinate classes: {report:?}"
    );
}

#[test]
#[ignore = "measurement"]
fn the_big_sweep() {
    let circuits = trials(1500);
    let cfg = Gen {
        max_n: 8,
        max_extra: 40,
        max_fanin: 12,
    };
    let (report, triples, satisfying) = sweep(circuits, 0xBEEF_0000_0000_0001, cfg);
    println!("{circuits} circuits, {triples} triples, {satisfying} with a true claim");
    println!("{report:?}");
}

#[test]
fn a_false_claim_never_satisfies_the_rows() {
    let mut false_claims = 0usize;
    for k in 0..trials(120) {
        let seed = 0x0FA1_5E00_0000_0001u64.wrapping_add(k as u64 * 0x9E37_79B9);
        let mut rng = Rng::new(seed);
        let c = random_circuit(&mut rng, Gen::default());
        let b = build(&c);
        for (_, o) in &oracle_zoo() {
            for _ in 0..4 {
                let x = random_input(&mut rng, c.n);
                let (y, a) = c.eval_with_answers(x.as_slice(), o);
                let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
                let ok = r1cs_satisfied(&b.sys, z.as_slice());
                assert_eq!(
                    ok, y,
                    "seed {seed}: the rows must accept the honest trace exactly when C^O(x) = 1"
                );
                if !y {
                    false_claims += 1;
                }
            }
        }
    }
    assert!(false_claims > 0, "no false claim was drawn");
}

#[test]
fn the_rows_accept_exactly_the_executions_of_some_answer_vector() {
    let mut drawn = 0usize;
    let mut accepting = 0usize;
    let mut k = 0u64;
    while drawn < trials(150) {
        let seed = 0x00DD_BA11_0000_0001u64.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        k += 1;
        let mut rng = Rng::new(seed);
        let c = random_circuit(
            &mut rng,
            Gen {
                max_n: 4,
                max_extra: 14,
                max_fanin: 5,
            },
        );
        let ell = c.ell();
        if ell == 0 || ell > 8 {
            continue;
        }
        drawn += 1;
        let b = build(&c);
        let x = random_input(&mut rng, c.n);
        for mask in 0..(1usize << ell) {
            let a = BitVec::from_bools((0..ell).map(|j| (mask >> j) & 1 == 1).collect());
            let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
            let want = b.dc.eval(x.as_slice(), a.as_slice());
            assert_eq!(
                relation_holds(&b, x.as_slice(), &z),
                want,
                "seed {seed}: mask {mask:b}"
            );
            if want {
                accepting += 1;
                loc_agrees_with_recomputation(&b, x.as_slice(), &z)
                    .unwrap_or_else(|e| panic!("seed {seed}: mask {mask:b}: {e}"));
                for i in b.layout.p + ell..b.layout.used_end() {
                    let mut bad = z.clone();
                    bad.set(i, !bad.get(i));
                    assert!(
                        !r1cs_satisfied(&b.sys, bad.as_slice()),
                        "seed {seed}: mask {mask:b}: coordinate {i} is not pinned"
                    );
                }
            }
        }
    }
    assert!(accepting > 0, "no accepting answer vector was reached");
}

#[test]
fn the_free_source_block_is_exactly_the_answers_and_the_tail() {
    for k in 0..trials(200) {
        let seed = 0xF2EE_0000_0000_0001u64.wrapping_add((k as u64).wrapping_mul(0x9E37_79B9));
        let mut rng = Rng::new(seed);
        let c = random_circuit(&mut rng, Gen::default());
        let (dc, _) = deoracle::deoraclize(&c);
        let layout = TraceLayout::new(&dc);
        let base = r1cs::emit(&dc, &layout, &R1csConfig::default());
        let blocked = r1cs::emit(
            &dc,
            &layout,
            &R1csConfig {
                booleanity: r1cs::Booleanity::AnswerBlockRows,
                force_square: false,
            },
        );
        let tail = layout.tail_pad_range();
        let expected: Vec<usize> = (0..layout.ell)
            .map(|j| layout.answer_coord(j))
            .chain(tail.clone())
            .collect();
        assert_eq!(
            blocked.m0,
            base.m0 + expected.len(),
            "seed {seed}: wrong number of Booleanity rows"
        );
        for (offset, &i) in expected.iter().enumerate() {
            let r = base.m0 + offset;
            assert!(
                blocked.a.contains(r, i) && blocked.b.contains(r, i) && blocked.c.contains(r, i),
                "seed {seed}: row {r} is not (e_{i}, e_{i}, e_{i})"
            );
        }
        for (_, o) in oracle_zoo() {
            let x = random_input(&mut rng, c.n);
            let (y, a) = c.eval_with_answers(x.as_slice(), &o);
            let z = trace::witness_trace(&dc, &layout, x.as_slice(), a.as_slice());
            assert_eq!(r1cs_satisfied(&blocked, z.as_slice()), y, "seed {seed}");
        }
    }
}

#[test]
fn edge_ell_zero_and_empty_query_are_rejected() {
    let no_oracle = OracleCircuit {
        n: 1,
        gates: vec![Gate::Input(0), Gate::Not(0)],
        output: 1,
    };
    let e = no_oracle.validate().expect_err("ell = 0 must be rejected");
    assert!(e.contains("ell = 0"), "{e}");

    let empty_query = OracleCircuit {
        n: 1,
        gates: vec![
            Gate::Input(0),
            Gate::Oracle {
                occurrence: 0,
                query: vec![],
            },
        ],
        output: 1,
    };
    let e = empty_query
        .validate()
        .expect_err("an empty query must be rejected");
    assert!(e.contains("empty oracle query"), "{e}");
}

#[test]
fn edge_smallest_circuit_n0_ell1() {
    let c = OracleCircuit {
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
    c.validate().expect("valid");
    let b = build(&c);
    assert_eq!(
        (b.layout.n, b.layout.ell, b.layout.k, b.layout.p),
        (0, 1, 0, 1)
    );
    assert_eq!(b.layout.g, 2);
    assert_eq!(b.layout.nu, 2);
    assert_eq!(b.layout.n_trace, 4);
    assert_eq!(b.layout.output_coord(), b.layout.answer_coord(0));
    assert_eq!(b.loc, vec![vec![2, 1]]);

    for (name, o) in oracle_zoo() {
        let x = BitVec::zeros(0);
        let (y, a) = c.eval_with_answers(x.as_slice(), &o);
        assert_eq!(y, a.get(0), "{name}: the output IS the answer");
        assert_eq!(y, o.query(&[true]), "{name}: query is the constant wire");
        let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
        assert_eq!(r1cs_satisfied(&b.sys, z.as_slice()), y, "{name}");
        let (read, ans) = trace::read_block(z.as_slice(), &b.loc[0]);
        assert_eq!(read.as_slice(), &[true], "{name}");
        assert_eq!(ans, y, "{name}");
    }
}

#[test]
fn edge_zero_length_input() {
    let mut rng = Rng::new(0x0000_1EAF_0000_0001);
    let mut seen = 0usize;
    for _ in 0..400 {
        let c = random_circuit(
            &mut rng,
            Gen {
                max_n: 0,
                ..Gen::default()
            },
        );
        assert_eq!(c.n, 0);
        let b = build(&c);
        assert_eq!((b.layout.k, b.layout.p), (0, 1));
        let x = BitVec::zeros(0);
        let p = trace::public_block(&b.layout, x.as_slice());
        assert_eq!(p.len(), 1);
        assert!(p.get(0));
        for (_, o) in oracle_zoo() {
            check_exactness_and_loc(&c, &b, &o, &x, 0x1EAF);
        }
        seen += 1;
    }
    assert_eq!(seen, 400);
}

#[test]
fn edge_maximal_fan_in() {
    let width = 512usize;
    let n = 4usize;
    let mut gates: Vec<Gate> = (0..n).map(Gate::Input).collect();
    gates.push(Gate::Const(false));
    gates.push(Gate::Const(true));
    let query: Vec<WireId> = (0..width).map(|i| i % 6).collect();
    gates.push(Gate::Oracle {
        occurrence: 0,
        query,
    });
    let c = OracleCircuit {
        n,
        gates,
        output: 6,
    };
    c.validate().expect("valid");
    let b = build(&c);
    assert_eq!(c.b_max(), width);
    assert_eq!(b.loc[0].len(), width + 1);
    let distinct: std::collections::BTreeSet<usize> = b.loc[0].iter().copied().collect();
    assert_eq!(
        distinct.len(),
        7,
        "6 query wires plus the answer, over {width} positions"
    );
    assert_eq!(loc::b_tr(&b.loc), width + 1);

    for (name, o) in oracle_zoo() {
        for v in 0..16u8 {
            let x = BitVec::from_bools((0..n).map(|i| (v >> i) & 1 == 1).collect());
            check_exactness_and_loc(&c, &b, &o, &x, u64::from(v));
            let (_, a) = c.eval_with_answers(x.as_slice(), &o);
            let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
            let (read, _) = trace::read_block(z.as_slice(), &b.loc[0]);
            assert_eq!(read.len(), width, "{name}: full-width bundle");
        }
    }
}

fn chain(n: usize, filler: usize) -> OracleCircuit {
    let mut gates: Vec<Gate> = (0..n).map(Gate::Input).collect();
    gates.push(Gate::Const(true));
    let cwire = gates.len() - 1;
    gates.push(Gate::Oracle {
        occurrence: 0,
        query: vec![cwire],
    });
    let awire = gates.len() - 1;
    let mut last = awire;
    for _ in 0..filler {
        gates.push(Gate::Or(last, cwire));
        last = gates.len() - 1;
    }
    let c = OracleCircuit {
        n,
        gates,
        output: last,
    };
    c.validate().expect("valid");
    c
}

#[test]
fn edge_trace_exactly_a_power_of_two_and_one_over() {
    let exact = chain(3, 2);
    let b = build(&exact);
    assert_eq!((b.layout.k, b.layout.p, b.layout.g), (2, 4, 4));
    assert_eq!(b.layout.used_end(), 8);
    assert_eq!(b.layout.nu, 3);
    assert_eq!(b.layout.n_trace, 8);
    assert!(
        b.layout.tail_pad_range().is_empty(),
        "an exact power of two has no trailing pad"
    );

    let over = chain(3, 3);
    let b2 = build(&over);
    assert_eq!(b2.layout.used_end(), 9);
    assert_eq!(b2.layout.nu, 4);
    assert_eq!(b2.layout.n_trace, 16);
    assert_eq!(b2.layout.tail_pad_range(), 9..16);

    for c in [&exact, &over] {
        let b = build(c);
        for (_, o) in oracle_zoo() {
            for v in 0..8u8 {
                let x = BitVec::from_bools((0..3).map(|i| (v >> i) & 1 == 1).collect());
                check_exactness_and_loc(c, &b, &o, &x, u64::from(v));
                let (_, a) = c.eval_with_answers(x.as_slice(), &o);
                let mut report = FlipReport::default();
                check_single_flips(&b, &x, &a, u64::from(v), &mut report);
            }
        }
    }
}

#[test]
fn edge_layout_sizes_match_their_closed_forms() {
    for n in 0..=130usize {
        let c = chain(n, 1);
        let b = build(&c);
        let k = (0..).find(|k| (1usize << k) > n).unwrap();
        assert_eq!(b.layout.k, k, "n = {n}");
        assert_eq!(b.layout.p, 1 << k, "n = {n}");
        let need = b.layout.p + b.layout.g;
        let nu = (0..).find(|nu| (1usize << nu) >= need).unwrap();
        assert_eq!(b.layout.nu, nu.max(k), "n = {n}, need = {need}");
        assert!(b.layout.n_trace >= b.layout.used_end(), "n = {n}");
        assert!(b.layout.nu > b.layout.k, "n = {n}: nu should exceed k");
    }
}
