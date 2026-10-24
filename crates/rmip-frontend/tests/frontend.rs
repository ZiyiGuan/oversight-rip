use rmip_frontend::bits::{BitSlice, BitVec};
use rmip_frontend::circuit::{CircuitBuilder, Gate, OracleCircuit};
use rmip_frontend::mnist::{label_bits, MnistInstance, SparseTableOracle, LABEL_BITS};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::r1cs::{Booleanity, R1csConfig};
use rmip_frontend::{deoracle, layout, levels, loc, r1cs, trace};

const VIEWS: usize = 16;

struct Built {
    dc: deoracle::OrdinaryCircuit,
    sites: Vec<deoracle::OracleSite>,
    layout: layout::TraceLayout,
    loc: Vec<Vec<usize>>,
    r1cs: r1cs::R1cs,
}

fn build(c: &OracleCircuit, cfg: &R1csConfig) -> Built {
    let (dc, sites) = deoracle::deoraclize(c);
    let lay = layout::TraceLayout::new(&dc);
    let l = loc::build(&sites, &lay);
    let sys = r1cs::emit(&dc, &lay, cfg);
    Built {
        dc,
        sites,
        layout: lay,
        loc: l,
        r1cs: sys,
    }
}

fn fixture() -> (MnistInstance, OracleCircuit, SparseTableOracle, Built) {
    let inst = MnistInstance::new(VIEWS);
    let c = inst.build();
    let o = inst.oracle();
    let b = build(&c, &R1csConfig::default());
    (inst, c, o, b)
}

fn all_labels() -> impl Iterator<Item = (usize, BitVec)> {
    (0..1usize << LABEL_BITS).map(|y| (y, label_bits(y)))
}

fn satisfied(sys: &r1cs::R1cs, z: &BitVec) -> bool {
    let az = sys.a.mul_vec(z.as_slice());
    let bz = sys.b.mul_vec(z.as_slice());
    let cz = sys.c.mul_vec(z.as_slice());
    (0..sys.m).all(|r| az[r] & bz[r] == cz[r])
}

struct FirstBit;

impl Oracle for FirstBit {
    fn query(&self, u: &BitSlice) -> bool {
        u.first().copied().unwrap_or(false)
    }
}

#[test]
fn validate_is_what_makes_b_widths_total() {
    let gates = |second: usize| {
        vec![
            Gate::Input(0),
            Gate::Oracle {
                occurrence: 0,
                query: vec![0],
            },
            Gate::Oracle {
                occurrence: second,
                query: vec![0, 1],
            },
            Gate::Or(1, 2),
        ]
    };
    let skipped = OracleCircuit {
        n: 1,
        gates: gates(2),
        output: 3,
    };
    let err = skipped
        .validate()
        .expect_err("a skipped occurrence id must be rejected");
    assert!(
        err.contains("occurrence 2 out of order, expected 1"),
        "the message must name the id and the expected one, got `{err}`"
    );
    assert_eq!(
        skipped.ell(),
        2,
        "ell counts gates, so it does not see the gap"
    );
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let panicked = std::panic::catch_unwind(|| skipped.b_widths());
    std::panic::set_hook(hook);
    assert!(
        panicked.is_err(),
        "b_widths must panic rather than return a wrong width"
    );

    let sound = OracleCircuit {
        n: 1,
        gates: gates(1),
        output: 3,
    };
    sound.validate().expect("call-order ids validate");
    assert_eq!(sound.b_widths(), vec![1, 2], "b_j in occurrence order");
    assert_eq!(sound.b_max(), 2);
}

#[test]
fn t1_exactness_forward() {
    let (inst, c, o, b) = fixture();
    for (y, x) in all_labels() {
        let (truth, a) = c.eval_with_answers(x.as_slice(), &o);
        assert_eq!(b.dc.eval(x.as_slice(), a.as_slice()), truth, "label {y}");
        let closed_form = y < 10 && inst.votes[y] >= inst.threshold;
        assert_eq!(truth, closed_form, "label {y}: the vote table decides");
        assert_eq!(a.len(), inst.ell(), "the full query plan on every label");
        if y < 10 {
            assert_eq!(a.as_slice().to_vec(), inst.honest_answers(y), "label {y}");
        }
    }
}

#[test]
fn t2_exactness_general() {
    let (_, c, o, b) = fixture();
    for (_, x) in all_labels() {
        let (truth, a) = c.eval_with_answers(x.as_slice(), &o);
        let bad = deoracle::bad_set(&b.dc, &b.sites, x.as_slice(), a.as_slice(), &o);
        assert!(
            bad.is_empty(),
            "the true execution is self-consistent by construction"
        );
        assert_eq!(b.dc.eval(x.as_slice(), a.as_slice()), truth);
    }
}

#[test]
fn t3_robust_local_gap_is_tight() {
    let (inst, c, o, b) = fixture();
    let ell = inst.ell();
    let eps = inst.eps_rob;

    let min_bad_over = |x: &BitVec, verdict_to_force: bool| -> usize {
        let mut min_bad = usize::MAX;
        for mask in 0u32..(1 << ell) {
            let mut a = BitVec::zeros(ell);
            for j in 0..ell {
                a.set(j, (mask >> j) & 1 == 1);
            }
            if b.dc.eval(x.as_slice(), a.as_slice()) != verdict_to_force {
                continue;
            }
            let bad = deoracle::bad_set(&b.dc, &b.sites, x.as_slice(), a.as_slice(), &o).len();
            assert!(
                bad as u64 * eps.den > eps.num * ell as u64,
                "robust local gap violated: |Bad| = {bad} on a trace that flips the verdict"
            );
            min_bad = min_bad.min(bad);
        }
        min_bad
    };

    let no = inst.no();
    assert!(!c.eval(no.as_slice(), &o), "must be a no-instance");
    let min_no = min_bad_over(&no, true);
    assert_eq!(
        min_no,
        inst.minimal_bad_set_size(),
        "no-instance: T - S_no corruptions are the cheapest satisfying vector"
    );
    assert_eq!(
        min_no,
        inst.threshold - inst.votes[inst.no_label],
        "the closed form is T - S_no"
    );

    let yes = inst.yes();
    assert!(c.eval(yes.as_slice(), &o), "must be a yes-instance");
    let min_yes = min_bad_over(&yes, false);
    assert_eq!(
        min_yes,
        inst.votes[inst.true_label] - inst.threshold + 1,
        "yes-instance: S_true - T + 1 corruptions are the cheapest rejecting vector"
    );

    assert_eq!(
        min_no.min(min_yes),
        eps.min_bad_set(ell),
        "the gap must be TIGHT: the radius is attained on one side"
    );
    assert_eq!(min_no.min(min_yes), inst.d_min, "and it is d_min");
}

#[test]
fn t4_loc_reads_the_query_block() {
    let (inst, c, o, b) = fixture();
    for (_, x) in all_labels() {
        let (_, a) = c.eval_with_answers(x.as_slice(), &o);
        let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());

        for j in 0..inst.ell() {
            let (u_read, a_read) = trace::read_block(z.as_slice(), &b.loc[j]);
            let u_true = b.dc.query_bundle(&b.sites, j, x.as_slice(), a.as_slice());
            assert_eq!(u_read, u_true, "loc({j}) query coordinates");
            assert_eq!(a_read, a.get(j), "loc({j}) answer coordinate");
        }
    }
}

#[test]
fn t5_honest_trace_satisfies_r1cs() {
    let (_, c, o, b) = fixture();
    let mut checked = 0;
    for (_, x) in all_labels() {
        if !c.eval(x.as_slice(), &o) {
            continue;
        }
        let (_, a) = c.eval_with_answers(x.as_slice(), &o);
        let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
        assert!(satisfied(&b.r1cs, &z), "unsatisfied by the honest trace");
        checked += 1;
    }
    assert!(checked > 0, "the true label is a yes-instance");
}

#[test]
fn t6_corrupt_a_gate_output_breaks_its_row() {
    let (inst, c, o, b) = fixture();
    let x = inst.yes();
    assert!(c.eval(x.as_slice(), &o));
    let (_, a) = c.eval_with_answers(x.as_slice(), &o);
    let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
    assert!(
        satisfied(&b.r1cs, &z),
        "the honest trace must satisfy the system"
    );

    let mut checked = 0;
    for (w, gate) in b.dc.gates.iter().enumerate() {
        if matches!(gate, Gate::Input(_) | Gate::Answer(_)) {
            continue;
        }
        let coord = b.layout.coord_of(w);
        let mut bad = z.clone();
        bad.set(coord, !bad.get(coord));
        assert!(
            !satisfied(&b.r1cs, &bad),
            "flipping gate {w} output (coord {coord}) left every row satisfied"
        );
        checked += 1;
    }
    assert!(checked > 0, "no gate outputs were exercised");

    let input_coord = b.layout.coord_of(0);
    let mut flipped = z.clone();
    flipped.set(input_coord, !flipped.get(input_coord));
    let _ = satisfied(&b.r1cs, &flipped);
}

#[test]
fn t7_subcube_layout() {
    let (inst, c, o, b) = fixture();
    let x = inst.yes();
    let (_, a) = c.eval_with_answers(x.as_slice(), &o);
    let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
    let p = trace::public_block(&b.layout, x.as_slice());

    assert_eq!(
        b.layout.k,
        (LABEL_BITS + 1).next_power_of_two().trailing_zeros() as usize
    );
    assert_eq!(b.layout.k, 3, "k = ceil(log2(1+4)) = 3");
    assert_eq!(b.layout.p, 1 << b.layout.k);

    for s in b.layout.public_range() {
        assert_eq!(z.get(s), p.get(s), "public subcube coordinate {s}");
    }
    assert!(z.get(0), "z_0 is the constant 1");
    for i in 0..LABEL_BITS {
        assert_eq!(z.get(1 + i), x.get(i));
    }
    for s in (1 + LABEL_BITS)..b.layout.p {
        assert!(!z.get(s), "inner pad coordinate {s} must be zero");
    }
    for j in 0..inst.ell() {
        assert_eq!(b.layout.answer_coord(j), b.layout.p + j);
        assert_eq!(z.get(b.layout.p + j), a.get(j));
    }
    for s in b.layout.tail_pad_range() {
        assert!(
            !z.get(s),
            "the canonical honest trace zeroes tail coordinate {s}"
        );
    }
}

#[test]
fn t7_unused_boolean_tail_is_free_of_r1cs_and_local_reads() {
    let mut builder = CircuitBuilder::new(1);
    let input = builder.input(0);
    let answer = builder.oracle(vec![input]);
    let negated = builder.not(answer);
    let output = builder.not(negated);
    let c = builder.finish(output);
    let o = FirstBit;
    let b = build(&c, &R1csConfig::default());
    let x = BitVec::from_bools(vec![true]);
    let (_, a) = c.eval_with_answers(x.as_slice(), &o);
    let z = trace::witness_trace(&b.dc, &b.layout, x.as_slice(), a.as_slice());
    let tail = b.layout.tail_pad_range();
    assert!(
        !tail.is_empty(),
        "the fixture must exercise a real unused tail"
    );
    assert!(satisfied(&b.r1cs, &z));

    let before_blocks: Vec<_> = b
        .loc
        .iter()
        .map(|loc_j| trace::read_block(z.as_slice(), loc_j))
        .collect();
    let mut alternate = z.clone();
    alternate.set(tail.start, true);

    assert!(
        satisfied(&b.r1cs, &alternate),
        "no R1CS row may constrain the unused tail"
    );
    assert_eq!(
        &alternate.as_slice()[b.layout.public_range()],
        &z.as_slice()[b.layout.public_range()],
        "the tail is outside the pinned public subcube"
    );
    let after_blocks: Vec<_> = b
        .loc
        .iter()
        .map(|loc_j| trace::read_block(alternate.as_slice(), loc_j))
        .collect();
    assert_eq!(
        after_blocks, before_blocks,
        "no loc block may reference the unused tail"
    );
}

#[test]
fn t8_level_decomposition() {
    let (_, c, _, _) = fixture();
    assert_eq!(
        levels::decompose(&c).depth(),
        1,
        "the MNIST circuit is one level"
    );

    let mut bld = CircuitBuilder::new(1);
    let input = bld.input(0);
    let first = bld.oracle(vec![input]);
    let second = bld.oracle(vec![input, first]);
    let out = bld.xor(first, second);
    let two = bld.finish(out);
    assert_eq!(
        levels::decompose(&two).depth(),
        2,
        "an answer feeding a later query is a second level"
    );
}

#[test]
fn t9_parameter_report() {
    let (inst, c, _, b) = fixture();
    let rep = rmip_frontend::report::ParamReport {
        n_trace: b.layout.n_trace,
        nu: b.layout.nu,
        m: b.r1cs.m,
        mu: b.r1cs.mu,
        z: b.r1cs.z,
        ell: inst.ell(),
        b_max: c.b_max(),
        k: b.layout.k,
        depth: levels::decompose(&c).depth(),
        b_tr: loc::b_tr(&b.loc),
        eps_rob: inst.eps_rob,
        booleanity: Booleanity::FromBinding,
    };

    assert_eq!(rep.ell, VIEWS);
    assert_eq!(rep.b_max, 64, "one 64-bit judge query per view");
    assert_eq!(rep.k, 3);
    assert_eq!(rep.eps_rob, rmip_frontend::Ratio::new(7, 16));
    assert_eq!(rep.t_for(0.125), 4, "t = 4 at sigma_loc = 1/8");
    let mut perfect_gap = rep.clone();
    perfect_gap.eps_rob = rmip_frontend::Ratio::new(1, 1);
    assert_eq!(perfect_gap.t_for(f64::MIN_POSITIVE), 1);
    assert_eq!(rep.k_max(4), 4 * (64 + 1));
    assert_eq!(rep.n_trace, 8 * VIEWS, "the trace is N = 8R");
    assert_eq!(rep.depth, 1);

    assert!(rep.m.is_power_of_two());
    assert!(rep.n_trace.is_power_of_two());

    eprintln!("{rep:#?}");
}

#[test]
fn t9b_booleanity_rows_are_switchable() {
    let (_, c, _, _) = fixture();
    let plain = build(&c, &R1csConfig::default());
    let source_rows = build(
        &c,
        &R1csConfig {
            booleanity: Booleanity::AnswerBlockRows,
            ..Default::default()
        },
    );
    let rows = build(
        &c,
        &R1csConfig {
            booleanity: Booleanity::ExplicitRows,
            ..Default::default()
        },
    );

    let free_sources = plain.dc.ell + plain.layout.tail_pad_range().len();
    assert_eq!(source_rows.r1cs.m0, plain.r1cs.m0 + free_sources);
    assert_eq!(source_rows.r1cs.z, plain.r1cs.z + 3 * free_sources);
    for (offset, coord) in (0..plain.dc.ell)
        .map(|j| plain.layout.answer_coord(j))
        .chain(plain.layout.tail_pad_range())
        .enumerate()
    {
        let row = plain.r1cs.m0 + offset;
        assert!(source_rows.r1cs.a.contains(row, coord));
        assert!(source_rows.r1cs.b.contains(row, coord));
        assert!(source_rows.r1cs.c.contains(row, coord));
    }

    assert!(rows.r1cs.m0 >= plain.r1cs.m0 + plain.layout.n_trace);
    assert_eq!(rows.r1cs.z, plain.r1cs.z + 3 * plain.layout.n_trace);
}
