use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::MnistInstance;
use rmip_frontend::r1cs::{Booleanity, R1csConfig};
use rmip_frontend::{deoracle, layout, r1cs, trace};

#[test]
fn every_gate_row_preserves_booleanity() {
    let one = F128::ONE;
    let b = [F128::ZERO, F128::ONE];
    let is_bool = |v: F128| v == F128::ZERO || v == F128::ONE;

    for &p in &b {
        for &q in &b {
            assert!(is_bool(p.mul(q)));
            assert!(is_bool(p.add(q)));
            assert!(is_bool(p.mul(q).add(p).add(q)));
        }
        assert!(is_bool(p.add(one)));
    }
    assert_eq!(F128::ONE.add(F128::ONE), F128::ZERO, "1 XOR 1 = 0");
    assert_eq!(
        F128::ONE.mul(F128::ONE).add(F128::ONE).add(F128::ONE),
        F128::ONE,
        "1 OR 1 = 1"
    );
    assert_eq!(
        F128::ZERO.mul(F128::ZERO).add(F128::ZERO).add(F128::ZERO),
        F128::ZERO,
        "0 OR 0 = 0"
    );
}

#[test]
fn source_rows_target_exactly_the_answers_and_unused_tail() {
    let mut with_a_tail = 0;
    let no_tail = {
        use rmip_frontend::circuit::CircuitBuilder;
        let mut b = CircuitBuilder::new(1);
        let x = b.input(0);
        let a = b.oracle(vec![x]);
        let g1 = b.not(a);
        let g2 = b.and(a, g1);
        let g3 = b.xor(g2, x);
        let g4 = b.or(g3, a);
        let g5 = b.not(g4);
        b.finish(g5)
    };
    for (name, circuit, expect_tail) in [
        ("one-input chain with P + g = N", no_tail, false),
        (
            "MnistInstance::new(1024)",
            MnistInstance::new(1024).build(),
            true,
        ),
    ] {
        let (dc, _) = deoracle::deoraclize(&circuit);
        let lay = layout::TraceLayout::new(&dc);
        let tail = lay.tail_pad_range().len();
        assert_eq!(tail > 0, expect_tail, "{name}: tail pad emptiness moved");
        if tail > 0 {
            with_a_tail += 1;
        }

        let base = r1cs::emit(&dc, &lay, &R1csConfig::default());
        let with = r1cs::emit(
            &dc,
            &lay,
            &R1csConfig {
                booleanity: Booleanity::AnswerBlockRows,
                ..Default::default()
            },
        );

        let free_sources = dc.ell + tail;
        assert_eq!(
            with.m0,
            base.m0 + free_sources,
            "{name}: one row per free source"
        );
        assert_eq!(
            with.z,
            base.z + 3 * free_sources,
            "{name}: three nonzeros each"
        );

        let row_support = |m: &r1cs::SparseF2, r: usize| -> Vec<usize> {
            m.nz.iter()
                .filter(|(rr, _)| *rr == r)
                .map(|(_, c)| *c)
                .collect()
        };
        let mut answers = 0;
        let mut tails = 0;
        for r in base.m0..with.m0 {
            let cols = row_support(&with.a, r);
            assert_eq!(
                cols.len(),
                1,
                "{name}: row {r} of A is not a single unit vector"
            );
            let coord = cols[0];
            if (0..dc.ell).any(|j| lay.answer_coord(j) == coord) {
                answers += 1;
            } else if lay.tail_pad_range().contains(&coord) {
                tails += 1;
            } else {
                panic!("{name}: row {r} constrains non-source coordinate {coord}");
            }
            assert_eq!(
                row_support(&with.b, r),
                vec![coord],
                "{name}: row {r} of B is not e_i"
            );
            assert_eq!(
                row_support(&with.c, r),
                vec![coord],
                "{name}: row {r} of C is not e_i"
            );
        }
        assert_eq!(answers, dc.ell, "{name}: one row per answer coordinate");
        assert_eq!(tails, tail, "{name}: one row per tail coordinate");
    }
    assert_eq!(
        with_a_tail, 1,
        "one of the two fixtures must exercise the tail rows"
    );
}

#[test]
fn non_boolean_answer_breaks_its_row() {
    let mnist = MnistInstance::new(16);
    let c = mnist.build();
    let (dc, _) = deoracle::deoraclize(&c);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(
        &dc,
        &lay,
        &R1csConfig {
            booleanity: Booleanity::AnswerBlockRows,
            ..Default::default()
        },
    );

    let x = mnist.yes();
    let (_, a) = c.eval_with_answers(x.as_slice(), &mnist.oracle());
    let z = trace::witness_trace(&dc, &lay, x.as_slice(), a.as_slice());

    let mut zf: Vec<F128> = z.as_slice().iter().map(|&b| F128::from_bool(b)).collect();
    let victim = lay.answer_coord(4);
    let free_sources = dc.ell + lay.tail_pad_range().len();
    let row = (sys.m0 - free_sources..sys.m0)
        .find(|&r| sys.a.contains(r, victim))
        .expect("the answer row exists");
    let dot = |m: &r1cs::SparseF2, r: usize, z: &[F128]| {
        m.nz.iter()
            .filter(|(rr, _)| *rr == r)
            .fold(F128::ZERO, |acc, (_, c)| acc.add(z[*c]))
    };

    assert_eq!(
        dot(&sys.a, row, &zf).mul(dot(&sys.b, row, &zf)),
        dot(&sys.c, row, &zf),
        "the honest Boolean trace must satisfy its own Booleanity row"
    );

    zf[victim] = F128::new(0xdead_beef);
    assert_ne!(
        dot(&sys.a, row, &zf).mul(dot(&sys.b, row, &zf)),
        dot(&sys.c, row, &zf),
        "x^2 = x has only the roots 0 and 1, so a non-Boolean value must fail"
    );
}

#[test]
fn free_source_rows_are_free_in_m_at_the_scan_shape() {
    let c = MnistInstance::new(1024).build();
    let (dc, _) = deoracle::deoraclize(&c);
    let lay = layout::TraceLayout::new(&dc);

    let base = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let with = r1cs::emit(
        &dc,
        &lay,
        &R1csConfig {
            booleanity: Booleanity::AnswerBlockRows,
            ..Default::default()
        },
    );
    let full = r1cs::emit(
        &dc,
        &lay,
        &R1csConfig {
            booleanity: Booleanity::ExplicitRows,
            ..Default::default()
        },
    );

    let free_sources = dc.ell + lay.tail_pad_range().len();
    assert_eq!(base.m, lay.n_trace, "the scan shape is square: M = N");
    assert!(
        base.m - base.m0 >= free_sources,
        "slack must absorb the source rows"
    );
    assert_eq!(
        base.m - base.m0 - free_sources,
        lay.p - 1,
        "the margin is the public subcube less the output row"
    );
    assert_eq!(with.m, base.m, "M unchanged");
    assert_eq!(with.mu, base.mu, "mu unchanged");
    assert!(full.m > base.m, "the N-row variant does NOT fit");
    assert_eq!(with.z - base.z, 3 * free_sources);
    assert_eq!(full.z - base.z, 3 * lay.n_trace);
    assert!(with.z - base.z < full.z - base.z);
}
