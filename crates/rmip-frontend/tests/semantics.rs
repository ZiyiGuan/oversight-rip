use rmip_frontend::bits::{BitSlice, BitVec};
use rmip_frontend::circuit::{CircuitBuilder, Gate};
use rmip_frontend::oracle::{encode, CountingOracle, Oracle};
use rmip_frontend::r1cs::{Booleanity, R1csConfig, SparseF2Builder};
use rmip_frontend::Ratio;
use rmip_frontend::{deoracle, layout, r1cs, trace};
use sha2::{Digest, Sha256};

struct ParityOfBits;

impl Oracle for ParityOfBits {
    fn query(&self, u: &BitSlice) -> bool {
        u.iter().filter(|b| **b).count() % 2 == 1
    }
}

struct ShaOverEncode;

impl Oracle for ShaOverEncode {
    fn query(&self, u: &BitSlice) -> bool {
        Sha256::digest(encode(u))[0] & 0x80 != 0
    }
}

#[test]
fn d1_repeated_term_cancels() {
    let mut b = SparseF2Builder::new(4, 4);
    b.add_term(0, 1);
    b.add_term(0, 1);
    let m = b.finish();
    assert_eq!(m.nnz(), 0, "e_1 + e_1 = 0 over F2");
    assert!(!m.contains(0, 1));
}

#[test]
fn d1_odd_multiplicity_survives() {
    for (count, expect) in [(1, true), (2, false), (3, true), (4, false)] {
        let mut b = SparseF2Builder::new(2, 2);
        for _ in 0..count {
            b.add_term(1, 0);
        }
        assert_eq!(b.finish().contains(1, 0), expect, "multiplicity {count}");
    }
}

#[test]
fn d1_degenerate_gate_rows_are_correct() {
    const ONE: usize = 0;
    const P: usize = 1;
    const O: usize = 2;
    let sat = |a: &rmip_frontend::r1cs::SparseF2,
               bb: &rmip_frontend::r1cs::SparseF2,
               c: &rmip_frontend::r1cs::SparseF2,
               z: &[bool]| {
        let (az, bz, cz) = (a.mul_vec(z), bb.mul_vec(z), c.mul_vec(z));
        (0..az.len()).all(|i| az[i] & bz[i] == cz[i])
    };

    let (mut a, mut b, mut c) = (
        SparseF2Builder::new(1, 3),
        SparseF2Builder::new(1, 3),
        SparseF2Builder::new(1, 3),
    );
    a.add_form(0, [ONE, ONE]);
    b.add_term(0, ONE);
    c.add_term(0, O);
    let (a, b, c) = (a.finish(), b.finish(), c.finish());
    assert_eq!(a.nnz(), 0, "A must cancel to the zero form");
    assert!(
        sat(&a, &b, &c, &[true, false, false]),
        "z_o = 0 must satisfy"
    );
    assert!(!sat(&a, &b, &c, &[true, false, true]), "z_o = 1 must not");

    let (mut a, mut b, mut c) = (
        SparseF2Builder::new(1, 3),
        SparseF2Builder::new(1, 3),
        SparseF2Builder::new(1, 3),
    );
    a.add_term(0, P);
    b.add_term(0, P);
    c.add_form(0, [O, P, P]);
    let (a, b, c) = (a.finish(), b.finish(), c.finish());
    assert_eq!(
        c.nz,
        vec![(0, O)],
        "the two e_p terms must cancel, leaving e_o"
    );
    for zp in [false, true] {
        assert!(sat(&a, &b, &c, &[true, zp, zp]), "z_o = z_p must satisfy");
        assert!(!sat(&a, &b, &c, &[true, zp, !zp]), "z_o != z_p must not");
    }
}

#[test]
fn d1_const_gate_rows_are_the_fifth_shape_and_never_free_sources() {
    const ONE: usize = 0;
    let mut bld = CircuitBuilder::new(1);
    let x_wire = bld.input(0);
    let _oracle_occurrence = bld.oracle(vec![x_wire]);
    let t_wire = bld.constant(true);
    let f_wire = bld.constant(false);
    let not_f = bld.not(f_wire);
    let and_t = bld.and(t_wire, not_f);
    let out = bld.and(and_t, x_wire);
    let circuit = bld.finish(out);

    let (dc, _) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());

    let support = |m: &rmip_frontend::SparseF2| -> Vec<Vec<usize>> {
        let mut rows = vec![Vec::new(); m.rows];
        for &(r, c) in &m.nz {
            rows[r].push(c);
        }
        rows
    };
    let (arows, brows, crows) = (support(&sys.a), support(&sys.b), support(&sys.c));

    let consts: Vec<(usize, bool)> = dc
        .gates
        .iter()
        .enumerate()
        .filter_map(|(i, g)| match g {
            Gate::Const(v) => Some((i, *v)),
            _ => None,
        })
        .collect();
    assert_eq!(
        consts.len(),
        2,
        "the fixture has one constant of each value"
    );

    let mut const_coords = Vec::new();
    for (w, v) in consts {
        let o = lay.coord_of(w);
        const_coords.push(o);
        let want_c: Vec<usize> = if v { vec![ONE] } else { Vec::new() };
        let hits: Vec<usize> = (0..sys.m)
            .filter(|&r| arows[r] == vec![o] && brows[r] == vec![ONE] && crows[r] == want_c)
            .collect();
        assert_eq!(
            hits.len(),
            1,
            "Const({v}) must carry exactly one (e_o, e_0, v·e_0) row"
        );
    }

    let x = BitVec::from_bools(vec![true]);
    let (decision, answers) = circuit.eval_with_answers(x.as_slice(), &ParityOfBits);
    assert!(
        decision,
        "the fixture accepts, so the z_out row is satisfiable"
    );
    let z = trace::witness_trace(&dc, &lay, x.as_slice(), answers.as_slice());
    let sat = |z: &[bool]| {
        let (az, bz, cz) = (sys.a.mul_vec(z), sys.b.mul_vec(z), sys.c.mul_vec(z));
        (0..sys.m).all(|r| az[r] & bz[r] == cz[r])
    };
    assert!(sat(z.as_slice()), "the honest trace must satisfy");
    for &o in &const_coords {
        let mut bad: Vec<bool> = z.as_slice().to_vec();
        bad[o] = !bad[o];
        assert!(
            !sat(&bad),
            "flipping constant coordinate {o} must break its own row"
        );
    }

    let ab = r1cs::emit(
        &dc,
        &lay,
        &R1csConfig {
            booleanity: Booleanity::AnswerBlockRows,
            force_square: false,
        },
    );
    let (aa, ba, ca) = (support(&ab.a), support(&ab.b), support(&ab.c));
    let block: Vec<usize> = (sys.m0..ab.m0)
        .map(|r| {
            assert_eq!(aa[r].len(), 1, "a Booleanity row is (e_i, e_i, e_i)");
            assert!(
                aa[r] == ba[r] && ba[r] == ca[r],
                "a Booleanity row is (e_i, e_i, e_i)"
            );
            aa[r][0]
        })
        .collect();
    let want: Vec<usize> = (0..dc.ell)
        .map(|j| lay.answer_coord(j))
        .chain(lay.tail_pad_range())
        .collect();
    assert_eq!(
        block, want,
        "the free sources are the answers and the unused tail"
    );
    for o in const_coords {
        assert!(
            !block.contains(&o),
            "constant coordinate {o} carries a gate row, so it is not free"
        );
    }
}

#[test]
fn d1_emitter_supports_repeated_input_or_and_xor() {
    let mut bld = CircuitBuilder::new(1);
    let x_wire = bld.input(0);
    let _oracle_occurrence = bld.oracle(vec![x_wire]);
    let same_or = bld.or(x_wire, x_wire);
    let same_xor = bld.xor(x_wire, x_wire);
    let not_xor = bld.not(same_xor);
    let out = bld.and(same_or, not_xor);
    let circuit = bld.finish(out);

    let x = BitVec::from_bools(vec![true]);
    let (decision, answers) = circuit.eval_with_answers(x.as_slice(), &ParityOfBits);
    assert!(decision);
    let (dc, _) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let system = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let z = trace::witness_trace(&dc, &lay, x.as_slice(), answers.as_slice());
    let (az, bz, cz) = (
        system.a.mul_vec(z.as_slice()),
        system.b.mul_vec(z.as_slice()),
        system.c.mul_vec(z.as_slice()),
    );
    assert!(
        (0..system.m).all(|row| az[row] & bz[row] == cz[row]),
        "the honest trace with repeated-input gates must satisfy its emitted R1CS"
    );
}

#[test]
fn d1_z_counts_the_support() {
    let mut b = SparseF2Builder::new(2, 4);
    b.add_form(0, [0, 1, 2]);
    b.add_form(1, [3, 3]);
    let m = b.finish();
    assert_eq!(m.nnz(), 3, "Z counts surviving terms, not terms added");
    assert_eq!(
        m.nz,
        vec![(0, 0), (0, 1), (0, 2)],
        "sorted and deduplicated"
    );
}

#[test]
fn d1_mul_vec_xor_accumulates() {
    let mut b = SparseF2Builder::new(1, 3);
    b.add_form(0, [0, 2]);
    let m = b.finish();
    assert_eq!(m.mul_vec(&[true, false, false]), vec![true]);
    assert_eq!(m.mul_vec(&[true, false, true]), vec![false], "1 XOR 1 = 0");
}

#[test]
fn d2_encoding_separates_widths() {
    let one = BitVec::zeros(1);
    let two = BitVec::zeros(2);
    assert_eq!(
        one.to_bytes_be(),
        two.to_bytes_be(),
        "the packing primitive alone is NOT injective -- this is why encode exists"
    );
    assert_ne!(encode(one.as_slice()), encode(two.as_slice()));

    let o = ShaOverEncode;
    assert!(o.query(one.as_slice()), "SHA-256(00..01 00)[0] is 1");
    assert!(!o.query(two.as_slice()), "SHA-256(00..02 00)[0] is 0");
}

#[test]
fn d2_encoding_is_length_prefixed_big_endian() {
    let u = BitVec::from_bools(vec![true, false, true]);
    assert_eq!(u.to_bytes_be(), vec![0xA0]);
    assert_eq!(encode(u.as_slice()), vec![0, 0, 0, 0, 0, 0, 0, 3, 0xA0]);
    assert_eq!(
        encode(&[]),
        vec![0u8; 8],
        "empty query is still well defined"
    );
}

#[test]
fn d2_from_index_is_big_endian() {
    assert_eq!(BitVec::from_index(1, 3).as_slice(), &[false, false, true]);
    assert_eq!(BitVec::from_index(4, 3).as_slice(), &[true, false, false]);
    assert_eq!(BitVec::from_index(6, 3).as_slice(), &[true, true, false]);
}

#[test]
fn d2_counting_oracle_separates_occurrences_from_distinct() {
    let o = CountingOracle::new(ShaOverEncode);
    for _ in 0..100 {
        o.query(&[true, false]);
    }
    o.query(&[true, true]);
    assert_eq!(o.occurrences(), 101);
    assert_eq!(o.distinct(), 2);
}

#[test]
fn d3_min_bad_set_uses_strict_inequality() {
    assert_eq!(Ratio::new(1, 2).min_bad_set(9), 5);
    assert_eq!(Ratio::new(1, 2).min_bad_set(9), (9 + 1_usize).div_ceil(2));

    assert_eq!(Ratio::new(1, 2).min_bad_set(10), 6, "strict > 5, so 6");

    assert_eq!(Ratio::new(7, 16).min_bad_set(16), 8);
    assert_eq!(Ratio::new(125, 256).min_bad_set(256), 126);
}
