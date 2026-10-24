use rmip_demo::mnist_bench::{sparse_instance, sparse_mnist_case};
use rmip_frontend::bits::BitVec;
use rmip_frontend::mnist::{sparse_query_bytes, LABEL_BITS};
use rmip_frontend::{Ratio, SparseF2};

fn label_bits(y: usize) -> BitVec {
    BitVec::from_bools((0..LABEL_BITS).map(|k| (y >> k) & 1 == 1).collect())
}

fn byte_bits(bytes: &[u8]) -> Vec<bool> {
    bytes
        .iter()
        .flat_map(|&b| (0..8).map(move |k| (b >> (7 - k)) & 1 == 1))
        .collect()
}

fn r1cs_satisfied(sys: &rmip_frontend::r1cs::R1cs, z: &[bool]) -> bool {
    let mul = |m: &SparseF2| {
        let mut v = vec![false; m.rows];
        for &(r, c) in &m.nz {
            if z[c] {
                v[r] ^= true;
            }
        }
        v
    };
    let (a, b, c) = (mul(&sys.a), mul(&sys.b), mul(&sys.c));
    (0..a.len()).all(|i| (a[i] && b[i]) == c[i])
}

#[test]
fn real_sparse_majority_votes_flip_distances_and_queries() {
    for r in [256usize, 1024, 4096] {
        let inst = sparse_instance(r);
        let (case, oracle) = sparse_mnist_case(r, "sparse-mnist");
        assert_eq!(case.circuit.ell(), r, "R = {r}: one oracle gate per mask");
        assert!(case.circuit.b_widths().iter().all(|&w| w == 64));

        let t = inst.threshold;
        let mut d_min = usize::MAX;
        for y in 0..16usize {
            let x = label_bits(y);
            let s_y = if y < 10 { inst.votes[y] } else { 0 };
            let s_direct = inst.table.iter().filter(|&&p| usize::from(p) == y).count();
            if y < 10 {
                assert_eq!(s_y, s_direct, "R = {r}: manifest votes match the table");
            }
            let want = y < 10 && s_y >= t;
            let (got, answers) = case.circuit.eval_with_answers(x.as_slice(), &oracle);
            assert_eq!(got, want, "R = {r} label {y}");
            assert_eq!(answers.len(), r, "full query plan on every label");
            assert_eq!(
                got,
                y == inst.true_label,
                "exactly the true label is accepted"
            );

            if y >= 10 {
                continue;
            }
            let d_y = if s_y >= t { s_y - t + 1 } else { t - s_y };
            d_min = d_min.min(d_y);
            let flip = |count: usize| -> bool {
                let mut a: Vec<bool> = (0..r).map(|j| answers.get(j)).collect();
                let mut left = count;
                for v in a.iter_mut() {
                    if left > 0 && *v == want {
                        *v = !*v;
                        left -= 1;
                    }
                }
                assert_eq!(left, 0);
                case.dc.eval(x.as_slice(), &a)
            };
            assert_eq!(
                flip(d_y - 1),
                want,
                "R = {r} label {y}: {} flips hold",
                d_y - 1
            );
            assert_eq!(flip(d_y), !want, "R = {r} label {y}: {d_y} flips reverse");
        }
        assert_eq!(
            d_min, inst.d_min,
            "R = {r}: the recorded whole-circuit radius is exact"
        );

        assert_eq!(
            case.eps_rob,
            Ratio::new(inst.d_min as u64 - 1, r as u64),
            "R = {r}: eps_rob is (d_min - 1) / R"
        );
        assert_eq!(
            case.eps_rob.min_bad_set(r),
            d_min,
            "R = {r}: eps_rob must be tight at the exact flip radius"
        );
        let strongest_false = (0..10usize)
            .filter(|&y| y != inst.true_label)
            .max_by_key(|&y| (inst.votes[y], usize::MAX - y))
            .expect("nine false labels");
        assert_eq!(
            case.no_instance_x,
            label_bits(strongest_false),
            "R = {r}: the no-instance input must be the strongest false label"
        );

        for (y, y0) in [(inst.true_label, inst.true_label as u8), (13usize, 0u8)] {
            let x = label_bits(y);
            let a = BitVec::zeros(r);
            for j in [0usize, 1, r / 2, r - 1] {
                let u = case
                    .dc
                    .query_bundle(&case.sites, j, x.as_slice(), a.as_slice());
                assert_eq!(
                    u.as_slice(),
                    &byte_bits(&sparse_query_bytes(inst.image_id, j as u32, y0))[..],
                    "R = {r} label {y} occurrence {j}"
                );
            }
        }
        println!(
            "sparse-mnist R={r}: image {} digit {} | votes {:?} | d_min {} | nu {} | Z {}",
            inst.image_id, inst.true_label, inst.votes, inst.d_min, case.layout.nu, case.sys.z
        );
    }
}

#[test]
fn real_sparse_wire_tampering_is_rejected_by_the_r1cs() {
    let inst = sparse_instance(256);
    let (case, oracle) = sparse_mnist_case(256, "sparse-mnist");
    let x = label_bits(inst.true_label);
    let (out, answers) = case.circuit.eval_with_answers(x.as_slice(), &oracle);
    assert!(out);
    let z = rmip_frontend::trace::witness_trace(
        &case.dc,
        &case.layout,
        x.as_slice(),
        answers.as_slice(),
    );
    assert!(r1cs_satisfied(&case.sys, z.as_slice()));
    let first = case.layout.answer_coord(case.dc.ell - 1) + 1;
    let end = case.layout.used_end();
    let mut coord = first;
    while coord < end {
        let mut bad = z.clone();
        bad.set(coord, !bad.get(coord));
        assert!(
            !r1cs_satisfied(&case.sys, bad.as_slice()),
            "coordinate {coord}"
        );
        coord += 13;
    }
}

#[test]
fn the_driver_accepts_the_sparse_circuit_at_both_shapes() {
    use rmip_iop::driver::{Cheat, Config, Statement};
    use rmip_iop::transcript::SeededChallenger;

    fn run(case: &rmip_demo::harness::Case, oracle: &impl rmip_frontend::oracle::Oracle) {
        let mut backend = rmip_backend_interleaved::Ligero::new(
            rmip_backend_interleaved::Shape::balanced(case.layout.nu, 4, 4.0),
        );
        let st = Statement {
            inst: &case.inst,
            sys: &case.sys,
            layout: &case.layout,
            circuit: &case.circuit,
            dc: &case.dc,
            holo: case
                .holo
                .as_deref()
                .map(|e| e as &dyn rmip_iop::mateval::HoloMatEval),
        };
        let mut ch = SeededChallenger::new(11);
        let (e, verdict) = rmip_iop::driver::execute_recording(
            &st,
            &mut backend,
            oracle,
            &Config {
                t: case.t,
                ..Config::default()
            },
            &Cheat::default(),
            &mut ch,
        );
        assert!(
            verdict.is_ok(),
            "{}: honest run rejected: {verdict:?}",
            case.family
        );
        assert!(!e.occurrences.is_empty());
    }

    let (c, o) = sparse_mnist_case(1024, "sparse-mnist");
    assert!(
        c.holo.is_none() && c.closed_form.is_some(),
        "nu = 12: closed form"
    );
    run(&c, &o);
    let (c, o) = sparse_mnist_case(4096, "sparse-mnist");
    assert!(
        c.holo.is_none() && c.closed_form.is_some(),
        "nu = 14: closed form"
    );
    run(&c, &o);
}
