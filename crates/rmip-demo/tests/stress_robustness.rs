use rmip_frontend::bits::BitVec;
use rmip_frontend::circuit::Gate;
use rmip_frontend::deoracle::{self, OracleSite, OrdinaryCircuit};
use rmip_frontend::mnist::{
    certify, label_bits, MnistInstance, SparseMajorityFamily, SparseTableOracle, LABEL_BITS,
};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::Ratio;

fn bad_of<O: Oracle>(
    dc: &OrdinaryCircuit,
    sites: &[OracleSite],
    x: &BitVec,
    a: &[bool],
    o: &O,
) -> usize {
    let w = dc.eval_all_wires(x.as_slice(), a);
    sites
        .iter()
        .filter(|s| {
            let u: Vec<bool> = s.query_wires.iter().map(|&q| w[q as usize]).collect();
            o.query(&u) != a[s.occurrence]
        })
        .count()
}

fn honest<O: Oracle>(dc: &OrdinaryCircuit, sites: &[OracleSite], x: &BitVec, o: &O) -> Vec<bool> {
    let mut a = vec![false; dc.ell];
    for _ in 0..=dc.ell {
        let w = dc.eval_all_wires(x.as_slice(), &a);
        let next: Vec<bool> = sites
            .iter()
            .map(|s| {
                o.query(
                    &s.query_wires
                        .iter()
                        .map(|&q| w[q as usize])
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        if next == a {
            return a;
        }
        a = next;
    }
    panic!("honest answers did not reach a fixpoint");
}

fn min_bad_for_false_claim<O: Oracle>(
    dc: &OrdinaryCircuit,
    sites: &[OracleSite],
    x: &BitVec,
    o: &O,
) -> usize {
    let ell = dc.ell;
    assert!(ell <= 16, "exhaustive search is 2^ell");
    let truth = dc.eval(x.as_slice(), &honest(dc, sites, x, o));
    let target = !truth;
    let mut best = usize::MAX;
    for mask in 0u32..(1u32 << ell) {
        let a: Vec<bool> = (0..ell).map(|j| (mask >> j) & 1 == 1).collect();
        if dc.eval(x.as_slice(), &a) != target {
            continue;
        }
        best = best.min(bad_of(dc, sites, x, &a, o));
    }
    assert!(
        best < usize::MAX,
        "no answer vector reaches the false claim"
    );
    best
}

fn assert_strict_gap(eps: Ratio, ell: usize, min_bad: usize) {
    assert!(
        min_bad as u128 * eps.den as u128 > eps.num as u128 * ell as u128,
        "declared eps = {}/{} at ell = {ell} needs |Bad| > eps*ell, but the \
         minimum over false claims is {min_bad}",
        eps.num,
        eps.den
    );
    assert_eq!(
        eps.min_bad_set(ell),
        min_bad,
        "min_bad_set must be exactly the measured minimum: tight, not merely safe"
    );
    let over = Ratio::new(min_bad as u64, ell as u64);
    assert!(
        over.min_bad_set(ell) > min_bad,
        "eps = min_bad/ell must be REJECTED by the strict inequality"
    );
}

fn two_sided_distance(votes: usize, threshold: usize) -> usize {
    if votes >= threshold {
        votes - threshold + 1
    } else {
        threshold - votes
    }
}

#[test]
fn the_mnist_circuit_queries_do_not_depend_on_answers() {
    let inst = MnistInstance::new(16);
    let (dc, sites) = deoracle::deoraclize(&inst.build());
    assert_eq!(dc.ell, inst.ell());
    let mut tainted = vec![false; dc.gates.len()];
    for (i, g) in dc.gates.iter().enumerate() {
        tainted[i] = match g {
            Gate::Answer(_) => true,
            Gate::Not(p) => tainted[*p],
            Gate::And(p, q) | Gate::Or(p, q) | Gate::Xor(p, q) => tainted[*p] || tainted[*q],
            Gate::Xor3(p, q, s) | Gate::Maj3(p, q, s) => tainted[*p] || tainted[*q] || tainted[*s],
            _ => false,
        };
    }
    assert!(
        tainted.iter().any(|&t| t),
        "the de-oracled circuit must read its answers somewhere"
    );
    for s in &sites {
        assert!(
            s.query_wires.iter().all(|&q| !tainted[q as usize]),
            "occurrence {} reads an answer-dependent wire: the circuit is not d = 1",
            s.occurrence
        );
    }
}

#[test]
fn the_mnist_circuit_rejects_invalid_labels_under_every_answer_vector() {
    let r = 9usize;
    let predicted: Vec<u8> = vec![3, 3, 3, 3, 3, 7, 7, 1, 0];
    let fam = SparseMajorityFamily::new(0, r);
    assert_eq!(fam.threshold, r / 2 + 1);
    let oracle = SparseTableOracle {
        image_id: 0,
        predicted: predicted.clone(),
    };
    let (dc, sites) = deoracle::deoraclize(&fam.build());
    assert_eq!(dc.ell, r);

    let mut votes = [0usize; 10];
    for &p in &predicted {
        votes[usize::from(p)] += 1;
    }
    let mut global_min = usize::MAX;
    let label_values = 1usize << LABEL_BITS;
    for y in 0..label_values {
        let x = BitVec::from_bools((0..LABEL_BITS).map(|k| (y >> k) & 1 == 1).collect());
        let a0 = honest(&dc, &sites, &x, &oracle);
        let truth = dc.eval(x.as_slice(), &a0);
        if y >= 10 {
            assert!(!truth);
            for mask in 0u32..(1u32 << r) {
                let a: Vec<bool> = (0..r).map(|j| (mask >> j) & 1 == 1).collect();
                assert!(!dc.eval(x.as_slice(), &a), "invalid label {y} accepted");
            }
            continue;
        }
        let v = *votes.get(y).expect("a valid label indexes the vote table");
        assert_eq!(a0.iter().filter(|&&b| b).count(), v, "vote recount");
        assert_eq!(truth, v >= fam.threshold);
        let d = min_bad_for_false_claim(&dc, &sites, &x, &oracle);
        let closed = two_sided_distance(v, fam.threshold);
        assert_eq!(d, closed, "label {y}: two-sided distance");
        global_min = global_min.min(d);
    }
    let certified = global_min - 1;
    let eps = Ratio::new(certified as u64, r as u64);
    assert_strict_gap(eps, r, global_min);
}

#[test]
fn the_compiled_in_table_at_sixteen_views_has_its_certified_radius_exhaustively() {
    let r = 16usize;
    let inst = MnistInstance::new(r);
    assert_eq!((inst.r, inst.ell(), inst.threshold), (r, r, r / 2 + 1));
    let oracle = inst.oracle();
    let (dc, sites) = deoracle::deoraclize(&inst.build());
    assert_eq!(dc.ell, r);
    let cert = certify(&inst.table, inst.threshold);
    assert_eq!(cert.votes, inst.votes);
    assert_eq!(cert.d_min, inst.d_min);

    let mut global_min = usize::MAX;
    for y in 0..(1usize << LABEL_BITS) {
        let x = label_bits(y);
        let a0 = honest(&dc, &sites, &x, &oracle);
        let truth = dc.eval(x.as_slice(), &a0);
        if y >= 10 {
            assert!(!truth, "invalid label {y} must reject honestly");
            for mask in 0u32..(1u32 << r) {
                let a: Vec<bool> = (0..r).map(|j| (mask >> j) & 1 == 1).collect();
                assert!(!dc.eval(x.as_slice(), &a), "invalid label {y} accepted");
            }
            continue;
        }
        assert_eq!(a0, inst.honest_answers(y), "label {y}: honest answers");
        assert_eq!(truth, y == inst.true_label, "label {y}: the honest verdict");
        let d = min_bad_for_false_claim(&dc, &sites, &x, &oracle);
        assert_eq!(
            d,
            two_sided_distance(inst.votes[y], inst.threshold),
            "label {y}: two-sided distance"
        );
        assert_eq!(
            d, cert.flip_distances[y],
            "label {y}: the certificate's distance"
        );
        global_min = global_min.min(d);
    }

    let no = inst.no();
    assert!(!dc.eval(no.as_slice(), &honest(&dc, &sites, &no, &oracle)));
    let searched = min_bad_for_false_claim(&dc, &sites, &no, &oracle);
    assert_eq!(searched, inst.threshold - inst.votes[inst.no_label]);
    assert_eq!(searched, inst.minimal_bad_set_size());
    let closed_form = inst.local_cheat_answers();
    assert!(
        dc.eval(no.as_slice(), &closed_form),
        "the closed-form vector must satisfy the circuit on the no-instance"
    );
    assert_eq!(
        bad_of(&dc, &sites, &no, &closed_form, &oracle),
        searched,
        "the closed-form vector must attain the searched minimum"
    );

    let yes = inst.yes();
    assert_eq!(
        min_bad_for_false_claim(&dc, &sites, &yes, &oracle),
        inst.votes[inst.true_label] - inst.threshold + 1
    );

    assert_eq!(global_min, inst.d_min, "the whole-circuit radius");
    assert_eq!(inst.eps_rob, Ratio::new(inst.d_min as u64 - 1, r as u64));
    assert_strict_gap(inst.eps_rob, r, global_min);
}
