use rmip_demo::harness::{mnist_case, Case};
use rmip_field::{BinaryField, F128};
use rmip_frontend::SparseF2;
use rmip_holo::{
    preprocess, preprocess_prover, preprocess_verifier, prove, verify, HoloIndex, Rejected,
};
use rmip_iop::matmle;
use rmip_iop::transcript::SeededChallenger;

fn assert_same_holo_index(x: &HoloIndex, y: &HoloIndex, tag: &str) {
    assert_eq!(x.mu, y.mu, "{tag}: mu");
    assert_eq!(x.nu, y.nu, "{tag}: nu");
    assert_eq!(x.zeta, y.zeta, "{tag}: zeta");
    assert_eq!(x.corrections, y.corrections, "{tag}: corrections");
    assert_eq!(x.v_pre, y.v_pre, "{tag}: v_pre");
    assert_eq!(x.groups, y.groups, "{tag}: groups");
    assert_eq!(x.codeword, y.codeword, "{tag}: codeword");
}

fn point(n: usize, seed: u64) -> Vec<F128> {
    (0..n)
        .map(|k| F128::new((seed as u128).wrapping_mul(0x9E37) + k as u128 * 104_729 + 5))
        .collect()
}

fn scan_cases() -> [(Case, &'static str); 2] {
    [(mnist_case(16).0, "R=16"), (mnist_case(1024).0, "R=1024")]
}

fn small_case() -> Case {
    mnist_case(16).0
}

#[test]
fn the_split_preprocessing_equals_the_joint_one() {
    let toy = || {
        let m = |nz: Vec<(usize, usize)>| SparseF2 {
            rows: 16,
            cols: 16,
            nz,
        };
        (
            m(vec![(1, 2), (3, 4), (5, 6)]),
            m(vec![(0, 0), (1, 1), (2, 2), (3, 3)]),
            m(vec![(7, 7), (8, 8), (9, 9), (10, 10), (11, 11)]),
            4,
            4,
        )
    };
    let [(small, small_tag), (large, large_tag)] = scan_cases();
    let shape = |c: &Case| {
        (
            c.sys.a.clone(),
            c.sys.b.clone(),
            c.sys.c.clone(),
            c.sys.mu,
            c.layout.nu,
        )
    };
    for ((a, b, c, mu, nu), tag) in [
        (shape(&small), small_tag),
        (shape(&large), large_tag),
        (toy(), "toy"),
    ] {
        let (joint_v, joint_p) = preprocess(&a, &b, &c, mu, nu);
        let split_v = preprocess_verifier(&a, &b, &c, mu, nu);
        let split_p = preprocess_prover(&a, &b, &c, mu, nu);
        assert_same_holo_index(&joint_v, &split_v, tag);
        assert_eq!(joint_p.tables, split_p.tables, "{tag}: tables");
        assert_eq!(joint_p.flat, split_p.flat, "{tag}: flat");
    }
}

#[test]
fn the_holographic_values_are_the_scanned_values() {
    for (case, tag) in scan_cases() {
        let (mu, nu) = (case.sys.mu, case.layout.nu);
        let (index, wit_index) = preprocess(&case.sys.a, &case.sys.b, &case.sys.c, mu, nu);
        for seed in [1u64, 7] {
            let r_row = point(mu, seed);
            let r_col = point(nu, seed + 100);
            let (proof, t) = prove(
                &index,
                &wit_index,
                [&case.sys.a, &case.sys.b, &case.sys.c],
                &r_row,
                &r_col,
                &mut SeededChallenger::new(seed),
            );
            let direct = matmle::eval_all(&case.sys.a, &case.sys.b, &case.sys.c, &r_row, &r_col);
            assert_eq!(
                (t[0], t[1], t[2]),
                direct,
                "{tag}: holo must equal the scan"
            );

            let got = verify(
                &index,
                &proof,
                &r_row,
                &r_col,
                &mut SeededChallenger::new(seed),
            )
            .expect("honest holographic proof rejected");
            assert_eq!(got, t, "{tag}: verify returns the proven values");
        }
    }
}

#[test]
fn padding_parity_is_handled_on_both_sides() {
    let mu = 4;
    let nu = 4;
    let a = SparseF2 {
        rows: 16,
        cols: 16,
        nz: vec![(1, 2), (3, 4), (5, 6)],
    };
    let b = SparseF2 {
        rows: 16,
        cols: 16,
        nz: vec![(0, 0), (1, 1), (2, 2), (3, 3)],
    };
    let c = SparseF2 {
        rows: 16,
        cols: 16,
        nz: vec![(7, 7), (8, 8), (9, 9), (10, 10), (11, 11)],
    };
    let (index, wit_index) = preprocess(&a, &b, &c, mu, nu);
    assert_eq!(
        index.zeta, 3,
        "the common zeta is the max over the three matrices"
    );
    assert!(
        index.corrections[0].0,
        "3 entries pad by 5 under the common zeta: odd"
    );
    assert!(
        !index.corrections[1].0,
        "4 entries pad by 4 under the common zeta: even"
    );
    assert!(
        index.corrections[2].0,
        "5 entries pad by 3 under the common zeta: odd"
    );

    let r_row = point(mu, 11);
    let r_col = point(nu, 12);
    let (proof, t) = prove(
        &index,
        &wit_index,
        [&a, &b, &c],
        &r_row,
        &r_col,
        &mut SeededChallenger::new(9),
    );
    assert_eq!(
        (t[0], t[1], t[2]),
        matmle::eval_all(&a, &b, &c, &r_row, &r_col),
        "padding must not move the value"
    );
    verify(
        &index,
        &proof,
        &r_row,
        &r_col,
        &mut SeededChallenger::new(9),
    )
    .unwrap();
}

#[test]
fn a_forged_t_value_is_rejected() {
    let case = small_case();
    let (mu, nu) = (case.sys.mu, case.layout.nu);
    let (index, wit_index) = preprocess(&case.sys.a, &case.sys.b, &case.sys.c, mu, nu);
    let r_row = point(mu, 3);
    let r_col = point(nu, 4);
    let (mut proof, _) = prove(
        &index,
        &wit_index,
        [&case.sys.a, &case.sys.b, &case.sys.c],
        &r_row,
        &r_col,
        &mut SeededChallenger::new(5),
    );
    proof.t[1] = proof.t[1].add(F128::ONE);
    match verify(
        &index,
        &proof,
        &r_row,
        &r_col,
        &mut SeededChallenger::new(5),
    ) {
        Err(Rejected::Endpoint { matrix: 1 }) => {}
        other => panic!("a forged T_B must fail matrix 1's endpoint, got {other:?}"),
    }
}

#[test]
fn a_consistent_endpoint_lie_is_caught_by_the_opening() {
    let case = small_case();
    let (mu, nu) = (case.sys.mu, case.layout.nu);
    let (index, wit_index) = preprocess(&case.sys.a, &case.sys.b, &case.sys.c, mu, nu);
    let r_row = point(mu, 21);
    let r_col = point(nu, 22);
    let (mut proof, _) = prove(
        &index,
        &wit_index,
        [&case.sys.a, &case.sys.b, &case.sys.c],
        &r_row,
        &r_col,
        &mut SeededChallenger::new(23),
    );
    let width = mu + nu;
    let lam = rmip_holo::pack::LAMBDA;
    let mut values: Vec<F128> = proof.endpoints[0]
        .chunks(lam)
        .flat_map(|chunk| rmip_holo::pack::unpack(chunk, lam).expect("full orbit chunks"))
        .collect();

    let k = width - 1;
    let old = values[k];
    let lie = old.add(F128::ONE);
    values[k] = lie;
    let rk = r_col[nu - 1];
    let f_old = F128::ONE.add(rk).add(old);
    let f_new = F128::ONE.add(rk).add(lie);
    let rj = r_row[0];
    let f_j = F128::ONE.add(rj).add(values[0]);
    let target_fj = f_j
        .mul(f_old)
        .mul(f_new.inv().expect("nonzero in general position"));
    values[0] = target_fj.add(F128::ONE).add(rj);

    values.resize(proof.endpoints[0].len(), F128::ZERO);
    proof.endpoints[0] = values
        .chunks(lam)
        .flat_map(rmip_holo::pack::evals_from_values)
        .collect();

    match verify(
        &index,
        &proof,
        &r_row,
        &r_col,
        &mut SeededChallenger::new(23),
    ) {
        Err(Rejected::OpeningMismatch) => {}
        other => panic!("a product-consistent endpoint lie must die at the opening, got {other:?}"),
    }
}

#[test]
fn a_tampered_round_dies_at_the_endpoint() {
    let case = small_case();
    let (mu, nu) = (case.sys.mu, case.layout.nu);
    let (index, wit_index) = preprocess(&case.sys.a, &case.sys.b, &case.sys.c, mu, nu);
    let r_row = point(mu, 31);
    let r_col = point(nu, 32);
    let (mut proof, _) = prove(
        &index,
        &wit_index,
        [&case.sys.a, &case.sys.b, &case.sys.c],
        &r_row,
        &r_col,
        &mut SeededChallenger::new(33),
    );
    proof.sumchecks[2].round_polys[0][0] = proof.sumchecks[2].round_polys[0][0].add(F128::ONE);
    match verify(
        &index,
        &proof,
        &r_row,
        &r_col,
        &mut SeededChallenger::new(33),
    ) {
        Err(Rejected::Endpoint { matrix: 2 }) => {}
        other => panic!("a tampered round must die at matrix 2's endpoint, got {other:?}"),
    }
}

#[test]
fn a_proof_missing_a_matrix_is_rejected_not_a_panic() {
    let case = small_case();
    let (mu, nu) = (case.sys.mu, case.layout.nu);
    let (index, wit_index) = preprocess(&case.sys.a, &case.sys.b, &case.sys.c, mu, nu);
    let r_row = point(mu, 41);
    let r_col = point(nu, 42);
    let (mut proof, _) = prove(
        &index,
        &wit_index,
        [&case.sys.a, &case.sys.b, &case.sys.c],
        &r_row,
        &r_col,
        &mut SeededChallenger::new(43),
    );
    verify(
        &index,
        &proof,
        &r_row,
        &r_col,
        &mut SeededChallenger::new(43),
    )
    .expect("the proof this test cuts down must itself be honest");

    let mut short = proof.clone();
    short.sumchecks.pop();
    assert_eq!(
        verify(
            &index,
            &short,
            &r_row,
            &r_col,
            &mut SeededChallenger::new(43)
        ),
        Err(Rejected::ProofShape),
        "a missing sumcheck must reject, not index out of range"
    );

    proof.endpoints.pop();
    assert_eq!(
        verify(
            &index,
            &proof,
            &r_row,
            &r_col,
            &mut SeededChallenger::new(43)
        ),
        Err(Rejected::ProofShape),
        "a missing endpoint frame must reject, not index out of range"
    );
}

#[test]
fn a_wrong_arity_point_is_rejected_not_a_panic() {
    let case = small_case();
    let (mu, nu) = (case.sys.mu, case.layout.nu);
    let (index, wit_index) = preprocess(&case.sys.a, &case.sys.b, &case.sys.c, mu, nu);
    let r_row = point(mu, 51);
    let r_col = point(nu, 52);
    let (proof, _) = prove(
        &index,
        &wit_index,
        [&case.sys.a, &case.sys.b, &case.sys.c],
        &r_row,
        &r_col,
        &mut SeededChallenger::new(53),
    );
    assert_eq!(
        verify(
            &index,
            &proof,
            &r_row[..mu - 1],
            &r_col,
            &mut SeededChallenger::new(53)
        ),
        Err(Rejected::Arity),
        "a short r_row must reject, not panic in the product"
    );
    let mut long = r_col.clone();
    long.push(F128::ONE);
    assert_eq!(
        verify(
            &index,
            &proof,
            &r_row,
            &long,
            &mut SeededChallenger::new(53)
        ),
        Err(Rejected::Arity),
        "a long r_col must reject: the correction reads all of it"
    );
}

#[test]
fn a_hand_built_inconsistent_index_is_refused() {
    let case = small_case();
    let (mu, nu) = (case.sys.mu, case.layout.nu);
    let (mut index, wit_index) = preprocess(&case.sys.a, &case.sys.b, &case.sys.c, mu, nu);
    let r_row = point(mu, 61);
    let r_col = point(nu, 62);
    let (proof, _) = prove(
        &index,
        &wit_index,
        [&case.sys.a, &case.sys.b, &case.sys.c],
        &r_row,
        &r_col,
        &mut SeededChallenger::new(63),
    );
    assert!(
        index.is_well_formed(),
        "what preprocess builds must pass its own predicate"
    );

    let dropped = index.corrections.pop().expect("three corrections");
    assert_eq!(
        verify(
            &index,
            &proof,
            &r_row,
            &r_col,
            &mut SeededChallenger::new(63)
        ),
        Err(Rejected::IndexShape),
        "two corrections must reject, not index out of range"
    );
    index.corrections.push(dropped);

    let zeta = index.zeta;
    index.zeta = index.v_pre + 1;
    assert_eq!(
        verify(
            &index,
            &proof,
            &r_row,
            &r_col,
            &mut SeededChallenger::new(63)
        ),
        Err(Rejected::IndexShape),
        "zeta above v_pre must reject, not underflow"
    );
    index.zeta = zeta;

    let groups = index.groups;
    index.groups = groups + 1;
    assert_eq!(
        verify(
            &index,
            &proof,
            &r_row,
            &r_col,
            &mut SeededChallenger::new(63)
        ),
        Err(Rejected::IndexShape),
        "a group count out of step with the width must reject"
    );
    index.groups = groups;

    index.codeword.pop();
    assert_eq!(
        verify(
            &index,
            &proof,
            &r_row,
            &r_col,
            &mut SeededChallenger::new(63)
        ),
        Err(Rejected::IndexShape),
        "a short codeword must reject up front"
    );
}
