use rmip_demo::harness::mnist_case;
use rmip_field::{BinaryField, F128};
use rmip_holo::{preprocess, prove, verify, Rejected};
use rmip_iop::transcript::SeededChallenger;

fn pt(n: usize, s: u64) -> Vec<F128> {
    (0..n)
        .map(|k| F128::new(s as u128 * 7919 + k as u128 * 104_729 + 3))
        .collect()
}

#[test]
fn a_corrupted_index_is_rejected_because_the_verifier_owns_it() {
    let (c, _o) = mnist_case(16);
    let (mu, nu) = (c.sys.mu, c.layout.nu);
    let (index, wit_index) = preprocess(&c.sys.a, &c.sys.b, &c.sys.c, mu, nu);
    let r_row = pt(mu, 1);
    let r_col = pt(nu, 2);
    let (proof, honest) = prove(
        &index,
        &wit_index,
        [&c.sys.a, &c.sys.b, &c.sys.c],
        &r_row,
        &r_col,
        &mut SeededChallenger::new(7),
    );
    verify(
        &index,
        &proof,
        &r_row,
        &r_col,
        &mut SeededChallenger::new(7),
    )
    .expect("the honest proof must verify against the honest index");

    let (mut tampered, _wit_tampered) = preprocess(&c.sys.a, &c.sys.b, &c.sys.c, mu, nu);
    for col in tampered.codeword.iter_mut() {
        col.0[0] = col.0[0].add(F128::ONE);
    }

    match verify(
        &tampered,
        &proof,
        &r_row,
        &r_col,
        &mut SeededChallenger::new(7),
    ) {
        Err(Rejected::OpeningMismatch) => {}
        other => panic!("a corrupted index must fail the opening, got {other:?}"),
    }
    let _ = honest;
}

#[test]
fn an_off_arity_point_is_refused_not_a_panic() {
    use rmip_holo::index::shape_for;
    use rmip_holo::opening;
    use rmip_iop::mle::eval_mle;

    let (c, _o) = mnist_case(16);
    let (mu, nu) = (c.sys.mu, c.layout.nu);
    let (index, wit_index) = preprocess(&c.sys.a, &c.sys.b, &c.sys.c, mu, nu);
    let shape = shape_for(index.v_pre);
    let point = pt(index.v_pre, 11);

    let row = opening::open(
        &wit_index.flat,
        shape,
        &point,
        &mut SeededChallenger::new(11),
    );
    let value = eval_mle(&row, &point[shape.nu_row..]);
    let cols = opening::columns(shape, &mut SeededChallenger::new(12));
    assert!(
        opening::check(&index.codeword, shape, &point, value, &row, &cols),
        "the honest opening at the honest arity must check"
    );

    for bad in [
        &point[..index.v_pre - 1],
        &point[..0],
        &point[..shape.nu_row],
    ] {
        assert!(
            !opening::check(&index.codeword, shape, bad, value, &row, &cols),
            "a point of {} coordinates must be refused, not panicked on",
            bad.len()
        );
    }
    let mut long = point.clone();
    long.push(F128::ONE);
    assert!(
        !opening::check(&index.codeword, shape, &long, value, &row, &cols),
        "an over-long point must be refused too"
    );
}
