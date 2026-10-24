use rmip_field::{BinaryField, F128};
use rmip_iop::mle::{eq_table, eval_mle, lift};

fn sqrt(x: F128) -> F128 {
    let mut r = x;
    for _ in 0..127 {
        r = r.square();
    }
    r
}

fn rng(seed: u64) -> impl FnMut() -> F128 {
    let mut s = seed | 1;
    move || {
        let mut hi = 0u64;
        for _ in 0..2 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            hi = s;
        }
        let lo = hi.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        F128::new(((hi as u128) << 64) | lo as u128)
    }
}

#[test]
fn sqrt_is_the_inverse_of_squaring() {
    let mut next = rng(1);
    for _ in 0..500 {
        let x = next();
        assert_eq!(sqrt(x).square(), x);
        assert_eq!(sqrt(x.square()), x);
    }
}

#[test]
fn eq_squares_under_the_square_root_point() {
    let mut next = rng(2);
    for nu in 1..=6 {
        let tau: Vec<F128> = (0..nu).map(|_| next()).collect();
        let sigma: Vec<F128> = tau.iter().map(|&t| sqrt(t)).collect();
        let et = eq_table(&tau);
        let es = eq_table(&sigma);
        for w in 0..(1usize << nu) {
            assert_eq!(es[w].square(), et[w], "nu = {nu}, w = {w}");
        }
    }
}

#[test]
fn booleanity_residual_has_a_closed_form() {
    let mut next = rng(3);
    for nu in 1..=8 {
        let n = 1usize << nu;
        for _ in 0..25 {
            let table: Vec<F128> = (0..n).map(|_| next()).collect();
            let tau: Vec<F128> = (0..nu).map(|_| next()).collect();
            let sigma: Vec<F128> = tau.iter().map(|&t| sqrt(t)).collect();

            let et = eq_table(&tau);
            let lhs = (0..n)
                .map(|w| et[w].mul(table[w].square().add(table[w])))
                .fold(F128::ZERO, |a, b| a.add(b));

            let rhs = eval_mle(&table, &sigma)
                .square()
                .add(eval_mle(&table, &tau));
            assert_eq!(lhs, rhs, "identity failed at nu = {nu}");
        }
    }
}

#[test]
fn the_closed_form_separates_boolean_from_non_boolean() {
    let mut next = rng(4);
    let nu = 7;
    let n = 1usize << nu;
    let bits: Vec<bool> = (0..n).map(|i| i % 5 < 2).collect();
    let boolean = lift(&bits);

    let mut vanished_on_corrupt = 0;
    for _ in 0..200 {
        let tau: Vec<F128> = (0..nu).map(|_| next()).collect();
        let sigma: Vec<F128> = tau.iter().map(|&t| sqrt(t)).collect();
        let residual = |t: &[F128]| eval_mle(t, &sigma).square().add(eval_mle(t, &tau));

        assert_eq!(residual(&boolean), F128::ZERO, "Boolean table must vanish");

        let mut bad = boolean.clone();
        bad[next().to_u128() as usize % n] = next();
        if residual(&bad) == F128::ZERO {
            vanished_on_corrupt += 1;
        }
    }
    assert_eq!(
        vanished_on_corrupt, 0,
        "a non-Boolean table slipped through"
    );
}

#[test]
fn protocol_form_needs_one_extra_claim_and_no_message() {
    let mut next = rng(5);
    let nu = 6;
    let n = 1usize << nu;
    let bits: Vec<bool> = (0..n).map(|i| i % 3 == 1).collect();
    let table = lift(&bits);

    let r_col: Vec<F128> = (0..nu).map(|_| next()).collect();
    let sigma: Vec<F128> = r_col.iter().map(|&t| sqrt(t)).collect();
    let v_z = eval_mle(&table, &r_col);

    let expected = sqrt(v_z);
    assert_eq!(
        eval_mle(&table, &sigma),
        expected,
        "z-hat(sigma) = sqrt(v_z)"
    );
    assert_eq!(eval_mle(&table, &sigma).square(), v_z);

    let mut bad = table.clone();
    bad[11] = F128::new(0xabcd_ef01);
    let v_bad = eval_mle(&bad, &r_col);
    assert_ne!(
        eval_mle(&bad, &sigma).square(),
        v_bad,
        "non-Boolean must fail"
    );
}

#[test]
fn library_helpers_match() {
    use rmip_iop::booleanity;
    let mut next = rng(6);
    let nu = 6;
    let n = 1usize << nu;
    let bits: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
    let table = lift(&bits);

    for _ in 0..50 {
        let x = next();
        assert_eq!(booleanity::sqrt(x), sqrt(x));
    }

    let r_col: Vec<F128> = (0..nu).map(|_| next()).collect();
    let v_z = eval_mle(&table, &r_col);
    let (sigma, v_sigma) = booleanity::claim_from_lincheck_endpoint(&r_col, v_z);

    assert_eq!(eval_mle(&table, &sigma), v_sigma);
    assert_eq!(booleanity::residual_at(&table, &r_col), F128::ZERO);

    let mut bad = table.clone();
    bad[7] = F128::new(0x5a5a);
    let v_bad = eval_mle(&bad, &r_col);
    let (sigma_b, v_sigma_b) = booleanity::claim_from_lincheck_endpoint(&r_col, v_bad);
    assert_ne!(
        eval_mle(&bad, &sigma_b),
        v_sigma_b,
        "the appended claim must be false"
    );
    assert_ne!(booleanity::residual_at(&bad, &r_col), F128::ZERO);
}

#[test]
fn closed_form_agrees_with_the_sumcheck_form() {
    use rmip_iop::booleanity;
    let mut next = rng(7);
    for nu in 2..=7 {
        let n = 1usize << nu;
        let table: Vec<F128> = (0..n).map(|_| next()).collect();
        let tau: Vec<F128> = (0..nu).map(|_| next()).collect();
        let s = booleanity::summand(&tau, &table);
        assert_eq!(
            s.current_sum(),
            booleanity::residual_at(&table, &tau),
            "sumcheck claim and closed form disagree at nu = {nu}"
        );
        assert_eq!(s.num_vars(), nu);
    }
}

#[test]
fn the_table_sqrt_is_the_ladder_sqrt() {
    use rmip_iop::booleanity::{sqrt, sqrt_ladder};
    for i in 0..128 {
        let b = F128::new(1u128 << i);
        assert_eq!(sqrt(b), sqrt_ladder(b), "basis bit {i}");
    }
    for seed in 0..64u128 {
        let x = F128::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (seed << 64) ^ 0xDEAD_BEEF);
        assert_eq!(sqrt(x), sqrt_ladder(x));
        assert_eq!(sqrt(x).square(), x, "sqrt must invert squaring");
    }
    assert_eq!(sqrt(F128::ZERO), F128::ZERO);
    assert_eq!(sqrt(F128::ONE), F128::ONE);
}
