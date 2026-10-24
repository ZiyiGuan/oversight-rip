use rmip_field::{BinaryField, F128};
use rmip_iop::booleanity;
use rmip_iop::mle::{eq_table, eval_mle, lift};
use rmip_iop::sumcheck;
use rmip_iop::transcript::SeededChallenger;

fn tau(nu: usize, seed: u64) -> Vec<F128> {
    let mut ch = SeededChallenger::new(seed);
    use rmip_iop::transcript::RawChallenger;
    (0..nu).map(|_| ch.sample()).collect()
}

#[test]
fn boolean_table_passes() {
    let nu = 6;
    let bits: Vec<bool> = (0..1 << nu).map(|i| i % 3 == 0).collect();
    let t = tau(nu, 1);
    let table = lift(&bits);

    let mut s = booleanity::summand(&t, &table);
    assert_eq!(
        s.current_sum(),
        F128::ZERO,
        "a Boolean table has no residual"
    );

    let mut ch = SeededChallenger::new(2);
    let (_p, claim) = sumcheck::run_mlecheck_recording(&mut s, &t, &mut ch);
    let v = s.row_claims().v_a;
    let _ = eval_mle(&eq_table(&t), &claim.point);
    assert!(booleanity::check_endpoint(v, claim.expected));
    assert_eq!(v, eval_mle(&table, &claim.point));
}

#[test]
fn one_non_boolean_coordinate_is_caught() {
    let nu = 6;
    let n = 1usize << nu;
    let bits: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let t = tau(nu, 3);

    for corrupt_at in [0usize, 1, 17, 63] {
        for bad in [
            F128::new(2),
            F128::new(3),
            F128::new(1 << 100),
            F128::new(u128::MAX),
        ] {
            let mut table = lift(&bits);
            table[corrupt_at] = bad;
            assert_ne!(bad.mul(bad), bad, "{bad:?} must not be idempotent");

            let s = booleanity::summand(&t, &table);
            assert_ne!(
                s.current_sum(),
                F128::ZERO,
                "non-Boolean value {bad:?} at {corrupt_at} left the residual zero"
            );

            let mut s = booleanity::summand(&t, &table);
            let mut ch = SeededChallenger::new(4);
            let (_p, claim) = sumcheck::run_mlecheck_recording(&mut s, &t, &mut ch);
            let v = s.row_claims().v_a;
            assert!(
                !booleanity::check_endpoint(v, claim.expected),
                "verifier accepted a non-Boolean table"
            );
        }
    }
}

#[test]
fn it_is_zerocheck_with_the_trace_in_all_three_slots() {
    let nu = 5;
    let bits: Vec<bool> = (0..1 << nu).map(|i| i % 2 == 0).collect();
    let t = tau(nu, 5);
    let table = lift(&bits);

    let mut a = booleanity::summand(&t, &table);
    let mut b =
        rmip_iop::zerocheck::Zerocheck::new(&t, table.clone(), table.clone(), table.clone());
    assert_eq!(a.current_sum(), b.current_sum());
    assert_eq!(a.num_vars(), b.num_vars());
    assert_eq!(a.round_message(), b.round_message());
}

#[test]
fn the_r1cs_alone_does_not_force_booleanity() {
    let (p, q) = (F128::new(0xdead_beef), F128::new(0x1234_5678));
    let o = p.mul(q);
    assert_ne!(p.mul(p), p, "p is not Boolean");
    assert_eq!(p.mul(q), o, "yet the AND row is satisfied");

    let table = vec![F128::ONE, p, q, o];
    let t = tau(2, 9);
    let s = booleanity::summand(&t, &table);
    assert_ne!(s.current_sum(), F128::ZERO);
}
