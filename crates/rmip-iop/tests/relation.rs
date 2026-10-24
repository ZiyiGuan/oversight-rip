use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::MnistInstance;
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, r1cs, trace};
use rmip_iop::lincheck::{Coefficients, Lincheck};
use rmip_iop::mle::{eq_table, eval_mle, lift};
use rmip_iop::sumcheck::{self, Summand};
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use rmip_iop::{matmle, zerocheck::Zerocheck};

const SEED: u64 = 0xC0FFEE;

const VIEWS: usize = 24;

struct Fixture {
    a: r1cs::SparseF2,
    b: r1cs::SparseF2,
    c: r1cs::SparseF2,
    z: Vec<bool>,
    mu: usize,
    nu: usize,
}

fn fixture() -> Fixture {
    let mnist = MnistInstance::new(VIEWS);
    let circuit = mnist.build();
    let o = mnist.oracle();
    let (dc, _sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());

    let x = mnist.yes();
    assert!(circuit.eval(x.as_slice(), &o), "need a yes-instance");
    let (_, a_vec) = circuit.eval_with_answers(x.as_slice(), &o);
    let z = trace::witness_trace(&dc, &lay, x.as_slice(), a_vec.as_slice());

    assert_ne!(
        sys.mu, lay.nu,
        "the fixture must separate the row width from the coordinate width"
    );
    Fixture {
        a: sys.a,
        b: sys.b,
        c: sys.c,
        z: z.as_slice().to_vec(),
        mu: sys.mu,
        nu: lay.nu,
    }
}

#[test]
fn honest_relation_reduction_accepts() {
    let f = fixture();
    let mut p = SeededChallenger::new(SEED);
    let mut v = SeededChallenger::new(SEED);

    let tau_p: Vec<F128> = (0..f.mu).map(|_| p.sample()).collect();
    let tau_v: Vec<F128> = (0..f.mu).map(|_| v.sample()).collect();
    assert_eq!(tau_p, tau_v, "coin streams must stay in lockstep");

    let az = lift(&f.a.mul_vec(&f.z));
    let bz = lift(&f.b.mul_vec(&f.z));
    let cz = lift(&f.c.mul_vec(&f.z));
    let mut zc = Zerocheck::new(&tau_p, az, bz, cz);

    assert_eq!(
        zc.current_sum(),
        F128::ZERO,
        "honest trace must zero the residual"
    );

    let (proof, zc_claim) = sumcheck::run_mlecheck_recording(&mut zc, &tau_p, &mut p);
    let r_row = zc_claim.point.clone();
    let (claim, bad) = sumcheck::verify_mlecheck_recording(&proof, &tau_v, F128::ZERO, 2, &mut v);
    assert!(bad.is_none(), "zerocheck verifies");
    assert_eq!(claim.point, r_row);

    let rc = zc.row_claims();
    let _unused = eval_mle(&eq_table(&tau_v), &r_row);
    assert_eq!(
        rc.v_a.mul(rc.v_b).add(rc.v_c),
        claim.expected,
        "zerocheck endpoint"
    );

    let coeffs_p = Coefficients::sample(&mut p);
    let coeffs_v = Coefficients::sample(&mut v);
    assert_eq!(coeffs_p, coeffs_v);
    assert_ne!(coeffs_p.lambda_a(), coeffs_p.lambda_b());
    assert_ne!(coeffs_p.lambda_b(), coeffs_p.lambda_c());
    assert_ne!(coeffs_p.lambda_a(), coeffs_p.lambda_c());
    let square = coeffs_p.lambda_a().mul(coeffs_p.lambda_a());
    assert_ne!(
        coeffs_p.lambda_b(),
        square,
        "lambda_b is not lambda_a squared"
    );
    assert_ne!(
        coeffs_p.lambda_c(),
        square.mul(coeffs_p.lambda_a()),
        "lambda_c is not lambda_a cubed"
    );

    let batched = coeffs_p.batch(rc);
    let mut lc = Lincheck::new(&f.a, &f.b, &f.c, &r_row, coeffs_p, lift(&f.z));
    assert_eq!(
        lc.current_sum(),
        batched,
        "lincheck claim must match the batched endpoints"
    );

    let (lproof, r_col, final_lc) = sumcheck::prove(&mut lc, &mut p);
    let lclaim = sumcheck::verify(&lproof, batched, f.nu, 2, &mut v).expect("lincheck verifies");
    assert_eq!(lclaim.point, r_col);
    assert_eq!(lclaim.expected, final_lc);

    let (ma, mb, mc) = matmle::eval_all(&f.a, &f.b, &f.c, &r_row, &r_col);
    let l_at = coeffs_v
        .lambda_a()
        .mul(ma)
        .add(coeffs_v.lambda_b().mul(mb))
        .add(coeffs_v.lambda_c().mul(mc));
    assert_eq!(
        l_at,
        lc.prover_l_at_point(),
        "verifier's L(r_col) must match the prover's"
    );

    let v_z = lc.z_claim();
    assert_eq!(l_at.mul(v_z), lclaim.expected, "lincheck endpoint");

    assert_eq!(v_z, eval_mle(&lift(&f.z), &r_col), "v_z = z-hat(r_col)");
}

#[test]
fn corrupted_trace_breaks_zerocheck() {
    const BROKEN_BY_A_FLIP: usize = 40;

    let f = fixture();
    let mut p = SeededChallenger::new(SEED);
    let tau: Vec<F128> = (0..f.mu).map(|_| p.sample()).collect();

    let touched = f.a.nz.iter().chain(f.b.nz.iter()).chain(f.c.nz.iter());
    let mut broke = 0;
    for (k, &(_, col)) in touched.take(40).enumerate() {
        let mut z = f.z.clone();
        z[col] = !z[col];
        let (az, bz, cz) = (
            lift(&f.a.mul_vec(&z)),
            lift(&f.b.mul_vec(&z)),
            lift(&f.c.mul_vec(&z)),
        );
        let residual_is_nonzero = (0..az.len()).any(|i| az[i].mul(bz[i]).add(cz[i]) != F128::ZERO);
        let zc = Zerocheck::new(&tau, az, bz, cz);
        assert_eq!(
            zc.current_sum() != F128::ZERO,
            residual_is_nonzero,
            "flip {k} at coordinate {col}: the eq-weighted claim must be nonzero \
             exactly when the residual is"
        );
        if residual_is_nonzero {
            broke += 1;
        }
    }
    assert_eq!(
        broke, BROKEN_BY_A_FLIP,
        "every one of the forty flips of a constrained coordinate must break the residual"
    );
}

#[test]
fn tampered_round_polynomial_diverts_the_terminal_claim() {
    let f = fixture();
    let mut p = SeededChallenger::new(SEED);
    let mut v = SeededChallenger::new(SEED);
    let tau: Vec<F128> = (0..f.mu).map(|_| p.sample()).collect();
    let _: Vec<F128> = (0..f.mu).map(|_| v.sample()).collect();

    let mut zc = Zerocheck::new(
        &tau,
        lift(&f.a.mul_vec(&f.z)),
        lift(&f.b.mul_vec(&f.z)),
        lift(&f.c.mul_vec(&f.z)),
    );
    let (mut proof, honest) = sumcheck::run_mlecheck_recording(&mut zc, &tau, &mut p);
    let honest_final = zc.row_claims();
    let honest_value = honest_final.v_a.mul(honest_final.v_b).add(honest_final.v_c);
    let _ = honest;
    proof.round_polys[0][0] = proof.round_polys[0][0].add(F128::ONE);

    let (out, bad) = sumcheck::verify_mlecheck_recording(&proof, &tau, F128::ZERO, 2, &mut v);
    assert!(
        bad.is_none(),
        "a well-formed stream verifies; the lie is carried"
    );
    assert_ne!(
        out.expected, honest_value,
        "the tamper must divert the terminal claim, or no endpoint check could reject it"
    );
}

#[test]
fn malformed_proofs_are_rejected_by_shape() {
    let f = fixture();
    let mut v = SeededChallenger::new(SEED);
    let tau: Vec<F128> = (0..f.mu).map(|i| F128::new(i as u128 + 1)).collect();
    let empty = sumcheck::SumcheckProof {
        round_polys: vec![],
    };
    assert!(matches!(
        sumcheck::verify_mlecheck_recording(&empty, &tau, F128::ZERO, 2, &mut v).1,
        Some(sumcheck::SumcheckError::WrongRoundCount { .. })
    ));

    let short = sumcheck::SumcheckProof {
        round_polys: vec![vec![F128::ZERO; 1]; f.mu],
    };
    assert!(matches!(
        sumcheck::verify_mlecheck_recording(&short, &tau, F128::ZERO, 2, &mut v).1,
        Some(sumcheck::SumcheckError::WrongDegree { .. })
    ));
}

#[test]
fn sumcheck_engine_is_generic() {
    use rmip_iop::mle::fold_in_place;

    struct Product {
        t: Vec<Vec<F128>>,
        vars: usize,
    }
    impl Summand for Product {
        fn num_vars(&self) -> usize {
            self.vars
        }
        fn degree(&self) -> usize {
            2
        }
        fn round_evals(&self) -> Vec<F128> {
            sumcheck::round_evals_over(&self.t, 2, |v| v[0].mul(v[1]))
        }
        fn fold(&mut self, r: F128) {
            for t in &mut self.t {
                fold_in_place(t, r);
            }
            self.vars -= 1;
        }
        fn final_value(&self) -> F128 {
            self.t[0][0].mul(self.t[1][0])
        }
    }

    let n = 16;
    let f: Vec<F128> = (0..n).map(|i| F128::new(i as u128 * 7 + 1)).collect();
    let g: Vec<F128> = (0..n).map(|i| F128::new(i as u128 * 13 + 5)).collect();
    let claim = (0..n)
        .map(|i| f[i].mul(g[i]))
        .fold(F128::ZERO, |a, b| a.add(b));

    let mut s = Product {
        t: vec![f, g],
        vars: 4,
    };
    let mut p = SeededChallenger::new(7);
    let mut v = SeededChallenger::new(7);
    let (proof, point, final_val) = sumcheck::prove(&mut s, &mut p);
    let out = sumcheck::verify(&proof, claim, 4, 2, &mut v).expect("verifies");
    assert_eq!(out.point, point);
    assert_eq!(out.expected, final_val);
}

#[test]
fn the_matrix_vector_product_is_the_field_one_not_the_boolean_one() {
    let m = r1cs::SparseF2 {
        rows: 1,
        cols: 2,
        nz: vec![(0, 0), (0, 1)],
    };

    for bits in [[false, false], [false, true], [true, false], [true, true]] {
        assert_eq!(
            matmle::mul_vec(&m, &lift(&bits)),
            lift(&m.mul_vec(&bits)),
            "the two products must agree on Boolean input {bits:?}"
        );
    }

    let z = vec![F128::new(2), F128::new(3)];
    let support: Vec<bool> = z.iter().map(|v| *v != F128::ZERO).collect();
    let field = matmle::mul_vec(&m, &z);
    let boolean = lift(&m.mul_vec(&support));
    assert_eq!(
        field,
        vec![F128::new(1)],
        "the field product is the sum in F"
    );
    assert_eq!(
        boolean,
        vec![F128::ZERO],
        "the Boolean product XORs the support"
    );
    assert_ne!(
        field, boolean,
        "a non-Boolean z is what separates the two products"
    );
}

#[test]
fn the_timing_replay_runs_the_mlecheck_verifier_on_the_runs_coins() {
    use rmip_iop::replay::{coins_since, CoinReplay};
    use rmip_iop::transcript::RawChallenger as _;
    let tau = vec![F128::new(7), F128::new(11)];
    let a = vec![F128::ONE; 4];
    let b = vec![F128::ONE, F128::ZERO, F128::ONE, F128::ZERO];
    let c = b.clone();
    let mut zc = rmip_iop::zerocheck::Zerocheck::new(&tau, a, b, c);
    let mut ch = SeededChallenger::new(23);
    let from = ch.transcript().entries().len();
    let (proof, expected) = sumcheck::run_mlecheck_recording(&mut zc, &tau, &mut ch);
    let coins = coins_since(ch.transcript(), from);
    assert_eq!(coins.len(), tau.len());
    let mut replay = CoinReplay::new(&coins);
    let (claim, fault) =
        sumcheck::verify_mlecheck_recording(&proof, &tau, F128::ZERO, 2, &mut replay);
    assert!(fault.is_none(), "{fault:?}");
    assert!(replay.replayed_exactly());
    assert_eq!(claim.point, expected.point);
    assert_eq!(claim.expected, expected.expected);
    let (_, stale) = sumcheck::verify_recording(
        &proof,
        F128::ZERO,
        proof.round_polys.len(),
        3,
        &mut SeededChallenger::new(23),
    );
    assert!(
        format!("{stale:?}").contains("WrongDegree"),
        "the plain verifier at degree 3 is the wrong replay: {stale:?}"
    );
}
