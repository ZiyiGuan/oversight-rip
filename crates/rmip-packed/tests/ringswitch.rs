use rmip_field::{F128, F64};
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use rmip_packed::ringswitch::{
    eval_words, frobenius, frobenius_inverse, prove, verify, BitClaim, RingSwitchError,
};
use rmip_packed::subfield::SkipTables;
use rmip_packed::zerocheck::{eq_table, eval_bits, PackedBits};
use rmip_packed::PackedField;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn random_point<F: PackedField>(n: usize, s: &mut u64) -> Vec<F> {
    (0..n)
        .map(|_| F::from_bits(((lcg(s) as u128) << 64) | lcg(s) as u128))
        .collect()
}

fn frobenius_round_trips<F: PackedField>() {
    let mut s = 0x44_u64;
    for j in [0usize, 1, 2, 7, F::BITS - 1] {
        let x = F::from_bits(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128);
        assert_eq!(frobenius(frobenius_inverse(x, j), j), x);
        assert_eq!(frobenius(x, 1), x.square());
    }
    let basis = F::frobenius_basis();
    for j in [0usize, 3, F::BITS - 1] {
        for r in [0usize, 1, 5, F::BITS - 1] {
            assert_eq!(basis.pow[j][r], frobenius(F::from_bits(1u128 << r), j));
        }
    }
}

fn ring_switching_yields_true_word_claims<F: PackedField>(m: usize, seed: u64) {
    let tables = SkipTables::<F>::new();
    let mut s = seed;
    let bits: Vec<bool> = (0..1usize << m).map(|_| lcg(&mut s) & 1 == 1).collect();
    let packed = PackedBits::<F>::from_bits(m, &bits);
    let z_skip = F::from_bits(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128);
    let rho = random_point::<F>(m - 6, &mut s);
    let value = eval_bits(&tables, &packed, z_skip, &rho);
    let claim = BitClaim::from_skip_point(&tables, z_skip, &rho, value);
    assert_eq!(
        claim.evaluate(&packed),
        value,
        "the word form of the claim agrees with the bit form"
    );
    assert_eq!(claim.point.len() + F::BITS.trailing_zeros() as usize, m);

    let mut pch = SeededChallenger::new(seed);
    let (proof, claims) = prove(&packed.words, &claim, &mut pch);
    assert_eq!(proof.s_hat.len(), F::BITS);
    let mut vch = SeededChallenger::new(seed);
    let verified = verify(&claim, &proof, &mut vch).expect("accepted");
    assert_eq!(verified, claims);
    assert_eq!(pch.transcript().canonical(), vch.transcript().canonical());
    assert_eq!(claims.len(), F::BITS);
    for (j, c) in claims.iter().enumerate() {
        assert_eq!(
            eval_words(&packed.words, &c.point),
            c.value,
            "word claim {j}"
        );
    }
    assert_eq!(claims[0].point, claim.point);
    let _ = eq_table::<F>(&claim.point);
}

fn a_wrong_partial_evaluation_is_rejected<F: PackedField>(m: usize, seed: u64) {
    let tables = SkipTables::<F>::new();
    let mut s = seed;
    let bits: Vec<bool> = (0..1usize << m).map(|_| lcg(&mut s) & 1 == 1).collect();
    let packed = PackedBits::<F>::from_bits(m, &bits);
    let z_skip = F::from_bits(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128);
    let rho = random_point::<F>(m - 6, &mut s);
    let value = eval_bits(&tables, &packed, z_skip, &rho);
    let claim = BitClaim::from_skip_point(&tables, z_skip, &rho, value);
    let (proof, _) = prove(&packed.words, &claim, &mut SeededChallenger::new(seed));
    let wrong_value = BitClaim {
        value: value + F::ONE,
        ..claim.clone()
    };
    assert_eq!(
        verify(&wrong_value, &proof, &mut SeededChallenger::new(seed)),
        Err(RingSwitchError::PartialEvaluationsMismatch)
    );
    let mut bad = proof.clone();
    bad.s_hat[2] += F::ONE;
    assert_eq!(
        verify(&claim, &bad, &mut SeededChallenger::new(seed)),
        Err(RingSwitchError::PartialEvaluationsMismatch)
    );
    let mut bad = proof.clone();
    let delta = F::from_bits(0x1234_5678_9abc_def1);
    bad.s_hat[3] += delta;
    let ratio = claim.coeffs[3] * rmip_field::BinaryField::inv(claim.coeffs[5]).unwrap();
    bad.s_hat[5] += delta * ratio;
    let claims =
        verify(&claim, &bad, &mut SeededChallenger::new(seed)).expect("the value check passes");
    assert!(
        claims
            .iter()
            .any(|c| eval_words(&packed.words, &c.point) != c.value),
        "some word claim is false"
    );
}

#[test]
fn frobenius_and_the_basis_table_at_both_fields() {
    frobenius_round_trips::<F64>();
    frobenius_round_trips::<F128>();
}

#[test]
fn ring_switching_yields_true_word_claims_at_both_fields() {
    for (m, seed) in [(9usize, 1u64), (12, 2), (14, 3)] {
        ring_switching_yields_true_word_claims::<F64>(m, seed);
        ring_switching_yields_true_word_claims::<F128>(m, seed);
    }
}

#[test]
fn a_wrong_partial_evaluation_is_rejected_at_both_fields() {
    a_wrong_partial_evaluation_is_rejected::<F64>(10, 11);
    a_wrong_partial_evaluation_is_rejected::<F128>(10, 12);
}
