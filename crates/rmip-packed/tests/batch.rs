use rmip_field::{F128, F64};
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use rmip_packed::batch::{prove, verify, BatchError, ConjugateFamily};
use rmip_packed::ringswitch::{eval_words, frobenius_inverse, WordClaim};
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

fn family<F: PackedField>(words: &[F], n: usize, s: &mut u64) -> ConjugateFamily<F> {
    let base = random_point::<F>(n, s);
    let mut point = base.clone();
    let mut values = Vec::with_capacity(F::BITS);
    for j in 0..F::BITS {
        if j > 0 {
            for x in point.iter_mut() {
                *x = frobenius_inverse(*x, 1);
            }
        }
        values.push(eval_words(words, &point));
    }
    ConjugateFamily { base, values }
}

fn boolean_claim<F: PackedField>(words: &[F], n: usize, index: usize) -> WordClaim<F> {
    let point: Vec<F> = (0..n)
        .map(|i| F::from_bool((index >> i) & 1 == 1))
        .collect();
    assert_eq!(eval_words(words, &point), words[index]);
    WordClaim {
        point,
        value: words[index],
    }
}

fn batching_ends_in_a_true_claim<F: PackedField>(n: usize, seed: u64) {
    let mut s = seed;
    let words: Vec<F> = (0..1usize << n)
        .map(|_| F::from_bits(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128))
        .collect();
    let families = [family(&words, n, &mut s), family(&words, n, &mut s)];
    let plain = [
        boolean_claim(&words, n, 3),
        boolean_claim(&words, n, (1 << n) - 1),
    ];
    type Case<'a, F> = (&'a [ConjugateFamily<F>], &'a [WordClaim<F>]);
    let cases: [Case<'_, F>; 3] = [
        (&families[..], &plain[..]),
        (&families[..1], &[]),
        (&[], &plain[..]),
    ];
    for (fams, pl) in cases {
        let mut pch = SeededChallenger::new(seed);
        let (proof, claim) = prove(&words, fams, pl, &mut pch);
        let mut vch = SeededChallenger::new(seed);
        let verified = verify(n, fams, pl, &proof, &mut vch).expect("accepted");
        assert_eq!(verified, claim);
        assert_eq!(pch.transcript().canonical(), vch.transcript().canonical());
        assert_eq!(
            eval_words(&words, &claim.point),
            claim.value,
            "the batched claim is true"
        );
    }
}

fn a_wrong_claim_is_rejected<F: PackedField>(n: usize, seed: u64) {
    let mut s = seed;
    let words: Vec<F> = (0..1usize << n)
        .map(|_| F::from_bits(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128))
        .collect();
    let mut families = [family(&words, n, &mut s)];
    let mut plain = [boolean_claim(&words, n, 5)];
    let (proof, _) = prove(&words, &families, &plain, &mut SeededChallenger::new(seed));
    families[0].values[7] += F::ONE;
    assert_eq!(
        verify(
            n,
            &families,
            &plain,
            &proof,
            &mut SeededChallenger::new(seed)
        ),
        Err(BatchError::RoundSumMismatch { round: 0 })
    );
    families[0].values[7] += F::ONE;
    plain[0].value += F::ONE;
    assert_eq!(
        verify(
            n,
            &families,
            &plain,
            &proof,
            &mut SeededChallenger::new(seed)
        ),
        Err(BatchError::RoundSumMismatch { round: 0 })
    );
    plain[0].value += F::ONE;
    let (mut proof, _) = prove(&words, &families, &plain, &mut SeededChallenger::new(seed));
    proof.final_value += F::ONE;
    assert_eq!(
        verify(
            n,
            &families,
            &plain,
            &proof,
            &mut SeededChallenger::new(seed)
        ),
        Err(BatchError::FinalMismatch)
    );
}

#[test]
fn batching_ends_in_a_true_claim_at_both_fields() {
    for (n, seed) in [(3usize, 1u64), (6, 2), (10, 3)] {
        batching_ends_in_a_true_claim::<F64>(n, seed);
        batching_ends_in_a_true_claim::<F128>(n, seed);
    }
}

#[test]
fn a_wrong_claim_is_rejected_at_both_fields() {
    a_wrong_claim_is_rejected::<F64>(7, 11);
    a_wrong_claim_is_rejected::<F128>(7, 12);
}

#[test]
fn a_short_live_prefix_runs_every_round() {
    let words = [F64::ONE, F64::ZERO, F64::ZERO, F64::ZERO];
    let plain = [WordClaim {
        point: vec![F64::ZERO, F64::ZERO],
        value: F64::ONE,
    }];
    let (proof, claim) = prove(&words, &[], &plain, &mut SeededChallenger::new(7));
    let verified = verify(2, &[], &plain, &proof, &mut SeededChallenger::new(7)).expect("accepted");
    assert_eq!(verified, claim);
    assert_eq!(claim.value, eval_words(&words, &claim.point));
    let wrong = [WordClaim {
        point: vec![F64::ZERO, F64::ZERO],
        value: F64::ZERO,
    }];
    assert!(verify(2, &[], &wrong, &proof, &mut SeededChallenger::new(7)).is_err());
}
