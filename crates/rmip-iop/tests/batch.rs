use rmip_field::{BinaryField, F128};
use rmip_iop::backend::Claim;
use rmip_iop::batch::{self, BatchMessageProver};
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::sumcheck::{RoundMessageProver, SumcheckError};
use rmip_iop::transcript::{Entry, RawChallenger, SeededChallenger, Transcript};

fn table(nu: usize, seed: u64) -> Vec<F128> {
    let mut s = seed | 1;
    (0..1usize << nu)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            F128::new(((s as u128) << 64) | s.wrapping_mul(0x9E37_79B9) as u128)
        })
        .collect()
}

fn point(nu: usize, k: u64) -> Vec<F128> {
    (0..nu)
        .map(|i| F128::new((k as u128 + 1) * 1_000_003 + i as u128 * 7))
        .collect()
}

#[derive(Default)]
struct OneChallenger {
    transcript: Transcript,
}

impl RawChallenger for OneChallenger {
    fn sample(&mut self) -> F128 {
        self.transcript.record_coin(F128::ONE);
        F128::ONE
    }

    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.transcript.record_ordinary(bytes);
    }

    fn post_oracle(&mut self, id: usize) {
        self.transcript.post_oracle(id);
    }

    fn transcript(&self) -> &Transcript {
        &self.transcript
    }
}

#[derive(Default)]
struct MalformedMessageProver {
    gammas: Option<Vec<F128>>,
    messages: usize,
    challenges: Vec<F128>,
    awaiting_challenge: bool,
}

impl BatchMessageProver for MalformedMessageProver {
    fn begin_batch(&mut self, _claims: &[Claim<F128>], gammas: &[F128]) {
        assert!(
            self.gammas.replace(gammas.to_vec()).is_none(),
            "gammas released twice"
        );
    }
}

impl RoundMessageProver for MalformedMessageProver {
    fn round_message(&mut self, round: usize) -> Vec<F128> {
        assert!(
            self.gammas.is_some(),
            "round requested before the batching coins"
        );
        assert_eq!(round, self.messages, "round requested out of order");
        assert!(
            !self.awaiting_challenge,
            "two messages requested without a challenge"
        );
        self.messages += 1;
        self.awaiting_challenge = true;
        vec![F128::ZERO]
    }

    fn receive_challenge(&mut self, round: usize, challenge: F128) {
        assert!(
            self.awaiting_challenge,
            "challenge released before a message"
        );
        assert_eq!(round + 1, self.messages, "challenge released out of order");
        self.awaiting_challenge = false;
        self.challenges.push(challenge);
    }
}

#[test]
fn honest_claims_reduce_to_a_true_claim() {
    for nu in [3usize, 6, 9] {
        let t = table(nu, 42);
        for k in [1usize, 2, 5, 40] {
            let claims: Vec<Claim<F128>> = (0..k)
                .map(|i| {
                    let p = point(nu, i as u64);
                    Claim {
                        value: eval_mle(&t, &p),
                        point: p,
                    }
                })
                .collect();
            let mut ch = SeededChallenger::new(7);
            let out = batch::reduce(&claims, &t, &mut ch).expect("honest reduction verifies");
            assert_eq!(out.value, eval_mle(&t, &out.point), "nu={nu} K={k}");
        }
    }
}

#[test]
fn separate_prover_and_verifier_derive_the_same_reduced_claim() {
    let nu = 6;
    let t = table(nu, 0x51);
    let claims: Vec<Claim<F128>> = (0..9)
        .map(|i| {
            let p = point(nu, i);
            Claim {
                value: eval_mle(&t, &p),
                point: p,
            }
        })
        .collect();
    let mut prover = SeededChallenger::new(99);
    let (proof, prover_claim) =
        batch::prove(&claims, &t, &mut prover).expect("honest batching prover");
    let mut verifier = SeededChallenger::new(99);
    let out = batch::verify_proof(&claims, nu, &proof, &mut verifier);
    assert_eq!(out.verdict, Ok(()));
    assert_eq!(out.claim, prover_claim);
    assert_eq!(out.claim.value, eval_mle(&t, &out.claim.point));
    assert_eq!(
        prover.transcript().entries(),
        verifier.transcript().entries()
    );
}

#[test]
fn live_verifier_owns_gamma_and_every_round_challenge() {
    let nu = 6;
    let t = table(nu, 0x5151);
    let claims: Vec<Claim<F128>> = (0..9)
        .map(|i| {
            let p = point(nu, i);
            Claim {
                value: eval_mle(&t, &p),
                point: p,
            }
        })
        .collect();

    let mut honest = batch::HonestBatchProver::new(&claims, &t);
    let mut verifier = SeededChallenger::new(0x9898);
    let live = batch::verify_interactive(&claims, nu, &mut honest, &mut verifier);
    assert_eq!(live.verification.verdict, Ok(()));
    assert_eq!(
        live.verification.claim.value,
        eval_mle(&t, &live.verification.claim.point)
    );
    assert_eq!(live.proof.sumcheck.round_polys.len(), nu);

    let mut replay = SeededChallenger::new(0x9898);
    let static_out = batch::prove_recording(&claims, &t, &mut replay);
    assert_eq!(live.proof, static_out.proof);
    assert_eq!(live.verification.claim, static_out.claim);
    assert_eq!(live.verification.verdict, static_out.verdict);
    assert_eq!(
        verifier.transcript().entries(),
        replay.transcript().entries()
    );
}

#[test]
fn live_verifier_finishes_the_coin_schedule_after_a_malformed_message() {
    let nu = 5;
    let t = table(nu, 0x6161);
    let p = point(nu, 3);
    let claims = vec![Claim {
        value: eval_mle(&t, &p),
        point: p,
    }];
    let mut malicious = MalformedMessageProver::default();
    let mut verifier = SeededChallenger::new(0x7171);

    let out = batch::verify_interactive(&claims, nu, &mut malicious, &mut verifier);
    assert_eq!(
        out.verification.verdict,
        Err(batch::BatchProofError::Sumcheck(
            SumcheckError::WrongDegree {
                round: 0,
                expected: 2,
                got: 1,
            }
        ))
    );
    assert_eq!(out.proof.sumcheck.round_polys.len(), nu);
    assert_eq!(malicious.messages, nu);
    assert_eq!(malicious.challenges.len(), nu);
    assert!(!malicious.awaiting_challenge);
    assert_eq!(
        verifier
            .transcript()
            .entries()
            .iter()
            .filter(|entry| matches!(entry, Entry::Coin(_)))
            .count(),
        batch::coins_for(claims.len()) + nu,
        "the batching coins plus one challenge for every scheduled round"
    );
}

#[test]
fn the_previously_fatal_duplicate_list_now_reduces_to_a_true_claim() {
    let nu = 4;
    let t = table(nu, 0x8181);
    let p = point(nu, 2);
    let value = eval_mle(&t, &p);
    let claims = vec![
        Claim {
            point: p.clone(),
            value,
        },
        Claim { point: p, value },
    ];

    let mut prover = OneChallenger::default();
    let output = batch::prove_recording(&claims, &t, &mut prover);
    assert_eq!(
        output.verdict,
        Ok(()),
        "duplicates cannot vanish the eq-tensor"
    );
    assert_eq!(output.proof.sumcheck.round_polys.len(), nu);
    assert_eq!(eval_mle(&t, &output.claim.point), output.claim.value);

    let mut verifier = OneChallenger::default();
    let checked = batch::verify_proof(&claims, nu, &output.proof, &mut verifier);
    assert_eq!(checked.verdict, Ok(()));
    assert_eq!(
        checked.claim, output.claim,
        "both parties derive the same claim"
    );
    assert_eq!(
        prover.transcript().entries(),
        verifier.transcript().entries()
    );
    assert_eq!(
        prover
            .transcript()
            .entries()
            .iter()
            .filter(|entry| matches!(entry, Entry::Coin(_)))
            .count(),
        nu + 1
    );
}

#[test]
fn a_corrupted_coefficient_diverts_the_reduced_claim() {
    let nu = 5;
    let t = table(nu, 0x91);
    let p = point(nu, 4);
    let claims = vec![Claim {
        value: eval_mle(&t, &p),
        point: p,
    }];
    let mut prover = SeededChallenger::new(13);
    let (mut proof, honest_claim) = batch::prove(&claims, &t, &mut prover).unwrap();
    proof.sumcheck.round_polys[0][0] = proof.sumcheck.round_polys[0][0].add(F128::ONE);

    let mut verifier = SeededChallenger::new(13);
    let out = batch::verify_proof(&claims, nu, &proof, &mut verifier);
    assert!(
        out.verdict.is_ok(),
        "recovery accepts any well-formed stream"
    );
    assert_ne!(
        out.claim.value, honest_claim.value,
        "the corruption must divert the claim, or the opening could never catch it"
    );
    assert_eq!(
        out.claim.point, honest_claim.point,
        "coins are unchanged by the corruption"
    );
}

#[test]
fn repeated_points_reduce_to_the_same_true_claim() {
    for nu in [3usize, 6, 9] {
        let t = table(nu, 42);
        let order = [0usize, 1, 0, 0, 2, 1, 0, 2, 2, 0, 1, 0];
        let pts: Vec<Vec<F128>> = (0..3).map(|i| point(nu, i as u64)).collect();
        let claims: Vec<Claim<F128>> = order
            .iter()
            .map(|&i| Claim {
                value: eval_mle(&t, &pts[i]),
                point: pts[i].clone(),
            })
            .collect();

        let mut ch = SeededChallenger::new(7);
        let out = batch::reduce(&claims, &t, &mut ch).expect("honest reduction verifies");
        assert_eq!(
            out.value,
            eval_mle(&t, &out.point),
            "nu={nu}, repeated points"
        );
    }
}

#[test]
fn a_false_repeat_is_not_absorbed_by_its_honest_twin() {
    let nu = 6;
    let t = table(nu, 42);
    let p = point(nu, 3);
    let honest = eval_mle(&t, &p);

    let claims = vec![
        Claim {
            point: p.clone(),
            value: honest,
        },
        Claim {
            point: p.clone(),
            value: honest.add(F128::ONE),
        },
        Claim {
            point: p.clone(),
            value: honest,
        },
    ];
    let mut ch = SeededChallenger::new(7);
    assert!(
        batch::reduce(&claims, &t, &mut ch).is_err(),
        "a false claim on a REPEATED point was absorbed: grouping points must \
         merge their gamma weights, never their values"
    );
}

#[test]
fn a_single_false_claim_is_caught() {
    let nu = 7;
    let t = table(nu, 9);
    let k = 30;
    for corrupt in [0usize, 7, 29] {
        let mut claims: Vec<Claim<F128>> = (0..k)
            .map(|i| {
                let p = point(nu, i as u64);
                Claim {
                    value: eval_mle(&t, &p),
                    point: p,
                }
            })
            .collect();
        claims[corrupt].value = claims[corrupt].value.add(F128::ONE);

        let mut rejected = 0;
        let mut survived_false = 0;
        for seed in 1..=20u64 {
            let mut ch = SeededChallenger::new(seed);
            match batch::reduce(&claims, &t, &mut ch) {
                Err(_) => rejected += 1,
                Ok(out) => {
                    if out.value != eval_mle(&t, &out.point) {
                        survived_false += 1;
                    }
                }
            }
        }
        assert_eq!(
            rejected + survived_false,
            20,
            "false claim at {corrupt} survived"
        );
        assert_eq!(
            rejected, 20,
            "reduce is expected to catch a false claim at the round-sum \
             self-check on every seed; {survived_false} of 20 escaped to the \
             opening at corrupt index {corrupt}"
        );
    }
}

#[test]
fn verifier_side_evaluation_matches_the_prover_table() {
    let nu = 6;
    let t = table(nu, 3);
    let k = 12;
    let claims: Vec<Claim<F128>> = (0..k)
        .map(|i| {
            let p = point(nu, i as u64);
            Claim {
                value: eval_mle(&t, &p),
                point: p,
            }
        })
        .collect();
    let gammas: Vec<F128> = (0..batch::coins_for(k))
        .map(|j| F128::new(0xfeed_face + j as u128))
        .collect();
    let coeffs = eq_table(&gammas);

    let mut e = vec![F128::ZERO; t.len()];
    for (i, c) in claims.iter().enumerate() {
        for (w, ev) in eq_table(&c.point).iter().enumerate() {
            e[w] = e[w].add(coeffs[i].mul(*ev));
        }
    }

    for probe in 0..5u64 {
        let r = point(nu, 900 + probe);
        assert_eq!(
            batch::eval_batching_poly(&claims, &gammas, &r),
            eval_mle(&e, &r),
            "verifier and prover disagree on E"
        );
    }
}

#[test]
fn a_vanishing_batching_polynomial_is_rejected_not_ignored() {
    let nu = 4;
    let t = table(nu, 5);

    let p = point(nu, 1);
    let honest = vec![Claim {
        value: eval_mle(&t, &p),
        point: p,
    }];
    let mut ch = SeededChallenger::new(2);
    let out = batch::reduce(&honest, &t, &mut ch).unwrap();
    assert_eq!(out.value, eval_mle(&t, &out.point));

    let bad_point = vec![F128::ZERO, F128::ONE, F128::new(5), F128::new(9)];
    let ones = vec![F128::ONE; nu];
    let claims = vec![Claim {
        value: eval_mle(&t, &bad_point),
        point: bad_point,
    }];
    assert_eq!(batch::coins_for(claims.len()), 0);
    assert_eq!(
        batch::eval_batching_poly(&claims, &[], &ones),
        F128::ZERO,
        "the fixture must actually make E(r) vanish"
    );
    assert!(F128::ZERO.inv().is_none());

    let mut prover_ch = OneChallenger::default();
    let recorded = batch::prove_recording(&claims, &t, &mut prover_ch);
    assert_eq!(
        recorded.verdict,
        Err(batch::BatchProofError::VanishingBatchingPolynomial),
        "the prover must report the vanishing polynomial, not divide by zero"
    );
    assert_eq!(recorded.claim.point, ones, "the bound point is 1^nu");
    assert_eq!(
        recorded.claim.value,
        F128::ZERO,
        "the canonical fallback value"
    );

    let mut verifier_ch = OneChallenger::default();
    let checked = batch::verify_proof(&claims, nu, &recorded.proof, &mut verifier_ch);
    assert_eq!(
        checked.verdict,
        Err(batch::BatchProofError::VanishingBatchingPolynomial),
        "the verifier must reach the same verdict from public data alone"
    );
    assert_eq!(
        checked.claim, recorded.claim,
        "both sides derive the same fallback claim"
    );

    let mut wrapper_ch = OneChallenger::default();
    assert_eq!(
        batch::prove(&claims, &t, &mut wrapper_ch).unwrap_err(),
        batch::BatchProofError::VanishingBatchingPolynomial
    );

    let mut joint_ch = OneChallenger::default();
    assert_eq!(
        batch::reduce(&claims, &t, &mut joint_ch).unwrap_err(),
        SumcheckError::RoundSumMismatch(nu),
        "reduce reports the vanishing polynomial as a last-round sum mismatch"
    );
}

#[test]
fn error_formula() {
    let e = batch::error(718, 13);
    assert!((e - (10.0 + 26.0) / 2f64.powi(128)).abs() < 1e-40);
    assert!(e < 2f64.powi(-118), "negligible at kappa = 128");
}

#[test]
fn the_boolean_fast_path_is_the_general_path() {
    let nu = 6;
    let t = table(nu, 0xB001);
    let mut claims = Vec::new();
    for s in [0usize, 1, 5, 63, 5] {
        let p: Vec<F128> = (0..nu)
            .map(|k| F128::new(((s >> (nu - 1 - k)) & 1) as u128))
            .collect();
        claims.push(Claim {
            value: eval_mle(&t, &p),
            point: p,
        });
    }
    for seed in [7u64, 8] {
        let p = point(nu, seed);
        claims.push(Claim {
            value: eval_mle(&t, &p),
            point: p,
        });
    }

    let mut prover = SeededChallenger::new(0xFA57);
    let (proof, claim) = batch::prove(&claims, &t, &mut prover).unwrap();
    assert_eq!(
        eval_mle(&t, &claim.point),
        claim.value,
        "the reduced claim must be true"
    );

    let mut verifier = SeededChallenger::new(0xFA57);
    let out = batch::verify_proof(&claims, nu, &proof, &mut verifier);
    assert_eq!(out.verdict, Ok(()));
    assert_eq!(out.claim, claim);

    let p1: Vec<F128> = (0..nu)
        .map(|k| if k == nu - 1 { F128::ONE } else { F128::ZERO })
        .collect();
    assert_eq!(eval_mle(&t, &p1), t[1]);
}

#[test]
fn the_boolean_prefix_fast_path_is_the_general_path() {
    let nu = 8;
    let t = table(nu, 0xB002);
    let suffix = point(3, 99);
    let mut claims = Vec::new();
    for slot in [0usize, 1, 7, 31, 7] {
        let mut p: Vec<F128> = (0..5)
            .map(|k| F128::new(((slot >> (4 - k)) & 1) as u128))
            .collect();
        p.extend_from_slice(&suffix);
        claims.push(Claim {
            value: eval_mle(&t, &p),
            point: p,
        });
    }
    let p = point(nu, 3);
    claims.push(Claim {
        value: eval_mle(&t, &p),
        point: p,
    });

    let mut prover = SeededChallenger::new(0xFA58);
    let (proof, claim) = batch::prove(&claims, &t, &mut prover).unwrap();
    assert_eq!(
        eval_mle(&t, &claim.point),
        claim.value,
        "the reduced claim must be true"
    );
    let mut verifier = SeededChallenger::new(0xFA58);
    let out = batch::verify_proof(&claims, nu, &proof, &mut verifier);
    assert_eq!(out.verdict, Ok(()));
    assert_eq!(out.claim, claim);
}
