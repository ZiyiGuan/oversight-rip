use crate::backend::Claim;
use crate::metrics::{self, Role};
use crate::mle::{fold_in_place, scaled_eq_table_into};
use crate::sumcheck::{
    self, round_evals_over, RoundMessageProver, SumcheckError, SumcheckProof, Summand,
};
use crate::transcript::RawChallenger;
use rmip_field::{BinaryField, F128};

pub struct Batched {
    tables: Vec<Vec<F128>>,
    vars: usize,
}

impl Summand for Batched {
    fn num_vars(&self) -> usize {
        self.vars
    }
    fn degree(&self) -> usize {
        2
    }
    fn round_evals(&self) -> Vec<F128> {
        round_evals_over(&self.tables, 2, |v| v[0].mul(v[1]))
    }
    fn fold(&mut self, r: F128) {
        for t in &mut self.tables {
            fold_in_place(t, r);
        }
        self.vars -= 1;
    }
    fn final_value(&self) -> F128 {
        self.tables[0][0].mul(self.tables[1][0])
    }
}

pub fn coins_for(k: usize) -> usize {
    k.next_power_of_two().trailing_zeros() as usize
}

fn coefficients(gammas: &[F128], k: usize) -> Vec<F128> {
    assert_eq!(
        gammas.len(),
        coins_for(k),
        "one coin per eq-tensor variable"
    );
    if gammas.is_empty() {
        return vec![F128::ONE];
    }
    let mut table = crate::mle::eq_table(gammas);
    table.truncate(k);
    table
}

fn accumulate_scaled_eq(point: &[F128], coeff: F128, acc: &mut [F128], scratch: &mut Vec<F128>) {
    assert_eq!(
        acc.len(),
        1usize << point.len(),
        "claim arity must match the table"
    );
    let mut index = 0usize;
    let mut boolean = true;
    for &c in point {
        index <<= 1;
        if c == F128::ONE {
            index |= 1;
        } else if c != F128::ZERO {
            boolean = false;
            break;
        }
    }
    if boolean {
        acc[index] = acc[index].add(coeff);
        return;
    }
    let mut prefix_bits = 0usize;
    let mut prefix_index = 0usize;
    for &c in point {
        if c == F128::ONE {
            prefix_index = (prefix_index << 1) | 1;
            prefix_bits += 1;
        } else if c == F128::ZERO {
            prefix_index <<= 1;
            prefix_bits += 1;
        } else {
            break;
        }
    }
    if prefix_bits > 0 {
        let suffix = &point[prefix_bits..];
        let block = 1usize << suffix.len();
        scaled_eq_table_into(suffix, coeff, scratch);
        let start = prefix_index * block;
        for (entry, value) in acc[start..start + block].iter_mut().zip(scratch.iter()) {
            *entry = entry.add(*value);
        }
        return;
    }
    scaled_eq_table_into(point, coeff, scratch);
    for (entry, value) in acc.iter_mut().zip(scratch.iter()) {
        *entry = entry.add(*value);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchProof {
    pub sumcheck: SumcheckProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchProofError {
    Sumcheck(SumcheckError),
    VanishingBatchingPolynomial,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchProverOutput {
    pub proof: BatchProof,
    pub claim: Claim<F128>,
    pub verdict: Result<(), BatchProofError>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchVerification {
    pub claim: Claim<F128>,
    pub verdict: Result<(), BatchProofError>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchInteraction {
    pub proof: BatchProof,
    pub verification: BatchVerification,
}

pub trait BatchMessageProver: RoundMessageProver {
    fn begin_batch(&mut self, claims: &[Claim<F128>], gammas: &[F128]);
}

pub struct HonestBatchProver {
    claims: Option<Vec<Claim<F128>>>,
    table: Vec<F128>,
    summand: Option<Batched>,
    next_round: usize,
    awaiting_challenge: bool,
}

impl HonestBatchProver {
    pub fn new(claims: &[Claim<F128>], table: &[F128]) -> Self {
        let _role = metrics::enter(Role::MainProver);
        validate_prover_inputs(claims, table);
        Self {
            claims: Some(claims.to_vec()),
            table: table.to_vec(),
            summand: None,
            next_round: 0,
            awaiting_challenge: false,
        }
    }

    pub fn for_table(table: &[F128]) -> Self {
        let _role = metrics::enter(Role::MainProver);
        assert!(
            table.len().is_power_of_two(),
            "committed table length must be a power of two"
        );
        Self {
            claims: None,
            table: table.to_vec(),
            summand: None,
            next_round: 0,
            awaiting_challenge: false,
        }
    }
}

impl BatchMessageProver for HonestBatchProver {
    fn begin_batch(&mut self, claims: &[Claim<F128>], gammas: &[F128]) {
        let _role = metrics::enter(Role::MainProver);
        assert!(self.summand.is_none(), "batching phase started twice");
        validate_prover_inputs(claims, &self.table);
        if let Some(expected) = &self.claims {
            assert_eq!(
                expected, claims,
                "live verifier supplied a different claim list"
            );
        } else {
            self.claims = Some(claims.to_vec());
        }
        let e = batching_table(claims, gammas, self.table.len());
        self.summand = Some(Batched {
            tables: vec![e, self.table.clone()],
            vars: self.table.len().trailing_zeros() as usize,
        });
    }
}

impl RoundMessageProver for HonestBatchProver {
    fn round_message(&mut self, round: usize) -> Vec<F128> {
        let _role = metrics::enter(Role::MainProver);
        assert_eq!(
            round, self.next_round,
            "batch rounds requested out of order"
        );
        assert!(
            !self.awaiting_challenge,
            "next message requested before releasing challenge"
        );
        self.awaiting_challenge = true;
        {
            let evals = self
                .summand
                .as_ref()
                .expect("batching phase was not started")
                .round_evals();
            let coeffs = crate::mle::monomial_from_evals(&evals);
            coeffs[..coeffs.len() - 1].to_vec()
        }
    }

    fn receive_challenge(&mut self, round: usize, challenge: F128) {
        let _role = metrics::enter(Role::MainProver);
        assert_eq!(
            round, self.next_round,
            "batch challenges released out of order"
        );
        assert!(
            self.awaiting_challenge,
            "challenge released before requesting a message"
        );
        self.summand
            .as_mut()
            .expect("batching phase was not started")
            .fold(challenge);
        self.awaiting_challenge = false;
        self.next_round += 1;
    }
}

fn validate_prover_inputs(claims: &[Claim<F128>], table: &[F128]) -> usize {
    assert!(!claims.is_empty(), "nothing to batch");
    assert!(
        table.len().is_power_of_two(),
        "committed table length must be a power of two"
    );
    let nu = table.len().trailing_zeros() as usize;
    assert!(
        claims.iter().all(|claim| claim.point.len() == nu),
        "claim arity mismatch"
    );
    nu
}

fn reduced_claim_or_fallback(
    claims: &[Claim<F128>],
    gammas: &[F128],
    point: Vec<F128>,
    expected: F128,
) -> (Claim<F128>, bool) {
    let e_at = eval_batching_poly(claims, gammas, &point);
    match e_at.inv() {
        Some(inverse) => (
            Claim {
                point,
                value: expected.mul(inverse),
            },
            false,
        ),
        None => (
            Claim {
                point,
                value: F128::ZERO,
            },
            true,
        ),
    }
}

fn batching_target(claims: &[Claim<F128>], gammas: &[F128]) -> F128 {
    claims
        .iter()
        .zip(coefficients(gammas, claims.len()))
        .map(|(claim, coefficient)| coefficient.mul(claim.value))
        .fold(F128::ZERO, |acc, value| acc.add(value))
}

fn batching_table(claims: &[Claim<F128>], gammas: &[F128], n: usize) -> Vec<F128> {
    let coefficients = coefficients(gammas, claims.len());
    let mut distinct: Vec<(&[F128], F128)> = Vec::new();
    for (claim, coefficient) in claims.iter().zip(&coefficients) {
        match distinct
            .iter_mut()
            .find(|(point, _)| *point == claim.point.as_slice())
        {
            Some((_, accumulated)) => *accumulated = accumulated.add(*coefficient),
            None => distinct.push((claim.point.as_slice(), *coefficient)),
        }
    }

    let mut table = vec![F128::ZERO; n];
    let mut scratch = Vec::new();
    for (point, coefficient) in distinct {
        assert_eq!(
            1usize << point.len(),
            n,
            "claim point arity must match the committed table"
        );
        accumulate_scaled_eq(point, coefficient, &mut table, &mut scratch);
    }
    table
}

pub fn prove_recording(
    claims: &[Claim<F128>],
    table: &[F128],
    ch: &mut dyn RawChallenger,
) -> BatchProverOutput {
    let _role = metrics::enter(Role::MainProver);
    let nu = validate_prover_inputs(claims, table);

    let gammas: Vec<F128> = {
        let _verifier = metrics::enter(Role::Verifier);
        (0..coins_for(claims.len())).map(|_| ch.sample()).collect()
    };
    let e = batching_table(claims, &gammas, table.len());
    let mut summand = Batched {
        tables: vec![e, table.to_vec()],
        vars: nu,
    };
    let (sumcheck, point, expected) = sumcheck::prove(&mut summand, ch);
    let (claim, vanishing) = reduced_claim_or_fallback(claims, &gammas, point, expected);
    let verdict = if vanishing {
        Err(BatchProofError::VanishingBatchingPolynomial)
    } else {
        Ok(())
    };
    BatchProverOutput {
        proof: BatchProof { sumcheck },
        claim,
        verdict,
    }
}

pub fn prove(
    claims: &[Claim<F128>],
    table: &[F128],
    ch: &mut dyn RawChallenger,
) -> Result<(BatchProof, Claim<F128>), BatchProofError> {
    let _role = metrics::enter(Role::MainProver);
    let BatchProverOutput {
        proof,
        claim,
        verdict,
    } = prove_recording(claims, table, ch);
    verdict?;
    Ok((proof, claim))
}

pub fn verify_proof(
    claims: &[Claim<F128>],
    nu: usize,
    proof: &BatchProof,
    ch: &mut dyn RawChallenger,
) -> BatchVerification {
    let _role = metrics::enter(Role::Verifier);
    assert!(!claims.is_empty(), "nothing to batch");
    assert!(
        claims.iter().all(|claim| claim.point.len() == nu),
        "claim arity mismatch"
    );
    let gammas: Vec<F128> = (0..coins_for(claims.len())).map(|_| ch.sample()).collect();
    let target = batching_target(claims, &gammas);
    let (out, sumcheck_error) = sumcheck::verify_recording(&proof.sumcheck, target, nu, 2, ch);
    let (claim, vanishing) = reduced_claim_or_fallback(claims, &gammas, out.point, out.expected);
    let verdict = if let Some(error) = sumcheck_error {
        Err(BatchProofError::Sumcheck(error))
    } else if vanishing {
        Err(BatchProofError::VanishingBatchingPolynomial)
    } else {
        Ok(())
    };
    BatchVerification { claim, verdict }
}

pub fn verify_interactive<P: BatchMessageProver + ?Sized>(
    claims: &[Claim<F128>],
    nu: usize,
    prover: &mut P,
    ch: &mut dyn RawChallenger,
) -> BatchInteraction {
    let _role = metrics::enter(Role::Verifier);
    assert!(!claims.is_empty(), "nothing to batch");
    assert!(
        claims.iter().all(|claim| claim.point.len() == nu),
        "claim arity mismatch"
    );

    let gammas: Vec<F128> = (0..coins_for(claims.len())).map(|_| ch.sample()).collect();
    {
        let _prover = metrics::enter(Role::MainProver);
        prover.begin_batch(claims, &gammas);
    }
    let target = batching_target(claims, &gammas);
    let (sumcheck, out, sumcheck_error) = sumcheck::verify_interactive(prover, target, nu, 2, ch);
    let (claim, vanishing) = reduced_claim_or_fallback(claims, &gammas, out.point, out.expected);
    let verdict = if let Some(error) = sumcheck_error {
        Err(BatchProofError::Sumcheck(error))
    } else if vanishing {
        Err(BatchProofError::VanishingBatchingPolynomial)
    } else {
        Ok(())
    };
    BatchInteraction {
        proof: BatchProof { sumcheck },
        verification: BatchVerification { claim, verdict },
    }
}

pub fn eval_batching_poly(claims: &[Claim<F128>], gammas: &[F128], point: &[F128]) -> F128 {
    let gs = coefficients(gammas, claims.len());
    let mut distinct: Vec<(&[F128], F128)> = Vec::new();
    for (c, g) in claims.iter().zip(&gs) {
        match distinct.iter_mut().find(|(p, _)| *p == c.point.as_slice()) {
            Some((_, acc)) => *acc = acc.add(*g),
            None => distinct.push((c.point.as_slice(), *g)),
        }
    }
    distinct
        .into_iter()
        .map(|(p, g)| {
            let e = p
                .iter()
                .zip(point)
                .map(|(p, r)| p.mul(*r).add(F128::ONE.add(*p).mul(F128::ONE.add(*r))))
                .fold(F128::ONE, |a, b| a.mul(b));
            g.mul(e)
        })
        .fold(F128::ZERO, |a, b| a.add(b))
}

pub fn reduce(
    claims: &[Claim<F128>],
    table: &[F128],
    ch: &mut dyn RawChallenger,
) -> Result<Claim<F128>, SumcheckError> {
    let _role = metrics::enter(Role::Verifier);
    assert!(!claims.is_empty(), "nothing to batch");
    let nu = table.len().trailing_zeros() as usize;
    let gammas: Vec<F128> = (0..coins_for(claims.len())).map(|_| ch.sample()).collect();
    let gs = coefficients(&gammas, claims.len());

    let _prover_table = metrics::enter(Role::MainProver);
    let mut distinct: Vec<(&[F128], F128)> = Vec::new();
    for (c, g) in claims.iter().zip(&gs) {
        match distinct.iter_mut().find(|(p, _)| *p == c.point.as_slice()) {
            Some((_, acc)) => *acc = acc.add(*g),
            None => distinct.push((c.point.as_slice(), *g)),
        }
    }
    let mut e = vec![F128::ZERO; table.len()];
    let mut scratch = Vec::new();
    for (point, coeff) in &distinct {
        accumulate_scaled_eq(point, *coeff, &mut e, &mut scratch);
    }
    drop(scratch);
    drop(distinct);
    drop(_prover_table);
    let target = claims
        .iter()
        .zip(&gs)
        .map(|(c, g)| g.mul(c.value))
        .fold(F128::ZERO, |a, b| a.add(b));

    let mut s = {
        let _prover = metrics::enter(Role::MainProver);
        Batched {
            tables: vec![e, table.to_vec()],
            vars: nu,
        }
    };
    let (_proof, out) = sumcheck::run(&mut s, target, ch)?;
    {
        let _prover = metrics::enter(Role::MainProver);
        drop(s);
    }

    let e_at = eval_batching_poly(claims, &gammas, &out.point);
    let v = match e_at.inv() {
        None => return Err(SumcheckError::RoundSumMismatch(nu)),
        Some(inv) => out.expected.mul(inv),
    };
    Ok(Claim {
        point: out.point,
        value: v,
    })
}

pub fn error(k: usize, nu: usize) -> f64 {
    crate::backend::div_error_upward((coins_for(k) + 2 * nu) as f64, 2f64.powi(128))
}

#[cfg(test)]
mod completeness_error_tests {
    use super::*;

    #[test]
    fn duplicate_points_no_longer_vanish_identically() {
        let nu = 6;
        let p: Vec<F128> = (0..nu).map(|i| F128::new(i as u128 * 7919 + 11)).collect();
        let dup = vec![
            Claim {
                value: F128::new(3),
                point: p.clone(),
            },
            Claim {
                value: F128::new(3),
                point: p.clone(),
            },
        ];
        let y: Vec<F128> = (0..nu).map(|i| F128::new(i as u128 + 5)).collect();

        assert_ne!(
            eval_batching_poly(&dup, &[F128::ONE], &y),
            F128::ZERO,
            "the eq-tensor must not vanish on duplicates at the all-ones coin"
        );

        let mut zeros = 0;
        for j in 0..64u128 {
            let g = [F128::new(j * 104_729 + 7)];
            if eval_batching_poly(&dup, &g, &y) == F128::ZERO {
                zeros += 1;
            }
        }
        assert_eq!(
            zeros, 0,
            "E vanished at {zeros}/64 sampled coins on a duplicated pair"
        );
    }

    fn ceil_log2(k: usize) -> usize {
        let mut bits = 0usize;
        let mut reach = 1usize;
        while reach < k {
            reach *= 2;
            bits += 1;
        }
        bits
    }

    #[test]
    fn the_declared_error_dominates_the_union_bound() {
        let field = 2f64.powi(128);
        for (k, nu) in [(718usize, 13usize), (21, 6), (2, 6), (1, 20)] {
            let terms = ceil_log2(k) + 2 * nu;
            let union = terms as f64 / field;
            assert!(
                error(k, nu) >= union,
                "declared error must be at least (ceil_log2(K)+2nu)/|F| = {terms}/|F| \
                 at K={k}, nu={nu}"
            );
            assert!(
                error(k, nu) < (terms + 1) as f64 / field,
                "declared error must be TIGHT: below ({terms}+1)/|F| at K={k}, nu={nu}"
            );
            if k > 64 {
                let old_union = (k as f64 - 1.0 + nu as f64) / field;
                assert!(
                    error(k, nu) < old_union / 20.0,
                    "the 72x-class improvement is real"
                );
            }
        }
    }
}
