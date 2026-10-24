use crate::metrics::{self, Role};
use crate::mle::{interpolation_points, monomial_from_evals};
use crate::transcript::RawChallenger;
use rmip_field::{BinaryField, F128};

pub trait Summand {
    fn num_vars(&self) -> usize;
    fn degree(&self) -> usize;
    fn round_evals(&self) -> Vec<F128>;
    fn fold(&mut self, r: F128);
    fn final_value(&self) -> F128;
}

pub trait RoundMessageProver {
    fn round_message(&mut self, round: usize) -> Vec<F128>;

    fn receive_challenge(&mut self, round: usize, challenge: F128);
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof {
    pub round_polys: Vec<Vec<F128>>,
}

fn record_round_message(ch: &mut dyn RawChallenger, evaluations: &[F128]) {
    let mut encoded = Vec::with_capacity(evaluations.len() * 16);
    for value in evaluations {
        encoded.extend_from_slice(&value.to_be_bytes());
    }
    ch.record_ordinary(&encoded);
}

pub fn recover(truncated: &[F128], sum: F128) -> Vec<F128> {
    let mut coeffs = truncated.to_vec();
    let first = coeffs.first().copied().unwrap_or(F128::ZERO);
    let total = coeffs.iter().fold(F128::ZERO, |a, &c| a.add(c));
    coeffs.push(sum.add(first).add(total));
    coeffs
}

pub fn run<S: Summand>(
    summand: &mut S,
    claim: F128,
    ch: &mut dyn RawChallenger,
) -> Result<(SumcheckProof, SumcheckClaim), SumcheckError> {
    let _role = metrics::enter(Role::Verifier);
    let (proof, sc, verdict) = run_recording(summand, claim, ch);
    match verdict {
        Some(e) => Err(e),
        None => Ok((proof, sc)),
    }
}

pub fn run_recording<S: Summand>(
    summand: &mut S,
    claim: F128,
    ch: &mut dyn RawChallenger,
) -> (SumcheckProof, SumcheckClaim, Option<SumcheckError>) {
    use crate::binius_channel::{down, ProverTranscript, SummandProver};

    let _role = metrics::enter(Role::Verifier);
    let mut running = claim;
    let mut verdict = None;
    let mut pc = ProverTranscript::new(ch);
    let out = binius_ip_prover::sumcheck::prove_single(
        SummandProver::new(summand, &mut running, &mut verdict),
        &mut pc,
    );
    let round_polys = pc.sent;
    (
        SumcheckProof { round_polys },
        SumcheckClaim {
            point: out.challenges.into_iter().map(down).collect(),
            expected: running,
        },
        verdict,
    )
}

pub fn prove<S: Summand>(
    summand: &mut S,
    ch: &mut dyn RawChallenger,
) -> (SumcheckProof, Vec<F128>, F128) {
    let _role = metrics::enter(Role::MainProver);
    let m = summand.num_vars();
    let mut round_polys = Vec::with_capacity(m);
    let mut point = Vec::with_capacity(m);

    for _ in 0..m {
        let evals = summand.round_evals();
        let coeffs = monomial_from_evals(&evals);
        let wire = coeffs[..coeffs.len() - 1].to_vec();
        record_round_message(ch, &wire);
        round_polys.push(wire);
        let r = {
            let _verifier = metrics::enter(Role::Verifier);
            ch.sample()
        };
        summand.fold(r);
        point.push(r);
    }

    (SumcheckProof { round_polys }, point, summand.final_value())
}

#[derive(Clone, Debug)]
pub struct SumcheckClaim {
    pub point: Vec<F128>,
    pub expected: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SumcheckError {
    RoundSumMismatch(usize),
    WrongRoundCount {
        expected: usize,
        got: usize,
    },
    WrongDegree {
        round: usize,
        expected: usize,
        got: usize,
    },
}

pub fn verify(
    proof: &SumcheckProof,
    claim: F128,
    num_vars: usize,
    degree: usize,
    ch: &mut dyn RawChallenger,
) -> Result<SumcheckClaim, SumcheckError> {
    let _role = metrics::enter(Role::Verifier);
    let (claim, verdict) = verify_recording(proof, claim, num_vars, degree, ch);
    match verdict {
        Some(e) => Err(e),
        None => Ok(claim),
    }
}

pub fn verify_recording(
    proof: &SumcheckProof,
    claim: F128,
    num_vars: usize,
    degree: usize,
    ch: &mut dyn RawChallenger,
) -> (SumcheckClaim, Option<SumcheckError>) {
    use crate::binius_channel::{down, up, RecordedSumcheck, B128};

    let _role = metrics::enter(Role::Verifier);
    let mut chan = RecordedSumcheck::new(&proof.round_polys, num_vars, degree, ch);
    let out = binius_ip::sumcheck::verify::<B128, _>(num_vars, degree, up(claim), &mut chan)
        .expect("RecordedSumcheck never returns Err");
    let verdict = chan.finish();
    (
        SumcheckClaim {
            point: out.challenges.into_iter().map(down).collect(),
            expected: down(out.eval),
        },
        verdict,
    )
}

pub fn run_mlecheck_recording(
    zc: &mut crate::zerocheck::Zerocheck,
    point: &[F128],
    ch: &mut dyn RawChallenger,
) -> (SumcheckProof, SumcheckClaim) {
    use crate::binius_channel::{down, up, B128};
    use binius_field::Field as _;

    let _role = metrics::enter(Role::Verifier);
    let num_vars = point.len();
    let mut round_polys = Vec::with_capacity(num_vars);
    let mut bound = Vec::with_capacity(num_vars);
    let mut eval = B128::ZERO;
    let alphas: Vec<B128> = point.iter().copied().map(up).collect();

    for (i, z_i) in alphas.iter().enumerate() {
        let wire = {
            let _prover = metrics::enter(Role::MainProver);
            let wire = zc.round_message();
            record_round_message(ch, &wire);
            wire
        };
        let r = ch.sample();
        {
            let _prover = metrics::enter(Role::MainProver);
            zc.receive_challenge(r);
        }

        let coeffs: Vec<B128> = wire.iter().copied().map(up).collect();
        let recovered = binius_ip::mlecheck::RoundProof(binius_ip::sumcheck::RoundCoeffs(coeffs))
            .recover(eval, *z_i);
        eval = recovered.evaluate(&up(r));

        bound.push(r);
        round_polys.push(wire);
        debug_assert!(i < num_vars);
    }

    (
        SumcheckProof { round_polys },
        SumcheckClaim {
            point: bound,
            expected: down(eval),
        },
    )
}

pub fn verify_mlecheck_interactive<P: RoundMessageProver + ?Sized>(
    prover: &mut P,
    point: &[F128],
    eval: F128,
    degree: usize,
    ch: &mut dyn RawChallenger,
) -> (SumcheckProof, SumcheckClaim, Option<SumcheckError>) {
    use crate::binius_channel::{down, up, LiveSumcheck, B128};

    let _role = metrics::enter(Role::Verifier);
    let rev: Vec<B128> = point.iter().rev().copied().map(up).collect();
    let mut chan = LiveSumcheck::new(prover, degree, ch);
    let out = binius_ip::mlecheck::verify::<B128, _>(&rev, degree, up(eval), &mut chan)
        .expect("LiveSumcheck never returns Err");
    let (round_polys, verdict) = chan.finish();
    (
        SumcheckProof { round_polys },
        SumcheckClaim {
            point: out.challenges.into_iter().map(down).collect(),
            expected: down(out.eval),
        },
        verdict,
    )
}

pub fn verify_mlecheck_recording(
    proof: &SumcheckProof,
    point: &[F128],
    eval: F128,
    degree: usize,
    ch: &mut dyn RawChallenger,
) -> (SumcheckClaim, Option<SumcheckError>) {
    use crate::binius_channel::{down, up, RecordedSumcheck, B128};

    let _role = metrics::enter(Role::Verifier);
    let n = point.len();
    let rev: Vec<B128> = point.iter().rev().copied().map(up).collect();
    let mut chan = RecordedSumcheck::new(&proof.round_polys, n, degree, ch);
    let out = binius_ip::mlecheck::verify::<B128, _>(&rev, degree, up(eval), &mut chan)
        .expect("RecordedSumcheck never returns Err");
    let verdict = chan.finish();
    (
        SumcheckClaim {
            point: out.challenges.into_iter().map(down).collect(),
            expected: down(out.eval),
        },
        verdict,
    )
}

pub fn verify_interactive<P: RoundMessageProver + ?Sized>(
    prover: &mut P,
    claim: F128,
    num_vars: usize,
    degree: usize,
    ch: &mut dyn RawChallenger,
) -> (SumcheckProof, SumcheckClaim, Option<SumcheckError>) {
    use crate::binius_channel::{down, up, LiveSumcheck, B128};

    let _role = metrics::enter(Role::Verifier);
    let mut chan = LiveSumcheck::new(prover, degree, ch);
    let out = binius_ip::sumcheck::verify::<B128, _>(num_vars, degree, up(claim), &mut chan)
        .expect("LiveSumcheck never returns Err");
    let (round_polys, verdict) = chan.finish();
    (
        SumcheckProof { round_polys },
        SumcheckClaim {
            point: out.challenges.into_iter().map(down).collect(),
            expected: down(out.eval),
        },
        verdict,
    )
}

pub fn round_evals_over<F>(tables: &[Vec<F128>], degree: usize, combine: F) -> Vec<F128>
where
    F: Fn(&[F128]) -> F128,
{
    let half = tables[0].len() / 2;
    let points = interpolation_points(degree);
    debug_assert!(
        points[0] == F128::ZERO && points[1] == F128::ONE,
        "the p=0 and p=1 shortcuts below read lo and hi directly, which is only \
         correct because `interpolation_points` starts 0, 1"
    );

    let mut acc = vec![F128::ZERO; degree + 1];
    let mut scratch = vec![F128::ZERO; tables.len()];
    let mut diffs = vec![F128::ZERO; tables.len()];
    for h in 0..half {
        for (t, table) in tables.iter().enumerate() {
            let lo = table[h];
            scratch[t] = lo;
            diffs[t] = lo.add(table[half + h]);
        }
        acc[0] = acc[0].add(combine(&scratch));
        for t in 0..tables.len() {
            scratch[t] = scratch[t].add(diffs[t]);
        }
        acc[1] = acc[1].add(combine(&scratch));
        for (pi, &p) in points.iter().enumerate().skip(2) {
            for (t, table) in tables.iter().enumerate() {
                scratch[t] = table[h].add(p.mul(diffs[t]));
            }
            acc[pi] = acc[pi].add(combine(&scratch));
        }
    }
    acc
}
