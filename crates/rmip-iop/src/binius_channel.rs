use binius_field::Field;
use binius_ip::channel::{Error as ChannelError, IPVerifierChannel};
use rmip_field::{BinaryField, F128};

use crate::metrics::{self, Role};
use crate::sumcheck::SumcheckError;
use crate::transcript::RawChallenger;

pub type B128 = binius_field::Ghash128b;

#[inline]
pub fn up(x: F128) -> B128 {
    B128::from(x.to_u128())
}

#[inline]
pub fn down(x: B128) -> F128 {
    F128::new(u128::from(x))
}

pub struct RecordedSumcheck<'a> {
    rounds: &'a [Vec<F128>],
    degree: usize,
    round: usize,
    within: usize,
    wire: Vec<F128>,
    ch: &'a mut dyn RawChallenger,
    fault: Option<SumcheckError>,
}

impl<'a> RecordedSumcheck<'a> {
    pub fn new(
        rounds: &'a [Vec<F128>],
        num_vars: usize,
        degree: usize,
        ch: &'a mut dyn RawChallenger,
    ) -> Self {
        assert!(
            degree >= 1,
            "a sumcheck round polynomial has degree at least 1"
        );
        let fault = (rounds.len() != num_vars).then_some(SumcheckError::WrongRoundCount {
            expected: num_vars,
            got: rounds.len(),
        });
        Self {
            rounds,
            degree,
            round: 0,
            within: 0,
            wire: Vec::new(),
            ch,
            fault,
        }
    }

    fn begin_round(&mut self) {
        let supplied = self
            .rounds
            .get(self.round)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        record_round_message(self.ch, supplied);
        if supplied.len() != self.degree && self.fault.is_none() {
            self.fault = Some(SumcheckError::WrongDegree {
                round: self.round,
                expected: self.degree,
                got: supplied.len(),
            });
        }
        self.wire = vec![F128::ZERO; self.degree];
        let copied = supplied.len().min(self.degree);
        self.wire[..copied].copy_from_slice(&supplied[..copied]);
        self.within = 0;
        self.round += 1;
    }

    pub fn finish(self) -> Option<SumcheckError> {
        for extra in self.rounds.iter().skip(self.round) {
            record_round_message(self.ch, extra);
        }
        self.fault
    }
}

fn record_round_message(ch: &mut dyn RawChallenger, evaluations: &[F128]) {
    let mut encoded = Vec::with_capacity(evaluations.len() * 16);
    for value in evaluations {
        encoded.extend_from_slice(&value.to_be_bytes());
    }
    ch.record_ordinary(&encoded);
}

impl IPVerifierChannel<B128> for RecordedSumcheck<'_> {
    type Elem = B128;

    fn recv_one(&mut self) -> Result<B128, ChannelError> {
        if self.within >= self.wire.len() {
            self.begin_round();
        }
        let v = self.wire.get(self.within).copied().unwrap_or(F128::ZERO);
        self.within += 1;
        Ok(up(v))
    }

    fn sample(&mut self) -> B128 {
        up(self.ch.sample())
    }

    fn observe_one(&mut self, val: B128) -> B128 {
        val
    }

    fn assert_zero(&mut self, val: B128) -> Result<(), ChannelError> {
        if val == B128::ZERO {
            Ok(())
        } else {
            Err(ChannelError::InvalidAssert)
        }
    }
}

pub struct LiveSumcheck<'a, P: ?Sized> {
    prover: &'a mut P,
    degree: usize,
    round: usize,
    within: usize,
    wire: Vec<F128>,
    supplied: Vec<Vec<F128>>,
    ch: &'a mut dyn RawChallenger,
    fault: Option<SumcheckError>,
}

impl<'a, P: crate::sumcheck::RoundMessageProver + ?Sized> LiveSumcheck<'a, P> {
    pub fn new(prover: &'a mut P, degree: usize, ch: &'a mut dyn RawChallenger) -> Self {
        assert!(
            degree >= 1,
            "a sumcheck round polynomial has degree at least 1"
        );
        Self {
            prover,
            degree,
            round: 0,
            within: 0,
            wire: Vec::new(),
            supplied: Vec::new(),
            ch,
            fault: None,
        }
    }

    fn begin_round(&mut self) {
        let msg = {
            let _prover = metrics::enter(Role::MainProver);
            self.prover.round_message(self.round)
        };
        record_round_message(self.ch, &msg);
        if msg.len() != self.degree && self.fault.is_none() {
            self.fault = Some(SumcheckError::WrongDegree {
                round: self.round,
                expected: self.degree,
                got: msg.len(),
            });
        }
        self.wire = vec![F128::ZERO; self.degree];
        let copied = msg.len().min(self.degree);
        self.wire[..copied].copy_from_slice(&msg[..copied]);
        self.supplied.push(msg);
        self.within = 0;
        self.round += 1;
    }

    pub fn finish(self) -> (Vec<Vec<F128>>, Option<SumcheckError>) {
        (self.supplied, self.fault)
    }
}

impl<P: crate::sumcheck::RoundMessageProver + ?Sized> IPVerifierChannel<B128>
    for LiveSumcheck<'_, P>
{
    type Elem = B128;

    fn recv_one(&mut self) -> Result<B128, ChannelError> {
        if self.within >= self.wire.len() {
            self.begin_round();
        }
        let v = self.wire.get(self.within).copied().unwrap_or(F128::ZERO);
        self.within += 1;
        Ok(up(v))
    }

    fn sample(&mut self) -> B128 {
        if self.round == 0 {
            self.begin_round();
        }
        let challenge = self.ch.sample();
        {
            let _prover = metrics::enter(Role::MainProver);
            self.prover.receive_challenge(self.round - 1, challenge);
        }
        up(challenge)
    }

    fn observe_one(&mut self, val: B128) -> B128 {
        val
    }

    fn assert_zero(&mut self, val: B128) -> Result<(), ChannelError> {
        if val == B128::ZERO {
            Ok(())
        } else {
            Err(ChannelError::InvalidAssert)
        }
    }
}

pub struct ProverTranscript<'a> {
    ch: &'a mut dyn RawChallenger,
    pub sent: Vec<Vec<F128>>,
}

impl<'a> ProverTranscript<'a> {
    pub fn new(ch: &'a mut dyn RawChallenger) -> Self {
        Self {
            ch,
            sent: Vec::new(),
        }
    }
}

impl binius_ip_prover::channel::IPProverChannel<B128> for ProverTranscript<'_> {
    fn send_one(&mut self, elem: B128) {
        self.send_many(&[elem]);
    }
    fn send_many(&mut self, elems: &[B128]) {
        let _role = metrics::enter(Role::MainProver);
        let vals: Vec<F128> = elems.iter().copied().map(down).collect();
        record_round_message(self.ch, &vals);
        self.sent.push(vals);
    }
    fn observe_one(&mut self, _val: B128) {}
    fn sample(&mut self) -> B128 {
        let _role = metrics::enter(Role::Verifier);
        up(self.ch.sample())
    }
}

pub struct SummandProver<'a, S: ?Sized> {
    pub inner: &'a mut S,
    running: &'a mut F128,
    verdict: &'a mut Option<SumcheckError>,
    last_coeffs: Vec<F128>,
    round: usize,
}

impl<'a, S: crate::sumcheck::Summand + ?Sized> SummandProver<'a, S> {
    pub fn new(
        inner: &'a mut S,
        running: &'a mut F128,
        verdict: &'a mut Option<SumcheckError>,
    ) -> Self {
        Self {
            inner,
            running,
            verdict,
            last_coeffs: Vec::new(),
            round: 0,
        }
    }
}

impl<S: crate::sumcheck::Summand + ?Sized> binius_ip_prover::sumcheck::common::SumcheckProver<B128>
    for SummandProver<'_, S>
{
    fn n_vars(&self) -> usize {
        self.inner.num_vars()
    }
    fn execute(&mut self) -> Vec<binius_ip::sumcheck::RoundCoeffs<B128>> {
        let _verifier = metrics::enter(Role::Verifier);
        let evals = {
            let _prover = metrics::enter(Role::MainProver);
            self.inner.round_evals()
        };
        if evals[0].add(evals[1]) != *self.running && self.verdict.is_none() {
            *self.verdict = Some(SumcheckError::RoundSumMismatch(self.round));
        }
        let _prover = metrics::enter(Role::MainProver);
        let coeffs = crate::mle::monomial_from_evals(&evals);
        self.last_coeffs = coeffs.clone();
        vec![binius_ip::sumcheck::RoundCoeffs(
            coeffs.into_iter().map(up).collect(),
        )]
    }
    fn fold(&mut self, challenge: B128) {
        let _verifier = metrics::enter(Role::Verifier);
        let r = down(challenge);
        {
            let _prover = metrics::enter(Role::MainProver);
            self.inner.fold(r);
        }
        *self.running = crate::mle::horner(&self.last_coeffs, r);
        self.round += 1;
    }
    fn finish(self) -> Vec<B128> {
        let _role = metrics::enter(Role::MainProver);
        vec![up(self.inner.final_value())]
    }
}

#[cfg(test)]
mod live_sumcheck_tests {

    use super::*;
    use crate::sumcheck::RoundMessageProver;
    use crate::transcript::SeededChallenger;

    struct RecordingProver {
        challenged: Vec<usize>,
    }

    impl RoundMessageProver for RecordingProver {
        fn round_message(&mut self, round: usize) -> Vec<F128> {
            vec![F128::new(round as u128 + 1), F128::ONE]
        }

        fn receive_challenge(&mut self, round: usize, _challenge: F128) {
            self.challenged.push(round);
        }
    }

    #[test]
    fn a_challenge_before_the_first_round_message_does_not_underflow() {
        let mut prover = RecordingProver {
            challenged: Vec::new(),
        };
        let mut ch = SeededChallenger::new(5);
        {
            let mut live = LiveSumcheck::new(&mut prover, 2, &mut ch);
            let _ = IPVerifierChannel::sample(&mut live);
            let (supplied, fault) = live.finish();
            assert_eq!(supplied.len(), 1, "round 0's message must have been pulled");
            assert_eq!(fault, None);
        }
        assert_eq!(prover.challenged, vec![0]);
        let entries = ch.transcript().entries();
        assert!(
            matches!(entries.first(), Some(crate::transcript::Entry::Ordinary(_))),
            "the round message must be recorded before its challenge"
        );
        assert!(matches!(
            entries.get(1),
            Some(crate::transcript::Entry::Coin(_))
        ));
    }

    #[test]
    fn the_guard_does_not_disturb_the_read_then_sample_order() {
        let mut prover = RecordingProver {
            challenged: Vec::new(),
        };
        let mut ch = SeededChallenger::new(5);
        {
            let mut live = LiveSumcheck::new(&mut prover, 2, &mut ch);
            for _ in 0..2 {
                let _ = live.recv_one().expect("the wire never fails");
                let _ = live.recv_one().expect("the wire never fails");
                let _ = IPVerifierChannel::sample(&mut live);
            }
            let (supplied, fault) = live.finish();
            assert_eq!(supplied.len(), 2);
            assert_eq!(fault, None);
        }
        assert_eq!(prover.challenged, vec![0, 1]);
    }
}
