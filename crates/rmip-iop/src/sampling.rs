use crate::transcript::RawChallenger;
use rmip_field::F128;

pub fn uniform_index(ch: &mut dyn RawChallenger, upper: usize) -> usize {
    uniform_index_observing(ch, upper, |_| {})
}

pub fn uniform_index_observing<F>(ch: &mut dyn RawChallenger, upper: usize, mut observe: F) -> usize
where
    F: FnMut(F128),
{
    assert!(upper > 0, "cannot sample from an empty range");
    let upper = upper as u128;
    let threshold = upper.wrapping_neg() % upper;
    loop {
        let coin = ch.sample();
        observe(coin);
        if coin.to_u128() >= threshold {
            return (coin.to_u128() % upper) as usize;
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::transcript::{OracleId, RawChallenger, Transcript};
    use rmip_field::F128;
    use std::collections::VecDeque;

    struct Scripted {
        coins: VecDeque<u128>,
        transcript: Transcript,
    }

    impl Scripted {
        fn new(coins: impl IntoIterator<Item = u128>) -> Self {
            Self {
                coins: coins.into_iter().collect(),
                transcript: Transcript::new(),
            }
        }
    }

    impl RawChallenger for Scripted {
        fn sample(&mut self) -> F128 {
            let coin = F128::new(self.coins.pop_front().expect("script exhausted"));
            self.transcript.record_coin(coin);
            coin
        }

        fn record_ordinary(&mut self, bytes: &[u8]) {
            self.transcript.record_ordinary(bytes);
        }

        fn post_oracle(&mut self, id: OracleId) {
            self.transcript.post_oracle(id);
        }

        fn transcript(&self) -> &Transcript {
            &self.transcript
        }
    }

    #[test]
    fn rejects_the_short_residue_classes_before_reducing() {
        let mut ch = Scripted::new([0, 3, 10]);
        assert_eq!(uniform_index(&mut ch, 6), 4);
        assert_eq!(ch.transcript.entries().len(), 3);
    }

    #[test]
    fn observer_receives_rejected_candidates_too() {
        let mut ch = Scripted::new([0, 3, 10]);
        let mut seen = Vec::new();
        assert_eq!(
            uniform_index_observing(&mut ch, 6, |coin| seen.push(coin)),
            4
        );
        assert_eq!(seen, vec![F128::new(0), F128::new(3), F128::new(10)]);
    }

    #[test]
    fn a_divisor_of_two_to_128_never_rejects() {
        let mut ch = Scripted::new([0]);
        assert_eq!(uniform_index(&mut ch, 64), 0);
        assert_eq!(ch.transcript.entries().len(), 1);
    }

    #[test]
    #[should_panic(expected = "empty range")]
    fn an_empty_range_is_rejected() {
        let mut ch = Scripted::new([]);
        let _ = uniform_index(&mut ch, 0);
    }
}
