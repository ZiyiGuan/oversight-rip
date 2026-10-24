use crate::backend::{OpeningVerifier, QueryPlan};
use crate::transcript::{Entry, OracleId, RawChallenger, Transcript};
use rmip_field::{BinaryField, F128};

pub trait CoinDeriver {
    fn absorb_ordinary(&mut self, bytes: &[u8]);
    fn absorb_post(&mut self, id: OracleId, root: Option<&[u8; 32]>);
    fn derive(&mut self) -> F128;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    ForgedCoin { index: usize },
    MessageMismatch { index: usize },
    ExpectedCoin { index: usize },
    ExpectedMessage { index: usize },
    ExpectedOraclePost { index: usize },
    TrailingEntries { remaining: usize },
    BackendSegment { index: usize },
}

pub trait Replay: RawChallenger {
    fn peek_ordinary(&self) -> Option<&[u8]>;
    fn peek_ordinaries(&self, n: usize) -> Vec<Vec<u8>>;
    fn take_ordinary(&mut self) -> Option<Vec<u8>>;
    fn take_post(&mut self) -> Option<OracleId>;
    fn fault_backend(&mut self);
    fn fault(&self) -> Option<&ReplayError>;
    fn remaining(&self) -> usize;
    fn posted(&self) -> &[OracleId];
    fn cursor(&self) -> usize;
}

pub struct Tape<'a, D: CoinDeriver> {
    entries: &'a [Entry],
    cursor: usize,
    deriver: D,
    roots: Vec<(OracleId, [u8; 32])>,
    posted: Vec<OracleId>,
    first: Option<ReplayError>,
    mirror: Transcript,
}

impl<'a, D: CoinDeriver> Tape<'a, D> {
    pub fn new(entries: &'a [Entry], roots: Vec<(OracleId, [u8; 32])>, deriver: D) -> Self {
        Self {
            entries,
            cursor: 0,
            deriver,
            roots,
            posted: Vec::new(),
            first: None,
            mirror: Transcript::new(),
        }
    }

    fn set(&mut self, e: ReplayError) {
        if self.first.is_none() {
            self.first = Some(e);
        }
    }

    fn root_of(&self, id: OracleId) -> Option<&[u8; 32]> {
        self.roots.iter().find(|(i, _)| *i == id).map(|(_, r)| r)
    }

    pub fn finish(&mut self) {
        let remaining = self.entries.len() - self.cursor;
        if remaining > 0 {
            self.set(ReplayError::TrailingEntries { remaining });
        }
    }

    pub fn into_fault(mut self) -> Option<ReplayError> {
        self.finish();
        self.first
    }
}

impl<'a, D: CoinDeriver> RawChallenger for Tape<'a, D> {
    fn sample(&mut self) -> F128 {
        let index = self.cursor;
        match self.entries.get(index) {
            Some(Entry::Coin(recorded)) => {
                self.cursor += 1;
                let derived = self.deriver.derive();
                if derived != *recorded {
                    self.set(ReplayError::ForgedCoin { index });
                }
                self.mirror.record_coin(*recorded);
                *recorded
            }
            _ => {
                self.set(ReplayError::ExpectedCoin { index });
                self.deriver.derive()
            }
        }
    }

    fn record_ordinary(&mut self, bytes: &[u8]) {
        let index = self.cursor;
        match self.entries.get(index) {
            Some(Entry::Ordinary(recorded)) => {
                self.cursor += 1;
                if recorded.as_slice() != bytes {
                    self.set(ReplayError::MessageMismatch { index });
                }
                let recorded = recorded.clone();
                self.deriver.absorb_ordinary(&recorded);
                self.mirror.record_ordinary(recorded);
            }
            _ => {
                self.set(ReplayError::ExpectedMessage { index });
                self.deriver.absorb_ordinary(bytes);
            }
        }
    }

    fn transcript(&self) -> &Transcript {
        &self.mirror
    }

    fn post_oracle(&mut self, id: OracleId) {
        let index = self.cursor;
        match self.entries.get(index) {
            Some(Entry::OraclePost(recorded)) if *recorded == id => {
                self.cursor += 1;
                self.posted.push(id);
                let root = self.root_of(id).copied();
                self.deriver.absorb_post(id, root.as_ref());
                self.mirror.post_oracle(id);
            }
            _ => self.set(ReplayError::ExpectedOraclePost { index }),
        }
    }
}

impl<'a, D: CoinDeriver> Replay for Tape<'a, D> {
    fn peek_ordinary(&self) -> Option<&[u8]> {
        match self.entries.get(self.cursor) {
            Some(Entry::Ordinary(b)) => Some(b),
            _ => None,
        }
    }

    fn peek_ordinaries(&self, n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map_while(|k| match self.entries.get(self.cursor + k) {
                Some(Entry::Ordinary(b)) => Some(b.clone()),
                _ => None,
            })
            .collect()
    }

    fn take_ordinary(&mut self) -> Option<Vec<u8>> {
        match self.entries.get(self.cursor) {
            Some(Entry::Ordinary(b)) => {
                let b = b.clone();
                self.cursor += 1;
                self.deriver.absorb_ordinary(&b);
                self.mirror.record_ordinary(b.clone());
                Some(b)
            }
            _ => {
                self.set(ReplayError::ExpectedMessage { index: self.cursor });
                None
            }
        }
    }

    fn take_post(&mut self) -> Option<OracleId> {
        match self.entries.get(self.cursor) {
            Some(Entry::OraclePost(id)) => {
                let id = *id;
                self.cursor += 1;
                self.posted.push(id);
                let root = self.root_of(id).copied();
                self.deriver.absorb_post(id, root.as_ref());
                self.mirror.post_oracle(id);
                Some(id)
            }
            _ => {
                self.set(ReplayError::ExpectedOraclePost { index: self.cursor });
                None
            }
        }
    }

    fn fault_backend(&mut self) {
        self.set(ReplayError::BackendSegment { index: self.cursor });
    }

    fn fault(&self) -> Option<&ReplayError> {
        self.first.as_ref()
    }

    fn remaining(&self) -> usize {
        self.entries.len() - self.cursor
    }

    fn posted(&self) -> &[OracleId] {
        &self.posted
    }

    fn cursor(&self) -> usize {
        self.cursor
    }
}

pub fn decode_elements(bytes: &[u8]) -> Vec<F128> {
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| F128::from_be_bytes(*c))
        .collect()
}

pub fn coins_since(transcript: &Transcript, from: usize) -> Vec<F128> {
    let entries = transcript.entries();
    entries[from.min(entries.len())..]
        .iter()
        .filter_map(|e| match e {
            Entry::Coin(c) => Some(*c),
            _ => None,
        })
        .collect()
}

pub struct CoinReplay<'a> {
    coins: &'a [F128],
    cursor: usize,
    overrun: usize,
    mirror: Transcript,
}

impl<'a> CoinReplay<'a> {
    pub fn new(coins: &'a [F128]) -> Self {
        Self {
            coins,
            cursor: 0,
            overrun: 0,
            mirror: Transcript::new(),
        }
    }

    pub fn overrun(&self) -> usize {
        self.overrun
    }

    pub fn unread(&self) -> usize {
        self.coins.len() - self.cursor
    }

    pub fn replayed_exactly(&self) -> bool {
        self.overrun == 0 && self.cursor == self.coins.len()
    }
}

impl RawChallenger for CoinReplay<'_> {
    fn sample(&mut self) -> F128 {
        match self.coins.get(self.cursor) {
            Some(c) => {
                self.cursor += 1;
                self.mirror.record_coin(*c);
                *c
            }
            None => {
                self.overrun += 1;
                self.mirror.record_coin(F128::ZERO);
                F128::ZERO
            }
        }
    }

    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.mirror.record_ordinary(bytes);
    }

    fn post_oracle(&mut self, id: OracleId) {
        self.mirror.post_oracle(id);
    }

    fn transcript(&self) -> &Transcript {
        &self.mirror
    }
}

pub struct TapeProver<'t> {
    entries: &'t [Entry],
    mirror: std::cell::Cell<usize>,
}

impl<'t> TapeProver<'t> {
    pub fn new(entries: &'t [Entry]) -> Self {
        Self::starting_at(entries, 0)
    }

    pub fn starting_at(entries: &'t [Entry], at: usize) -> Self {
        Self {
            entries,
            mirror: std::cell::Cell::new(at),
        }
    }

    fn next_message(&self) -> Vec<F128> {
        let mut i = self.mirror.get();
        while let Some(e) = self.entries.get(i) {
            match e {
                Entry::Ordinary(b) => {
                    self.mirror.set(i + 1);
                    return decode_elements(b);
                }
                _ => i += 1,
            }
        }
        Vec::new()
    }

    fn next_messages(&self, n: usize) -> Vec<Vec<F128>> {
        (0..n).map(|_| self.next_message()).collect()
    }
}

impl<'t> crate::relation_protocol::RelationMessageProver for TapeProver<'t> {
    fn observe_public_input_point(&mut self, _r_pub: &[F128]) {}
    fn begin_zerocheck(&mut self, _tau: &[F128]) {}
    fn zerocheck_round_message(&mut self, _round: usize) -> Vec<F128> {
        self.next_message()
    }
    fn receive_zerocheck_challenge(&mut self, _round: usize, _challenge: F128) {}
    fn zerocheck_endpoint(&mut self) -> Vec<F128> {
        self.next_message()
    }
    fn begin_lincheck(&mut self, _row_point: &[F128], _c: crate::lincheck::Coefficients) {}
    fn lincheck_round_message(&mut self, _round: usize) -> Vec<F128> {
        self.next_message()
    }
    fn receive_lincheck_challenge(&mut self, _round: usize, _challenge: F128) {}
    fn lincheck_endpoint(&mut self) -> Vec<F128> {
        self.next_message()
    }
    fn holo_frame(&mut self) -> Vec<F128> {
        self.next_message()
    }
}

impl<'t> crate::local_protocol::LocalMessageProver for TapeProver<'t> {
    fn answer_blocks(&mut self, _coins: &[F128], sampled: &[usize]) -> Vec<Vec<F128>> {
        self.next_messages(sampled.len())
    }
}

impl<'t> crate::batch::BatchMessageProver for TapeProver<'t> {
    fn begin_batch(&mut self, _claims: &[crate::backend::Claim<F128>], _gammas: &[F128]) {}
}

impl<'t> crate::sumcheck::RoundMessageProver for TapeProver<'t> {
    fn round_message(&mut self, _round: usize) -> Vec<F128> {
        self.next_message()
    }
    fn receive_challenge(&mut self, _round: usize, _challenge: F128) {}
}

pub trait TranscriptVerifier {
    type Opening: OpeningVerifier<F128>;

    fn replay_commit(&mut self, tape: &mut dyn Replay);
    fn replay_open(&mut self, point: &[F128], tape: &mut dyn Replay);
    fn query_plan(&self, tape: &mut dyn Replay) -> Box<dyn QueryPlan<F128>>;
    fn opening_verifier(&self) -> Self::Opening;
    fn q_long(&self) -> usize;
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::transcript::Entry;

    struct ConstantDeriver;

    impl CoinDeriver for ConstantDeriver {
        fn absorb_ordinary(&mut self, _bytes: &[u8]) {}
        fn absorb_post(&mut self, _id: OracleId, _root: Option<&[u8; 32]>) {}
        fn derive(&mut self) -> F128 {
            F128::ZERO
        }
    }

    #[test]
    fn trailing_entries_are_caught_even_without_an_explicit_finish() {
        let entries = vec![Entry::Ordinary(vec![1, 2, 3]), Entry::Ordinary(vec![4, 5])];

        let mut tape = Tape::new(&entries, Vec::new(), ConstantDeriver);
        tape.record_ordinary(&[1, 2, 3]);
        assert_eq!(
            tape.into_fault(),
            Some(ReplayError::TrailingEntries { remaining: 1 })
        );

        let mut complete = Tape::new(&entries, Vec::new(), ConstantDeriver);
        complete.record_ordinary(&[1, 2, 3]);
        complete.record_ordinary(&[4, 5]);
        assert_eq!(complete.into_fault(), None);
    }

    #[test]
    fn the_first_fault_still_wins_over_trailing_entries() {
        let entries = vec![Entry::Ordinary(vec![1, 2, 3]), Entry::Ordinary(vec![4, 5])];

        let mut tape = Tape::new(&entries, Vec::new(), ConstantDeriver);
        tape.record_ordinary(&[9, 9, 9]);
        assert_eq!(
            tape.into_fault(),
            Some(ReplayError::MessageMismatch { index: 0 })
        );

        let mut explicit = Tape::new(&entries, Vec::new(), ConstantDeriver);
        explicit.record_ordinary(&[1, 2, 3]);
        explicit.finish();
        assert_eq!(
            explicit.into_fault(),
            Some(ReplayError::TrailingEntries { remaining: 1 })
        );
    }
}
