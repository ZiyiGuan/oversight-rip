pub mod encoder;
pub mod interactive;
pub mod reset;

use flock_core::challenger::Challenger;
use flock_field::{F128 as Theirs, F256};
use rmip_field::F128 as Ours;
use rmip_iop::transcript::RawChallenger;

pub fn up(x: Ours) -> Theirs {
    let v = x.to_u128();
    {
        Theirs::from_u64(v as u64)
    }
}

pub fn down(x: Theirs) -> Ours {
    Ours::new(x.to_u128())
}

pub fn bytes_of(x: Theirs) -> [u8; Theirs::BYTES] {
    x.to_le_bytes()
}

pub struct FlockChallenger<'a, C: RawChallenger + Send> {
    inner: &'a mut C,
}

impl<'a, C: RawChallenger + Send> FlockChallenger<'a, C> {
    pub fn new(inner: &'a mut C) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &C {
        self.inner
    }
}

impl<C: RawChallenger + Send> Challenger for FlockChallenger<'_, C> {
    fn observe_label(&mut self, label: &[u8]) {
        self.inner.record_ordinary(label);
    }

    fn observe_f128(&mut self, value: Theirs) {
        self.inner.record_ordinary(&bytes_of(value));
    }

    fn observe_f128_slice(&mut self, values: &[Theirs]) {
        let mut bytes = Vec::with_capacity(Theirs::BYTES * values.len());
        for v in values {
            bytes.extend_from_slice(&bytes_of(*v));
        }
        self.inner.record_ordinary(&bytes);
    }

    fn observe_f256(&mut self, value: F256) {
        self.observe_f128_slice(&value.coordinates());
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.inner.record_ordinary(bytes);
    }

    fn sample_f128(&mut self) -> Theirs {
        up(self.inner.sample())
    }

    fn fork_from_seed(&self, _seed: [Theirs; 2], label: &'static [u8]) -> Self {
        panic!(
            "the protocol's transcript does not fork (label {:?}): a forked child would expand one seed into many coins",
            String::from_utf8_lossy(label)
        )
    }
}
