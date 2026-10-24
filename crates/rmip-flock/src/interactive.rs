use std::sync::{Arc, Mutex};

use flock_core::challenger::{Challenger, Compilation};
use flock_field::{F128, F256};
use flock_hash::HashKind;
use rmip_iop::transcript::{RawChallenger, Transcript};

#[derive(Clone, Debug, Default)]
pub struct Stream {
    pub parent: Option<usize>,
    pub label: Vec<u8>,
    pub offset: usize,
    pub seed: [F128; 2],
    pub seeded: bool,
    pub coins: Vec<F128>,
}

pub type CoinLog = Arc<Mutex<Vec<Stream>>>;

pub struct Recording<C: Challenger> {
    inner: C,
    log: CoinLog,
    stream: usize,
}

impl<C: Challenger> Recording<C> {
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            log: Arc::new(Mutex::new(vec![Stream::default()])),
            stream: 0,
        }
    }

    pub fn log(&self) -> CoinLog {
        Arc::clone(&self.log)
    }

    pub fn into_inner(self) -> C {
        self.inner
    }

    fn push(&self, value: F128) {
        self.log.lock().expect("coin log")[self.stream]
            .coins
            .push(value);
    }

    fn push_all(&self, values: &[F128]) {
        self.log.lock().expect("coin log")[self.stream]
            .coins
            .extend_from_slice(values);
    }

    fn open_stream(&self, label: &'static [u8], seed: [F128; 2], seeded: bool) -> usize {
        let mut log = self.log.lock().expect("coin log");
        let offset = log[self.stream].coins.len();
        log.push(Stream {
            parent: Some(self.stream),
            label: label.to_vec(),
            offset,
            seed,
            seeded,
            coins: Vec::new(),
        });
        log.len() - 1
    }
}

impl<C: Challenger> Challenger for Recording<C> {
    fn supports_fused_pow_squeeze(&self) -> bool {
        self.inner.supports_fused_pow_squeeze()
    }
    fn hash_kind(&self) -> HashKind {
        self.inner.hash_kind()
    }
    fn observe_label(&mut self, label: &[u8]) {
        self.inner.observe_label(label);
    }
    fn observe_f128(&mut self, value: F128) {
        self.inner.observe_f128(value);
    }
    fn observe_f128_slice(&mut self, values: &[F128]) {
        self.inner.observe_f128_slice(values);
    }
    fn observe_f256(&mut self, value: F256) {
        self.inner.observe_f256(value);
    }
    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.inner.observe_bytes(bytes);
    }
    fn sample_f128(&mut self) -> F128 {
        let v = self.inner.sample_f128();
        self.push(v);
        v
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        let v = self.inner.sample_f128_vec(n);
        self.push_all(&v);
        v
    }
    fn sample_f256(&mut self) -> F256 {
        let v = self.inner.sample_f256();
        self.push_all(&v.coordinates());
        v
    }
    fn grind_pow(&mut self, bits: u32) -> u64 {
        self.inner.grind_pow(bits)
    }
    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.inner.verify_pow(nonce, bits)
    }
    fn grind_pow_and_sample_f128(&mut self, bits: u32) -> (u64, F128) {
        let (nonce, v) = self.inner.grind_pow_and_sample_f128(bits);
        self.push(v);
        (nonce, v)
    }
    fn verify_pow_and_sample_f128(&mut self, nonce: u64, bits: u32) -> Option<F128> {
        let r = self.inner.verify_pow_and_sample_f128(nonce, bits);
        if let Some(v) = r {
            self.push(v);
        }
        r
    }
    fn grind_pow_and_sample_f128_vec(&mut self, bits: u32, n: usize) -> (u64, Vec<F128>) {
        let (nonce, v) = self.inner.grind_pow_and_sample_f128_vec(bits, n);
        self.push_all(&v);
        (nonce, v)
    }
    fn verify_pow_and_sample_f128_vec(
        &mut self,
        nonce: u64,
        bits: u32,
        n: usize,
    ) -> Option<Vec<F128>> {
        let r = self.inner.verify_pow_and_sample_f128_vec(nonce, bits, n);
        if let Some(v) = &r {
            self.push_all(v);
        }
        r
    }
    fn fork(&mut self, label: &'static [u8]) -> Self {
        if self.inner.compilation() == Compilation::Interactive {
            let inner = self.inner.fork(label);
            let stream = self.open_stream(label, [F128::ZERO; 2], false);
            return Self {
                inner,
                log: Arc::clone(&self.log),
                stream,
            };
        }
        let seed = [self.sample_f128(), self.sample_f128()];
        self.fork_from_seed(seed, label)
    }
    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        let inner = self.inner.fork_from_seed(seed, label);
        let stream = self.open_stream(label, seed, true);
        Self {
            inner,
            log: Arc::clone(&self.log),
            stream,
        }
    }
    fn merge_child(&mut self, child: Self) {
        self.inner.merge_child(child.inner);
    }
    fn compilation(&self) -> Compilation {
        self.inner.compilation()
    }
    fn post_oracle(&mut self, table: &[F128], row_len: usize) -> usize {
        self.inner.post_oracle(table, row_len)
    }
    fn wants_oracle_coefficients(&self) -> bool {
        self.inner.wants_oracle_coefficients()
    }
    fn uses_lazy_oracles(&self) -> bool {
        self.inner.uses_lazy_oracles()
    }
    fn defers_oracle_reads(&self) -> bool {
        self.inner.defers_oracle_reads()
    }
    fn post_oracle_from_coefficients(
        &mut self,
        row_len: usize,
        rows: usize,
        coefficients: &[F128],
    ) -> usize {
        self.inner
            .post_oracle_from_coefficients(row_len, rows, coefficients)
    }
    fn post_oracle_with_coefficients(
        &mut self,
        table: &[F128],
        row_len: usize,
        coefficients: &[F128],
    ) -> usize {
        self.inner
            .post_oracle_with_coefficients(table, row_len, coefficients)
    }
    fn read_oracle(&mut self, oracle: usize, position: usize) -> Option<Vec<F128>> {
        self.inner.read_oracle(oracle, position)
    }
    fn read_oracle_rows(&mut self, oracle: usize, positions: &[usize]) -> Option<Vec<Vec<F128>>> {
        self.inner.read_oracle_rows(oracle, positions)
    }
    fn native_query_tuple(&mut self, oracle: usize, positions: &[usize]) {
        self.inner.native_query_tuple(oracle, positions);
    }
    fn expect_oracle(&mut self, oracle: usize, row_len: usize, rows: usize) -> bool {
        self.inner.expect_oracle(oracle, row_len, rows)
    }
}

#[derive(Clone, Debug)]
struct Oracle {
    row_len: usize,
    table: Vec<F128>,
}

impl Oracle {
    fn rows(&self) -> usize {
        self.table.len() / self.row_len
    }
}

#[derive(Clone, Debug, Default)]
pub struct OracleStore {
    inner: Arc<Mutex<Vec<Oracle>>>,
}

impl OracleStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn post(&self, table: &[F128], row_len: usize) -> usize {
        assert!(row_len >= 1 && table.len().is_multiple_of(row_len));
        let mut store = self.inner.lock().expect("oracle store");
        store.push(Oracle {
            row_len,
            table: table.to_vec(),
        });
        store.len() - 1
    }

    pub fn read(&self, oracle: usize, position: usize) -> Option<Vec<F128>> {
        let store = self.inner.lock().expect("oracle store");
        let o = store.get(oracle)?;
        if position >= o.rows() {
            return None;
        }
        Some(o.table[position * o.row_len..(position + 1) * o.row_len].to_vec())
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("oracle store").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn shape(&self, oracle: usize) -> Option<(usize, usize)> {
        let store = self.inner.lock().expect("oracle store");
        store.get(oracle).map(|o| (o.row_len, o.rows()))
    }
}

#[derive(Debug, Default)]
struct Consumption {
    position: Vec<usize>,
    claimed: Vec<bool>,
}

pub struct Replay {
    log: CoinLog,
    stream: usize,
    consumption: Arc<Mutex<Consumption>>,
    kind: HashKind,
    store: Option<OracleStore>,
}

impl Replay {
    pub fn new(log: CoinLog, kind: HashKind) -> Self {
        let streams = log.lock().expect("coin log").len();
        let mut claimed = vec![false; streams];
        claimed[0] = true;
        Self {
            log,
            stream: 0,
            consumption: Arc::new(Mutex::new(Consumption {
                position: vec![0; streams],
                claimed,
            })),
            kind,
            store: None,
        }
    }

    pub fn interactive(log: CoinLog, kind: HashKind, store: OracleStore) -> Self {
        Self {
            store: Some(store),
            ..Self::new(log, kind)
        }
    }

    pub fn consumed(&self) -> (usize, usize) {
        let log = self.log.lock().expect("coin log");
        let c = self.consumption.lock().expect("consumption");
        (
            c.position.iter().sum(),
            log.iter().map(|s| s.coins.len()).sum(),
        )
    }

    pub fn assert_exhausted(&self) {
        let streams: Vec<(Option<usize>, String, usize)> = {
            let log = self.log.lock().expect("coin log");
            log.iter()
                .map(|s| {
                    (
                        s.parent,
                        String::from_utf8_lossy(&s.label).into_owned(),
                        s.coins.len(),
                    )
                })
                .collect()
        };
        let (position, claimed) = {
            let c = self.consumption.lock().expect("consumption");
            (c.position.clone(), c.claimed.clone())
        };
        for (i, (parent, label, recorded)) in streams.iter().enumerate() {
            assert!(
                claimed[i],
                "replay: recorded stream {i} (label {label:?}, forked from {parent:?}) was never forked"
            );
            assert_eq!(
                position[i], *recorded,
                "replay: stream {i} (label {label:?}) consumed {} of {recorded} recorded coins",
                position[i]
            );
        }
    }

    fn next(&mut self) -> F128 {
        let taken = {
            let log = self.log.lock().expect("coin log");
            let mut c = self.consumption.lock().expect("consumption");
            let stream = &log[self.stream];
            let position = c.position[self.stream];
            match stream.coins.get(position) {
                Some(&v) => {
                    c.position[self.stream] = position + 1;
                    Ok(v)
                }
                None => Err((
                    String::from_utf8_lossy(&stream.label).into_owned(),
                    stream.coins.len(),
                    position,
                )),
            }
        };
        match taken {
            Ok(v) => v,
            Err((label, len, position)) => panic!(
                "replay exhausted: stream {} (label {label:?}) has {len} coins, position {position}",
                self.stream
            ),
        }
    }
}

impl Replay {
    fn claim_fork(&self, label: &'static [u8], seed: Option<[F128; 2]>) -> Option<usize> {
        let log = self.log.lock().expect("coin log");
        let mut c = self.consumption.lock().expect("consumption");
        let position = c.position[self.stream];
        let found = log.iter().enumerate().position(|(i, s)| {
            !c.claimed[i]
                && s.parent == Some(self.stream)
                && s.label == label
                && s.offset == position
                && match seed {
                    None => !s.seeded,
                    Some(seed) => s.seeded && s.seed == seed,
                }
        });
        if let Some(i) = found {
            c.claimed[i] = true;
        }
        found
    }
    fn child(&self, stream: usize) -> Self {
        Self {
            log: Arc::clone(&self.log),
            stream,
            consumption: Arc::clone(&self.consumption),
            kind: self.kind,
            store: self.store.clone(),
        }
    }
}

impl Challenger for Replay {
    fn supports_fused_pow_squeeze(&self) -> bool {
        true
    }
    fn hash_kind(&self) -> HashKind {
        self.kind
    }
    fn observe_f128(&mut self, _value: F128) {}
    fn sample_f128(&mut self) -> F128 {
        self.next()
    }
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        0
    }
    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        true
    }
    fn fork(&mut self, label: &'static [u8]) -> Self {
        if let Some(stream) = self.claim_fork(label, None) {
            return self.child(stream);
        }
        let seed = [self.sample_f128(), self.sample_f128()];
        self.fork_from_seed(seed, label)
    }
    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        let Some(stream) = self.claim_fork(label, Some(seed)) else {
            let position = self.consumption.lock().expect("consumption").position[self.stream];
            panic!(
                "replay: no unclaimed recorded fork from stream {} with label {:?} at coin {} with this seed",
                self.stream,
                String::from_utf8_lossy(label),
                position
            )
        };
        self.child(stream)
    }
    fn merge_child(&mut self, _child: Self) {}
    fn compilation(&self) -> Compilation {
        if self.store.is_some() {
            Compilation::Interactive
        } else {
            Compilation::Snarg
        }
    }
    fn post_oracle(&mut self, table: &[F128], row_len: usize) -> usize {
        self.store
            .as_ref()
            .expect("an interactive replay posts oracles")
            .post(table, row_len)
    }
    fn read_oracle(&mut self, oracle: usize, position: usize) -> Option<Vec<F128>> {
        self.store
            .as_ref()
            .expect("an interactive replay reads oracles")
            .read(oracle, position)
    }
    fn expect_oracle(&mut self, oracle: usize, row_len: usize, rows: usize) -> bool {
        match &self.store {
            Some(store) => store.shape(oracle) == Some((row_len, rows)),
            None => true,
        }
    }
}

fn oracle_statement(oracle: usize, row_len: usize, rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(40);
    out.extend_from_slice(b"rmip-oracle-v0");
    for w in [oracle, row_len, rows] {
        out.extend_from_slice(&(w as u64).to_le_bytes());
    }
    out
}

pub struct Interactive<C: RawChallenger + Send> {
    source: Arc<Mutex<C>>,
    store: OracleStore,
}

impl<C: RawChallenger + Send> Interactive<C> {
    pub fn new(source: C, store: OracleStore) -> Self {
        Self {
            source: Arc::new(Mutex::new(source)),
            store,
        }
    }

    pub fn with_transcript<R>(&self, f: impl FnOnce(&Transcript) -> R) -> R {
        f(self.source.lock().expect("coin source").transcript())
    }

    pub fn store(&self) -> &OracleStore {
        &self.store
    }

    pub fn with_source<R>(&self, f: impl FnOnce(&C) -> R) -> R {
        f(&self.source.lock().expect("coin source"))
    }

    fn record(&self, bytes: &[u8]) {
        self.source
            .lock()
            .expect("coin source")
            .record_ordinary(bytes);
    }
}

impl<C: RawChallenger + Send> Challenger for Interactive<C> {
    fn compilation(&self) -> Compilation {
        Compilation::Interactive
    }
    fn verify_pow(&mut self, nonce: u64, _bits: u32) -> bool {
        nonce == 0
    }
    fn observe_label(&mut self, label: &[u8]) {
        self.record(label);
    }
    fn observe_f128(&mut self, value: F128) {
        self.record(&crate::bytes_of(value));
    }
    fn observe_f128_slice(&mut self, values: &[F128]) {
        let mut bytes = Vec::with_capacity(F128::BYTES * values.len());
        for v in values {
            bytes.extend_from_slice(&crate::bytes_of(*v));
        }
        self.record(&bytes);
    }
    fn observe_f256(&mut self, value: F256) {
        self.observe_f128_slice(&value.coordinates());
    }
    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.record(bytes);
    }
    fn sample_f128(&mut self) -> F128 {
        crate::up(self.source.lock().expect("coin source").sample())
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.source
            .lock()
            .expect("coin source")
            .sample_many(n)
            .into_iter()
            .map(crate::up)
            .collect()
    }
    fn fork(&mut self, label: &'static [u8]) -> Self {
        self.fork_from_seed([F128::ZERO; 2], label)
    }
    fn fork_from_seed(&self, _seed: [F128; 2], label: &'static [u8]) -> Self {
        let mut event = b"rmip-fork-v0:".to_vec();
        event.extend_from_slice(label);
        self.record(&event);
        Self {
            source: Arc::clone(&self.source),
            store: self.store.clone(),
        }
    }
    fn merge_child(&mut self, _child: Self) {
        self.record(b"rmip-merge-v0");
    }
    fn post_oracle(&mut self, table: &[F128], row_len: usize) -> usize {
        let id = self.store.post(table, row_len);
        let rows = table.len() / row_len;
        self.record(&oracle_statement(id, row_len, rows));
        self.source.lock().expect("coin source").post_oracle(id);
        id
    }
    fn read_oracle(&mut self, oracle: usize, position: usize) -> Option<Vec<F128>> {
        self.store.read(oracle, position)
    }
    fn expect_oracle(&mut self, oracle: usize, row_len: usize, rows: usize) -> bool {
        self.record(&oracle_statement(oracle, row_len, rows));
        self.source.lock().expect("coin source").post_oracle(oracle);
        self.store.shape(oracle) == Some((row_len, rows))
    }
}
