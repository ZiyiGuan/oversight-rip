use rmip_field::{BinaryField, F128};

pub type OracleId = usize;

pub const RESERVED_ORACLE_ID: OracleId = usize::MAX;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    Ordinary(Vec<u8>),
    Coin(F128),
    OraclePost(OracleId),
}

impl Entry {
    const TAG_ORDINARY: u8 = 1;
    const TAG_COIN: u8 = 2;
    const TAG_ORACLE: u8 = 3;

    fn decode_from(rest: &mut &[u8]) -> Result<Self, PrefixDecodeError> {
        fn take<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [u8], PrefixDecodeError> {
            if rest.len() < n {
                return Err(PrefixDecodeError::Truncated);
            }
            let (head, tail) = rest.split_at(n);
            *rest = tail;
            Ok(head)
        }
        let tag = take(rest, 1)?[0];
        match tag {
            Self::TAG_ORDINARY => {
                let n = u64::from_be_bytes(take(rest, 8)?.try_into().unwrap());
                let n = usize::try_from(n).map_err(|_| PrefixDecodeError::Truncated)?;
                Ok(Entry::Ordinary(take(rest, n)?.to_vec()))
            }
            Self::TAG_COIN => {
                let b: [u8; 16] = take(rest, 16)?.try_into().unwrap();
                Ok(Entry::Coin(F128::from_be_bytes(b)))
            }
            Self::TAG_ORACLE => {
                let id = u64::from_be_bytes(take(rest, 8)?.try_into().unwrap());
                Ok(Entry::OraclePost(id as OracleId))
            }
            other => Err(PrefixDecodeError::BadTag(other)),
        }
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Entry::Ordinary(bytes) => {
                out.push(Self::TAG_ORDINARY);
                out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                out.extend_from_slice(bytes);
            }
            Entry::Coin(c) => {
                out.push(Self::TAG_COIN);
                out.extend_from_slice(&c.to_be_bytes());
            }
            Entry::OraclePost(id) => {
                out.push(Self::TAG_ORACLE);
                out.extend_from_slice(&(*id as u64).to_be_bytes());
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixDecodeError {
    Truncated,
    BadTag(u8),
    EntryCountMismatch { declared: usize, decoded: usize },
}

impl std::fmt::Display for PrefixDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "transcript prefix ended inside an entry"),
            Self::BadTag(t) => write!(f, "transcript prefix has unknown entry tag {t}"),
            Self::EntryCountMismatch { declared, decoded } => {
                write!(
                    f,
                    "prefix declares {declared} entries but decodes to {decoded}"
                )
            }
        }
    }
}

impl std::error::Error for PrefixDecodeError {}

#[derive(Clone)]
pub struct TranscriptPrefix {
    bytes: std::sync::Arc<[u8]>,
    len: usize,
    entries: usize,
}

impl PartialEq for TranscriptPrefix {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries && self.as_bytes() == other.as_bytes()
    }
}

impl Eq for TranscriptPrefix {}

impl std::hash::Hash for TranscriptPrefix {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
        self.entries.hash(state);
    }
}

pub fn decode_entries(bytes: &[u8]) -> Result<Vec<Entry>, PrefixDecodeError> {
    let mut rest = bytes;
    let mut out = Vec::new();
    while !rest.is_empty() {
        out.push(Entry::decode_from(&mut rest)?);
    }
    Ok(out)
}

impl TranscriptPrefix {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    pub fn shares_snapshot(&self, other: &TranscriptPrefix) -> bool {
        std::sync::Arc::ptr_eq(&self.bytes, &other.bytes)
    }

    pub fn entry_count(&self) -> usize {
        self.entries
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn decode(&self) -> Result<Vec<Entry>, PrefixDecodeError> {
        let mut out = decode_entries(self.as_bytes())?;
        out.shrink_to_fit();
        if out.len() != self.entries {
            return Err(PrefixDecodeError::EntryCountMismatch {
                declared: self.entries,
                decoded: out.len(),
            });
        }
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PrefixDecodeError> {
        let mut rest = bytes;
        let mut entries = 0usize;
        while !rest.is_empty() {
            Entry::decode_from(&mut rest)?;
            entries += 1;
        }
        Ok(Self {
            bytes: std::sync::Arc::from(bytes),
            len: bytes.len(),
            entries,
        })
    }

    pub fn coins(&self) -> Result<Vec<F128>, PrefixDecodeError> {
        Ok(self
            .decode()?
            .into_iter()
            .filter_map(|e| match e {
                Entry::Coin(c) => Some(c),
                _ => None,
            })
            .collect())
    }
}

impl std::fmt::Debug for TranscriptPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TranscriptPrefix({} entries, {} bytes)",
            self.entries, self.len
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct Transcript {
    entries: Vec<Entry>,
    encoded: Vec<u8>,
    ends: Vec<usize>,
    posts: Vec<(OracleId, usize)>,
    snapshot: Snapshot,
}

#[derive(Default)]
struct Snapshot(std::sync::Mutex<Option<std::sync::Arc<[u8]>>>);

impl Clone for Snapshot {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Snapshot")
    }
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, e: Entry) {
        if self.ends.is_empty() {
            self.ends.push(0);
        }
        e.encode_into(&mut self.encoded);
        self.ends.push(self.encoded.len());
        self.entries.push(e);
        self.drop_snapshot();
    }

    fn drop_snapshot(&mut self) {
        *self
            .snapshot
            .0
            .get_mut()
            .expect("the snapshot lock is not poisoned") = None;
    }

    pub fn record_ordinary(&mut self, bytes: impl Into<Vec<u8>>) {
        self.push(Entry::Ordinary(bytes.into()));
    }

    pub fn record_coin(&mut self, c: F128) {
        self.push(Entry::Coin(c));
    }

    pub fn record_coins(&mut self, coins: &[F128]) {
        if self.ends.is_empty() {
            self.ends.push(0);
        }
        self.encoded.reserve(coins.len() * 17);
        self.ends.reserve(coins.len());
        self.entries.reserve(coins.len());
        for &c in coins {
            let e = Entry::Coin(c);
            e.encode_into(&mut self.encoded);
            self.ends.push(self.encoded.len());
            self.entries.push(e);
        }
        self.drop_snapshot();
    }

    pub fn post_oracle(&mut self, id: OracleId) {
        assert!(
            !self.posts.iter().any(|&(x, _)| x == id),
            "oracle {id} posted twice; identifiers must be unique within an execution"
        );
        self.posts.push((id, self.entries.len()));
        self.push(Entry::OraclePost(id));
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn prefix_at(&self, id: OracleId) -> TranscriptPrefix {
        self.try_prefix_at(id)
            .unwrap_or_else(|| panic!("oracle {id} was never posted"))
    }

    pub fn try_prefix_at(&self, id: OracleId) -> Option<TranscriptPrefix> {
        let &(_, pos) = self.posts.iter().find(|&&(x, _)| x == id)?;
        debug_assert!(matches!(self.entries[pos], Entry::OraclePost(x) if x == id));
        Some(self.prefix_of_first(pos))
    }

    fn prefix_of_first(&self, n: usize) -> TranscriptPrefix {
        let end = if n == 0 { 0 } else { self.ends[n] };
        debug_assert_eq!(end, {
            let mut bytes = Vec::new();
            for e in &self.entries[..n] {
                e.encode_into(&mut bytes);
            }
            bytes.len()
        });
        let bytes = self
            .snapshot
            .0
            .lock()
            .expect("the snapshot lock is not poisoned")
            .get_or_insert_with(|| std::sync::Arc::from(&self.encoded[..]))
            .clone();
        TranscriptPrefix {
            bytes,
            len: end,
            entries: n,
        }
    }

    pub fn canonical(&self) -> TranscriptPrefix {
        self.prefix_of_first(self.entries.len())
    }
}

pub trait RawChallenger {
    fn sample(&mut self) -> F128;
    fn sample_many(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.sample()).collect()
    }
    fn record_ordinary(&mut self, bytes: &[u8]);
    fn post_oracle(&mut self, id: OracleId);
    fn post_long_oracle(&mut self, oracle: &crate::backend::LongOracle<F128>) {
        self.post_oracle(oracle.id);
    }
    fn transcript(&self) -> &Transcript;
}

pub struct SeededChallenger {
    state: u64,
    transcript: Transcript,
}

impl SeededChallenger {
    pub fn new(seed: u64) -> Self {
        let mut x = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x2545_F491_4F6C_DD1D);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        Self {
            state: if x == 0 { 1 } else { x },
            transcript: Transcript::new(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

impl RawChallenger for SeededChallenger {
    fn sample(&mut self) -> F128 {
        let hi = self.next_u64();
        let lo = self.next_u64();
        let c = F128::new(((hi as u128) << 64) | lo as u128);
        self.transcript.record_coin(c);
        c
    }

    fn sample_many(&mut self, n: usize) -> Vec<F128> {
        let coins: Vec<F128> = (0..n)
            .map(|_| {
                let hi = self.next_u64();
                let lo = self.next_u64();
                F128::new(((hi as u128) << 64) | lo as u128)
            })
            .collect();
        self.transcript.record_coins(&coins);
        coins
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

pub struct OsChallenger {
    buf: Vec<u8>,
    used: usize,
    stream: Option<crate::chacha::ChaCha20>,
    transcript: Transcript,
}

impl Default for OsChallenger {
    fn default() -> Self {
        Self::new()
    }
}

impl OsChallenger {
    const BLOCK: usize = 65536;

    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            used: 0,
            stream: None,
            transcript: Transcript::new(),
        }
    }

    fn refill(&mut self) {
        let stream = self.stream.get_or_insert_with(|| {
            use std::io::Read;
            let mut f = std::fs::File::open("/dev/urandom")
                .expect("the verifier's coins require an entropy source");
            let mut seed = [0u8; 40];
            f.read_exact(&mut seed)
                .expect("short read from /dev/urandom");
            let key: [u8; 32] = seed[..32].try_into().expect("32 bytes");
            let nonce = u64::from_le_bytes(seed[32..].try_into().expect("8 bytes"));
            crate::chacha::ChaCha20::new(&key, nonce)
        });
        self.buf.resize(Self::BLOCK, 0);
        stream.fill(&mut self.buf);
        self.used = 0;
    }

    fn next_bytes(&mut self) -> [u8; 16] {
        if self.used + 16 > self.buf.len() {
            self.refill();
        }
        let mut out = [0u8; 16];
        out.copy_from_slice(&self.buf[self.used..self.used + 16]);
        self.used += 16;
        out
    }
}

impl RawChallenger for OsChallenger {
    fn sample(&mut self) -> F128 {
        let c = F128::from_be_bytes(self.next_bytes());
        self.transcript.record_coin(c);
        c
    }

    fn sample_many(&mut self, n: usize) -> Vec<F128> {
        let coins: Vec<F128> = (0..n)
            .map(|_| F128::from_be_bytes(self.next_bytes()))
            .collect();
        self.transcript.record_coins(&coins);
        coins
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
