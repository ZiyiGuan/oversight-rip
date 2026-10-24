use std::cell::{Ref, RefCell};
use std::sync::Arc;

use crate::schedule::{Half, Kind, LadderShape, Reply, RowSource, Schedule};
use crate::tuple;
use rmip_field::F128;
use rmip_iop::backend::{
    div_error_upward, mul_error_upward, pow_error_upward, sum_errors_upward, Alphabet, AuditMode,
    BackendParams, BooleanBinding, Claim, CommitmentBackend, Coord, LevelParams, LongOracle,
    OpeningVerifier, QueryPlan, Regime, Symbol,
};
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::transcript::{Entry, OracleId, RawChallenger, TranscriptPrefix};
use rmip_opt::encode::{encode_columns_shifted, eval_columns_shifted};
use rmip_opt::ntt::NttField;

pub const ECHO_TAG: [u8; 8] = *b"LADDER-P";

pub const EAGER_DOMAIN_LIMIT: u128 = 1 << 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableState<F: NttField = F128> {
    pub rows: Arc<Vec<Vec<F>>>,
    pub pin_coins: Vec<Vec<F>>,
    pub pin_replies: Vec<Vec<F>>,
    pub prox_coins: Vec<F>,
    pub prox_reply: Vec<F>,
    pub open_coeffs: Vec<F>,
    pub open_reply: Vec<F>,
}

impl<F: NttField> Default for TableState<F> {
    fn default() -> Self {
        Self {
            rows: Arc::default(),
            pin_coins: Vec::new(),
            pin_replies: Vec::new(),
            prox_coins: Vec::new(),
            prox_reply: Vec::new(),
            open_coeffs: Vec::new(),
            open_reply: Vec::new(),
        }
    }
}

fn combine<F: NttField>(rows: &[Vec<F>], c: &[F]) -> Vec<F> {
    assert_eq!(rows.len(), c.len(), "one coefficient per row");
    let n = rows.first().map_or(0, Vec::len);
    let mut acc = vec![F::Unreduced::default(); n];
    for (ci, row) in c.iter().zip(rows) {
        if *ci == F::ZERO {
            continue;
        }
        F::axpy_unreduced(&mut acc, *ci, row);
    }
    F::reduce_all(&acc)
}

const COMBINE_BUDGET_BYTES: usize = 1 << 20;

fn tile_width(n: usize, functionals: usize, unreduced_bytes: usize) -> usize {
    let per_element = functionals.max(1) * unreduced_bytes.max(1);
    let fit = (COMBINE_BUDGET_BYTES / per_element)
        .max(1024)
        .next_power_of_two();
    if fit >= n {
        n.max(1)
    } else {
        fit
    }
}

fn combine_many<F: NttField>(rows: &[Vec<F>], coeffs: &[Vec<F>]) -> Vec<Vec<F>> {
    let n = rows.first().map_or(0, Vec::len);
    for c in coeffs {
        assert_eq!(rows.len(), c.len(), "one coefficient per row");
    }
    let mut accs: Vec<Vec<F::Unreduced>> = coeffs
        .iter()
        .map(|_| vec![F::Unreduced::default(); n])
        .collect();
    let tile = tile_width(n, coeffs.len(), std::mem::size_of::<F::Unreduced>());
    for start in (0..n).step_by(tile) {
        let end = (start + tile).min(n);
        for (i, row) in rows.iter().enumerate() {
            for (acc, c) in accs.iter_mut().zip(coeffs) {
                if c[i] != F::ZERO {
                    F::axpy_unreduced(&mut acc[start..end], c[i], &row[start..end]);
                }
            }
        }
    }
    accs.iter().map(|acc| F::reduce_all(acc)).collect()
}

fn halves<F: NttField>(v: &[F]) -> (Vec<F>, Vec<F>) {
    (
        v.iter().step_by(2).copied().collect(),
        v.iter().skip(1).step_by(2).copied().collect(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Echo,
    PinCoins(OracleId),
    ProxCoins(OracleId),
    OpenCoeffs(OracleId),
    BuildAndPost(OracleId),
    ClearPin(OracleId),
    ClearProx(OracleId),
    ClearOpen(OracleId),
    BatchCoin,
}

pub fn commit_steps(s: &Schedule) -> Vec<Step> {
    let d = s.depth;
    let mut v = vec![Step::BuildAndPost(0)];
    for k in 1..=d {
        v.push(Step::PinCoins(s.a(k - 1)));
        v.push(Step::BuildAndPost(s.a(k)));
    }
    v.push(Step::PinCoins(s.a(d)));
    v.push(Step::ClearPin(s.a(d)));
    v
}

pub fn open_steps(s: &Schedule) -> Vec<Step> {
    let d = s.depth;
    let mut v = vec![Step::Echo, Step::OpenCoeffs(0), Step::ProxCoins(0)];
    if d == 0 {
        v.push(Step::ClearProx(0));
        v.push(Step::ClearOpen(0));
        v.push(Step::BatchCoin);
        return v;
    }
    v.push(Step::BuildAndPost(s.o(1)));
    for k in 2..=d {
        v.push(Step::PinCoins(s.o(k - 1)));
        if k >= 3 {
            v.push(Step::PinCoins(s.p(k - 1)));
        }
        v.push(Step::BuildAndPost(s.p(k)));
        v.push(Step::ProxCoins(s.o(k - 1)));
        if k >= 3 {
            v.push(Step::ProxCoins(s.p(k - 1)));
        }
        v.push(Step::OpenCoeffs(s.o(k - 1)));
        v.push(Step::ProxCoins(s.a(k - 1)));
        v.push(Step::BuildAndPost(s.o(k)));
    }
    v.push(Step::PinCoins(s.o(d)));
    if d >= 2 {
        v.push(Step::PinCoins(s.p(d)));
    }
    v.push(Step::ClearPin(s.o(d)));
    if d >= 2 {
        v.push(Step::ClearPin(s.p(d)));
    }
    v.push(Step::ProxCoins(s.o(d)));
    if d >= 2 {
        v.push(Step::ProxCoins(s.p(d)));
    }
    v.push(Step::OpenCoeffs(s.o(d)));
    v.push(Step::ProxCoins(s.a(d)));
    v.push(Step::ClearProx(s.o(d)));
    if d >= 2 {
        v.push(Step::ClearProx(s.p(d)));
    }
    v.push(Step::ClearOpen(s.o(d)));
    v.push(Step::ClearProx(s.a(d)));
    v.push(Step::BatchCoin);
    v
}

pub fn events_of(s: &Schedule, step: Step) -> (usize, usize, usize) {
    let rows = |id: OracleId| s.table(id).rows.len();
    match step {
        Step::Echo => (0, 1, 0),
        Step::PinCoins(id) => (s.shape.pins * rows(id), 0, 0),
        Step::ProxCoins(id) => (rows(id), 0, 0),
        Step::OpenCoeffs(_) => (0, 0, 0),
        Step::BuildAndPost(_) => (0, 0, 1),
        Step::ClearPin(_) => (0, s.shape.pins, 0),
        Step::ClearProx(_) | Step::ClearOpen(_) => (0, 1, 0),
        Step::BatchCoin => (1, 0, 0),
    }
}

pub struct LadderOf<F: NttField = F128> {
    sched: Schedule,
    materialise: bool,
    states: Vec<TableState<F>>,
    posted: Vec<OracleId>,
    oracles: Vec<LongOracle<F128>>,
    point: Vec<F>,
    ordinary_bytes: usize,
    cache: RefCell<Option<ReplayCache<F>>>,
    level_domains: Vec<Coord>,
    batch_coin: F,
    posting_prefixes: Vec<Option<TranscriptPrefix>>,
    verified: RefCell<Option<(TranscriptPrefix, usize)>>,
}

pub type Ladder = LadderOf<F128>;
pub type Ladder64 = LadderOf<rmip_field::F64>;

#[derive(Clone)]
struct ReplayCache<F: NttField> {
    prefix: Vec<u8>,
    entries: Vec<Entry>,
    states: Vec<TableState<F>>,
    done: usize,
    point: Vec<F>,
}

impl<F: NttField> LadderOf<F> {
    pub fn new(shape: LadderShape) -> Self {
        let mut b = Self::lazy(shape);
        for t in &b.sched.tables {
            let dom = tuple::domain_size(b.sched.shape.n_enc_at(t.level), b.sched.queries)
                .expect("tuple domain fits 1024 bits");
            assert!(
                dom <= Coord::from(EAGER_DOMAIN_LIMIT),
                "Ladder::new materialises |D|^q = {dom} symbols at level {}; use Ladder::lazy",
                t.level
            );
        }
        b.materialise = true;
        b
    }

    pub fn lazy(shape: LadderShape) -> Self {
        let sched = shape.schedule();
        for t in &sched.tables {
            let bits = (usize::BITS
                - sched
                    .shape
                    .n_enc_at(t.level)
                    .saturating_sub(1)
                    .leading_zeros()) as usize;
            assert!(
                sched.queries * bits <= 1000,
                "q·log2(n_enc) = {} exceeds the 1024-bit coordinate limit at level {}",
                sched.queries * bits,
                t.level
            );
        }
        let states = vec![TableState::<F>::default(); sched.tables.len()];
        let tables = sched.tables.len();
        let level_domains = (0..=sched.depth)
            .map(|k| {
                tuple::domain_size(sched.shape.n_enc_at(k), sched.queries)
                    .expect("tuple domain fits 1024 bits")
            })
            .collect();
        Self {
            sched,
            materialise: false,
            states,
            posted: Vec::new(),
            oracles: Vec::new(),
            point: Vec::new(),
            ordinary_bytes: 0,
            cache: RefCell::new(None),
            level_domains,
            batch_coin: F::ZERO,
            posting_prefixes: vec![None; tables],
            verified: RefCell::new(None),
        }
    }

    pub fn shape(&self) -> LadderShape {
        self.sched.shape
    }
    pub fn schedule(&self) -> &Schedule {
        &self.sched
    }
    pub fn states(&self) -> &[TableState<F>] {
        &self.states
    }
    pub fn is_committed(&self) -> bool {
        !self.states[0].rows.is_empty()
    }
    pub fn domain_of(&self, id: OracleId) -> Coord {
        self.level_domains[self.sched.table(id).level]
    }
    pub fn alphabet_of(&self, id: OracleId) -> Alphabet {
        Alphabet::Column {
            elements: self.sched.queries * self.sched.table(id).rows.len(),
        }
    }
    pub fn posted(&self) -> &[OracleId] {
        &self.posted
    }
    pub fn tuple_at(&self, id: OracleId, theta: Coord) -> Option<Vec<usize>> {
        tuple::try_decode(
            theta,
            self.sched.shape.n_enc_at(self.sched.table(id).level),
            self.sched.queries,
        )
    }
    pub fn coord_of_tuple(&self, id: OracleId, j: &[usize]) -> Coord {
        tuple::encode(j, self.sched.shape.n_enc_at(self.sched.table(id).level))
    }

    fn open_coeffs(s: &Schedule, id: OracleId, p: &[F]) -> Vec<F> {
        let t = s.table(id);
        match t.kind {
            Kind::Trace => eq_table(&p[..s.shape.log_m0]),
            Kind::Open => {
                let c = p[s.fold_coordinate(t.level)];
                let mut e = vec![F::ZERO; t.rows.len()];
                for (r, src) in t.rows.iter().enumerate() {
                    if let RowSource::Half {
                        reply: Reply::Open,
                        half,
                        ..
                    } = src
                    {
                        e[r] = match half {
                            Half::Even => F::ONE.add(c),
                            Half::Odd => c,
                        };
                    }
                }
                e
            }
            Kind::Pin | Kind::OpenPin => panic!("table {id} carries no claim"),
        }
    }

    fn build_rows(s: &Schedule, states: &[TableState<F>], id: OracleId) -> Vec<Vec<F>> {
        s.table(id)
            .rows
            .iter()
            .map(|src| match src {
                RowSource::Trace(_) => panic!("trace rows are set by commit"),
                RowSource::Half {
                    parent,
                    reply,
                    half,
                } => {
                    let st = &states[*parent];
                    let v: &[F] = match reply {
                        Reply::Pin(i) => &st.pin_replies[*i],
                        Reply::Prox => &st.prox_reply,
                        Reply::Open => &st.open_reply,
                    };
                    assert!(
                        !v.is_empty(),
                        "parent {parent}'s {reply:?} reply is not ready"
                    );
                    let (e, o) = halves(v);
                    match half {
                        Half::Even => e,
                        Half::Odd => o,
                    }
                }
            })
            .collect()
    }

    fn execute(
        s: &Schedule,
        states: &mut [TableState<F>],
        point: &[F],
        step: Step,
        coins: &mut dyn FnMut(usize) -> Vec<F>,
        sink: &mut dyn FnMut(Sink),
    ) {
        match step {
            Step::Echo => {
                let mut m = ECHO_TAG.to_vec();
                m.extend(F::to_bytes(point));
                sink(Sink::Ordinary(m));
            }
            Step::PinCoins(id) => {
                let rows = s.table(id).rows.len();
                let st = &mut states[id];
                assert!(!st.rows.is_empty(), "table {id} has no rows yet");
                st.pin_coins = (0..s.shape.pins).map(|_| coins(rows)).collect();
                st.pin_replies = combine_many(&st.rows, &st.pin_coins);
            }
            Step::ProxCoins(id) => {
                let rows = s.table(id).rows.len();
                let st = &mut states[id];
                assert!(!st.rows.is_empty(), "table {id} has no rows yet");
                st.prox_coins = coins(rows);
                st.prox_reply = combine(&st.rows, &st.prox_coins);
            }
            Step::OpenCoeffs(id) => {
                let e = Self::open_coeffs(s, id, point);
                let st = &mut states[id];
                st.open_coeffs = e;
                st.open_reply = combine(&st.rows, &st.open_coeffs);
            }
            Step::BuildAndPost(id) => {
                if id != 0 {
                    states[id].rows = Arc::new(Self::build_rows(s, states, id));
                }
                sink(Sink::Post(id));
            }
            Step::ClearPin(id) => {
                for w in &states[id].pin_replies {
                    sink(Sink::Ordinary(F::to_bytes(w)));
                }
            }
            Step::ClearProx(id) => sink(Sink::Ordinary(F::to_bytes(&states[id].prox_reply))),
            Step::ClearOpen(id) => sink(Sink::Ordinary(F::to_bytes(&states[id].open_reply))),
            Step::BatchCoin => {
                let c = coins(1);
                debug_assert_eq!(c.len(), 1);
            }
        }
    }

    fn post(&mut self, id: OracleId, ch: &mut dyn RawChallenger) {
        let t = self.sched.table(id);
        let (level, q) = (t.level, self.sched.queries);
        let n_enc = self.sched.shape.n_enc_at(level);
        let domain = self.domain_of(id);
        let oracle = if self.materialise {
            let cols = encode_columns_shifted(&self.states[id].rows, level, n_enc);
            let count = u64::try_from(domain).expect("an eager domain is below the limit");
            let symbols = (0..count)
                .map(|theta| {
                    let theta = Coord::from(theta);
                    let j = tuple::try_decode(theta, n_enc, q).expect("θ < |D|^q");
                    Symbol(
                        j.iter()
                            .flat_map(|&jk| cols[jk].iter().map(|v| v.to_word()))
                            .collect(),
                    )
                })
                .collect();
            LongOracle::new(id, self.alphabet_of(id), symbols)
        } else {
            LongOracle::new_lazy_wide(id, self.alphabet_of(id), domain)
        };
        self.posting_prefixes[id] = Some(ch.transcript().canonical());
        ch.post_long_oracle(&oracle);
        self.oracles.push(oracle);
        self.posted.push(id);
    }

    fn walk(&mut self, steps: &[Step], ch: &mut dyn RawChallenger) {
        for &step in steps {
            if let Step::BuildAndPost(id) = step {
                if id != 0 {
                    self.states[id].rows =
                        Arc::new(Self::build_rows(&self.sched, &self.states, id));
                }
                self.post(id, ch);
                continue;
            }
            if let Step::BatchCoin = step {
                self.batch_coin = F::from_coin(ch.sample());
                continue;
            }
            let sched = &self.sched;
            let mut messages: Vec<Vec<u8>> = Vec::new();
            Self::execute(
                sched,
                &mut self.states,
                &self.point,
                step,
                &mut |n| ch.sample_many(n).into_iter().map(F::from_coin).collect(),
                &mut |s| match s {
                    Sink::Ordinary(m) => messages.push(m),
                    Sink::Post(_) => unreachable!("posting is handled above"),
                },
            );
            for m in messages {
                ch.record_ordinary(&m);
                self.ordinary_bytes += m.len();
            }
        }
    }

    fn bundle(s: &Schedule, states: &[TableState<F>], id: OracleId, j: &[usize]) -> Symbol<F128> {
        let level = s.table(id).level;
        let rows = &states[id].rows;
        let mut out = Vec::with_capacity(j.len() * rows.len());
        for col in eval_columns_shifted(rows, level, j) {
            out.extend(col);
        }
        Symbol(F::to_words(&out))
    }

    fn replay(&self, target: OracleId, h: &TranscriptPrefix) -> Ref<'_, ReplayCache<F>> {
        let s = &self.sched;
        let commit = commit_steps(s);
        let open = open_steps(s);
        let is_open_table = !matches!(s.table(target).kind, Kind::Trace | Kind::Pin);

        let mut states: Vec<TableState<F>>;
        let mut point: Vec<F>;
        let mut done: usize;
        let mut prefix: Vec<u8>;
        let mut entries: Vec<Entry>;
        let taken = self.cache.borrow_mut().take();
        match taken {
            Some(c) if h.as_bytes().starts_with(&c.prefix) && c.done > 0 => {
                states = c.states;
                point = c.point;
                done = c.done;
                prefix = c.prefix;
                entries = c.entries;
                entries.extend(
                    rmip_iop::transcript::decode_entries(&h.as_bytes()[prefix.len()..])
                        .expect("the checker was handed an undecodable prefix"),
                );
                prefix.extend_from_slice(&h.as_bytes()[prefix.len()..]);
            }
            _ => {
                states = vec![TableState::<F>::default(); s.tables.len()];
                states[0].rows = Arc::clone(&self.states[0].rows);
                point = Vec::new();
                done = 0;
                prefix = h.as_bytes().to_vec();
                entries = h
                    .decode()
                    .expect("the checker was handed an undecodable prefix");
            }
        }
        assert!(
            !states[0].rows.is_empty(),
            "the checker must have committed to the trace before answering"
        );

        let g = entries
            .iter()
            .position(|e| matches!(e, Entry::OraclePost(0)));
        if target == 0 {
            assert!(g.is_none(), "G_0's prefix precedes its own posting");
            return self.store(ReplayCache {
                prefix,
                entries,
                states,
                done,
                point,
            });
        }
        let g = g.expect("an opening-chain prefix contains the trace posting");
        let commit_len: usize = commit
            .iter()
            .map(|&st| {
                let (c, m, p) = events_of(s, st);
                c + m + p
            })
            .sum();
        let commit_start = g;
        let open_before_target: usize = open
            .iter()
            .take_while(|&&st| st != Step::BuildAndPost(target))
            .map(|&st| {
                let (c, m, p) = events_of(s, st);
                c + m + p
            })
            .sum();
        let open_start = if is_open_table {
            assert!(
                entries.len() >= open_before_target,
                "prefix too short for the opening phase"
            );
            Some(entries.len() - open_before_target)
        } else {
            None
        };

        let all: Vec<Step> = commit.iter().chain(open.iter()).copied().collect();
        let n_commit = commit.len();
        let cursor = |i: usize| -> usize {
            if i < n_commit {
                commit_start
                    + commit[..i]
                        .iter()
                        .map(|&st| {
                            let (c, m, p) = events_of(s, st);
                            c + m + p
                        })
                        .sum::<usize>()
            } else {
                open_start.expect("open-phase step on a commit-phase question")
                    + open[..i - n_commit]
                        .iter()
                        .map(|&st| {
                            let (c, m, p) = events_of(s, st);
                            c + m + p
                        })
                        .sum::<usize>()
            }
        };
        let _ = commit_len;

        let stop = all
            .iter()
            .position(|&st| st == Step::BuildAndPost(target))
            .expect("the target is posted by some step");
        assert!(done <= stop, "cache ran past the target");
        while done < stop {
            let step = all[done];
            let at = cursor(done);
            let (coins, msgs, posts) = events_of(s, step);
            let mut k = 0usize;
            let mut next_coins = |n: usize| -> Vec<F> {
                (0..n)
                    .map(|_| {
                        let e = entries
                            .get(at + k)
                            .unwrap_or_else(|| panic!("prefix ends inside step {step:?}"));
                        k += 1;
                        match e {
                            Entry::Coin(c) => F::from_coin(*c),
                            other => panic!("expected a coin at {}, found {other:?}", at + k - 1),
                        }
                    })
                    .collect()
            };
            if let Step::Echo = step {
                match entries.get(at) {
                    Some(Entry::Ordinary(m))
                        if m.starts_with(&ECHO_TAG) && (m.len() - 8) % F::BYTES == 0 =>
                    {
                        point = F::from_bytes(&m[8..]);
                        assert_eq!(
                            point.len(),
                            s.shape.nu,
                            "echoed point has the shape's length"
                        );
                    }
                    other => panic!("expected the echoed point at {at}, found {other:?}"),
                }
            }
            Self::execute(s, &mut states, &point, step, &mut next_coins, &mut |_| {});
            assert_eq!(
                k, coins,
                "step {step:?} consumed {k} coins, schedule says {coins}"
            );
            for i in 0..msgs {
                let e = entries.get(at + coins + i);
                assert!(
                    matches!(e, Some(Entry::Ordinary(_))),
                    "expected an ordinary message at {}, found {e:?}",
                    at + coins + i
                );
            }
            for i in 0..posts {
                let e = entries.get(at + coins + msgs + i);
                assert!(
                    matches!((step, e), (Step::BuildAndPost(id), Some(Entry::OraclePost(x))) if *x == id),
                    "expected a posting at {}, found {e:?}",
                    at + coins + msgs + i
                );
            }
            done += 1;
        }
        if target != 0 {
            states[target].rows = Arc::new(Self::build_rows(s, &states, target));
        }
        self.store(ReplayCache {
            prefix,
            entries,
            states,
            done,
            point,
        })
    }

    fn is_own_posting(&self, id: OracleId, h: &TranscriptPrefix) -> bool {
        let Some(own) = self.posting_prefixes[id].as_ref() else {
            return false;
        };
        if self.states[id].rows.is_empty()
            || own.len() != h.len()
            || own.entry_count() != h.entry_count()
        {
            return false;
        }
        let (own_bytes, bytes) = (own.as_bytes(), h.as_bytes());
        let mut verified = self.verified.borrow_mut();
        let from = match verified.as_ref() {
            Some((last, n)) if last.shares_snapshot(h) && *n <= bytes.len() => *n,
            _ => 0,
        };
        if own_bytes[from..] != bytes[from..] {
            return false;
        }
        *verified = Some((h.clone(), bytes.len()));
        true
    }

    fn store(&self, c: ReplayCache<F>) -> Ref<'_, ReplayCache<F>> {
        *self.cache.borrow_mut() = Some(c);
        Ref::map(self.cache.borrow(), |o| o.as_ref().expect("just stored"))
    }

    fn open_rounds(&self) -> usize {
        let mut rounds = 0usize;
        let mut in_run = false;
        for st in open_steps(&self.sched) {
            let (c, m, p) = events_of(&self.sched, st);
            if c > 0 {
                in_run = false;
            }
            if (m > 0 || p > 0) && !in_run {
                rounds += 1;
                in_run = true;
            }
        }
        rounds
    }
}

enum Sink {
    Ordinary(Vec<u8>),
    Post(#[allow(dead_code)] OracleId),
}

struct PathPlan {
    coords: std::collections::VecDeque<(OracleId, Coord)>,
}

impl QueryPlan<F128> for PathPlan {
    fn next(&mut self, _prev: Option<&Symbol<F128>>) -> Option<(OracleId, Coord)> {
        self.coords.pop_front()
    }
}

pub struct LadderVerifierOf<F: NttField = F128> {
    sched: Schedule,
    #[cfg(feature = "test-hooks")]
    pub states: Vec<TableState<F>>,
    #[cfg(not(feature = "test-hooks"))]
    states: Vec<TableState<F>>,
    #[cfg(feature = "test-hooks")]
    pub point: Vec<F>,
    #[cfg(not(feature = "test-hooks"))]
    point: Vec<F>,
    #[cfg(feature = "test-hooks")]
    pub batch_coin: F,
    #[cfg(not(feature = "test-hooks"))]
    batch_coin: F,
}

pub type LadderVerifier = LadderVerifierOf<F128>;

impl<F: NttField> LadderVerifierOf<F> {
    fn columns(&self, answers: &[(OracleId, Coord, Symbol<F128>)]) -> Option<Vec<Vec<F>>> {
        let s = &self.sched;
        let q = s.queries;
        if answers.len() != s.tables.len() {
            return None;
        }
        let mut base: Option<Vec<usize>> = None;
        let mut cols: Vec<Vec<F>> = Vec::with_capacity(answers.len());
        for (k, (id, theta, sym)) in answers.iter().enumerate() {
            if *id != k {
                return None;
            }
            let t = s.table(*id);
            let j = tuple::try_decode(*theta, s.shape.n_enc_at(t.level), q)?;
            match &base {
                None => base = Some(j.clone()),
                Some(b) => {
                    if j.iter().zip(b).any(|(x, b0)| *x != b0 >> t.level) {
                        return None;
                    }
                }
            }
            let rows = t.rows.len();
            if sym.0.len() != q * rows {
                return None;
            }
            cols.push(F::from_words(&sym.0)?);
        }
        Some(cols)
    }
}

impl<F: NttField> OpeningVerifier<F128> for LadderVerifierOf<F> {
    fn check(&self, claims: &[Claim<F128>], answers: &[(OracleId, Coord, Symbol<F128>)]) -> bool {
        let s = &self.sched;
        let sh = s.shape;
        let (nu, pins, d) = (sh.nu, sh.pins, s.depth);
        if claims.len() != 1 {
            return false;
        }
        let (Some(point), Some(value)) = (
            F::from_words(&claims[0].point),
            F::from_word(claims[0].value),
        ) else {
            return false;
        };
        if point.len() != nu || self.point != point {
            return false;
        }
        if self.states.len() != s.tables.len() {
            return false;
        }
        for t in &s.tables {
            let st = &self.states[t.id];
            let (rows, n) = (t.rows.len(), sh.n_at(t.level));
            if st.pin_coins.len() != pins
                || st.pin_coins.iter().any(|c| c.len() != rows)
                || st.prox_coins.len() != rows
                || (t.has_open() && st.open_coeffs.len() != rows)
            {
                return false;
            }
            if t.level == d
                && (st.pin_replies.len() != pins
                    || st.pin_replies.iter().any(|w| w.len() != n)
                    || st.prox_reply.len() != n
                    || (t.has_open() && st.open_reply.len() != n))
            {
                return false;
            }
        }
        for t in &s.tables {
            if t.has_open()
                && self.states[t.id].open_coeffs != LadderOf::<F>::open_coeffs(s, t.id, &point)
            {
                return false;
            }
        }
        let Some(cols) = self.columns(answers) else {
            return false;
        };
        let tuple0 = tuple::try_decode(answers[0].1, sh.n_enc_at(0), s.queries)
            .expect("columns() decoded it");
        let tables = F::ntt_tables();
        let q = s.queries;
        let gamma = self.batch_coin;
        let slots = pins + 2;
        let mut powers = Vec::with_capacity(slots);
        let mut g = F::ONE;
        for _ in 0..slots {
            powers.push(g);
            g = g.mul(gamma);
        }
        let log_n_bottom = sh.n_at(d).trailing_zeros() as usize;
        let mut bottom_bases: Vec<Option<Vec<F>>> = vec![None; q];
        for t in &s.tables {
            let st = &self.states[t.id];
            let level = t.level;
            let rows = t.rows.len();
            let ys = &cols[t.id];
            let xs: Vec<usize> = tuple0.iter().map(|&x0| x0 >> level).collect();
            let child_rows = |child: OracleId| s.table(child).rows.len();
            let w: Vec<F> = xs.iter().map(|&x| tables.w_hat(level, x)).collect();
            let mut replies: Vec<(Reply, &[F])> = st
                .pin_coins
                .iter()
                .enumerate()
                .map(|(i, c)| (Reply::Pin(i), c.as_slice()))
                .collect();
            replies.push((Reply::Prox, st.prox_coins.as_slice()));
            if t.has_open() {
                replies.push((Reply::Open, st.open_coeffs.as_slice()));
            }
            let mut combined = vec![F::ZERO; rows];
            let mut children: Vec<(OracleId, usize, usize, F)> = Vec::with_capacity(slots);
            let mut clear: Option<Vec<F>> = None;
            for (i, (reply, coins)) in replies.iter().enumerate() {
                F::axpy(&mut combined, powers[i], coins);
                match s.child_rows(t.id, *reply) {
                    Some((child, e, o)) => children.push((child, e, o, powers[i])),
                    None => {
                        if level != d {
                            return false;
                        }
                        let v: &[F] = match reply {
                            Reply::Pin(i) => &st.pin_replies[*i],
                            Reply::Prox => &st.prox_reply,
                            Reply::Open => &st.open_reply,
                        };
                        let acc = clear.get_or_insert_with(|| vec![F::ZERO; v.len()]);
                        if acc.len() != v.len() {
                            return false;
                        }
                        F::axpy(acc, powers[i], v);
                    }
                }
            }
            for path in 0..q {
                let lhs = F::dot(&combined, &ys[path * rows..(path + 1) * rows]);
                let (mut even, mut odd) = (F::ZERO, F::ZERO);
                for &(child, e, o, gi) in &children {
                    let y = &cols[child][path * child_rows(child)..];
                    even = even.add(gi.mul(y[e]));
                    odd = odd.add(gi.mul(y[o]));
                }
                let mut rhs = even.add(w[path].mul(odd));
                if let Some(v) = &clear {
                    let basis = bottom_bases[path].get_or_insert_with(|| {
                        tables.shifted_basis_at(level, xs[path], log_n_bottom)
                    });
                    rhs = rhs.add(F::dot(v, basis));
                }
                if lhs != rhs {
                    return false;
                }
            }
        }
        let bottom_open = if d == 0 { 0 } else { s.o(d) };
        let u = &self.states[bottom_open].open_reply;
        let remaining = &point[sh.log_m0..nu - d];
        if u.len() != 1usize << remaining.len() {
            return false;
        }
        eval_mle(u, remaining) == value
    }
}

impl<F: NttField> CommitmentBackend for LadderOf<F> {
    type Field = F128;
    type Commitment = usize;
    type Verifier = LadderVerifierOf<F>;

    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
        let (m, n) = (self.sched.shape.m0(), self.sched.shape.n0());
        assert_eq!(z.len(), m * n, "trace length must match the shape");
        let rows: Vec<Vec<F>> = z
            .chunks(n)
            .map(|r| {
                F::from_words(r).expect("the committed words carry elements of the backend's field")
            })
            .collect();
        self.states[0].rows = Arc::new(rows);
        *self.cache.borrow_mut() = None;
        self.posting_prefixes = vec![None; self.sched.tables.len()];
        self.posted.clear();
        self.oracles.clear();
        self.ordinary_bytes = 0;
        *self.verified.borrow_mut() = None;
        let steps = commit_steps(&self.sched);
        self.walk(&steps, ch);
        m * n
    }

    fn open(&mut self, claims: &[Claim<F128>], ch: &mut dyn RawChallenger) {
        assert_eq!(claims.len(), 1, "the ladder opens one batched claim");
        assert_eq!(claims[0].point.len(), self.sched.shape.nu);
        self.point = F::from_words(&claims[0].point)
            .expect("the opened point carries elements of the backend's field");
        let steps = open_steps(&self.sched);
        self.walk(&steps, ch);
    }

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
        let j = tuple::draw_tuple(ch, self.sched.shape.n_enc_at(0), self.sched.queries);
        let coords = self
            .sched
            .tables
            .iter()
            .map(|t| {
                let jk: Vec<usize> = j.iter().map(|&c| c >> t.level).collect();
                (t.id, self.coord_of_tuple(t.id, &jk))
            })
            .collect();
        Box::new(PathPlan { coords })
    }

    fn oracles(&self) -> &[LongOracle<F128>] {
        assert!(
            self.materialise,
            "Ladder::lazy() materialises no oracle: drive it with rmip-mip-prover, \
             which reads coordinates through `reconstruct`"
        );
        &self.oracles
    }

    fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
        assert!(id < self.sched.tables.len(), "no table {id} in this ladder");
        let j = self
            .tuple_at(id, coord)
            .expect("tuple coordinate outside the posted domain");
        if self.is_own_posting(id, h) {
            return Self::bundle(&self.sched, &self.states, id, &j);
        }
        let cache = self.replay(id, h);
        Self::bundle(&self.sched, &cache.states, id, &j)
    }

    fn verifier(&self) -> LadderVerifierOf<F> {
        let d = self.sched.depth;
        let states = self
            .sched
            .tables
            .iter()
            .map(|t| {
                let st = &self.states[t.id];
                let bottom = t.level == d;
                TableState {
                    rows: Arc::default(),
                    pin_coins: st.pin_coins.clone(),
                    pin_replies: if bottom {
                        st.pin_replies.clone()
                    } else {
                        Vec::new()
                    },
                    prox_coins: st.prox_coins.clone(),
                    prox_reply: if bottom {
                        st.prox_reply.clone()
                    } else {
                        Vec::new()
                    },
                    open_coeffs: st.open_coeffs.clone(),
                    open_reply: if bottom {
                        st.open_reply.clone()
                    } else {
                        Vec::new()
                    },
                }
            })
            .collect();
        LadderVerifierOf {
            sched: self.sched.clone(),
            states,
            point: self.point.clone(),
            batch_coin: self.batch_coin,
        }
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        assert_eq!(
            trace_len,
            self.sched.shape.m0() * self.sched.shape.n0(),
            "trace length must match the public backend shape"
        );
        self.params(claims)
    }

    fn params(&self, _claims: usize) -> BackendParams {
        let s = &self.sched;
        let sh = s.shape;
        let q = s.queries;
        let eps_open = opening_error(sh, F::KAPPA);
        let mut l_long = Coord::ZERO;
        let mut a_long: u64 = 0;
        let mut alphabet = Vec::with_capacity(s.tables.len());
        for t in &s.tables {
            let qm = q * t.rows.len();
            l_long = l_long
                .checked_add(
                    self.domain_of(t.id)
                        .checked_mul_u64(qm as u64)
                        .expect("|D|^q · q · rows fits 1024 bits"),
                )
                .expect("L_long fits 1024 bits");
            a_long += qm as u64;
            alphabet.push(self.alphabet_of(t.id));
        }
        let schedule = (0..=s.depth)
            .map(|k| LevelParams {
                dim: sh.log_n0() - k,
                rate_inv: sh.rate_inv,
                queries: q,
                lambda_bits: sh.lambda_bits,
            })
            .collect();
        BackendParams {
            name: "bundled-ladder",
            q_long: s.oracle_count(),
            eps_bind: 0.0,
            eps_open,
            regime: Regime::UniqueDecoding,
            boolean_binding: BooleanBinding::ProvedByRelation,
            schedule,
            l_long,
            a_long,
            alphabet,
            ordinary_bytes: self.ordinary_bytes,
            r_open: self.open_rounds(),
            audit: AuditMode::Sequential,
        }
    }
}

pub fn eps_col(shape: LadderShape, level: usize) -> f64 {
    eps_col_at(shape, level, 128)
}

pub const SINGLE_SHOT_COLLISION_BUDGET: f64 = 1.0 / (1u64 << 30) as f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SingleShotTarget {
    pub bits: u32,
}

impl SingleShotTarget {
    pub const DEFAULT: Self = Self { bits: 20 };
    pub const ENV: &'static str = "RMIP_TARGET_BITS";

    pub fn lambda_bits(self) -> f64 {
        f64::from(self.bits) + 3.0
    }
    pub fn local_budget(self) -> f64 {
        self.power_of_two_below(1)
    }
    pub fn collision_budget(self) -> f64 {
        self.power_of_two_below(10)
    }
    pub fn ladder_shape(
        self,
        nu: usize,
        depth: Option<usize>,
        kappa: usize,
    ) -> Option<LadderShape> {
        let target = self.power_of_two_below(0);
        (0..=21u32).find_map(|extra| {
            let lambda = self.lambda_bits() + f64::from(extra);
            let mut shape = match depth {
                Some(d) => LadderShape::for_trace_at_depth(nu, lambda, d),
                None => LadderShape::for_trace(nu, lambda),
            };
            shape.pins = pins_for_budget(shape, kappa, self.collision_budget());
            (opening_error(shape, kappa) <= target).then_some(shape)
        })
    }

    fn power_of_two_below(self, extra: u32) -> f64 {
        let exponent = self.bits.checked_add(extra).expect("target bits");
        assert!(
            exponent <= 1000,
            "a 2^-{} target is beyond the f64 accounting",
            self.bits
        );
        2f64.powi(-(exponent as i32))
    }
    pub fn from_env() -> Self {
        Self::from_env_value(std::env::var(Self::ENV).ok().as_deref())
    }
    fn from_env_value(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            None | Some("") => Self::DEFAULT,
            Some(text) => {
                let bits = text
                    .parse::<u32>()
                    .ok()
                    .filter(|&bits| bits > 0)
                    .unwrap_or_else(|| {
                        panic!("{}: a positive integer in bits, not {text:?}", Self::ENV)
                    });
                Self { bits }
            }
        }
    }
}

pub fn opening_error(shape: LadderShape, kappa: usize) -> f64 {
    let sched = shape.schedule();
    let q = sched.queries;
    let two_f = 2f64.powi(kappa as i32);
    let escape = mul_error_upward(2.0, pow_error_upward(1.0 - shape.delta(), q));
    let mut terms = vec![escape];
    let per_level: Vec<(f64, f64)> = (0..=sched.depth)
        .map(|level| {
            (
                eps_col_at(shape, level, kappa),
                div_error_upward(mul_error_upward(2.0, shape.n_enc_at(level) as f64), two_f),
            )
        })
        .collect();
    for t in &sched.tables {
        let (collision, field) = per_level[t.level];
        terms.push(collision);
        terms.push(field);
    }
    terms.push(div_error_upward(
        mul_error_upward(q as f64, (shape.pins + 1) as f64),
        two_f,
    ));
    sum_errors_upward(&terms)
}

pub fn eps_col_log2_at(shape: LadderShape, level: usize, kappa: usize) -> f64 {
    let q = shape.queries();
    assert!(q >= 1);
    assert!(shape.pins >= 1, "at least one pinning functional");
    let n_enc = shape.n_enc_at(0);
    assert!(
        n_enc.is_power_of_two(),
        "the codeword length is a power of two"
    );
    let log_n_enc = i64::from(n_enc.trailing_zeros());
    let exact = level as i64 + (q as i64 - 1) * log_n_enc - (kappa * shape.pins) as i64;
    let log_q_squared = next_up(((q * q) as f64).log2());
    next_up(log_q_squared + exact as f64)
}

pub fn collision_total_log2(shape: LadderShape, kappa: usize) -> f64 {
    let sched = shape.schedule();
    let logs: Vec<f64> = sched
        .tables
        .iter()
        .map(|t| eps_col_log2_at(shape, t.level, kappa))
        .collect();
    assert!(!logs.is_empty(), "a schedule has at least the trace table");
    let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled: Vec<f64> = logs
        .iter()
        .map(|&l| next_up(next_up(l - max).exp2()))
        .collect();
    let sum = sum_errors_upward(&scaled);
    next_up(max + next_up(sum.log2()))
}

pub fn collision_total(shape: LadderShape, kappa: usize) -> f64 {
    power_of_two_upward(collision_total_log2(shape, kappa))
}

pub fn pins_for_budget(shape: LadderShape, kappa: usize, budget: f64) -> usize {
    assert!(
        budget.is_finite() && budget > 0.0,
        "a positive collision budget"
    );
    let budget_log2 = next_down(budget.log2());
    (1..)
        .find(|&pins| {
            let mut candidate = shape;
            candidate.pins = pins;
            collision_total_log2(candidate, kappa) <= budget_log2
        })
        .expect("the collision total shrinks by 2^kappa per pin")
}

pub fn eps_col_at(shape: LadderShape, level: usize, kappa: usize) -> f64 {
    power_of_two_upward(eps_col_log2_at(shape, level, kappa))
}

fn power_of_two_upward(log2: f64) -> f64 {
    if log2 > 1023.0 {
        return f64::INFINITY;
    }
    next_up(log2.exp2()).max(f64::from_bits(1))
}

fn next_up(x: f64) -> f64 {
    assert!(x.is_finite(), "next_up of a non-finite value");
    if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

#[cfg(test)]
fn eps_col_at_direct(shape: LadderShape, level: usize, kappa: usize) -> f64 {
    let q = shape.queries();
    assert!(q >= 1);
    assert!(shape.pins >= 1, "at least one pinning functional");
    let mut value = mul_error_upward(
        mul_error_upward((q * q) as f64, 2f64.powi(level as i32)),
        pow_error_upward(shape.n_enc_at(0) as f64, q - 1),
    );
    for _ in 0..shape.pins {
        value = mul_error_upward(value, 2f64.powi(-(kappa as i32)));
    }
    value.max(f64::from_bits(1))
}

pub fn require_unpadded(cfg: &rmip_iop::driver::Config) {
    assert_eq!(
        cfg.pad_queries_to, None,
        "the ladder must not be padded: every occurrence is audited"
    );
}

#[cfg(test)]
mod collision_tests {
    use super::*;

    fn shape_at(nu: usize, lambda: f64, depth: Option<usize>) -> LadderShape {
        match depth {
            Some(d) => LadderShape::for_trace_at_depth(nu, lambda, d),
            None => LadderShape::for_trace(nu, lambda),
        }
    }

    type Row = (
        usize,
        usize,
        f64,
        i32,
        Option<usize>,
        usize,
        usize,
        usize,
        f64,
    );

    const DIRECT_TABLE: &[Row] = &[
        (24, 64, 23.0, 30, Some(0), 34, 8, 1, -39.8),
        (24, 64, 23.0, 30, None, 34, 12, 45, -47.2),
        (24, 64, 40.0, 47, Some(0), 59, 14, 1, -72.2),
        (23, 128, 23.0, 30, Some(0), 34, 4, 1, -39.8),
        (23, 128, 23.0, 30, None, 34, 6, 42, -81.2),
        (23, 128, 40.0, 47, Some(0), 59, 7, 1, -72.2),
        (26, 64, 23.0, 30, Some(0), 34, 9, 1, -70.8),
        (26, 64, 23.0, 30, None, 34, 13, 51, -43.2),
        (26, 64, 40.0, 47, Some(0), 59, 15, 1, -78.2),
        (25, 128, 23.0, 30, Some(0), 34, 5, 1, -134.8),
        (25, 128, 23.0, 30, None, 34, 7, 48, -141.2),
        (25, 128, 40.0, 47, Some(0), 59, 8, 1, -142.2),
    ];

    #[test]
    fn pins_for_budget_in_the_log_domain_reproduces_the_direct_table() {
        for &(nu, kappa, lambda, budget_log, depth, queries, pins, tables, collision) in
            DIRECT_TABLE
        {
            let label = format!("nu={nu} kappa={kappa} lambda={lambda} depth={depth:?}");
            let mut shape = shape_at(nu, lambda, depth);
            assert_eq!(shape.queries(), queries, "{label}: queries");
            let budget = 2f64.powi(-budget_log);
            let chosen = pins_for_budget(shape, kappa, budget);
            assert_eq!(chosen, pins, "{label}: pins");
            shape.pins = chosen;
            assert_eq!(shape.schedule().tables.len(), tables, "{label}: tables");
            let total_log2 = collision_total_log2(shape, kappa);
            assert!(
                (total_log2 - collision).abs() <= 0.05,
                "{label}: collision 2^{total_log2:.3} against 2^{collision}"
            );
            assert!(total_log2 <= budget.log2(), "{label}: within the budget");
            assert!(
                collision_total(shape, kappa) <= budget,
                "{label}: the linear total within the budget"
            );
            let linear_log2 = collision_total(shape, kappa).log2();
            assert!(
                (linear_log2 - total_log2).abs() <= 1e-9,
                "{label}: linear 2^{linear_log2} against log 2^{total_log2}"
            );
            if chosen > 1 {
                let mut fewer = shape;
                fewer.pins = chosen - 1;
                assert!(
                    collision_total_log2(fewer, kappa) > budget.log2(),
                    "{label}: one pin fewer breaks the budget"
                );
            }
            println!(
                "{label}: queries={queries} pins={chosen} tables={tables} collision=2^{total_log2:.1}"
            );
        }
    }

    #[test]
    fn the_43_bit_full_depth_shapes_no_longer_overflow() {
        let target = SingleShotTarget { bits: 40 };
        for nu in [24usize, 26] {
            let mut shape = LadderShape::for_trace(nu, target.lambda_bits());
            let pins = pins_for_budget(shape, 64, target.collision_budget());
            shape.pins = pins;
            let total_log2 = collision_total_log2(shape, 64);
            println!(
                "nu={nu} kappa=64 lambda={} budget=2^-50 depth=None: queries={} pins={pins} tables={} collision=2^{total_log2:.1}",
                target.lambda_bits(),
                shape.queries(),
                shape.schedule().tables.len()
            );
            assert!(total_log2 <= -50.0, "within the budget");
            assert!((12..=32).contains(&pins), "pins = {pins}");
        }
    }

    #[test]
    fn the_log_domain_agrees_with_the_direct_product_at_small_shapes() {
        for nu in [14usize, 16, 18] {
            for kappa in [64usize, 128] {
                for depth in [Some(0), None] {
                    let mut shape = shape_at(nu, 23.0, depth);
                    for pins in [1usize, 2, 3] {
                        shape.pins = pins;
                        for level in 0..=shape.depth() {
                            let direct = eps_col_at_direct(shape, level, kappa);
                            let direct_log2 = direct.log2();
                            let log2 = eps_col_log2_at(shape, level, kappa);
                            let scale = direct_log2.abs().max(1.0);
                            assert!(
                                (log2 - direct_log2).abs() <= 1e-9 * scale,
                                "nu={nu} kappa={kappa} depth={depth:?} pins={pins} level={level}: {log2} against {direct_log2}"
                            );
                            let value = eps_col_at(shape, level, kappa);
                            assert!(
                                ((value - direct) / direct).abs() <= 1e-12,
                                "nu={nu} kappa={kappa} depth={depth:?} pins={pins} level={level}: {value:e} against {direct:e}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn next_up_and_next_down_match_the_standard_library() {
        let samples = [
            0.0,
            -0.0,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            1.0,
            -1.0,
            -39.8,
            -141.2,
            10.087,
            1e300,
            -1e300,
            f64::MAX,
            f64::MIN,
        ];
        for x in samples {
            assert_eq!(
                next_up(x).to_bits(),
                x.next_up().to_bits(),
                "next_up({x:e})"
            );
            assert_eq!(
                next_down(x).to_bits(),
                x.next_down().to_bits(),
                "next_down({x:e})"
            );
            assert!(next_up(x) > x, "next_up({x:e}) is above");
            assert!(next_down(x) < x, "next_down({x:e}) is below");
        }
    }

    #[test]
    fn the_single_shot_target_reproduces_the_registry_numbers() {
        let target = SingleShotTarget::DEFAULT;
        assert_eq!(target.bits, 20);
        assert_eq!(target.collision_budget(), SINGLE_SHOT_COLLISION_BUDGET);
        assert_eq!(target.lambda_bits(), 23.0);
        assert_eq!(target.local_budget(), 2f64.powi(-21));
        for bits in [20u32, 40] {
            let target = SingleShotTarget { bits };
            let escape = 2.0 * 2f64.powi(-(target.lambda_bits() as i32));
            let sum = escape + target.local_budget() + target.collision_budget();
            assert!(
                sum < 2f64.powi(-(bits as i32)),
                "at {bits} bits the three budgets sum to {sum:e}"
            );
        }
    }

    #[test]
    fn the_target_is_read_from_the_environment_as_a_positive_integer() {
        assert_eq!(
            SingleShotTarget::from_env_value(None),
            SingleShotTarget::DEFAULT
        );
        assert_eq!(
            SingleShotTarget::from_env_value(Some("")),
            SingleShotTarget::DEFAULT
        );
        assert_eq!(
            SingleShotTarget::from_env_value(Some(" 40 ")),
            SingleShotTarget { bits: 40 }
        );
        assert_eq!(
            SingleShotTarget::from_env_value(Some("20")),
            SingleShotTarget::DEFAULT
        );
    }

    #[test]
    #[should_panic(expected = "RMIP_TARGET_BITS: a positive integer")]
    fn a_zero_target_is_refused() {
        let _ = SingleShotTarget::from_env_value(Some("0"));
    }

    #[test]
    #[should_panic(expected = "RMIP_TARGET_BITS: a positive integer")]
    fn a_non_numeric_target_is_refused() {
        let _ = SingleShotTarget::from_env_value(Some("forty"));
    }

    fn opening_error_at(
        target: SingleShotTarget,
        nu: usize,
        kappa: usize,
        depth: Option<usize>,
    ) -> (LadderShape, f64) {
        let mut shape = shape_at(nu, target.lambda_bits(), depth);
        shape.pins = pins_for_budget(shape, kappa, target.collision_budget());
        let error = opening_error(shape, kappa);
        println!(
            "target 2^-{} nu={nu} kappa={kappa} depth={depth:?}: queries={} pins={} tables={} opening_error=2^{:.2}",
            target.bits,
            shape.queries(),
            shape.pins,
            shape.schedule().tables.len(),
            error.log2()
        );
        (shape, error)
    }

    #[test]
    fn the_single_shot_opening_error_is_within_the_target_at_depth_zero() {
        let (shape, error) = opening_error_at(SingleShotTarget::DEFAULT, 24, 64, Some(0));
        assert!(error < 2f64.powi(-20), "opening error {error:e}");
        let params = crate::Ladder64::lazy(shape).params(1);
        assert_eq!(params.eps_open, error, "the backend reports opening_error");
    }

    #[test]
    fn the_40_bit_target_is_met_at_depth_zero_or_at_128_bits_only() {
        let target = SingleShotTarget { bits: 40 };
        let bound = 2f64.powi(-40);
        for (nu, kappa, depth, within) in [
            (24usize, 64usize, Some(0usize), true),
            (24, 64, None, false),
            (26, 64, None, false),
            (24, 128, Some(0), true),
            (24, 128, None, true),
            (26, 128, None, true),
        ] {
            let (_, error) = opening_error_at(target, nu, kappa, depth);
            assert_eq!(
                error <= bound,
                within,
                "nu={nu} kappa={kappa} depth={depth:?}: opening error 2^{:.2}",
                error.log2()
            );
        }
    }
}
