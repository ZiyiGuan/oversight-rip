use std::sync::{Arc, Mutex};

use flock_core::{
    challenger::{Challenger, Compilation},
    pcs::{ligerito::LigeritoProof, PcsParams},
    proof::{R1csProofCircuitMergedAg, R1csProofMergedLigeritoAg},
};
use flock_field::{F128, F256};
use rmip_iop::transcript::{RawChallenger, Transcript, TranscriptPrefix};

use crate::encoder::TableShape;
use checker::OracleSnapshot;

pub mod checker;

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleGeometry {
    pub shape: TableShape,
    pub queries: usize,
}

pub fn geometry(params: &PcsParams, max_words: usize) -> Result<Vec<OracleGeometry>> {
    let config = params.ligerito_prover_config()?;
    if config.queries.len() != config.recursive_steps + 1
        || config.recursive_log_msg_cols.len() != config.recursive_steps
        || config.recursive_ks.len() != config.recursive_steps
        || config.log_inv_rates.len() != config.queries.len()
        || config.recursive_steps == 0
    {
        return Err("reset: inconsistent native ladder".into());
    }
    let mut out = Vec::with_capacity(config.queries.len());
    for (oracle, &queries) in config.queries.iter().enumerate() {
        let coefficient_log = if oracle == 0 {
            config.initial_log_msg_cols
        } else {
            config.recursive_log_msg_cols[oracle - 1]
        };
        let domain_log = coefficient_log
            .checked_add(config.log_inv_rates[oracle])
            .ok_or("reset: dimension overflow")?;
        let coefficient_count = 1usize
            .checked_shl(coefficient_log as u32)
            .ok_or("reset: coefficient dimension overflow")?;
        let lanes = if oracle == 0 {
            params.num_ntts()
        } else {
            1usize
                .checked_shl(config.recursive_ks[oracle - 1] as u32)
                .ok_or("reset: lane dimension overflow")?
        };
        let shape =
            TableShape::new(domain_log, 0, coefficient_count, lanes).map_err(|e| e.to_string())?;
        if queries == 0 || shape.encoded_words() > max_words {
            return Err("reset: unsupported query count or resource limit".into());
        }
        out.push(OracleGeometry { shape, queries });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowQuery {
    pub oracle: usize,
    pub position: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Posting {
    oracle: usize,
    shape: TableShape,
    prefix: TranscriptPrefix,
}
impl Posting {
    pub fn oracle(&self) -> usize {
        self.oracle
    }
    pub fn shape(&self) -> TableShape {
        self.shape
    }
    pub fn prefix(&self) -> &TranscriptPrefix {
        &self.prefix
    }
}

#[derive(Clone, Debug)]
pub struct ResetSchedule {
    geometry: Vec<OracleGeometry>,
    postings: Vec<Posting>,
    queries: Vec<RowQuery>,
    transcript: Transcript,
}
impl ResetSchedule {
    pub fn postings(&self) -> &[Posting] {
        &self.postings
    }
    pub fn queries(&self) -> &[RowQuery] {
        &self.queries
    }
    pub fn transcript(&self) -> &Transcript {
        &self.transcript
    }
}

struct Rewind {
    max_words: usize,
    snapshots: Vec<Arc<OracleSnapshot>>,
    build: std::time::Duration,
}

enum Role {
    Main {
        postings: Vec<Posting>,
        queries: Vec<RowQuery>,
        rewind: Option<Rewind>,
    },
    Verify {
        schedule: ResetSchedule,
        answers: Vec<Vec<F128>>,
        next_row: usize,
    },
}
struct State<C> {
    source: C,
    geometry: Vec<OracleGeometry>,
    role: Role,
    posted: usize,
    fault: Option<String>,
}

pub struct ResetChallenger<C: RawChallenger + Send> {
    state: Arc<Mutex<State<C>>>,
}

fn announcement(oracle: usize, lanes: usize, rows: usize) -> Vec<u8> {
    let mut out = b"rmip-oracle-v0".to_vec();
    for word in [oracle, lanes, rows] {
        out.extend_from_slice(&(word as u64).to_le_bytes());
    }
    out
}

fn validate_geometry(geometry: &[OracleGeometry]) -> Result<()> {
    if geometry.len() < 2 || geometry.iter().any(|g| g.queries == 0) {
        return Err("reset: missing native oracle or empty query phase".into());
    }
    Ok(())
}

impl<C: RawChallenger + Send> ResetChallenger<C> {
    pub fn main(source: C, geometry: Vec<OracleGeometry>) -> Result<Self> {
        validate_geometry(&geometry)?;
        Ok(Self {
            state: Arc::new(Mutex::new(State {
                source,
                geometry,
                role: Role::Main {
                    postings: Vec::new(),
                    queries: Vec::new(),
                    rewind: None,
                },
                posted: 0,
                fault: None,
            })),
        })
    }

    pub fn rewinding_main(
        source: C,
        geometry: Vec<OracleGeometry>,
        max_words: usize,
    ) -> Result<Self> {
        let challenger = Self::main(source, geometry)?;
        if let Role::Main { rewind, .. } =
            &mut challenger.state.lock().expect("reset transport").role
        {
            *rewind = Some(Rewind {
                max_words,
                snapshots: Vec::new(),
                build: std::time::Duration::ZERO,
            });
        }
        Ok(challenger)
    }

    pub fn take_snapshots(&self) -> Result<(Vec<Arc<OracleSnapshot>>, std::time::Duration)> {
        let mut state = self.state.lock().map_err(|_| "reset transport poisoned")?;
        if let Some(error) = &state.fault {
            return Err(error.clone());
        }
        let posted = state.posted;
        let Role::Main { rewind, .. } = &mut state.role else {
            return Err("reset: take_snapshots on verifier".into());
        };
        let rewind = rewind
            .take()
            .ok_or("reset: no rewinding snapshots (not a rewinding main, or already taken)")?;
        if rewind.snapshots.len() != posted {
            return Err("reset: a posting has no snapshot".into());
        }
        Ok((rewind.snapshots, rewind.build))
    }

    pub fn verifier(
        source: C,
        geometry: Vec<OracleGeometry>,
        schedule: ResetSchedule,
        answers: Vec<Vec<F128>>,
    ) -> Result<Self> {
        validate_geometry(&geometry)?;
        if geometry != schedule.geometry || answers.len() != schedule.queries.len() {
            return Err("reset: verifier geometry or answer count differs".into());
        }
        for (query, answer) in schedule.queries.iter().zip(&answers) {
            if answer.len() != geometry[query.oracle].shape.lanes() {
                return Err("reset: checker answer width differs".into());
            }
        }
        Ok(Self {
            state: Arc::new(Mutex::new(State {
                source,
                geometry,
                role: Role::Verify {
                    schedule,
                    answers,
                    next_row: 0,
                },
                posted: 0,
                fault: None,
            })),
        })
    }

    pub fn with_source<R>(&self, f: impl FnOnce(&C) -> R) -> R {
        f(&self.state.lock().expect("reset transport").source)
    }

    pub fn transcript(&self) -> Transcript {
        self.with_source(|source| source.transcript().clone())
    }

    fn record(&self, bytes: &[u8]) {
        self.state
            .lock()
            .expect("reset transport")
            .source
            .record_ordinary(bytes);
    }

    pub fn finish_main(&self) -> Result<ResetSchedule> {
        let state = self.state.lock().map_err(|_| "reset transport poisoned")?;
        if let Some(error) = &state.fault {
            return Err(error.clone());
        }
        let Role::Main {
            postings, queries, ..
        } = &state.role
        else {
            return Err("reset: finish_main on verifier".into());
        };
        if state.posted != state.geometry.len() || postings.len() != state.geometry.len() {
            return Err("reset: incomplete native postings".into());
        }
        let mut counts = vec![0usize; state.geometry.len()];
        let mut previous = 0;
        for query in queries {
            if query.oracle < previous || query.oracle >= counts.len() {
                return Err("reset: native reads are not in oracle order".into());
            }
            counts[query.oracle] += 1;
            previous = query.oracle;
        }
        if counts
            .iter()
            .zip(&state.geometry)
            .any(|(a, b)| *a != b.queries)
        {
            return Err("reset: incomplete native row occurrences".into());
        }
        Ok(ResetSchedule {
            geometry: state.geometry.clone(),
            postings: postings.clone(),
            queries: queries.clone(),
            transcript: state.source.transcript().clone(),
        })
    }

    pub fn finish_verifier(&self) -> Result<()> {
        let state = self.state.lock().map_err(|_| "reset transport poisoned")?;
        if let Some(error) = &state.fault {
            return Err(error.clone());
        }
        let Role::Verify {
            schedule, next_row, ..
        } = &state.role
        else {
            return Err("reset: finish_verifier on main".into());
        };
        if state.posted != state.geometry.len() || *next_row != schedule.queries.len() {
            return Err("reset: verifier did not consume every posting and row".into());
        }
        if state.source.transcript().canonical() != schedule.transcript.canonical() {
            return Err("reset: verifier transcript differs from transport record".into());
        }
        Ok(())
    }
}

impl<C: RawChallenger + Send> Challenger for ResetChallenger<C> {
    fn compilation(&self) -> Compilation {
        Compilation::Interactive
    }
    fn wants_oracle_coefficients(&self) -> bool {
        true
    }
    fn uses_lazy_oracles(&self) -> bool {
        true
    }
    fn defers_oracle_reads(&self) -> bool {
        matches!(
            self.state.lock().expect("reset transport").role,
            Role::Main { .. }
        )
    }
    fn verify_pow(&mut self, nonce: u64, _: u32) -> bool {
        nonce == 0
    }
    fn observe_label(&mut self, bytes: &[u8]) {
        self.record(bytes);
    }
    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.record(bytes);
    }
    fn observe_f128(&mut self, value: F128) {
        self.record(&crate::bytes_of(value));
    }
    fn observe_f128_slice(&mut self, values: &[F128]) {
        let bytes: Vec<_> = values.iter().flat_map(|v| crate::bytes_of(*v)).collect();
        self.record(&bytes);
    }
    fn observe_f256(&mut self, value: F256) {
        self.observe_f128_slice(&value.coordinates());
    }
    fn sample_f128(&mut self) -> F128 {
        crate::up(self.state.lock().expect("reset transport").source.sample())
    }
    fn fork(&mut self, label: &'static [u8]) -> Self {
        self.fork_from_seed([F128::ZERO; 2], label)
    }
    fn fork_from_seed(&self, _: [F128; 2], label: &'static [u8]) -> Self {
        let mut bytes = b"rmip-fork-v0:".to_vec();
        bytes.extend_from_slice(label);
        self.record(&bytes);
        Self {
            state: self.state.clone(),
        }
    }
    fn merge_child(&mut self, _: Self) {
        self.record(b"rmip-merge-v0");
    }
    fn post_oracle(&mut self, _: &[F128], _: usize) -> usize {
        panic!("reset: native coefficient-only posting required")
    }
    fn post_oracle_from_coefficients(
        &mut self,
        lanes: usize,
        rows: usize,
        coefficients: &[F128],
    ) -> usize {
        let mut state = self.state.lock().expect("reset transport");
        let oracle = state.posted;
        let shape = state.geometry[oracle].shape;
        assert_eq!(
            (lanes, rows, coefficients.len()),
            (shape.lanes(), shape.rows(), shape.coefficient_words())
        );
        assert!(matches!(state.role, Role::Main { .. }));
        state
            .source
            .record_ordinary(&announcement(oracle, lanes, rows));
        state.source.post_oracle(oracle);
        let prefix = state.source.transcript().prefix_at(oracle);
        let queries = state.geometry[oracle].queries;
        let Role::Main {
            postings, rewind, ..
        } = &mut state.role
        else {
            unreachable!()
        };
        let built = rewind.as_mut().map(|rewind| {
            let start = std::time::Instant::now();
            let snapshot = OracleSnapshot::from_posting(
                oracle,
                prefix.clone(),
                shape,
                coefficients,
                queries,
                rewind.max_words,
            );
            rewind.build += start.elapsed();
            snapshot.map(|snapshot| rewind.snapshots.push(Arc::new(snapshot)))
        });
        postings.push(Posting {
            oracle,
            shape,
            prefix,
        });
        if let Some(Err(error)) = built {
            state.fault = Some(error);
        }
        state.posted += 1;
        oracle
    }
    fn read_oracle(&mut self, oracle: usize, position: usize) -> Option<Vec<F128>> {
        let mut state = self.state.lock().ok()?;
        let shape = state.geometry.get(oracle)?.shape;
        if oracle >= state.posted || position >= shape.rows() {
            state.fault = Some("reset: read outside posted oracle".into());
            return None;
        }
        match &mut state.role {
            Role::Main { queries, .. } => {
                queries.push(RowQuery { oracle, position });
                Some(vec![F128::ZERO; shape.lanes()])
            }
            Role::Verify {
                schedule,
                answers,
                next_row,
            } => {
                if schedule.queries.get(*next_row) != Some(&RowQuery { oracle, position }) {
                    state.fault =
                        Some("reset: derived query differs from recorded occurrence".into());
                    return None;
                }
                let answer = answers.get(*next_row)?.clone();
                *next_row += 1;
                Some(answer)
            }
        }
    }
    fn expect_oracle(&mut self, oracle: usize, lanes: usize, rows: usize) -> bool {
        let mut state = self.state.lock().expect("reset transport");
        if oracle != state.posted
            || state
                .geometry
                .get(oracle)
                .is_none_or(|g| (g.shape.lanes(), g.shape.rows()) != (lanes, rows))
        {
            state.fault = Some("reset: native posting identity or shape differs".into());
            return false;
        }
        state
            .source
            .record_ordinary(&announcement(oracle, lanes, rows));
        state.source.post_oracle(oracle);
        let prefix = state.source.transcript().prefix_at(oracle);
        let Role::Verify { schedule, .. } = &state.role else {
            return false;
        };
        if schedule.postings[oracle].prefix != prefix {
            state.fault = Some("reset: posting prefix differs from verifier replay".into());
            return false;
        }
        state.posted += 1;
        true
    }
}

pub fn hydrate_proof(
    proof: &mut R1csProofMergedLigeritoAg,
    schedule: &ResetSchedule,
    answers: &[Vec<F128>],
) -> Result<()> {
    hydrate_native_rows(&mut proof.pcs_open.inner.ligerito, schedule, answers)
}

pub fn hydrate_circuit_proof(
    proof: &mut R1csProofCircuitMergedAg,
    schedule: &ResetSchedule,
    answers: &[Vec<F128>],
) -> Result<()> {
    hydrate_native_rows(&mut proof.pcs_open.inner.ligerito, schedule, answers)
}

fn hydrate_native_rows(
    inner: &mut LigeritoProof,
    schedule: &ResetSchedule,
    answers: &[Vec<F128>],
) -> Result<()> {
    if inner.recursive_proofs.len() + 2 != schedule.geometry.len()
        || answers.len() != schedule.queries.len()
    {
        return Err("reset: hydrated proof has wrong oracle or answer count".into());
    }
    let mut groups: Vec<_> = std::iter::once(&mut inner.initial_proof.opened_rows)
        .chain(
            inner
                .recursive_proofs
                .iter_mut()
                .map(|p| &mut p.opened_rows),
        )
        .chain(std::iter::once(&mut inner.final_proof.opened_rows))
        .collect();
    for (rows, geometry) in groups.iter().zip(&schedule.geometry) {
        if rows.len() != geometry.queries
            || rows.iter().any(|row| {
                row.len() != geometry.shape.lanes() || row.iter().any(|v| *v != F128::ZERO)
            })
        {
            return Err("reset: noncanonical placeholder row slots".into());
        }
    }
    for (query, answer) in schedule.queries.iter().zip(answers) {
        if answer.len() != schedule.geometry[query.oracle].shape.lanes() {
            return Err("reset: hydrated answer width differs".into());
        }
    }
    let mut next = vec![0usize; groups.len()];
    for (query, answer) in schedule.queries.iter().zip(answers) {
        groups[query.oracle][next[query.oracle]] = answer.clone();
        next[query.oracle] += 1;
    }
    Ok(())
}
