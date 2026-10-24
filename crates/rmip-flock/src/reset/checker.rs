use std::sync::Arc;
use std::time::{Duration, Instant};

use flock_core::pcs::{ligerito::ProverConfig, statement_bytes, PcsParams};
use flock_field::{F128, F256};
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};
use rmip_iop::transcript::{decode_entries, Entry, TranscriptPrefix};

use crate::encoder::{Coefficients, PartialTable, ShiftedEncoder, TableShape};

const OPENER: &[u8] = b"flock-ligerito-basis-f256-split-v0";
const NATIVE_PREFIX_SUFFIX_BYTES: usize = 16 << 20;

type Result<T> = std::result::Result<T, String>;
type DenseWitness = Box<dyn FnOnce() -> Result<Vec<F128>> + Send>;

pub fn native_shapes(params: &PcsParams, max_words: usize) -> Result<Vec<TableShape>> {
    if max_words == 0
        || params.m < F128::LOG_BITS
        || params.log_batch_size >= usize::BITS as usize
        || params.log_batch_size > params.m - F128::LOG_BITS
        || params
            .num_lanes
            .is_some_and(|n| n == 0 || n > 1usize << params.log_batch_size)
    {
        return Err("reset checker: unsupported native geometry".into());
    }
    let config = params.ligerito_prover_config()?;
    if config.queries.len() != config.recursive_steps + 1
        || config.recursive_log_msg_cols.len() != config.recursive_steps
        || config.recursive_ks.len() != config.recursive_steps
        || config.log_inv_rates.len() != config.queries.len()
        || config.queries.contains(&0)
    {
        return Err("reset checker: inconsistent native configuration".into());
    }
    (0..config.queries.len())
        .map(|native| {
            let coefficient_log = if native == 0 {
                config.initial_log_msg_cols
            } else {
                config.recursive_log_msg_cols[native - 1]
            };
            let domain_log = coefficient_log
                .checked_add(config.log_inv_rates[native])
                .ok_or("reset checker: domain dimension overflow")?;
            let coefficient_count = 1usize
                .checked_shl(coefficient_log as u32)
                .ok_or("reset checker: coefficient dimension overflow")?;
            let lanes = if native == 0 {
                params.num_ntts()
            } else {
                1usize
                    .checked_shl(config.recursive_ks[native - 1] as u32)
                    .ok_or("reset checker: lane dimension overflow")?
            };
            let shape = TableShape::new(domain_log, 0, coefficient_count, lanes)
                .map_err(|error| error.to_string())?;
            if shape.encoded_words() > max_words {
                return Err("reset checker: native oracle exceeds resource limit".into());
            }
            Ok(shape)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PreparationTimings {
    pub witness: Duration,
    pub fold: Duration,
    pub encoding: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EncodingPlan {
    #[default]
    Prefix,
    Complete,
}

pub struct OracleSnapshot {
    oracle: usize,
    prefix: TranscriptPrefix,
    coefficients: Coefficients,
    encoder: ShiftedEncoder,
    table: PartialTable,
    timings: PreparationTimings,
}

impl OracleSnapshot {
    pub fn from_posting(
        oracle: usize,
        prefix: TranscriptPrefix,
        shape: TableShape,
        coefficients: &[F128],
        queries: usize,
        max_words: usize,
    ) -> Result<Self> {
        let start = Instant::now();
        let coefficients =
            Coefficients::new(shape, coefficients.to_vec()).map_err(|error| error.to_string())?;
        let encoder = ShiftedEncoder::new(shape.native_domain_log(), max_words)
            .map_err(|error| error.to_string())?;
        let layer = encoder
            .prefix_layer(shape, queries)
            .map_err(|error| error.to_string())?;
        let table = encoder
            .encode_prefix(&coefficients, layer)
            .map_err(|error| error.to_string())?;
        let timings = PreparationTimings {
            encoding: start.elapsed(),
            ..PreparationTimings::default()
        };
        Ok(Self {
            oracle,
            prefix,
            coefficients,
            encoder,
            table,
            timings,
        })
    }

    pub fn oracle(&self) -> usize {
        self.oracle
    }

    pub fn shape(&self) -> TableShape {
        self.coefficients.shape()
    }

    pub fn coefficient_words(&self) -> usize {
        self.coefficients.values().len()
    }

    pub fn table_words(&self) -> usize {
        self.table.values().len()
    }

    pub fn prefix_layer(&self) -> usize {
        self.table.layer()
    }

    pub fn preparation_timings(&self) -> PreparationTimings {
        self.timings
    }

    pub fn answer(&self, position: usize) -> Result<Vec<F128>> {
        self.encoder
            .finish_row(&self.coefficients, &self.table, position)
            .map_err(|error| error.to_string())
    }

    pub fn answer_direct(&self, position: usize) -> Result<Vec<F128>> {
        self.encoder
            .evaluate(&self.coefficients, position)
            .map_err(|error| error.to_string())
    }
}

struct Cursor<'a> {
    events: &'a [Entry],
    at: usize,
}

impl Cursor<'_> {
    fn exact(&mut self, expected: Entry) -> Result<()> {
        if self.events.get(self.at) != Some(&expected) {
            return Err(format!("reset checker: unexpected event {}", self.at));
        }
        self.at += 1;
        Ok(())
    }

    fn ordinary(&mut self, size: usize) -> Result<()> {
        match self.events.get(self.at) {
            Some(Entry::Ordinary(bytes)) if bytes.len() == size => self.at += 1,
            _ => return Err(format!("reset checker: invalid message at {}", self.at)),
        }
        Ok(())
    }

    fn coin(&mut self) -> Result<F128> {
        let Some(Entry::Coin(value)) = self.events.get(self.at) else {
            return Err(format!("reset checker: missing coin at {}", self.at));
        };
        self.at += 1;
        Ok(crate::up(*value))
    }

    fn coins(&mut self, count: usize) -> Result<()> {
        for _ in 0..count {
            self.coin()?;
        }
        Ok(())
    }

    fn quadratic(&mut self) -> Result<()> {
        self.ordinary(2 * F128::BYTES)?;
        self.ordinary(2 * F128::BYTES)
    }

    fn folds(&mut self, count: usize) -> Result<Vec<F256>> {
        let mut challenges = Vec::with_capacity(count);
        for _ in 0..count {
            challenges.push(F256::new(self.coin()?, self.coin()?));
            self.quadratic()?;
        }
        Ok(challenges)
    }

    fn seek_opener(&mut self) -> Result<()> {
        loop {
            match self.events.get(self.at) {
                Some(Entry::Ordinary(bytes)) if bytes == OPENER => {
                    self.at += 1;
                    return Ok(());
                }
                Some(Entry::Ordinary(_)) | Some(Entry::Coin(_)) => self.at += 1,
                _ => return Err("reset checker: missing opener or unexpected posting".into()),
            }
        }
    }
}

fn announcement(oracle: usize, shape: TableShape) -> Vec<u8> {
    let mut bytes = b"rmip-oracle-v0".to_vec();
    for value in [oracle, shape.lanes(), shape.rows()] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    bytes
}

pub struct SnapshotChecker {
    params: PcsParams,
    binding_messages: Vec<Vec<u8>>,
    derive_dense: Option<DenseWitness>,
    config: ProverConfig,
    shapes: Vec<TableShape>,
    snapshots: Vec<Arc<OracleSnapshot>>,
    max_words: usize,
    prefix_limit: usize,
    plan: EncodingPlan,
    failed: bool,
}

impl SnapshotChecker {
    pub fn new(setup: Blake3Setup, witness: Vec<Compression>, max_words: usize) -> Result<Self> {
        if witness.len() != setup.n_blocks || witness.is_empty() {
            return Err("reset checker: witness count differs from statement".into());
        }
        let params = setup.pcs_params.clone();
        let binding_messages = vec![
            b"flock-mixed-v1".to_vec(),
            setup.registry.digest().to_vec(),
            (setup.n_blocks as u64).to_le_bytes().to_vec(),
            Vec::new(),
            statement_bytes(&params),
        ];
        Self::from_native(
            params,
            binding_messages,
            move || Ok(setup.dense_witness_from_compressions(&witness)),
            max_words,
        )
    }

    pub fn from_native(
        params: PcsParams,
        binding_messages: Vec<Vec<u8>>,
        derive_dense: impl FnOnce() -> Result<Vec<F128>> + Send + 'static,
        max_words: usize,
    ) -> Result<Self> {
        if binding_messages.last() != Some(&statement_bytes(&params)) {
            return Err("reset checker: common binding does not end with native parameters".into());
        }
        let binding_bytes = binding_messages
            .iter()
            .try_fold(0usize, |sum, message| {
                sum.checked_add(9)?.checked_add(message.len())
            })
            .ok_or("reset checker: common binding length overflow")?;
        let binding_budget = max_words
            .checked_mul(F128::BYTES)
            .ok_or("reset checker: common binding resource limit overflow")?;
        if binding_bytes > binding_budget {
            return Err("reset checker: common binding exceeds resource limit".into());
        }
        let prefix_limit = binding_bytes
            .checked_add(NATIVE_PREFIX_SUFFIX_BYTES)
            .filter(|&limit| limit <= isize::MAX as usize)
            .ok_or("reset checker: prefix transport limit overflow")?;
        let shapes = native_shapes(&params, max_words)?;
        let dense_words = 1usize
            .checked_shl((params.m - F128::LOG_BITS) as u32)
            .ok_or("reset checker: dense witness dimension overflow")?;
        if dense_words > max_words {
            return Err("reset checker: dense witness exceeds resource limit".into());
        }
        let retained = shapes.iter().try_fold(0usize, |total, shape| {
            total
                .checked_add(shape.coefficient_words())
                .and_then(|sum| sum.checked_add(shape.encoded_words()))
        });
        if retained.is_none_or(|words| words > max_words) {
            return Err("reset checker: retained checkpoints exceed resource limit".into());
        }
        let config = params.ligerito_prover_config()?;
        Ok(Self {
            params,
            binding_messages,
            derive_dense: Some(Box::new(derive_dense)),
            config,
            shapes,
            snapshots: Vec::new(),
            max_words,
            prefix_limit,
            plan: EncodingPlan::default(),
            failed: false,
        })
    }

    pub fn set_encoding_plan(&mut self, plan: EncodingPlan) -> Result<()> {
        if !self.snapshots.is_empty() || self.failed {
            return Err("reset checker: encoding plan fixed after the first checkpoint".into());
        }
        self.plan = plan;
        Ok(())
    }

    pub fn encoding_plan(&self) -> EncodingPlan {
        self.plan
    }

    pub fn oracle_count(&self) -> usize {
        self.shapes.len()
    }

    pub fn prepared_count(&self) -> usize {
        self.snapshots.len()
    }

    pub fn prefix_limit_bytes(&self) -> usize {
        self.prefix_limit
    }

    pub fn prepare(
        &mut self,
        oracle: usize,
        prefix: &TranscriptPrefix,
    ) -> Result<Arc<OracleSnapshot>> {
        if self.failed {
            return Err("reset checker: checkpoint preparation already failed".into());
        }
        let result = self.prepare_inner(oracle, prefix);
        if result.is_err() {
            self.failed = true;
            self.derive_dense = None;
        }
        result
    }

    fn prepare_inner(
        &mut self,
        oracle: usize,
        prefix: &TranscriptPrefix,
    ) -> Result<Arc<OracleSnapshot>> {
        if let Some(snapshot) = self.snapshots.get(oracle) {
            return if &snapshot.prefix == prefix {
                Ok(Arc::clone(snapshot))
            } else {
                Err("reset checker: cached oracle's prefix changed".into())
            };
        }
        if oracle != self.snapshots.len() || oracle >= self.shapes.len() {
            return Err("reset checker: checkpoints require native posting order".into());
        }
        if prefix.len() > self.prefix_limit {
            return Err("reset checker: prefix exceeds instance transport bound".into());
        }
        let previous = self.snapshots.last();
        let previous_len = previous.map_or(0, |snapshot| snapshot.prefix.len());
        if previous
            .is_some_and(|snapshot| !prefix.as_bytes().starts_with(snapshot.prefix.as_bytes()))
        {
            return Err("reset checker: prefix changed its exact checkpoint".into());
        }
        let events = decode_entries(&prefix.as_bytes()[previous_len..])
            .map_err(|error| error.to_string())?;
        let mut cursor = Cursor {
            events: &events,
            at: 0,
        };
        let challenges = if oracle == 0 {
            self.binding(&mut cursor)?;
            Vec::new()
        } else {
            cursor.exact(Entry::OraclePost(oracle - 1))?;
            self.fold_challenges(&mut cursor, oracle)?
        };
        let shape = self.shapes[oracle];
        cursor.exact(Entry::Ordinary(announcement(oracle, shape)))?;
        if cursor.at != events.len() {
            return Err("reset checker: future events after posting boundary".into());
        }

        let mut timings = PreparationTimings::default();
        let start = Instant::now();
        let values = if oracle == 0 {
            let values = self.first_coefficients()?;
            timings.witness = start.elapsed();
            values
        } else {
            let values = self.fold_coefficients(oracle, &challenges)?;
            timings.fold = start.elapsed();
            values
        };
        let coefficients = Coefficients::new(shape, values).map_err(|error| error.to_string())?;
        let start = Instant::now();
        let encoder = ShiftedEncoder::new(shape.native_domain_log(), self.max_words)
            .map_err(|error| error.to_string())?;
        let layer = match self.plan {
            EncodingPlan::Prefix => encoder
                .prefix_layer(shape, self.config.queries[oracle])
                .map_err(|error| error.to_string())?,
            EncodingPlan::Complete => shape.domain_log(),
        };
        let table = encoder
            .encode_prefix(&coefficients, layer)
            .map_err(|error| error.to_string())?;
        timings.encoding = start.elapsed();
        let snapshot = Arc::new(OracleSnapshot {
            oracle,
            prefix: prefix.clone(),
            coefficients,
            encoder,
            table,
            timings,
        });
        self.snapshots.push(Arc::clone(&snapshot));
        Ok(snapshot)
    }

    fn binding(&self, cursor: &mut Cursor<'_>) -> Result<()> {
        for message in &self.binding_messages {
            cursor.exact(Entry::Ordinary(message.clone()))?;
        }
        Ok(())
    }

    fn first_coefficients(&mut self) -> Result<Vec<F128>> {
        let derive_dense = self
            .derive_dense
            .take()
            .ok_or("reset checker: witness already consumed")?;
        let dense = derive_dense()?;
        let shape = self.shapes[0];
        let expected_words = 1usize << (self.params.m - F128::LOG_BITS);
        if dense.len() != expected_words || dense.len() < shape.coefficient_words() {
            return Err("reset checker: native dense witness length differs".into());
        }
        if self.params.num_lanes.is_none() {
            if dense.len() != shape.coefficient_words() {
                return Err("reset checker: full-lane coefficient count differs".into());
            }
            return Ok(dense);
        }
        if dense[shape.coefficient_words()..]
            .iter()
            .any(|&word| word != F128::ZERO)
        {
            return Err("reset checker: invalid omitted native witness tail".into());
        }
        let count = shape.coefficient_count();
        let mut values = vec![F128::ZERO; shape.coefficient_words()];
        const TILE: usize = 1024;
        for start in (0..count).step_by(TILE) {
            let end = (start + TILE).min(count);
            for lane in 0..shape.lanes() {
                for degree in start..end {
                    values[degree * shape.lanes() + lane] = dense[lane * count + degree];
                }
            }
        }
        Ok(values)
    }

    fn fold_challenges(&self, cursor: &mut Cursor<'_>, native: usize) -> Result<Vec<F256>> {
        if native == 1 {
            cursor.seek_opener()?;
            cursor.ordinary(F128::BYTES)?;
            cursor.ordinary(0)?;
            let log_n = self.params.m - F128::LOG_BITS;
            for _ in 0..self.config.ood_samples[0] {
                cursor.coins(log_n)?;
                cursor.ordinary(F128::BYTES)?;
                cursor.coin()?;
            }
            cursor.quadratic()?;
            cursor.folds(self.config.initial_k)
        } else {
            let previous = native - 1;
            cursor.ordinary(0)?;
            let shape = self.shapes[previous];
            let split_dim = shape.coefficient_count().trailing_zeros() as usize
                + shape.lanes().trailing_zeros() as usize;
            for _ in 0..self.config.ood_samples[previous] {
                cursor.coins(split_dim)?;
                cursor.ordinary(F128::BYTES)?;
                cursor.quadratic()?;
                cursor.coin()?;
            }
            let count = self.config.queries[previous - 1];
            cursor.coins(count)?;
            cursor.coins((usize::BITS - (count - 1).leading_zeros()) as usize)?;
            cursor.quadratic()?;
            cursor.coin()?;
            cursor.folds(self.config.recursive_ks[previous - 1])
        }
    }

    fn fold_coefficients(&self, native: usize, challenges: &[F256]) -> Result<Vec<F128>> {
        let source = self.snapshots[native - 1].coefficients.values();
        let weights = eq_weights(challenges);
        let width = if native == 1 {
            self.shapes[0].lanes()
        } else {
            weights.len()
        };
        if width > weights.len() || !source.len().is_multiple_of(width) {
            return Err("reset checker: invalid native fold geometry".into());
        }
        let count = (source.len() / width)
            .checked_mul(2)
            .ok_or("reset checker: folded length overflow")?;
        if count != self.shapes[native].coefficient_words() || count > self.max_words {
            return Err("reset checker: folded coefficient count differs".into());
        }
        let mut words = Vec::with_capacity(count);
        for chunk in source.chunks_exact(width) {
            let value = chunk
                .iter()
                .zip(&weights)
                .fold(F256::ZERO, |acc, (&word, &weight)| acc + weight * word);
            words.extend_from_slice(&value.coordinates());
        }
        Ok(words)
    }
}

fn eq_weights(point: &[F256]) -> Vec<F256> {
    let mut weights = vec![F256::ONE];
    for &r in point {
        let size = weights.len();
        weights.resize(2 * size, F256::ZERO);
        for index in 0..size {
            let high = weights[index] * r;
            weights[index + size] = high;
            weights[index] += high;
        }
    }
    weights
}
