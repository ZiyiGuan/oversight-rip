use crate::backend::Claim;
use crate::driver::boolean_point;
use crate::sampling::{uniform_index, uniform_index_observing};
use crate::transcript::RawChallenger;
use rmip_field::{BinaryField, F128};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::Instance;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalProof {
    pub blocks: Vec<Vec<F128>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalError {
    WrongBlockCount {
        expected: usize,
        got: usize,
    },
    WrongBlockWidth {
        occurrence: usize,
        expected: usize,
        got: usize,
    },
    NonBooleanBlock {
        occurrence: usize,
    },
    OracleMismatch {
        occurrence: usize,
    },
}

pub struct LocalVerification {
    pub sampled: Vec<usize>,
    pub claims: Vec<Claim<F128>>,
    pub oracle_calls: usize,
    pub verdict: Result<(), LocalError>,
}

pub struct LocalProverOutput {
    pub proof: LocalProof,
    pub claims: Vec<Claim<F128>>,
}

pub trait LocalMessageProver {
    fn answer_blocks(&mut self, sampling_coins: &[F128], sampled: &[usize]) -> Vec<Vec<F128>>;
}

pub struct HonestLocalProver<'a> {
    inst: &'a Instance,
    table: &'a [F128],
}

impl<'a> HonestLocalProver<'a> {
    pub fn new(inst: &'a Instance, table: &'a [F128]) -> Self {
        assert_eq!(
            table.len(),
            inst.layout.n_trace,
            "trace length must match the layout"
        );
        Self { inst, table }
    }
}

impl LocalMessageProver for HonestLocalProver<'_> {
    fn answer_blocks(&mut self, _sampling_coins: &[F128], sampled: &[usize]) -> Vec<Vec<F128>> {
        sampled
            .iter()
            .map(|&index| {
                self.inst
                    .loc_row(index)
                    .iter()
                    .map(|&coord| self.table[coord])
                    .collect()
            })
            .collect()
    }
}

fn set_first(slot: &mut Option<LocalError>, error: LocalError) {
    if slot.is_none() {
        *slot = Some(error);
    }
}

fn sample(inst: &Instance, t: usize, ch: &mut dyn RawChallenger) -> Vec<usize> {
    (0..t).map(|_| uniform_index(ch, inst.ell)).collect()
}

fn sample_with_coins(
    inst: &Instance,
    t: usize,
    ch: &mut dyn RawChallenger,
) -> (Vec<usize>, Vec<F128>) {
    let mut sampled = Vec::with_capacity(t);
    let mut coins = Vec::with_capacity(t);
    for _ in 0..t {
        sampled.push(uniform_index_observing(ch, inst.ell, |coin| {
            coins.push(coin)
        }));
    }
    (sampled, coins)
}

pub fn prove(inst: &Instance, table: &[F128], t: usize, ch: &mut dyn RawChallenger) -> LocalProof {
    prove_with_claims(inst, table, t, ch).proof
}

pub fn prove_with_claims(
    inst: &Instance,
    table: &[F128],
    t: usize,
    ch: &mut dyn RawChallenger,
) -> LocalProverOutput {
    assert_eq!(
        table.len(),
        inst.layout.n_trace,
        "trace length must match the layout"
    );
    let sampled = sample(inst, t, ch);
    let mut blocks = Vec::with_capacity(t);
    let mut claims = Vec::new();
    for index in sampled {
        let row = inst.loc_row(index);
        let mut block = Vec::with_capacity(row.len());
        for &coord in &row {
            let value = table[coord];
            block.push(value);
            claims.push(Claim {
                point: boolean_point(coord, inst.layout.nu),
                value,
            });
        }
        let mut encoded = Vec::with_capacity(block.len() * 16);
        for value in &block {
            encoded.extend_from_slice(&value.to_be_bytes());
        }
        ch.record_ordinary(&encoded);
        blocks.push(block);
    }
    LocalProverOutput {
        proof: LocalProof { blocks },
        claims,
    }
}

fn verify_blocks<O: Oracle>(
    inst: &Instance,
    oracle: &O,
    t: usize,
    sampled: Vec<usize>,
    blocks: &[Vec<F128>],
    ch: &mut dyn RawChallenger,
) -> LocalVerification {
    let mut first = (blocks.len() != t).then_some(LocalError::WrongBlockCount {
        expected: t,
        got: blocks.len(),
    });
    let mut claims = Vec::new();

    for (occurrence, &index) in sampled.iter().enumerate() {
        let supplied = blocks.get(occurrence).map(Vec::as_slice).unwrap_or(&[]);
        let mut encoded = Vec::with_capacity(supplied.len() * 16);
        for value in supplied {
            encoded.extend_from_slice(&value.to_be_bytes());
        }
        ch.record_ordinary(&encoded);
        let coords = inst.loc_row(index);
        if supplied.len() != coords.len() {
            set_first(
                &mut first,
                LocalError::WrongBlockWidth {
                    occurrence,
                    expected: coords.len(),
                    got: supplied.len(),
                },
            );
        }
        let mut values = vec![F128::ZERO; coords.len()];
        let copied = supplied.len().min(values.len());
        values[..copied].copy_from_slice(&supplied[..copied]);
        let is_boolean = values
            .iter()
            .all(|value| *value == F128::ZERO || *value == F128::ONE);
        if !is_boolean {
            set_first(&mut first, LocalError::NonBooleanBlock { occurrence });
        }

        for (&coord, &value) in coords.iter().zip(&values) {
            claims.push(Claim {
                point: boolean_point(coord, inst.layout.nu),
                value,
            });
        }

        let bits: Vec<bool> = values[..values.len() - 1]
            .iter()
            .map(|value| *value == F128::ONE)
            .collect();
        let claimed_answer = values[values.len() - 1] == F128::ONE;
        let oracle_matches = oracle.query(&bits) == claimed_answer;
        if !is_boolean || !oracle_matches {
            set_first(&mut first, LocalError::OracleMismatch { occurrence });
        }
    }

    LocalVerification {
        sampled,
        claims,
        oracle_calls: t,
        verdict: first.map_or(Ok(()), Err),
    }
}

pub fn verify<O: Oracle>(
    inst: &Instance,
    oracle: &O,
    t: usize,
    proof: &LocalProof,
    ch: &mut dyn RawChallenger,
) -> LocalVerification {
    let sampled = sample(inst, t, ch);
    verify_blocks(inst, oracle, t, sampled, &proof.blocks, ch)
}

pub fn verify_interactive<O, P>(
    inst: &Instance,
    oracle: &O,
    t: usize,
    prover: &mut P,
    ch: &mut dyn RawChallenger,
) -> (LocalProof, LocalVerification)
where
    O: Oracle,
    P: LocalMessageProver + ?Sized,
{
    let (sampled, sampling_coins) = sample_with_coins(inst, t, ch);
    let blocks = prover.answer_blocks(&sampling_coins, &sampled);
    let verification = verify_blocks(inst, oracle, t, sampled, &blocks, ch);
    (LocalProof { blocks }, verification)
}
