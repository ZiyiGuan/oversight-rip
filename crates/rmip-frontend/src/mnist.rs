use crate::bits::{BitSlice, BitVec};
use crate::circuit::{CircuitBuilder, OracleCircuit, WireId};
use crate::oracle::Oracle;
use crate::Ratio;

pub const LABEL_BITS: usize = 4;

pub const SPARSE_TAG: u8 = 0x02;

pub fn sparse_query_bytes(image_id: u16, mask_id: u32, label: u8) -> [u8; 8] {
    let id = image_id.to_le_bytes();
    let m = mask_id.to_le_bytes();
    [SPARSE_TAG, id[0], id[1], m[0], m[1], m[2], m[3], label]
}

fn byte_at(u: &BitSlice, i: usize) -> u8 {
    (0..8).fold(0u8, |acc, k| (acc << 1) | u8::from(u[i * 8 + k]))
}

fn const_byte(bld: &mut CircuitBuilder, b: u8) -> Vec<WireId> {
    (0..8)
        .map(|k| bld.constant((b >> (7 - k)) & 1 == 1))
        .collect()
}

pub struct SparseMajorityFamily {
    pub image_id: u16,
    pub r: usize,
    pub threshold: usize,
}

impl SparseMajorityFamily {
    pub fn new(image_id: u16, r: usize) -> Self {
        assert!(
            r >= 2,
            "at least two judgments: the popcount chain needs two inputs"
        );
        Self {
            image_id,
            r,
            threshold: r / 2 + 1,
        }
    }

    pub fn build(&self) -> OracleCircuit {
        self.build_with_shape().0
    }

    pub fn build_with_shape(&self) -> (OracleCircuit, crate::circuit::ChainShape) {
        assert!(
            self.threshold >= 1 && self.threshold <= self.r,
            "threshold in [1, r]"
        );
        assert!(self.r <= u32::MAX as usize, "mask id is four bytes");
        let mut bld = CircuitBuilder::new(LABEL_BITS);
        let ybits: Vec<WireId> = (0..LABEL_BITS).map(|i| bld.input(i)).collect();
        let ge = bld.ge_const(&ybits, 10);
        let valid = bld.not(ge);
        let y0: Vec<WireId> = ybits.iter().map(|&y| bld.and(y, valid)).collect();
        let label_byte: Vec<WireId> = {
            let zero = bld.constant(false);
            vec![zero, zero, zero, zero, y0[3], y0[2], y0[1], y0[0]]
        };

        let answers: Vec<WireId> = (0..self.r)
            .map(|mask_id| {
                let bytes = sparse_query_bytes(self.image_id, mask_id as u32, 0);
                let mut q: Vec<WireId> = bytes[..7]
                    .iter()
                    .flat_map(|&b| const_byte(&mut bld, b))
                    .collect();
                q.extend_from_slice(&label_byte);
                bld.oracle(q)
            })
            .collect();

        let (pc, shape) = bld.popcount_chain(&answers);
        let accept = bld.ge_const(&pc, self.threshold);
        let out = bld.and(valid, accept);
        (bld.finish(out), shape)
    }
}

pub struct SparseTableOracle {
    pub image_id: u16,
    pub predicted: Vec<u8>,
}

pub fn majority_direct(predicted: &[u8], masks: &[u32], label: u8, threshold: usize) -> bool {
    let votes = masks
        .iter()
        .filter(|&&m| predicted.get(m as usize) == Some(&label))
        .count();
    votes >= threshold
}

pub fn majority_direct_in_order(predicted: &[u8], r: usize, label: u8, threshold: usize) -> bool {
    let votes = predicted[..r].iter().filter(|&&p| p == label).count();
    votes >= threshold
}

impl Oracle for SparseTableOracle {
    fn query(&self, u: &BitSlice) -> bool {
        if u.len() != 64 || byte_at(u, 0) != SPARSE_TAG {
            return false;
        }
        let id = u16::from_le_bytes([byte_at(u, 1), byte_at(u, 2)]);
        let mask = u32::from_le_bytes([byte_at(u, 3), byte_at(u, 4), byte_at(u, 5), byte_at(u, 6)])
            as usize;
        let label = byte_at(u, 7);
        id == self.image_id && self.predicted.get(mask).is_some_and(|&p| p == label)
    }

    fn query_word(&self, u: u64, width: usize) -> bool {
        if width != 64 {
            return false;
        }
        let b: [u8; 8] = u.to_le_bytes().map(u8::reverse_bits);
        if b[0] != SPARSE_TAG {
            return false;
        }
        let id = u16::from_le_bytes([b[1], b[2]]);
        let mask = u32::from_le_bytes([b[3], b[4], b[5], b[6]]) as usize;
        let label = b[7];
        id == self.image_id && self.predicted.get(mask).is_some_and(|&p| p == label)
    }
}

const SPARSE_256: &str = include_str!("../../../benchmarks/artifacts/sparse_instance_R256.txt");
const SPARSE_1024: &str = include_str!("../../../benchmarks/artifacts/sparse_instance_R1024.txt");
const SPARSE_4096: &str = include_str!("../../../benchmarks/artifacts/sparse_instance_R4096.txt");

pub const MAX_COMPILED_VIEWS: usize = 4096;

pub const MIN_COMPILED_VIEWS: usize = 3;

fn kv<'a>(text: &'a str, key: &str) -> &'a str {
    text.lines()
        .find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix(' ')))
        .unwrap_or_else(|| panic!("instance file lacks `{key}`"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowError {
    Empty,
    BadChar { index: usize, ch: char },
}

impl std::fmt::Display for RowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RowError::Empty => write!(f, "the row is empty"),
            RowError::BadChar { index, ch } => {
                write!(f, "character {index} is {ch:?}, outside the row's alphabet")
            }
        }
    }
}

pub fn parse_digit_row(row: &str) -> Result<Vec<u8>, RowError> {
    if row.is_empty() {
        return Err(RowError::Empty);
    }
    row.chars()
        .enumerate()
        .map(|(index, ch)| {
            ch.to_digit(10)
                .filter(|_| ch.is_ascii_digit())
                .map(|d| d as u8)
                .ok_or(RowError::BadChar { index, ch })
        })
        .collect()
}

pub struct SparseInstance {
    pub image_id: u16,
    pub true_label: usize,
    pub r: usize,
    pub threshold: usize,
    pub votes: [usize; 10],
    pub d_min: usize,
    pub table: Vec<u8>,
}

pub fn sparse_instance(r: usize) -> SparseInstance {
    let text = match r {
        256 => SPARSE_256,
        1024 => SPARSE_1024,
        4096 => SPARSE_4096,
        _ => panic!("no prepared sparse instance at R = {r}"),
    };
    let votes_v: Vec<usize> = kv(text, "votes")
        .split(' ')
        .map(|v| v.parse().expect("vote count"))
        .collect();
    let inst = SparseInstance {
        image_id: kv(text, "image_id").parse().expect("image_id"),
        true_label: kv(text, "true_label").parse().expect("true_label"),
        r: kv(text, "r").parse().expect("r"),
        threshold: kv(text, "threshold").parse().expect("threshold"),
        votes: votes_v.try_into().expect("ten vote counts"),
        d_min: kv(text, "d_min").parse().expect("d_min"),
        table: parse_digit_row(kv(text, "table")).expect("the `table` row is ASCII digits only"),
    };
    assert_eq!(inst.r, r);
    assert_eq!(inst.table.len(), r, "one prediction per registered mask");
    inst
}

pub fn label_bits(y: usize) -> BitVec {
    BitVec::from_bools((0..LABEL_BITS).map(|k| (y >> k) & 1 == 1).collect())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate {
    pub votes: [usize; 10],
    pub threshold: usize,
    pub accepted: Option<usize>,
    pub flip_distances: [usize; 10],
    pub d_min: usize,
}

pub fn certify(table: &[u8], threshold: usize) -> Certificate {
    let mut votes = [0usize; 10];
    for &p in table {
        votes[usize::from(p)] += 1;
    }
    let mut flip_distances = [0usize; 10];
    for y in 0..10 {
        flip_distances[y] = if votes[y] >= threshold {
            votes[y] - threshold + 1
        } else {
            threshold - votes[y]
        };
    }
    let accepted = (0..10).find(|&y| votes[y] >= threshold);
    Certificate {
        votes,
        threshold,
        accepted,
        flip_distances,
        d_min: *flip_distances.iter().min().expect("ten labels"),
    }
}

#[derive(Clone, Debug)]
pub struct MnistInstance {
    pub image_id: u16,
    pub r: usize,
    pub threshold: usize,
    pub true_label: usize,
    pub no_label: usize,
    pub votes: [usize; 10],
    pub d_min: usize,
    pub eps_rob: Ratio,
    pub table: Vec<u8>,
}

impl MnistInstance {
    pub fn new(r: usize) -> Self {
        assert!(
            (MIN_COMPILED_VIEWS..=MAX_COMPILED_VIEWS).contains(&r),
            "the compiled-in MNIST table covers {MIN_COMPILED_VIEWS}..={MAX_COMPILED_VIEWS} views, not {r}"
        );
        let full = sparse_instance(MAX_COMPILED_VIEWS);
        let table = full.table[..r].to_vec();
        let threshold = r / 2 + 1;
        let cert = certify(&table, threshold);
        assert_eq!(
            cert.accepted,
            Some(full.true_label),
            "image {} must be correctly accepted at {r} views",
            full.image_id
        );
        assert!(cert.d_min >= 2, "a positive certified radius at {r} views");
        if let 256 | 1024 = r {
            let own = sparse_instance(r);
            assert_eq!(own.table, table, "the {r}-view table is a prefix");
            assert_eq!(own.votes, cert.votes);
            assert_eq!(own.d_min, cert.d_min);
            assert_eq!(own.threshold, threshold);
        }
        if r == MAX_COMPILED_VIEWS {
            assert_eq!(full.votes, cert.votes);
            assert_eq!(full.d_min, cert.d_min);
            assert_eq!(full.threshold, threshold);
        }
        let no_label = (0..10)
            .filter(|&y| y != full.true_label)
            .max_by_key(|&y| (cert.votes[y], usize::MAX - y))
            .expect("nine false labels");
        let eps_rob = Ratio::new(cert.d_min as u64 - 1, r as u64);
        assert_eq!(eps_rob.min_bad_set(r), cert.d_min, "the radius is tight");
        Self {
            image_id: full.image_id,
            r,
            threshold,
            true_label: full.true_label,
            no_label,
            votes: cert.votes,
            d_min: cert.d_min,
            eps_rob,
            table,
        }
    }

    pub fn family(&self) -> SparseMajorityFamily {
        let fam = SparseMajorityFamily::new(self.image_id, self.r);
        assert_eq!(fam.threshold, self.threshold);
        fam
    }

    pub fn build(&self) -> OracleCircuit {
        self.family().build()
    }

    pub fn build_with_shape(&self) -> (OracleCircuit, crate::circuit::ChainShape) {
        self.family().build_with_shape()
    }

    pub fn oracle(&self) -> SparseTableOracle {
        SparseTableOracle {
            image_id: self.image_id,
            predicted: self.table.clone(),
        }
    }

    pub fn ell(&self) -> usize {
        self.r
    }

    pub fn yes(&self) -> BitVec {
        label_bits(self.true_label)
    }

    pub fn no(&self) -> BitVec {
        label_bits(self.no_label)
    }

    pub fn honest_answers(&self, y: usize) -> Vec<bool> {
        self.table.iter().map(|&p| usize::from(p) == y).collect()
    }

    pub fn minimal_bad_set_size(&self) -> usize {
        self.threshold - self.votes[self.no_label]
    }

    pub fn local_cheat_answers(&self) -> Vec<bool> {
        let mut a = self.honest_answers(self.no_label);
        let mut left = self.minimal_bad_set_size();
        for v in a.iter_mut() {
            if left == 0 {
                break;
            }
            if !*v {
                *v = true;
                left -= 1;
            }
        }
        assert_eq!(left, 0, "enough false answers to flip");
        a
    }
}

#[cfg(test)]
mod direct_loop_tests {
    use super::*;

    #[test]
    fn majority_direct_agrees_with_the_interpreter() {
        for r in [16usize, 64, 256, 1024] {
            let inst = MnistInstance::new(r);
            let circuit = inst.build();
            let oracle = inst.oracle();
            let masks: Vec<u32> = (0..r).map(|m| m as u32).collect();
            let mut accepted = 0;
            for y in 0..10usize {
                let by_gates = circuit.eval(label_bits(y).as_slice(), &oracle);
                let by_loop = majority_direct(&inst.table, &masks, y as u8, inst.threshold);
                let by_scan = majority_direct_in_order(&inst.table, r, y as u8, inst.threshold);
                assert_eq!(by_gates, by_loop, "r = {r}, label {y}");
                assert_eq!(by_gates, by_scan, "r = {r}, label {y}");
                accepted += usize::from(by_gates);
            }
            assert_eq!(accepted, 1, "exactly the true label is accepted at r = {r}");
            assert!(majority_direct(
                &inst.table,
                &masks,
                inst.true_label as u8,
                inst.threshold
            ));
        }
    }
}
