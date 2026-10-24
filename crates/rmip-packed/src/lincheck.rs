use rmip_iop::transcript::RawChallenger;

use crate::circuit::transpose_64x64;
use crate::phases::Phases;
use crate::ringswitch::BitClaim;
use crate::subfield::SkipTables;
use crate::zerocheck::{coin, eq_table, record, PackedBits, ZerocheckClaim};
use crate::PackedField;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrix {
    pub k: usize,
    pub rows: Vec<Vec<usize>>,
}

impl SparseMatrix {
    pub fn identity(k: usize) -> Self {
        Self {
            k,
            rows: (0..k).map(|i| vec![i]).collect(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }
}

#[derive(Clone, Debug)]
pub struct BlockR1cs {
    pub m: usize,
    pub k_log: usize,
    pub a0: SparseMatrix,
    pub b0: SparseMatrix,
}

impl BlockR1cs {
    pub fn new(m: usize, k_log: usize, a0: SparseMatrix, b0: SparseMatrix) -> Self {
        assert!(k_log >= 6 && k_log <= m, "a block holds whole skip lanes");
        assert!(a0.k == 1 << k_log && b0.k == 1 << k_log);
        Self { m, k_log, a0, b0 }
    }

    pub fn k(&self) -> usize {
        1 << self.k_log
    }

    pub fn n_log(&self) -> usize {
        self.m - self.k_log
    }

    pub fn apply<F: PackedField>(&self, matrix: &SparseMatrix, z: &PackedBits<F>) -> PackedBits<F> {
        let k = self.k();
        let mut out = vec![false; 1 << self.m];
        for j_out in 0..1usize << self.n_log() {
            for (i_in, row) in matrix.rows.iter().enumerate() {
                let mut bit = false;
                for &j_in in row {
                    bit ^= z.bit(j_in + k * j_out);
                }
                out[i_in + k * j_out] = bit;
            }
        }
        PackedBits::from_bits(self.m, &out)
    }

    pub fn satisfies<F: PackedField>(&self, z: &PackedBits<F>) -> bool {
        let a = self.apply(&self.a0, z);
        let b = self.apply(&self.b0, z);
        (0..1usize << self.m).all(|i| (a.bit(i) & b.bit(i)) == z.bit(i))
    }
}

pub fn quirky_eq<F: PackedField>(tables: &SkipTables<F>, z_skip: F, rest: &[F]) -> Vec<F> {
    let lanes = tables.fold_weights(z_skip);
    let eq = eq_table(rest);
    let mut out = Vec::with_capacity(64 * eq.len());
    for &e in &eq {
        for &l in &lanes {
            out.push(l * e);
        }
    }
    out
}

fn column_weights<F: PackedField>(matrix: &SparseMatrix, qeq: &[F]) -> Vec<F> {
    let mut w = vec![F::ZERO; matrix.k];
    for (i, row) in matrix.rows.iter().enumerate() {
        for &j in row {
            w[j] += qeq[i];
        }
    }
    w
}

fn inner_product<F: PackedField>(x: &[F], y: &[F]) -> F {
    F::dot(x, y)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LincheckProof<F> {
    pub z_vec: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LincheckClaim<F> {
    pub r_skip: F,
    pub r_rest: Vec<F>,
    pub outer: Vec<F>,
    pub w: F,
}

impl<F: PackedField> LincheckClaim<F> {
    pub fn bit_claim(&self, tables: &SkipTables<F>) -> BitClaim<F> {
        let mut rho = self.r_rest.clone();
        rho.extend_from_slice(&self.outer);
        BitClaim::from_skip_point(tables, self.r_skip, &rho, self.w)
    }

    pub fn partial_evaluations(&self, z_vec: &[F]) -> Vec<F> {
        let per_word = F::BITS / 64;
        let in_word = per_word.trailing_zeros() as usize;
        let eq = eq_table(&self.r_rest[in_word..]);
        assert_eq!(
            eq.len() * F::BITS,
            z_vec.len(),
            "Z holds one block of k bits"
        );
        let mut s_hat = vec![F::ZERO; F::BITS];
        for (t, &e) in eq.iter().enumerate() {
            for p in 0..per_word {
                let base = 64 * (t * per_word + p);
                for l in 0..64 {
                    s_hat[64 * p + l] += e * z_vec[base + l];
                }
            }
        }
        s_hat
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LincheckError {
    BadLength { expected: usize, got: usize },
    AMismatch,
    BMismatch,
}

pub fn prove<F: PackedField>(
    tables: &SkipTables<F>,
    r1cs: &BlockR1cs,
    z: &PackedBits<F>,
    rho: &[F],
    ch: &mut dyn RawChallenger,
) -> (LincheckProof<F>, LincheckClaim<F>) {
    prove_with_phases(tables, r1cs, z, rho, ch, None)
}

pub fn prove_with_phases<F: PackedField>(
    tables: &SkipTables<F>,
    r1cs: &BlockR1cs,
    z: &PackedBits<F>,
    rho: &[F],
    ch: &mut dyn RawChallenger,
    phases: Option<&mut Phases>,
) -> (LincheckProof<F>, LincheckClaim<F>) {
    assert_eq!(rho.len(), r1cs.m - 6);
    let started = std::time::Instant::now();
    let (_, outer) = rho.split_at(r1cs.k_log - 6);
    let z_vec = fold_rows(r1cs.k(), z, outer);
    if let Some(p) = phases {
        p.add("lincheck fold", started.elapsed());
    }
    record(ch, &z_vec);
    let r_skip: F = coin(ch);
    let r_rest: Vec<F> = (0..r1cs.k_log - 6).map(|_| coin::<F>(ch)).collect();
    let w = inner_product(&quirky_eq(tables, r_skip, &r_rest), &z_vec);
    (
        LincheckProof { z_vec },
        LincheckClaim {
            r_skip,
            r_rest,
            outer: outer.to_vec(),
            w,
        },
    )
}

pub fn fold_rows<F: PackedField>(k: usize, z: &PackedBits<F>, outer: &[F]) -> Vec<F> {
    let blocks_per_row = k / 64;
    let rows = z.blocks() / blocks_per_row;
    assert_eq!(1usize << outer.len(), rows);
    let n = outer.len();
    let (lo, rest) = outer.split_at(n.min(3));
    let (mid, hi) = rest.split_at(rest.len().min(3));
    let eq_lo = eq_table(lo);
    let eq_mid = eq_table(mid);
    let eq_hi = eq_table(hi);
    let mut tables = [[F::ZERO; 256]; 8];
    for (b, table) in tables.iter_mut().enumerate() {
        let Some(&scale) = eq_mid.get(b) else { break };
        for (i, &w) in eq_lo.iter().enumerate() {
            let scaled = scale * w;
            let (low, high) = table.split_at_mut(1 << i);
            for (dst, &src) in high[..1 << i].iter_mut().zip(low.iter()) {
                *dst = src + scaled;
            }
        }
    }
    let mut acc = vec![F::Unreduced::default(); k];
    let mut tile = [0u64; 64];
    for (g, &w_hi) in eq_hi.iter().enumerate() {
        let in_group = 64.min(rows - 64 * g);
        for t in 0..blocks_per_row {
            for (j, slot) in tile.iter_mut().enumerate() {
                *slot = if j < in_group {
                    z.block((64 * g + j) * blocks_per_row + t)
                } else {
                    0
                };
            }
            transpose_64x64(&mut tile);
            let mut partial = [F::ZERO; 64];
            for (b, table) in tables.iter().enumerate() {
                for (o, &mask) in partial.iter_mut().zip(&tile) {
                    *o += table[((mask >> (8 * b)) & 0xFF) as usize];
                }
            }
            for (a, &p) in acc[64 * t..64 * t + 64].iter_mut().zip(&partial) {
                F::accumulate(a, w_hi, p);
            }
        }
    }
    F::reduce_all(&acc)
}

pub fn fold_rows_by_bits<F: PackedField>(k: usize, z: &PackedBits<F>, weights: &[F]) -> Vec<F> {
    let blocks_per_row = k / 64;
    let mut z_vec = vec![F::ZERO; k];
    for x in 0..z.blocks() {
        let weight = weights[x / blocks_per_row];
        let base = 64 * (x % blocks_per_row);
        let mut bits = z.block(x);
        while bits != 0 {
            let s = bits.trailing_zeros() as usize;
            z_vec[base + s] += weight;
            bits &= bits - 1;
        }
    }
    z_vec
}

pub fn verify<F: PackedField>(
    tables: &SkipTables<F>,
    r1cs: &BlockR1cs,
    zc: &ZerocheckClaim<F>,
    proof: &LincheckProof<F>,
    ch: &mut dyn RawChallenger,
) -> Result<LincheckClaim<F>, LincheckError> {
    let (z_skip, rho, v_a, v_b) = (zc.z, zc.rho.as_slice(), zc.a_eval, zc.b_eval);
    let k = r1cs.k();
    if proof.z_vec.len() != k {
        return Err(LincheckError::BadLength {
            expected: k,
            got: proof.z_vec.len(),
        });
    }
    let (inner_rest, outer) = rho.split_at(r1cs.k_log - 6);
    let qeq = quirky_eq(tables, z_skip, inner_rest);
    if inner_product(&column_weights(&r1cs.a0, &qeq), &proof.z_vec) != v_a {
        return Err(LincheckError::AMismatch);
    }
    if inner_product(&column_weights(&r1cs.b0, &qeq), &proof.z_vec) != v_b {
        return Err(LincheckError::BMismatch);
    }
    record(ch, &proof.z_vec);
    let r_skip: F = coin(ch);
    let r_rest: Vec<F> = (0..r1cs.k_log - 6).map(|_| coin::<F>(ch)).collect();
    let w = inner_product(&quirky_eq(tables, r_skip, &r_rest), &proof.z_vec);
    Ok(LincheckClaim {
        r_skip,
        r_rest,
        outer: outer.to_vec(),
        w,
    })
}
