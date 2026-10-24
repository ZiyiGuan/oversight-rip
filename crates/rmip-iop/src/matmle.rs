use crate::mle::eq_table;
use rmip_field::{BinaryField, F128};
use rmip_frontend::r1cs::SparseF2;

pub fn mul_vec(m: &SparseF2, z: &[F128]) -> Vec<F128> {
    assert_eq!(z.len(), m.cols, "vector length must match column count");
    let mut out = vec![F128::ZERO; m.rows];
    for &(r, c) in &m.nz {
        out[r] = out[r].add(z[c]);
    }
    out
}

pub fn eval(m: &SparseF2, eq_row: &[F128], eq_col: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for &(i, j) in &m.nz {
        acc = acc.add(eq_row[i].mul(eq_col[j]));
    }
    acc
}

pub fn eval_all(
    a: &SparseF2,
    b: &SparseF2,
    c: &SparseF2,
    r_row: &[F128],
    r_col: &[F128],
) -> (F128, F128, F128) {
    let eq_row = eq_table(r_row);
    let eq_col = eq_table(r_col);
    (
        eval(a, &eq_row, &eq_col),
        eval(b, &eq_row, &eq_col),
        eval(c, &eq_row, &eq_col),
    )
}

pub fn row_restriction(m: &SparseF2, eq_row: &[F128], cols: usize) -> Vec<F128> {
    let mut out = vec![F128::ZERO; cols];
    for &(i, j) in &m.nz {
        out[j] = out[j].add(eq_row[i]);
    }
    out
}
