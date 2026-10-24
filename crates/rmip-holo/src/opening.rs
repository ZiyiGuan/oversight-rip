use rmip_backend_interleaved::{code, draw_column, Shape};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::Symbol;
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::transcript::RawChallenger;

fn encode_row(values: &[F128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 16);
    for v in values {
        out.extend_from_slice(&v.to_be_bytes());
    }
    out
}

fn combine(flat: &[F128], shape: Shape, coeffs: &[F128]) -> Vec<F128> {
    let n = shape.n();
    let mut out = vec![F128::ZERO; n];
    for (i, row) in flat.chunks(n).enumerate() {
        let ci = coeffs[i];
        if ci == F128::ZERO {
            continue;
        }
        for (j, v) in row.iter().enumerate() {
            out[j] = out[j].add(ci.mul(*v));
        }
    }
    out
}

pub fn open(flat: &[F128], shape: Shape, point: &[F128], ch: &mut dyn RawChallenger) -> Vec<F128> {
    let coeffs = eq_table(&point[..shape.nu_row]);
    let row = combine(flat, shape, &coeffs);
    ch.record_ordinary(&encode_row(&row));
    row
}

pub fn columns(shape: Shape, ch: &mut dyn RawChallenger) -> Vec<usize> {
    (0..shape.queries())
        .map(|_| draw_column(ch, shape.n_enc))
        .collect()
}

pub fn check(
    codeword: &[Symbol<F128>],
    shape: Shape,
    point: &[F128],
    value: F128,
    row: &[F128],
    cols: &[usize],
) -> bool {
    if row.len() != shape.n() || codeword.len() != shape.n_enc {
        return false;
    }
    if point.len() != shape.nu_row + shape.nu_col {
        return false;
    }
    let coeffs = eq_table(&point[..shape.nu_row]);
    for &col in cols {
        let Some(sym) = codeword.get(col) else {
            return false;
        };
        if sym.0.len() != shape.m() {
            return false;
        }
        let dot = coeffs
            .iter()
            .zip(&sym.0)
            .map(|(a, b)| a.mul(*b))
            .fold(F128::ZERO, |x, y| x.add(y));
        if dot != code::eval_at(row, F128::new(col as u128)) {
            return false;
        }
    }
    eval_mle(row, &point[shape.nu_row..]) == value
}
