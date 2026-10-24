use crate::ntt::NttField;

pub fn encode_columns<F: NttField>(rows: &[Vec<F>], dom: &[F]) -> Vec<Vec<F>> {
    let m = rows.len();
    if m == 0 {
        return dom.iter().map(|_| Vec::new()).collect();
    }
    let n = rows[0].len();
    let n_enc = dom.len();
    debug_assert!(rows.iter().all(|r| r.len() == n), "ragged matrix");
    debug_assert!(
        n_enc.is_power_of_two() && n_enc >= n,
        "n_enc must be a power of two >= n"
    );
    debug_assert!(
        dom.iter()
            .enumerate()
            .all(|(c, x)| x.to_bits() == c as u128),
        "encode_columns is the additive NTT: its domain is fixed at 0, 1, 2, ... \
         and `dom` must be those n_enc points, in that order"
    );

    let mut data = vec![F::ZERO; m * n_enc];

    let fast = n > 0 && n.is_power_of_two() && n_enc.is_multiple_of(n);
    let start_layer = if fast {
        (n_enc / n).trailing_zeros() as usize
    } else {
        0
    };
    let replicas = if fast { n_enc / n } else { 1 };
    for (i, r) in rows.iter().enumerate() {
        for (k, v) in r.iter().enumerate() {
            for t in 0..replicas {
                data[(t * n + k) * m + i] = *v;
            }
        }
    }

    crate::ntt::AdditiveNtt::<F>::standard(n_enc.trailing_zeros() as usize)
        .forward_interleaved_from_layer(&mut data, m, start_layer);

    data.chunks(m).map(<[F]>::to_vec).collect()
}

pub fn eval_column<F: NttField>(rows: &[Vec<F>], x: F) -> Vec<F> {
    let m = rows.len();
    if m == 0 {
        return Vec::new();
    }
    let n = rows[0].len();
    debug_assert!(rows.iter().all(|r| r.len() == n), "ragged matrix");

    let idx = x.to_bits() as usize;
    let log_n = n.next_power_of_two().trailing_zeros() as usize;
    let basis = F::ntt_tables().novel_basis_at(idx, log_n);

    rows.iter().map(|r| F::dot(r, &basis)).collect()
}

#[cfg(test)]
fn encode_rows<F: NttField>(rows: &[Vec<F>], dom: &[F]) -> Vec<Vec<F>> {
    let cols = encode_columns(rows, dom);
    (0..rows.len())
        .map(|i| cols.iter().map(|c| c[i]).collect())
        .collect()
}

#[cfg(test)]
fn domain<F: NttField>(n_enc: usize) -> Vec<F> {
    (0..n_enc).map(|i| F::from_bits(i as u128)).collect()
}

pub fn encode_columns_shifted<F: NttField>(
    rows: &[Vec<F>],
    offset: usize,
    n_enc: usize,
) -> Vec<Vec<F>> {
    let m = rows.len();
    if m == 0 {
        return (0..n_enc).map(|_| Vec::new()).collect();
    }
    let n = rows[0].len();
    assert!(rows.iter().all(|r| r.len() == n), "ragged matrix");
    assert!(
        n.is_power_of_two() && n_enc.is_power_of_two() && n_enc >= n,
        "a level's message length and codeword length are powers of two with n_enc >= n"
    );
    let log_d = n_enc.trailing_zeros() as usize;
    let start_layer = (n_enc / n).trailing_zeros() as usize;
    let replicas = n_enc / n;
    let mut data = vec![F::ZERO; m * n_enc];
    for (i, r) in rows.iter().enumerate() {
        for (k, v) in r.iter().enumerate() {
            for t in 0..replicas {
                data[(t * n + k) * m + i] = *v;
            }
        }
    }
    F::ntt_tables()
        .shifted(offset, log_d)
        .forward_interleaved_from_layer(&mut data, m, start_layer);
    data.chunks(m).map(<[F]>::to_vec).collect()
}

pub fn eval_column_shifted<F: NttField>(rows: &[Vec<F>], offset: usize, idx: usize) -> Vec<F> {
    let m = rows.len();
    if m == 0 {
        return Vec::new();
    }
    let n = rows[0].len();
    debug_assert!(rows.iter().all(|r| r.len() == n), "ragged matrix");
    let log_n = n.next_power_of_two().trailing_zeros() as usize;
    let basis = F::ntt_tables().shifted_basis_at(offset, idx, log_n);
    rows.iter().map(|row| F::dot(row, &basis)).collect()
}

const COLUMN_BUDGET_BYTES: usize = 1 << 20;

fn column_tile<F>(n: usize, columns: usize) -> usize {
    let per_element = columns.max(1) * std::mem::size_of::<F>().max(1);
    let fit = (COLUMN_BUDGET_BYTES / per_element)
        .max(1024)
        .next_power_of_two();
    if fit >= n {
        n.max(1)
    } else {
        fit
    }
}

pub fn eval_columns_shifted<F: NttField>(
    rows: &[Vec<F>],
    offset: usize,
    idxs: &[usize],
) -> Vec<Vec<F>> {
    let m = rows.len();
    if m == 0 {
        return idxs.iter().map(|_| Vec::new()).collect();
    }
    let n = rows[0].len();
    debug_assert!(rows.iter().all(|r| r.len() == n), "ragged matrix");
    let log_n = n.next_power_of_two().trailing_zeros() as usize;
    let bases: Vec<Vec<F>> = idxs
        .iter()
        .map(|&idx| F::ntt_tables().shifted_basis_at(offset, idx, log_n))
        .collect();
    let mut cols: Vec<Vec<F>> = bases.iter().map(|_| vec![F::ZERO; m]).collect();
    let tile = column_tile::<F>(n, bases.len());
    for start in (0..n).step_by(tile) {
        let end = (start + tile).min(n);
        for (i, row) in rows.iter().enumerate() {
            for (col, basis) in cols.iter_mut().zip(&bases) {
                col[i] = col[i].add(F::dot(&row[start..end], &basis[start..end]));
            }
        }
    }
    cols
}

pub fn eval_at_shifted<F: NttField>(row: &[F], offset: usize, idx: usize) -> F {
    let log_n = row.len().next_power_of_two().trailing_zeros() as usize;
    let basis = F::ntt_tables().shifted_basis_at(offset, idx, log_n);
    F::dot(row, &basis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmip_field::{BinaryField, F128};

    #[test]
    fn shifted_columns_agree_with_the_shifted_basis() {
        for offset in [0usize, 1, 3] {
            for (m, n, n_enc) in [
                (1usize, 4usize, 16usize),
                (3, 8, 32),
                (5, 16, 64),
                (64, 256, 512),
                (2, 1024, 2048),
            ] {
                let mut s = 0xDEAD_BEEF_CAFE_F00Du64 ^ (offset as u64 * 977);
                let rows: Vec<Vec<F128>> = (0..m)
                    .map(|_| {
                        (0..n)
                            .map(|_| {
                                s ^= s << 13;
                                s ^= s >> 7;
                                s ^= s << 17;
                                F128::from_parts(s, s.rotate_left(11))
                            })
                            .collect()
                    })
                    .collect();
                let cols = encode_columns_shifted(&rows, offset, n_enc);
                assert_eq!(cols.len(), n_enc);
                let picked = [0usize, n_enc - 1, 3 % n_enc, 3 % n_enc];
                assert_eq!(
                    eval_columns_shifted(&rows, offset, &picked),
                    picked
                        .iter()
                        .map(|&x| eval_column_shifted(&rows, offset, x))
                        .collect::<Vec<_>>(),
                    "one pass gives the per-column values, duplicates included"
                );
                for (x, col) in cols.iter().enumerate() {
                    assert_eq!(
                        col,
                        &eval_column_shifted(&rows, offset, x),
                        "offset={offset} x={x}"
                    );
                    for (i, row) in rows.iter().enumerate() {
                        assert_eq!(col[i], eval_at_shifted(row, offset, x));
                    }
                }
                if offset == 0 {
                    let dom: Vec<F128> = (0..n_enc).map(|c| F128::new(c as u128)).collect();
                    assert_eq!(
                        cols,
                        encode_columns(&rows, &dom),
                        "level 0 is the standard code"
                    );
                }
            }
        }
    }

    fn reference_encode(row: &[F128], dom: &[F128]) -> Vec<F128> {
        let log_n = row.len().next_power_of_two().trailing_zeros() as usize;
        let ntt = crate::ntt::tables();
        dom.iter()
            .map(|&x| {
                let basis = ntt.novel_basis_at(x.to_u128() as usize, log_n);
                row.iter()
                    .zip(&basis)
                    .fold(F128::ZERO, |a, (c, b)| a.add(c.mul(*b)))
            })
            .collect()
    }

    #[test]
    fn agrees_with_the_reference_encoder() {
        for (m, n, n_enc) in [
            (1usize, 1usize, 2usize),
            (3, 4, 16),
            (8, 8, 32),
            (64, 128, 512),
            (4, 8, 8),
            (4, 3, 16),
            (5, 6, 32),
        ] {
            let dom = domain(n_enc);
            let rows: Vec<Vec<F128>> = (0..m)
                .map(|i| {
                    (0..n)
                        .map(|k| F128::new((i as u128 + 1) * 7919 + k as u128 * 31))
                        .collect()
                })
                .collect();

            let mine = encode_rows(&rows, &dom);
            for (i, r) in rows.iter().enumerate() {
                assert_eq!(
                    mine[i],
                    reference_encode(r, &dom),
                    "shape ({m},{n},{n_enc}) row {i}"
                );
            }

            let cols = encode_columns(&rows, &dom);
            for c in 0..n_enc {
                for i in 0..m {
                    assert_eq!(cols[c][i], mine[i][c]);
                }
            }
        }
    }

    #[test]
    fn eval_column_agrees_with_the_full_encoding() {
        let dom = domain(64);
        let rows: Vec<Vec<F128>> = (0..16)
            .map(|i| {
                (0..16)
                    .map(|k| F128::new((i as u128 + 3) * 101 + k as u128))
                    .collect()
            })
            .collect();
        let all = encode_columns(&rows, &dom);
        for (c, &x) in dom.iter().enumerate() {
            assert_eq!(eval_column(&rows, x), all[c], "column {c}");
        }
        assert!(eval_column(&[], F128::ONE).is_empty());
    }

    #[test]
    fn eval_column_agrees_with_the_encoder_on_a_non_power_of_two_message() {
        for (m, n, n_enc) in [(4usize, 3usize, 16usize), (1, 5, 8), (5, 6, 32)] {
            let dom = domain(n_enc);
            let rows: Vec<Vec<F128>> = (0..m)
                .map(|i| {
                    (0..n)
                        .map(|k| F128::new((i as u128 + 2) * 6997 + k as u128 * 13))
                        .collect()
                })
                .collect();
            let all = encode_columns(&rows, &dom);
            for (c, &x) in dom.iter().enumerate() {
                assert_eq!(
                    eval_column(&rows, x),
                    all[c],
                    "shape ({m},{n},{n_enc}) column {c}"
                );
            }
            for (i, r) in rows.iter().enumerate() {
                let want = reference_encode(r, &dom);
                for (c, &x) in dom.iter().enumerate() {
                    assert_eq!(eval_column(&rows, x)[i], want[c], "row {i} column {c}");
                }
            }
        }
        assert_eq!(
            eval_column(&[Vec::new(), Vec::new()], F128::new(3)),
            vec![F128::ZERO; 2]
        );
    }

    #[test]
    fn degenerate_shapes() {
        let dom = domain(4);
        assert_eq!(encode_columns(&[], &dom).len(), 4);
        let one = vec![vec![F128::ONE]];
        for c in encode_columns(&one, &dom) {
            assert_eq!(c, vec![F128::ONE]);
        }
    }

    #[test]
    fn replicate_and_skip_matches_zero_padded_transform() {
        for (m, n, n_enc) in [
            (1usize, 1usize, 2usize),
            (3, 4, 16),
            (8, 8, 32),
            (64, 128, 512),
            (5, 2, 64),
            (4, 8, 8),
            (4, 3, 16),
            (5, 6, 32),
        ] {
            let rows: Vec<Vec<F128>> = (0..m)
                .map(|i| {
                    (0..n)
                        .map(|k| F128::new((i as u128 + 3) * 104_729 + k as u128 * 97))
                        .collect()
                })
                .collect();

            let mut want = vec![F128::ZERO; m * n_enc];
            for (i, r) in rows.iter().enumerate() {
                for (k, v) in r.iter().enumerate() {
                    want[k * m + i] = *v;
                }
            }
            crate::ntt::AdditiveNtt::standard(n_enc.trailing_zeros() as usize)
                .forward_interleaved(&mut want, m);
            let want: Vec<Vec<F128>> = want.chunks(m).map(<[F128]>::to_vec).collect();

            assert_eq!(
                encode_columns(&rows, &domain(n_enc)),
                want,
                "shape ({m},{n},{n_enc})"
            );
        }
    }

    #[test]
    fn the_skipped_layers_are_exactly_replication() {
        for (m, n, n_enc) in [(3usize, 8usize, 32usize), (1, 1, 2), (5, 2, 64), (4, 8, 64)] {
            let rows: Vec<Vec<F128>> = (0..m)
                .map(|i| {
                    (0..n)
                        .map(|k| F128::new((i as u128 + 1) * 31 + k as u128))
                        .collect()
                })
                .collect();

            let log_d = n_enc.trailing_zeros() as usize;
            let ntt = crate::ntt::AdditiveNtt::standard(log_d);
            let start_layer = (n_enc / n).trailing_zeros() as usize;

            let mut replicated = vec![F128::ZERO; m * n_enc];
            for (i, r) in rows.iter().enumerate() {
                for (k, v) in r.iter().enumerate() {
                    for t in 0..(n_enc / n) {
                        replicated[(t * n + k) * m + i] = *v;
                    }
                }
            }

            let mut padded = vec![F128::ZERO; m * n_enc];
            for (i, r) in rows.iter().enumerate() {
                for (k, v) in r.iter().enumerate() {
                    padded[k * m + i] = *v;
                }
            }
            for layer in 0..start_layer {
                let block_size = 1usize << (log_d - layer);
                let half = block_size >> 1;
                for block in 0..(1usize << layer) {
                    let tw = ntt.twiddle(layer, block);
                    let start = block * block_size * m;
                    for row in 0..half {
                        let top = start + row * m;
                        let bot = top + half * m;
                        for lane in 0..m {
                            let v = padded[bot + lane];
                            let u = padded[top + lane].add(v.mul(tw));
                            padded[top + lane] = u;
                            padded[bot + lane] = v.add(u);
                        }
                    }
                }
            }
            assert_eq!(
                padded, replicated,
                "shape ({m},{n},{n_enc}): the first {start_layer} layers are not replication"
            );

            let mut got = replicated;
            ntt.forward_interleaved_from_layer(&mut got, m, start_layer);
            let got: Vec<Vec<F128>> = got.chunks(m).map(<[F128]>::to_vec).collect();
            assert_eq!(
                got,
                encode_columns(&rows, &domain(n_enc)),
                "shape ({m},{n},{n_enc})"
            );
        }
    }

    #[test]
    fn batched_encoding_is_still_f_linear() {
        let dom = domain(32);
        let a: Vec<F128> = (0..8).map(|i| F128::new(i * 13 + 1)).collect();
        let b: Vec<F128> = (0..8).map(|i| F128::new(i * 29 + 4)).collect();
        let c = F128::new(0x5eed);
        let combo: Vec<F128> = a.iter().zip(&b).map(|(x, y)| x.add(c.mul(*y))).collect();
        let e = encode_rows(&[a, b, combo], &dom);
        for (j, _) in dom.iter().enumerate() {
            assert_eq!(e[2][j], e[0][j].add(c.mul(e[1][j])));
        }
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use rmip_field::{OpeningField, F64};

    #[test]
    #[ignore = "measurement"]
    fn encoder_parts_at_a_depth_zero_shape() {
        use std::time::Instant;
        let (m, n, n_enc) = (4096usize, 4096usize, 8192usize);
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut rows: Vec<Vec<F64>> = (0..m)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        F64::from_bits(u128::from(s))
                    })
                    .collect()
            })
            .collect();
        let t = Instant::now();
        let cols = encode_columns_shifted(&rows, 0, n_enc);
        println!(
            "encode_columns_shifted: {:.1} ms ({} cols)",
            t.elapsed().as_secs_f64() * 1e3,
            cols.len()
        );
        let t = Instant::now();
        let mut data = vec![F64::ZERO; m * n_enc];
        for (i, r) in rows.iter().enumerate() {
            for (k, v) in r.iter().enumerate() {
                for rep in 0..2 {
                    data[(rep * n + k) * m + i] = *v;
                }
            }
        }
        println!(
            "interleaved layout: {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        let t = Instant::now();
        F64::ntt_tables()
            .shifted(0, 13)
            .forward_interleaved_from_layer(&mut data, m, 1);
        println!(
            "interleaved NTT from layer 1: {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        let t = Instant::now();
        let out: Vec<Vec<F64>> = data.chunks(m).map(<[F64]>::to_vec).collect();
        println!(
            "split into columns: {:.1} ms ({})",
            t.elapsed().as_secs_f64() * 1e3,
            out.len()
        );
        let t = Instant::now();
        let ntt = F64::ntt_tables().shifted(0, 13);
        let mut enc_rows: Vec<Vec<F64>> = rows
            .iter()
            .map(|r| {
                let mut d = vec![F64::ZERO; n_enc];
                d[..n].copy_from_slice(r);
                d[n..].copy_from_slice(r);
                d
            })
            .collect();
        for d in enc_rows.iter_mut() {
            ntt.forward_interleaved_from_layer(d, 1, 1);
        }
        println!(
            "row-wise NTTs (num_ntts = 1): {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(enc_rows[7][123], cols[123][7]);
        let c = rows[1][2];
        let (a, b) = rows.split_at_mut(1);
        let t = Instant::now();
        for _ in 0..4096 {
            F64::axpy(&mut a[0], c, &b[0]);
        }
        println!(
            "axpy 2^24 elements: {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}
