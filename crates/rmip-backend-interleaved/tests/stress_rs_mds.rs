use rmip_backend_interleaved::code;
use rmip_field::{BinaryField, F128};
use rmip_opt::encode;
use rmip_opt::ntt;
use std::collections::HashSet;

fn rng(seed: u64) -> impl FnMut() -> F128 {
    let mut s = seed | 1;
    move || {
        let mut o = [0u64; 2];
        for w in o.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *w = s;
        }
        F128::new(((o[0] as u128) << 64) | o[1] as u128)
    }
}

fn solve(a: &[Vec<F128>], b: &[Vec<F128>]) -> Option<Vec<Vec<F128>>> {
    let k = a.len();
    let mut m: Vec<Vec<F128>> = (0..k)
        .map(|i| a[i].iter().chain(b[i].iter()).copied().collect())
        .collect();
    for col in 0..k {
        let piv = (col..k).find(|&row| m[row][col] != F128::ZERO)?;
        m.swap(col, piv);
        let inv = m[col][col].inv().unwrap();
        for v in m[col].iter_mut() {
            *v = v.mul(inv);
        }
        for row in 0..k {
            if row == col {
                continue;
            }
            let f = m[row][col];
            if f == F128::ZERO {
                continue;
            }
            let (pivot, target): (&[F128], &mut [F128]) = if row < col {
                let (lo, hi) = m.split_at_mut(col);
                (hi[0].as_slice(), lo[row].as_mut_slice())
            } else {
                let (lo, hi) = m.split_at_mut(row);
                (lo[col].as_slice(), hi[0].as_mut_slice())
            };
            for (t, p) in target.iter_mut().zip(pivot).skip(col) {
                *t = t.add(p.mul(f));
            }
        }
    }
    Some(m.into_iter().map(|row| row[k..].to_vec()).collect())
}

fn is_invertible(a: &[Vec<F128>]) -> bool {
    let k = a.len();
    let rhs: Vec<Vec<F128>> = (0..k).map(|_| vec![F128::ZERO]).collect();
    solve(a, &rhs).is_some()
}

fn generator_row(c: usize, log_k: usize) -> Vec<F128> {
    ntt::tables().novel_basis_at(c, log_k)
}

fn powers(x: F128, k: usize) -> Vec<F128> {
    let mut p = F128::ONE;
    (0..k)
        .map(|_| {
            let cur = p;
            p = p.mul(x);
            cur
        })
        .collect()
}

#[test]
fn domain_points_are_distinct() {
    for n_enc in [32usize, 256, 512, 1024, 4096, 32768] {
        let seen: HashSet<u128> = (0..n_enc)
            .map(|c| code::domain_point::<F128>(n_enc, c).to_u128())
            .collect();
        assert_eq!(seen.len(), n_enc, "n_enc = {n_enc}");
    }
    let dom = code::domain::<F128>(512);
    for (c, x) in dom.iter().enumerate() {
        assert_eq!(*x, code::domain_point(512, c));
        assert_eq!(x.to_u128(), c as u128);
    }
}

#[test]
fn the_transform_position_index_is_the_domain_point_bit_pattern() {
    let ntt = ntt::tables();
    for idx in [0usize, 1, 2, 3, 17, 255, 256, 4095, 65535, 1 << 20] {
        assert_eq!(
            ntt.novel_basis_at(idx, 1)[1],
            F128::new(idx as u128),
            "X̂_1(ω_{idx}) must be ω_{idx} itself"
        );
        assert_eq!(ntt.novel_basis_at(idx, 1)[0], F128::ONE);
    }
    for idx in 0..4096usize {
        assert_eq!(ntt.novel_basis_at(idx, 1)[1], F128::new(idx as u128));
    }
}

#[test]
fn the_code_is_reed_solomon_of_degree_less_than_k() {
    for (log_k, rate_inv) in [
        (3usize, 4usize),
        (5, 4),
        (6, 4),
        (7, 4),
        (8, 4),
        (7, 2),
        (7, 8),
    ] {
        let k = 1usize << log_k;
        let n_enc = k * rate_inv;

        let vand: Vec<Vec<F128>> = (0..k)
            .map(|c| powers(code::domain_point(n_enc, c), k))
            .collect();
        let b: Vec<Vec<F128>> = (0..k).map(|c| generator_row(c, log_k)).collect();

        let coef = solve(&vand, &b).expect("Vandermonde on distinct points is invertible");
        for (d, row) in coef.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                if d > j {
                    assert_eq!(*v, F128::ZERO, "deg X̂_{j} > {j} (k={k})");
                }
            }
        }
        for (j, row) in coef.iter().enumerate() {
            assert_ne!(row[j], F128::ZERO, "deg X̂_{j} < {j} (k={k})");
        }

        for c in 0..n_enc {
            let pow = powers(code::domain_point(n_enc, c), k);
            let got = generator_row(c, log_k);
            for (j, g) in got.iter().enumerate() {
                let want = (0..=j).fold(F128::ZERO, |a, d| a.add(coef[d][j].mul(pow[d])));
                assert_eq!(*g, want, "k={k} n_enc={n_enc} col={c} coord={j}");
            }
        }
    }
}

#[test]
fn the_degree_check_rejects_a_reversed_tensor_order() {
    let (log_k, n_enc) = (5usize, 128usize);
    let k = 1usize << log_k;
    let rev = |j: usize| -> usize {
        (0..log_k).fold(0usize, |a, t| a | (((j >> t) & 1) << (log_k - 1 - t)))
    };
    let vand: Vec<Vec<F128>> = (0..k)
        .map(|c| powers(code::domain_point(n_enc, c), k))
        .collect();
    let b: Vec<Vec<F128>> = (0..k)
        .map(|c| {
            let row = generator_row(c, log_k);
            (0..k).map(|j| row[rev(j)]).collect()
        })
        .collect();
    let coef = solve(&vand, &b).expect("Vandermonde is invertible");
    let triangular = (0..k).all(|j| (j + 1..k).all(|d| coef[d][j] == F128::ZERO));
    assert!(
        !triangular,
        "a reversed tensor order still produced a degree-ordered basis; \
         the positive test would then prove nothing"
    );
}

fn every_k_subset_is_invertible(log_k: usize, n_enc: usize, trials: usize) {
    let k = 1usize << log_k;
    let mut next = rng(0xa5a5_0000 + log_k as u64);
    let mut subsets: Vec<Vec<usize>> = Vec::new();
    subsets.push((n_enc - k..n_enc).collect());
    subsets.push((0..k).map(|i| i * (n_enc / k)).collect());
    for _ in 0..trials {
        let mut seen = HashSet::new();
        while seen.len() < k {
            seen.insert((next().to_u128() as usize) % n_enc);
        }
        subsets.push(seen.into_iter().collect());
    }
    for s in subsets {
        let g: Vec<Vec<F128>> = s.iter().map(|&c| generator_row(c, log_k)).collect();
        assert!(
            is_invertible(&g),
            "singular k-subset at k={k}, n_enc={n_enc}"
        );
    }
}

#[test]
fn any_k_columns_determine_the_message() {
    every_k_subset_is_invertible(7, 512, 8);
}

#[test]
#[ignore = "measurement"]
fn any_k_columns_determine_the_message_at_the_larger_shipped_shape() {
    every_k_subset_is_invertible(10, 4096, 2);
}

#[test]
fn nonzero_messages_have_at_most_k_minus_one_zero_columns() {
    for (k, n_enc) in [(8usize, 32usize), (128, 512), (1024, 4096)] {
        let dom = code::domain(n_enc);
        let mut next = rng(0x5eed_0000 + k as u64);
        for t in 0..4 {
            let row: Vec<F128> = (0..k)
                .map(|i| {
                    if t > 0 && i % 7 != 0 {
                        F128::ZERO
                    } else {
                        next()
                    }
                })
                .collect();
            if row.iter().all(|v| *v == F128::ZERO) {
                continue;
            }
            let cols = encode::encode_columns(&[row], &dom);
            let zeros = cols.iter().filter(|c| c[0] == F128::ZERO).count();
            assert!(
                zeros < k,
                "k={k} n_enc={n_enc} trial {t}: {zeros} zero symbols, MDS allows at most {}",
                k - 1
            );
        }
    }
}

#[test]
fn encoder_is_f_linear_at_the_shipped_shape() {
    let (m, n, n_enc) = (64usize, 128usize, 512usize);
    let dom = code::domain(n_enc);
    let mut next = rng(0xdeadbeef);
    let a: Vec<Vec<F128>> = (0..m).map(|_| (0..n).map(|_| next()).collect()).collect();
    let b: Vec<Vec<F128>> = (0..m).map(|_| (0..n).map(|_| next()).collect()).collect();
    let s = next();
    let combo: Vec<Vec<F128>> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| x.iter().zip(y).map(|(u, v)| u.add(s.mul(*v))).collect())
        .collect();
    let (ea, eb, ec) = (
        encode::encode_columns(&a, &dom),
        encode::encode_columns(&b, &dom),
        encode::encode_columns(&combo, &dom),
    );
    for (c, ((col_c, col_a), col_b)) in ec.iter().zip(&ea).zip(&eb).enumerate() {
        for (i, ((vc, va), vb)) in col_c.iter().zip(col_a).zip(col_b).enumerate() {
            assert_eq!(*vc, va.add(s.mul(*vb)), "col {c} row {i}");
        }
    }
}

#[test]
fn additivity_is_weaker_than_f_linearity() {
    let n = 64usize;
    let tw = F128::new(0x9e37_79b9_7f4a_7c15);
    let twisted = |m: &[F128]| -> Vec<F128> {
        let half = n / 2;
        let mut d = m.to_vec();
        for r in 0..half {
            let v = d[r + half];
            let u = d[r].add(v.square().mul(tw));
            d[r] = u;
            d[r + half] = v.add(u);
        }
        d
    };
    let mut next = rng(0xfeed_face);
    let a: Vec<F128> = (0..n).map(|_| next()).collect();
    let b: Vec<F128> = (0..n).map(|_| next()).collect();
    let sum: Vec<F128> = a.iter().zip(&b).map(|(x, y)| x.add(*y)).collect();
    let (ea, eb, es) = (twisted(&a), twisted(&b), twisted(&sum));
    for ((x, y), s) in ea.iter().zip(&eb).zip(&es) {
        assert_eq!(x.add(*y), *s, "additivity holds for the twist too");
    }
    let s = F128::new(3);
    let scaled: Vec<F128> = a.iter().map(|x| s.mul(*x)).collect();
    let e_scaled = twisted(&scaled);
    assert!(
        e_scaled.iter().zip(&ea).any(|(v, e)| *v != s.mul(*e)),
        "the twist would have been F-linear after all"
    );
}

#[test]
fn eval_column_aliases_off_domain_points_instead_of_rejecting_them() {
    let (m, n, n_enc) = (4usize, 16usize, 64usize);
    let dom = code::domain(n_enc);
    let mut next = rng(0x0bad_c0de);
    let rows: Vec<Vec<F128>> = (0..m).map(|_| (0..n).map(|_| next()).collect()).collect();
    let cols = encode::encode_columns(&rows, &dom);

    assert_eq!(
        encode::eval_column(&rows, F128::new((1u128 << 64) | 3)),
        cols[3],
        "an element outside the domain evaluated as column 3"
    );
    assert_eq!(
        code::eval_at(&rows[0], F128::new((1u128 << 64) | 3)),
        cols[3][0]
    );

    let far = encode::eval_column(&rows, F128::new(4095));
    assert_eq!(far.len(), m);
}

#[test]
fn the_three_evaluators_agree_at_the_shipped_shape() {
    let (m, n, n_enc) = (8usize, 128usize, 512usize);
    let dom = code::domain(n_enc);
    let mut next = rng(0x1234_5678);
    let rows: Vec<Vec<F128>> = (0..m).map(|_| (0..n).map(|_| next()).collect()).collect();
    let cols = encode::encode_columns(&rows, &dom);
    for c in (0..n_enc).step_by(37) {
        let x = code::domain_point(n_enc, c);
        let ec = encode::eval_column(&rows, x);
        for (i, (got, row)) in ec.iter().zip(&rows).enumerate() {
            assert_eq!(*got, cols[c][i], "eval_column col {c} row {i}");
            assert_eq!(code::eval_at(row, x), cols[c][i], "eval_at col {c} row {i}");
        }
    }
}
