// Copyright 2024-2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Vendored from Flock (`crates/flock-core/src/ntt/additive_ntt_f128.rs`) at
// commit e636760, whose own header records that the iterative LCH skeleton
// with neighbours-last ordering is derived from binius64's
// `NeighborsLastReference` (https://github.com/binius-zk/binius64,
// `crates/math/src/ntt/reference.rs`).
//
// Taken: `generate_evals_from_subspace`, `span_get`, `AdditiveNttF128::{new,
// standard, log_domain_size, twiddle}`, `forward_transform_interleaved_scalar_
// from_layer`, `log2_pow2`. The SIMD kernels, the rayon paths, the A/B toggles
// and the inverse transform are NOT taken -- see the module docs.
//
// Changes: `F128::new(lo, hi)` -> `F128::from_parts`, `+`/`*`/`+=`/`*=` ->
// `BinaryField::{add, mul}`, and `.inv()` -> `.inv().expect(..)` because this
// crate's inverse returns `Option`. `novel_basis_at` is ours and is not from
// Flock. The arithmetic is otherwise unmodified.
use rmip_field::{BinaryField, OpeningField, F128, F64};

fn generate_evals_from_subspace<F: BinaryField>(basis: &[F]) -> Vec<Vec<F>> {
    let l = basis.len();
    let mut evals: Vec<Vec<F>> = Vec::with_capacity(l);
    evals.push(basis.to_vec());
    for i in 1..l {
        let mut row = Vec::with_capacity(l - i);
        for k in 1..evals[i - 1].len() {
            let v = evals[i - 1][k];
            row.push(v.mul(v.add(evals[i - 1][0])));
        }
        evals.push(row);
    }
    for row in evals.iter_mut() {
        let inv = row[0]
            .inv()
            .expect("W_i(beta_i) is nonzero for an F_2-basis");
        for v in row.iter_mut() {
            *v = v.mul(inv);
        }
    }
    evals
}

#[inline]
fn span_get<F: BinaryField>(basis: &[F], idx: usize) -> F {
    let mut acc = F::ZERO;
    for (j, &b) in basis.iter().enumerate() {
        if (idx >> j) & 1 == 1 {
            acc = acc.add(b);
        }
    }
    acc
}

#[inline]
fn log2_pow2(n: usize) -> usize {
    assert!(
        n.is_power_of_two() && n > 0,
        "length must be a positive power of 2"
    );
    n.trailing_zeros() as usize
}

#[derive(Clone, Debug)]
pub struct AdditiveNtt<F: BinaryField = F128> {
    evals: Vec<Vec<F>>,
}

impl<F: OpeningField> AdditiveNtt<F> {
    pub fn new(basis: &[F]) -> Self {
        assert!(
            !basis.is_empty(),
            "the additive NTT needs a basis of at least one element"
        );
        Self {
            evals: generate_evals_from_subspace(basis),
        }
    }

    pub fn standard(dim: usize) -> Self {
        assert!(
            (1..=64).contains(&dim) && dim <= F::KAPPA,
            "standard NTT requires 1 <= dim <= 64"
        );
        let basis: Vec<F> = (0..dim).map(|i| F::from_bits(1u128 << i)).collect();
        Self::new(&basis)
    }

    pub fn log_domain_size(&self) -> usize {
        self.evals.len()
    }

    pub fn twiddle(&self, layer: usize, block: usize) -> F {
        let v = &self.evals[self.log_domain_size() - layer - 1];
        span_get(&v[1..], block)
    }

    pub fn forward_interleaved(&self, data: &mut [F], num_ntts: usize) {
        self.forward_interleaved_from_layer(data, num_ntts, 0)
    }

    pub fn forward_interleaved_from_layer(
        &self,
        data: &mut [F],
        num_ntts: usize,
        start_layer: usize,
    ) {
        let log_d = log2_pow2(data.len() / num_ntts);
        for layer in start_layer..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let half = block_size >> 1;
            let stride = block_size * num_ntts;
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let start = block * stride;
                for row in 0..half {
                    let top = start + row * num_ntts;
                    let bot = top + half * num_ntts;
                    for lane in 0..num_ntts {
                        let v = data[bot + lane];
                        let u = data[top + lane].add(v.mul(twiddle));
                        data[top + lane] = u;
                        data[bot + lane] = v.add(u);
                    }
                }
            }
        }
    }

    pub fn novel_basis_at(&self, idx: usize, log_n: usize) -> Vec<F> {
        assert!(log_n <= self.log_domain_size());
        let mut basis = vec![F::ONE; 1usize << log_n];
        let mut span = 1usize;
        for i in 0..log_n {
            let w_i = span_get(&self.evals[i], idx >> i);
            for j in 0..span {
                basis[j + span] = basis[j].mul(w_i);
            }
            span <<= 1;
        }
        basis
    }
    pub fn w_hat(&self, i: usize, idx: usize) -> F {
        assert!(i < self.log_domain_size(), "no basis function Ŵ_{i}");
        span_get(&self.evals[i], idx)
    }

    pub fn shifted_basis_at(&self, offset: usize, idx: usize, log_n: usize) -> Vec<F> {
        assert!(offset + log_n <= self.log_domain_size());
        let mut basis = vec![F::ONE; 1usize << log_n];
        let mut span = 1usize;
        for i in 0..log_n {
            let w_i = span_get(&self.evals[offset + i], idx >> i);
            for j in 0..span {
                basis[j + span] = basis[j].mul(w_i);
            }
            span <<= 1;
        }
        basis
    }

    pub fn shifted(&self, offset: usize, dim: usize) -> AdditiveNtt<F> {
        assert!(offset + dim <= self.log_domain_size());
        AdditiveNtt {
            evals: self.evals[offset..offset + dim].to_vec(),
        }
    }
}

pub fn tables() -> &'static AdditiveNtt<F128> {
    F128::ntt_tables()
}

pub trait NttField: OpeningField {
    fn ntt_tables() -> &'static AdditiveNtt<Self>;
}

impl NttField for F128 {
    fn ntt_tables() -> &'static AdditiveNtt<Self> {
        static NTT: std::sync::OnceLock<AdditiveNtt<F128>> = std::sync::OnceLock::new();
        NTT.get_or_init(|| AdditiveNtt::standard(32))
    }
}

impl NttField for F64 {
    fn ntt_tables() -> &'static AdditiveNtt<Self> {
        static NTT: std::sync::OnceLock<AdditiveNtt<F64>> = std::sync::OnceLock::new();
        NTT.get_or_init(|| AdditiveNtt::standard(32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_basis_evaluates_what_the_transform_computes() {
        for (log_n, log_d) in [(2usize, 4usize), (3, 5), (5, 7), (6, 8)] {
            let (n, n_enc) = (1usize << log_n, 1usize << log_d);
            let ntt = AdditiveNtt::standard(log_d);

            let mut s = 0x243f_6a88_85a3_08d3u64;
            let row: Vec<F128> = (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    F128::from_parts(s, s.rotate_left(17))
                })
                .collect();

            let mut data = vec![F128::ZERO; n_enc];
            data[..n].copy_from_slice(&row);
            ntt.forward_interleaved(&mut data, 1);

            for (c, &want) in data.iter().enumerate() {
                let basis = ntt.novel_basis_at(c, log_n);
                let got = row
                    .iter()
                    .zip(&basis)
                    .fold(F128::ZERO, |a, (r, b)| a.add(r.mul(*b)));
                assert_eq!(got, want, "log_n={log_n} log_d={log_d} col={c}");
            }
        }
    }

    #[test]
    fn the_shifted_transform_computes_the_shifted_basis() {
        let tables = AdditiveNtt::standard(14);
        for offset in [0usize, 1, 2, 3, 5] {
            for (log_n, log_d) in [(2usize, 4usize), (3, 5), (4, 7), (6, 8)] {
                let (n, n_enc) = (1usize << log_n, 1usize << log_d);
                let sub = tables.shifted(offset, log_d);
                let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ (offset as u64) << 40;
                let row: Vec<F128> = (0..n)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        F128::from_parts(s, s.rotate_left(29))
                    })
                    .collect();
                let mut data = vec![F128::ZERO; n_enc];
                data[..n].copy_from_slice(&row);
                sub.forward_interleaved(&mut data, 1);
                for (c, &want) in data.iter().enumerate() {
                    let basis = tables.shifted_basis_at(offset, c, log_n);
                    let got = row
                        .iter()
                        .zip(&basis)
                        .fold(F128::ZERO, |a, (r, b)| a.add(r.mul(*b)));
                    assert_eq!(
                        got, want,
                        "offset={offset} log_n={log_n} log_d={log_d} col={c}"
                    );
                }
                if offset == 0 {
                    assert_eq!(
                        tables.shifted_basis_at(0, 5 % n_enc, log_n),
                        tables.novel_basis_at(5 % n_enc, log_n)
                    );
                }
            }
        }
    }

    #[test]
    fn even_and_odd_halves_evaluate_the_whole() {
        let tables = AdditiveNtt::standard(14);
        for k in [0usize, 1, 2, 4] {
            for (log_n, log_d) in [(1usize, 3usize), (3, 5), (5, 7)] {
                let (n, n_enc) = (1usize << log_n, 1usize << log_d);
                let mut s = 0x2545_F491_4F6C_DD1Du64 ^ (k as u64) << 33;
                let w: Vec<F128> = (0..n)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        F128::from_parts(s.rotate_left(7), s)
                    })
                    .collect();
                let even: Vec<F128> = w.iter().step_by(2).copied().collect();
                let odd: Vec<F128> = w.iter().skip(1).step_by(2).copied().collect();
                let dot = |m: &[F128], b: &[F128]| {
                    m.iter()
                        .zip(b)
                        .fold(F128::ZERO, |a, (r, v)| a.add(r.mul(*v)))
                };
                for x in 0..n_enc {
                    let whole = dot(&w, &tables.shifted_basis_at(k, x, log_n));
                    let e = dot(&even, &tables.shifted_basis_at(k + 1, x >> 1, log_n - 1));
                    let o = dot(&odd, &tables.shifted_basis_at(k + 1, x >> 1, log_n - 1));
                    assert_eq!(
                        whole,
                        e.add(tables.w_hat(k, x).mul(o)),
                        "k={k} log_n={log_n} log_d={log_d} x={x}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_constant_basis_element_is_one() {
        let ntt = AdditiveNtt::<F128>::standard(6);
        for c in 0..64 {
            assert_eq!(ntt.novel_basis_at(c, 4)[0], F128::ONE, "col={c}");
        }
    }

    #[test]
    fn the_transform_is_linear() {
        let (log_d, n_enc) = (6usize, 64usize);
        let ntt = AdditiveNtt::standard(log_d);
        let mk = |seed: u64| -> Vec<F128> {
            let mut s = seed | 1;
            (0..n_enc)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    F128::from_parts(s, 0)
                })
                .collect()
        };
        let enc = |m: &[F128]| {
            let mut d = m.to_vec();
            ntt.forward_interleaved(&mut d, 1);
            d
        };
        let (a, b) = (mk(11), mk(22));
        let sum: Vec<F128> = a.iter().zip(&b).map(|(x, y)| x.add(*y)).collect();
        let (ea, eb, es) = (enc(&a), enc(&b), enc(&sum));
        for i in 0..n_enc {
            assert_eq!(ea[i].add(eb[i]), es[i], "additive: col={i}");
        }

        let c = F128::from_parts(0x9E37_79B9_7F4A_7C15, 0x0123_4567_89AB_CDEF);
        assert!(c != F128::ZERO && c != F128::ONE, "c must be a real scalar");
        let combo: Vec<F128> = a.iter().zip(&b).map(|(x, y)| x.add(c.mul(*y))).collect();
        let ec = enc(&combo);
        for i in 0..n_enc {
            assert_eq!(ea[i].add(c.mul(eb[i])), ec[i], "F-linear: col={i}");
        }
    }
}
