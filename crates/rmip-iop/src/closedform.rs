use crate::mateval::{HoloMatEval, MatEvalFault};
use crate::transcript::RawChallenger;
use rmip_field::{BinaryField, F128};
use rmip_frontend::circuit::{ChainShape, Gate, OracleCircuit};
use rmip_frontend::deoracle::OrdinaryCircuit;
use rmip_frontend::layout::TraceLayout;
use rmip_frontend::SparseF2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Family {
    pub row_base: usize,
    pub row_shift: usize,
    pub col_base: usize,
    pub col_shift: Option<usize>,
    pub log_count: usize,
}

#[derive(Clone, Debug)]
pub struct ChainMatEval {
    mu: usize,
    nu: usize,
    families: [Vec<Family>; 3],
    explicit: [Vec<(usize, usize)>; 3],
}

#[inline]
fn eq_bit(b: bool, r: F128) -> F128 {
    if b {
        r
    } else {
        r.add(F128::ONE)
    }
}

pub fn eq_at(idx: usize, r: &[F128]) -> F128 {
    let n = r.len();
    assert!(
        n == usize::BITS as usize || idx >> n == 0,
        "index out of range"
    );
    let mut acc = F128::ONE;
    for (t, &rt) in r.iter().enumerate() {
        let bit = (idx >> (n - 1 - t)) & 1 == 1;
        acc = acc.mul(eq_bit(bit, rt));
    }
    acc
}

pub fn affine_pair_sum(f: Family, mu: usize, nu: usize, r_row: &[F128], r_col: &[F128]) -> F128 {
    assert_eq!(r_row.len(), mu);
    assert_eq!(r_col.len(), nu);
    let (rs, cs) = (f.row_shift, f.col_shift.unwrap_or(usize::MAX));
    assert!(
        f.col_shift.is_none() || (cs >= rs && cs - rs <= 3),
        "column stride must be the row stride times 1, 2, 4 or 8"
    );
    let d = if f.col_shift.is_some() { cs - rs } else { 0 };
    let t = f.log_count;
    let positions = mu.max(nu) + rs.max(if f.col_shift.is_some() { cs } else { 0 }) + t + 2;
    let states = 4usize << d;
    let mut cur = vec![F128::ZERO; states];
    cur[0] = F128::ONE;
    for tau in 0..positions {
        let mut next = vec![F128::ZERO; states];
        let rb = (f.row_base >> tau) & 1;
        let cb = (f.col_base >> tau) & 1;
        let row_in_free = tau >= rs && tau - rs < t;
        for (state, &weight) in cur.iter().enumerate() {
            if weight == F128::ZERO {
                continue;
            }
            let carry_r = state & 1;
            let carry_c = (state >> 1) & 1;
            let delay = state >> 2;
            let choices: &[usize] = if row_in_free { &[0, 1] } else { &[0] };
            for &bit in choices {
                let sr = rb + bit + carry_r;
                let row_bit = sr & 1;
                let ncarry_r = sr >> 1;
                let col_in = if f.col_shift.is_some() {
                    if d == 0 {
                        bit
                    } else {
                        (delay >> (d - 1)) & 1
                    }
                } else {
                    0
                };
                let sc = cb + col_in + carry_c;
                let col_bit = sc & 1;
                let ncarry_c = sc >> 1;
                let mut w = weight;
                if tau < mu {
                    w = w.mul(eq_bit(row_bit == 1, r_row[mu - 1 - tau]));
                } else if row_bit == 1 {
                    continue;
                }
                if tau < nu {
                    w = w.mul(eq_bit(col_bit == 1, r_col[nu - 1 - tau]));
                } else if col_bit == 1 {
                    continue;
                }
                let ndelay = if d == 0 {
                    0
                } else {
                    ((delay << 1) | bit) & ((1 << d) - 1)
                };
                let ns = ncarry_r | (ncarry_c << 1) | (ndelay << 2);
                next[ns] = next[ns].add(w);
            }
        }
        cur = next;
    }
    cur[0]
}

fn dc_remap(c: &OracleCircuit) -> Vec<usize> {
    let n = c.n;
    let ell = c.ell();
    let mut ordinary = 0usize;
    c.gates
        .iter()
        .map(|g| match g {
            Gate::Input(k) => *k,
            Gate::Oracle { occurrence, .. } => n + occurrence,
            _ => {
                ordinary += 1;
                n + ell + ordinary - 1
            }
        })
        .collect()
}

impl ChainMatEval {
    pub fn new(
        c: &OracleCircuit,
        dc: &OrdinaryCircuit,
        layout: &TraceLayout,
        shape: &ChainShape,
        mu: usize,
    ) -> Self {
        let n = dc.n;
        let ell = dc.ell;
        let nu = layout.nu;
        let remap = dc_remap(c);
        let chain_first = remap[shape.first_gate];
        let chain_gates = shape.gates();
        let row_of = |w: usize| -> usize {
            assert!(w >= n + ell, "sources have no row");
            w - (n + ell)
        };
        let row0 = row_of(chain_first);
        let ordinary = dc.gates.len() - (n + ell);
        let m0 = ordinary + 1;
        assert!(m0 <= 1 << mu, "mu too small for {m0} rows");
        let zero_col = layout.coord_of(remap[shape.zero]);
        let coord_q = layout.p + ell;
        let k = shape.log_inputs;
        let r_padded = 1usize << k;
        let inputs: Vec<usize> = shape
            .inputs
            .iter()
            .map(|&w| layout.coord_of(remap[w]))
            .collect();
        assert_eq!(inputs.len(), r_padded);

        {
            let mut level_inputs: Vec<usize> = shape.inputs.iter().map(|&w| remap[w]).collect();
            let zero = remap[shape.zero];
            let mut g = chain_first;
            loop {
                let kw = level_inputs.len();
                let mut prev = zero;
                let mut carries = Vec::with_capacity(kw / 2);
                for i in 0..kw / 2 {
                    let (a, b) = (level_inputs[2 * i], level_inputs[2 * i + 1]);
                    assert!(
                        matches!(dc.gates[g], Gate::Xor3(p, q, s) if p == prev && q == a && s == b),
                        "chain gate {g} is not the predicted Xor3"
                    );
                    assert!(
                        matches!(dc.gates[g + 1], Gate::Maj3(p, q, s) if p == prev && q == a && s == b),
                        "chain gate {} is not the predicted Maj3",
                        g + 1
                    );
                    prev = g;
                    carries.push(g + 1);
                    g += 2;
                }
                if kw == 2 {
                    break;
                }
                level_inputs = carries;
            }
            assert_eq!(g, chain_first + chain_gates, "chain length");
        }

        let mut families: [Vec<Family>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut explicit: [Vec<(usize, usize)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        const A: usize = 0;
        const B: usize = 1;
        const C: usize = 2;

        let mut base_w = row0;
        for w in 0..k {
            let kw = r_padded >> w;
            let t = k - w - 1;
            let sum = Family {
                row_base: base_w,
                row_shift: 1,
                col_base: 0,
                col_shift: None,
                log_count: t,
            };
            let carry = Family {
                row_base: base_w + 1,
                ..sum
            };
            let with_col = |f: Family, col_base: usize, col_shift: usize| Family {
                col_base,
                col_shift: Some(col_shift),
                ..f
            };
            let prev_base = coord_q + base_w - 2;
            let (a_base, b_base, in_shift) = if w == 0 {
                (layout.p, layout.p + 1, 1)
            } else {
                let base_prev = base_w - 2 * kw;
                (coord_q + base_prev + 1, coord_q + base_prev + 3, 2)
            };
            families[A].push(with_col(sum, prev_base, 1));
            families[A].push(with_col(sum, a_base, in_shift));
            families[A].push(with_col(sum, b_base, in_shift));
            families[B].push(sum);
            families[C].push(with_col(sum, coord_q + base_w, 1));
            families[A].push(with_col(carry, prev_base, 1));
            families[A].push(with_col(carry, a_base, in_shift));
            families[B].push(with_col(carry, prev_base, 1));
            families[B].push(with_col(carry, b_base, in_shift));
            families[C].push(with_col(carry, coord_q + base_w + 1, 1));
            families[C].push(with_col(carry, prev_base, 1));
            for (m, row) in [
                (A, base_w),
                (A, base_w + 1),
                (B, base_w + 1),
                (C, base_w + 1),
            ] {
                explicit[m].push((row, prev_base));
                explicit[m].push((row, zero_col));
            }
            if w == 0 {
                for i in 0..kw / 2 {
                    let (ra, rb) = (base_w + 2 * i, base_w + 2 * i + 1);
                    let a_pred = layout.p + 2 * i;
                    let b_pred = layout.p + 2 * i + 1;
                    if inputs[2 * i] != a_pred {
                        for (m, row) in [(A, ra), (A, rb)] {
                            explicit[m].push((row, a_pred));
                            explicit[m].push((row, inputs[2 * i]));
                        }
                    }
                    if inputs[2 * i + 1] != b_pred {
                        for (m, row) in [(A, ra), (B, rb)] {
                            explicit[m].push((row, b_pred));
                            explicit[m].push((row, inputs[2 * i + 1]));
                        }
                    }
                }
            }
            base_w += kw;
        }
        assert_eq!(base_w, row0 + chain_gates);

        let one = 0usize;
        for w in (n + ell)..dc.gates.len() {
            if (chain_first..chain_first + chain_gates).contains(&w) {
                continue;
            }
            let r = row_of(w);
            let o = layout.coord_of(w);
            let co = |p: &usize| layout.coord_of(*p);
            match &dc.gates[w] {
                Gate::Input(_) | Gate::Answer(_) | Gate::Oracle { .. } => unreachable!(),
                Gate::Const(v) => {
                    explicit[A].push((r, o));
                    explicit[B].push((r, one));
                    if *v {
                        explicit[C].push((r, one));
                    }
                }
                Gate::Not(p) => {
                    explicit[A].push((r, co(p)));
                    explicit[A].push((r, one));
                    explicit[B].push((r, one));
                    explicit[C].push((r, o));
                }
                Gate::And(p, q) => {
                    explicit[A].push((r, co(p)));
                    explicit[B].push((r, co(q)));
                    explicit[C].push((r, o));
                }
                Gate::Or(p, q) => {
                    explicit[A].push((r, co(p)));
                    explicit[B].push((r, co(q)));
                    explicit[C].push((r, o));
                    explicit[C].push((r, co(p)));
                    explicit[C].push((r, co(q)));
                }
                Gate::Xor(p, q) => {
                    explicit[A].push((r, co(p)));
                    explicit[A].push((r, co(q)));
                    explicit[B].push((r, one));
                    explicit[C].push((r, o));
                }
                Gate::Xor3(p, q, s) => {
                    explicit[A].push((r, co(p)));
                    explicit[A].push((r, co(q)));
                    explicit[A].push((r, co(s)));
                    explicit[B].push((r, one));
                    explicit[C].push((r, o));
                }
                Gate::Maj3(p, q, s) => {
                    explicit[A].push((r, co(p)));
                    explicit[A].push((r, co(q)));
                    explicit[B].push((r, co(p)));
                    explicit[B].push((r, co(s)));
                    explicit[C].push((r, o));
                    explicit[C].push((r, co(p)));
                }
            }
        }
        let out_row = ordinary;
        explicit[A].push((out_row, layout.output_coord()));
        explicit[B].push((out_row, one));
        explicit[C].push((out_row, one));

        Self {
            mu,
            nu,
            families,
            explicit,
        }
    }

    pub fn eval(&self, r_row: &[F128], r_col: &[F128]) -> [F128; 3] {
        assert_eq!(r_row.len(), self.mu);
        assert_eq!(r_col.len(), self.nu);
        let mut out = [F128::ZERO; 3];
        let mut row_eq: std::collections::HashMap<usize, F128> = std::collections::HashMap::new();
        let mut col_eq: std::collections::HashMap<usize, F128> = std::collections::HashMap::new();
        for (m, slot) in out.iter_mut().enumerate() {
            let mut acc = F128::ZERO;
            for f in &self.families[m] {
                acc = acc.add(affine_pair_sum(*f, self.mu, self.nu, r_row, r_col));
            }
            for &(r, c) in &self.explicit[m] {
                let er = *row_eq.entry(r).or_insert_with(|| eq_at(r, r_row));
                let ec = *col_eq.entry(c).or_insert_with(|| eq_at(c, r_col));
                acc = acc.add(er.mul(ec));
            }
            *slot = acc;
        }
        out
    }

    pub fn sizes(&self) -> (usize, usize) {
        (
            self.families.iter().map(Vec::len).sum(),
            self.explicit.iter().map(Vec::len).sum(),
        )
    }
}

impl HoloMatEval for ChainMatEval {
    fn prove_frames(
        &self,
        _a: &SparseF2,
        _b: &SparseF2,
        _c: &SparseF2,
        r_row: &[F128],
        r_col: &[F128],
        _ch: &mut dyn RawChallenger,
    ) -> ([F128; 3], Vec<Vec<F128>>) {
        (self.eval(r_row, r_col), Vec::new())
    }

    fn verify_frames(
        &self,
        _next: &mut dyn FnMut() -> Vec<F128>,
        r_row: &[F128],
        r_col: &[F128],
        _ch: &mut dyn RawChallenger,
    ) -> Result<[F128; 3], MatEvalFault> {
        Ok(self.eval(r_row, r_col))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_affine_pair_sum_is_the_direct_sum() {
        let (mu, nu) = (7usize, 9usize);
        let r_row: Vec<F128> = (0..mu)
            .map(|i| F128::new(0x1234_5678 + 3 * i as u128))
            .collect();
        let r_col: Vec<F128> = (0..nu)
            .map(|i| F128::new(0x9ABC_DEF0 + 7 * i as u128))
            .collect();
        for (row_shift, col_shift) in [
            (0usize, Some(0usize)),
            (1, Some(1)),
            (1, Some(2)),
            (0, Some(3)),
            (1, None),
        ] {
            for (row_base, col_base, log_count) in
                [(3usize, 5usize, 3usize), (37, 61, 4), (0, 0, 0), (13, 2, 2)]
            {
                let f = Family {
                    row_base,
                    row_shift,
                    col_base,
                    col_shift,
                    log_count,
                };
                let mut direct = F128::ZERO;
                for i in 0..1usize << log_count {
                    let row = row_base + (i << row_shift);
                    let col = match col_shift {
                        Some(s) => col_base + (i << s),
                        None => col_base,
                    };
                    if row >= 1 << mu || col >= 1 << nu {
                        continue;
                    }
                    direct = direct.add(eq_at(row, &r_row).mul(eq_at(col, &r_col)));
                }
                let got = affine_pair_sum(f, mu, nu, &r_row, &r_col);
                assert_eq!(got, direct, "{f:?}");
            }
        }
    }
}
