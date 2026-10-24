use crate::mle::{eq_table, fold_in_place};
use rmip_field::{BinaryField, F128};

pub struct Zerocheck {
    tables: [Vec<F128>; 3],
    tau: Vec<F128>,
    round: usize,
}

impl Zerocheck {
    pub fn new(tau: &[F128], a: Vec<F128>, b: Vec<F128>, c: Vec<F128>) -> Self {
        assert_eq!(a.len(), b.len());
        assert_eq!(b.len(), c.len());
        assert_eq!(a.len(), 1 << tau.len());
        Self {
            tables: [a, b, c],
            tau: tau.to_vec(),
            round: 0,
        }
    }

    pub fn current_sum(&self) -> F128 {
        let eq = eq_table(&self.tau[self.round..]);
        (0..self.tables[0].len())
            .map(|h| {
                let (a, b, c) = (self.tables[0][h], self.tables[1][h], self.tables[2][h]);
                eq[h].mul(a.mul(b).add(c))
            })
            .fold(F128::ZERO, |acc, v| acc.add(v))
    }

    pub fn round_message(&mut self) -> Vec<F128> {
        let half = self.tables[0].len() / 2;
        let eq_rest = eq_table(&self.tau[self.round + 1..]);
        debug_assert_eq!(eq_rest.len(), half.max(1));

        let mut r0 = F128::ZERO;
        let mut lead = F128::ZERO;
        let mut r1 = F128::ZERO;
        let [a, b, c] = &self.tables;
        for v in 0..half {
            let w = eq_rest[v];
            let (la, ha) = (a[v], a[half + v]);
            let (lb, hb) = (b[v], b[half + v]);
            let (lc, hc) = (c[v], c[half + v]);
            r0 = r0.add(w.mul(la.mul(lb).add(lc)));
            r1 = r1.add(w.mul(ha.mul(hb).add(hc)));
            lead = lead.add(w.mul(la.add(ha).mul(lb.add(hb))));
        }
        let a0 = r0;
        let a1 = r1.add(a0).add(lead);
        vec![a1, lead]
    }

    pub fn receive_challenge(&mut self, challenge: F128) {
        for t in &mut self.tables {
            fold_in_place(t, challenge);
        }
        self.round += 1;
    }

    pub fn num_vars(&self) -> usize {
        self.tau.len() - self.round
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowClaims {
    pub v_a: F128,
    pub v_b: F128,
    pub v_c: F128,
}

impl Zerocheck {
    pub fn row_claims(&mut self) -> RowClaims {
        assert_eq!(
            self.round,
            self.tau.len(),
            "row claims exist after the last round"
        );
        RowClaims {
            v_a: self.tables[0][0],
            v_b: self.tables[1][0],
            v_c: self.tables[2][0],
        }
    }
}
