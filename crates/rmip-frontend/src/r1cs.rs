use crate::deoracle::OrdinaryCircuit;
use crate::layout::TraceLayout;

#[derive(Clone, Debug, Default)]
pub struct SparseF2 {
    pub rows: usize,
    pub cols: usize,
    pub nz: Vec<(usize, usize)>,
}

impl SparseF2 {
    pub fn mul_vec(&self, z: &[bool]) -> Vec<bool> {
        assert_eq!(z.len(), self.cols, "vector length must match column count");
        let mut out = vec![false; self.rows];
        for &(r, c) in &self.nz {
            out[r] ^= z[c];
        }
        out
    }

    pub fn nnz(&self) -> usize {
        self.nz.len()
    }

    pub fn contains(&self, r: usize, c: usize) -> bool {
        self.nz.binary_search(&(r, c)).is_ok()
    }
}

#[derive(Clone, Debug)]
pub struct SparseF2Builder {
    rows: usize,
    cols: usize,
    terms: Vec<(usize, usize)>,
}

impl SparseF2Builder {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            terms: Vec::new(),
        }
    }

    pub fn add_term(&mut self, r: usize, c: usize) {
        debug_assert!(r < self.rows && c < self.cols, "({r},{c}) out of bounds");
        self.terms.push((r, c));
    }

    pub fn add_form(&mut self, r: usize, cols: impl IntoIterator<Item = usize>) {
        for c in cols {
            self.add_term(r, c);
        }
    }

    pub fn set_rows(&mut self, rows: usize) {
        assert!(rows >= self.rows);
        self.rows = rows;
    }

    pub fn finish(mut self) -> SparseF2 {
        self.terms.sort_unstable();
        let mut nz: Vec<(usize, usize)> = Vec::with_capacity(self.terms.len());
        let mut i = 0;
        while i < self.terms.len() {
            let mut j = i;
            while j < self.terms.len() && self.terms[j] == self.terms[i] {
                j += 1;
            }
            if (j - i) % 2 == 1 {
                nz.push(self.terms[i]);
            }
            i = j;
        }
        SparseF2 {
            rows: self.rows,
            cols: self.cols,
            nz,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Booleanity {
    FromBinding,
    ClosedForm,
    AnswerBlockRows,
    ExplicitRows,
}

#[derive(Clone, Debug)]
pub struct R1csConfig {
    pub booleanity: Booleanity,
    pub force_square: bool,
}

impl Default for R1csConfig {
    fn default() -> Self {
        Self {
            booleanity: Booleanity::ClosedForm,
            force_square: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct R1cs {
    pub booleanity: Booleanity,
    pub a: SparseF2,
    pub b: SparseF2,
    pub c: SparseF2,
    pub m0: usize,
    pub mu: usize,
    pub m: usize,
    pub z: usize,
}

pub fn emit(dc: &OrdinaryCircuit, layout: &TraceLayout, cfg: &R1csConfig) -> R1cs {
    use crate::circuit::Gate;

    const ONE: usize = 0;
    let n_cols = layout.n_trace;
    let mut a = SparseF2Builder::new(0, n_cols);
    let mut b = SparseF2Builder::new(0, n_cols);
    let mut c = SparseF2Builder::new(0, n_cols);
    let mut row = 0usize;

    let mut new_row =
        |a: &mut SparseF2Builder, b: &mut SparseF2Builder, c: &mut SparseF2Builder| {
            let r = row;
            row += 1;
            a.set_rows(row);
            b.set_rows(row);
            c.set_rows(row);
            r
        };

    for (w, gate) in dc.gates.iter().enumerate() {
        let o = layout.coord_of(w);
        match gate {
            Gate::Input(_) | Gate::Answer(_) => {}
            Gate::Const(v) => {
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_term(r, o);
                b.add_term(r, ONE);
                if *v {
                    c.add_term(r, ONE);
                }
            }
            Gate::Not(p) => {
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_form(r, [layout.coord_of(*p), ONE]);
                b.add_term(r, ONE);
                c.add_term(r, o);
            }
            Gate::And(p, q) => {
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_term(r, layout.coord_of(*p));
                b.add_term(r, layout.coord_of(*q));
                c.add_term(r, o);
            }
            Gate::Or(p, q) => {
                let (cp, cq) = (layout.coord_of(*p), layout.coord_of(*q));
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_term(r, cp);
                b.add_term(r, cq);
                c.add_form(r, [o, cp, cq]);
            }
            Gate::Xor(p, q) => {
                let (cp, cq) = (layout.coord_of(*p), layout.coord_of(*q));
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_form(r, [cp, cq]);
                b.add_term(r, ONE);
                c.add_term(r, o);
            }
            Gate::Xor3(p, q, s) => {
                let (cp, cq, cs) = (
                    layout.coord_of(*p),
                    layout.coord_of(*q),
                    layout.coord_of(*s),
                );
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_form(r, [cp, cq, cs]);
                b.add_term(r, ONE);
                c.add_term(r, o);
            }
            Gate::Maj3(p, q, s) => {
                let (cp, cq, cs) = (
                    layout.coord_of(*p),
                    layout.coord_of(*q),
                    layout.coord_of(*s),
                );
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_form(r, [cp, cq]);
                b.add_form(r, [cp, cs]);
                c.add_form(r, [o, cp]);
            }
            Gate::Oracle { .. } => panic!("Oracle gate survived de-oraclization"),
        }
    }

    let r = new_row(&mut a, &mut b, &mut c);
    a.add_term(r, layout.output_coord());
    b.add_term(r, ONE);
    c.add_term(r, ONE);

    match cfg.booleanity {
        Booleanity::FromBinding | Booleanity::ClosedForm => {}
        Booleanity::AnswerBlockRows => {
            let free_sources = (0..dc.ell)
                .map(|j| layout.answer_coord(j))
                .chain(layout.tail_pad_range());
            for i in free_sources {
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_term(r, i);
                b.add_term(r, i);
                c.add_term(r, i);
            }
        }
        Booleanity::ExplicitRows => {
            for i in 0..n_cols {
                let r = new_row(&mut a, &mut b, &mut c);
                a.add_term(r, i);
                b.add_term(r, i);
                c.add_term(r, i);
            }
        }
    }

    let m0 = row;
    let mut m = m0.next_power_of_two().max(1);
    if cfg.force_square {
        m = m.max(n_cols);
    }
    let mu = m.trailing_zeros() as usize;
    a.set_rows(m);
    b.set_rows(m);
    c.set_rows(m);

    let (a, b, c) = (a.finish(), b.finish(), c.finish());
    let z = a.nnz() + b.nnz() + c.nnz();
    R1cs {
        booleanity: cfg.booleanity,
        a,
        b,
        c,
        m0,
        mu,
        m,
        z,
    }
}
