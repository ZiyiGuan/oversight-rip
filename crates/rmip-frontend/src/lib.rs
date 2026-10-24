pub mod bits;
pub mod circuit;
pub mod deoracle;
pub mod layout;
pub mod levels;
pub mod loc;
pub mod mnist;
pub mod oracle;
pub mod r1cs;
pub mod report;
pub mod trace;

pub use bits::BitVec;
pub use layout::TraceLayout;
pub use oracle::Oracle;
pub use r1cs::SparseF2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio {
    pub num: u64,
    pub den: u64,
}

impl Ratio {
    pub const fn new(num: u64, den: u64) -> Self {
        Self { num, den }
    }

    pub fn min_bad_set(&self, ell: usize) -> usize {
        (self.num as u128 * ell as u128 / self.den as u128) as usize + 1
    }
}

#[derive(Clone, Debug)]
pub struct Instance {
    pub layout: TraceLayout,
    pub loc: crate::loc::Loc,
    pub ell: usize,
    pub b_widths: Vec<usize>,
    pub b_max: usize,
    pub x: BitVec,
    pub eps_rob: Ratio,
}

impl Instance {
    pub fn b_max(&self) -> usize {
        self.b_max
    }

    pub fn max_width(b_widths: &[usize]) -> usize {
        b_widths.iter().copied().max().unwrap_or(0)
    }

    pub fn loc_row(&self, j: usize) -> Vec<usize> {
        self.loc.row(j, &self.layout)
    }

    pub fn loc_rows(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        (0..self.loc.len()).map(move |j| self.loc_row(j))
    }

    pub fn b_tr(&self) -> usize {
        self.loc.b_tr(&self.layout)
    }
}
