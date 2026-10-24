use crate::deoracle::OracleSite;
use crate::layout::TraceLayout;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum Loc {
    Rows(Vec<Vec<usize>>),
    Sites(Arc<[OracleSite]>),
}

impl Loc {
    pub fn from_rows(rows: Vec<Vec<usize>>) -> Self {
        Loc::Rows(rows)
    }

    pub fn from_sites(sites: Arc<[OracleSite]>) -> Self {
        Loc::Sites(sites)
    }

    pub fn len(&self) -> usize {
        match self {
            Loc::Rows(r) => r.len(),
            Loc::Sites(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn row(&self, j: usize, layout: &TraceLayout) -> Vec<usize> {
        match self {
            Loc::Rows(r) => r[j].clone(),
            Loc::Sites(s) => row_of(&s[j], layout),
        }
    }

    pub fn b_tr(&self, layout: &TraceLayout) -> usize {
        match self {
            Loc::Rows(r) => b_tr(r),
            Loc::Sites(s) => {
                let _ = layout;
                s.iter().map(|s| s.query_wires.len() + 1).sum()
            }
        }
    }
}

fn row_of(s: &OracleSite, layout: &TraceLayout) -> Vec<usize> {
    let mut v: Vec<usize> = s
        .query_wires
        .iter()
        .map(|&w| layout.coord_of(w as usize))
        .collect();
    v.push(layout.coord_of(s.answer_wire));
    v
}

pub fn build(sites: &[OracleSite], layout: &TraceLayout) -> Vec<Vec<usize>> {
    sites
        .iter()
        .map(|s| {
            let mut v: Vec<usize> = s
                .query_wires
                .iter()
                .map(|&w| layout.coord_of(w as usize))
                .collect();
            v.push(layout.coord_of(s.answer_wire));
            v
        })
        .collect()
}

pub fn b_tr(loc: &[Vec<usize>]) -> usize {
    loc.iter().map(|v| v.len()).sum()
}
