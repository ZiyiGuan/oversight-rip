use crate::bits::{BitSlice, BitVec};
use crate::deoracle::OrdinaryCircuit;
use crate::layout::TraceLayout;

pub fn witness_trace(
    dc: &OrdinaryCircuit,
    layout: &TraceLayout,
    x: &BitSlice,
    a: &BitSlice,
) -> BitVec {
    let w = dc.eval_all_wires(x, a);
    let mut z = BitVec::zeros(layout.n_trace);
    z.set(0, true);
    for (wire, value) in w.iter().enumerate() {
        z.set(layout.coord_of(wire), *value);
    }
    z
}

pub fn public_block(layout: &TraceLayout, x: &BitSlice) -> BitVec {
    assert_eq!(x.len(), layout.n);
    let mut p = BitVec::zeros(layout.p);
    p.set(0, true);
    for (i, b) in x.iter().enumerate() {
        p.set(1 + i, *b);
    }
    p
}

pub fn read_block(z: &BitSlice, loc_j: &[usize]) -> (BitVec, bool) {
    let (query, answer) = loc_j.split_at(loc_j.len() - 1);
    (
        BitVec::from_bools(query.iter().map(|&i| z[i]).collect()),
        z[answer[0]],
    )
}
