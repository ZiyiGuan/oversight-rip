use crate::circuit::{Gate, WireId};
use crate::deoracle::OrdinaryCircuit;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceLayout {
    pub n: usize,
    pub ell: usize,
    pub g: usize,
    pub k: usize,
    pub p: usize,
    pub nu: usize,
    pub n_trace: usize,
    wire_to_coord: Vec<usize>,
    output: WireId,
}

impl TraceLayout {
    pub fn new(dc: &OrdinaryCircuit) -> Self {
        let n = dc.n;
        let ell = dc.ell;
        let ordinary = dc.gates.len() - (n + ell);
        let g = ell + ordinary;

        let k = usize::BITS as usize - (n + 1 - 1).leading_zeros() as usize;
        let k = if (n + 1).is_power_of_two() {
            (n + 1).trailing_zeros() as usize
        } else {
            k
        };
        let p = 1usize << k;
        let need = p + g;
        let nu = if need.is_power_of_two() {
            need.trailing_zeros() as usize
        } else {
            usize::BITS as usize - need.leading_zeros() as usize
        };
        let nu = nu.max(k);
        let n_trace = 1usize << nu;

        let mut wire_to_coord = vec![usize::MAX; dc.gates.len()];
        let mut next_ordinary = p + ell;
        for (w, gate) in dc.gates.iter().enumerate() {
            wire_to_coord[w] = match gate {
                Gate::Input(i) => 1 + *i,
                Gate::Answer(j) => p + *j,
                _ => {
                    let c = next_ordinary;
                    next_ordinary += 1;
                    c
                }
            };
        }
        debug_assert!(
            next_ordinary <= n_trace,
            "trace overflow: {next_ordinary} > {n_trace}"
        );

        Self {
            n,
            ell,
            g,
            k,
            p,
            nu,
            n_trace,
            wire_to_coord,
            output: dc.output,
        }
    }

    pub fn coord_of(&self, w: WireId) -> usize {
        self.wire_to_coord[w]
    }

    pub fn answer_coord(&self, j: usize) -> usize {
        assert!(j < self.ell);
        self.p + j
    }

    pub fn output_coord(&self) -> usize {
        self.wire_to_coord[self.output]
    }

    pub fn used_end(&self) -> usize {
        self.p + self.g
    }

    pub fn public_range(&self) -> std::ops::Range<usize> {
        0..self.p
    }

    pub fn tail_pad_range(&self) -> std::ops::Range<usize> {
        (self.p + self.g)..self.n_trace
    }
}
