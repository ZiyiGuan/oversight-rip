use crate::bits::{BitSlice, BitVec};
use crate::circuit::{Gate, OracleCircuit, WireId};

#[derive(Clone, Debug)]
pub struct OracleSite {
    pub occurrence: usize,
    pub query_wires: Vec<u32>,
    pub answer_wire: WireId,
}

#[derive(Clone, Debug)]
pub struct OrdinaryCircuit {
    pub n: usize,
    pub ell: usize,
    pub gates: Vec<Gate>,
    pub output: WireId,
}

impl OrdinaryCircuit {
    pub fn eval(&self, x: &BitSlice, a: &BitSlice) -> bool {
        let w = self.eval_all_wires(x, a);
        w[self.output]
    }

    pub fn eval_all_wires(&self, x: &BitSlice, a: &BitSlice) -> Vec<bool> {
        assert_eq!(x.len(), self.n, "input width");
        assert_eq!(a.len(), self.ell, "answer-vector length");
        let mut w = vec![false; self.gates.len()];
        for (i, g) in self.gates.iter().enumerate() {
            w[i] = match g {
                Gate::Input(k) => x[*k],
                Gate::Answer(j) => a[*j],
                Gate::Const(b) => *b,
                Gate::Not(p) => !w[*p],
                Gate::And(p, q) => w[*p] && w[*q],
                Gate::Or(p, q) => w[*p] || w[*q],
                Gate::Xor(p, q) => w[*p] ^ w[*q],
                Gate::Xor3(p, q, s) => w[*p] ^ w[*q] ^ w[*s],
                Gate::Maj3(p, q, s) => (w[*p] & w[*q]) | (w[*p] & w[*s]) | (w[*q] & w[*s]),
                Gate::Oracle { .. } => panic!("Oracle gate survived de-oraclization"),
            };
        }
        w
    }

    pub fn query_bundle(
        &self,
        sites: &[OracleSite],
        j: usize,
        x: &BitSlice,
        a: &BitSlice,
    ) -> BitVec {
        let w = self.eval_all_wires(x, a);
        BitVec::from_bools(
            sites[j]
                .query_wires
                .iter()
                .map(|&q| w[q as usize])
                .collect(),
        )
    }
}

pub fn deoraclize(c: &OracleCircuit) -> (OrdinaryCircuit, Vec<OracleSite>) {
    c.validate().expect("de-oraclizing an invalid circuit");
    let n = c.n;
    let ell = c.ell();

    let mut gates: Vec<Gate> = (0..n).map(Gate::Input).collect();
    gates.extend((0..ell).map(Gate::Answer));

    let mut remap = vec![usize::MAX; c.gates.len()];
    let mut sites: Vec<Option<OracleSite>> = vec![None; ell];

    for (i, g) in c.gates.iter().enumerate() {
        match g {
            Gate::Input(k) => remap[i] = *k,
            Gate::Oracle { occurrence, query } => {
                let answer_wire = n + *occurrence;
                remap[i] = answer_wire;
                sites[*occurrence] = Some(OracleSite {
                    occurrence: *occurrence,
                    query_wires: query
                        .iter()
                        .map(|&q| u32::try_from(remap[q]).expect("a de-oracled wire id fits u32"))
                        .collect(),
                    answer_wire,
                });
            }
            other => {
                let mapped = match other {
                    Gate::Const(b) => Gate::Const(*b),
                    Gate::Not(p) => Gate::Not(remap[*p]),
                    Gate::And(p, q) => Gate::And(remap[*p], remap[*q]),
                    Gate::Or(p, q) => Gate::Or(remap[*p], remap[*q]),
                    Gate::Xor(p, q) => Gate::Xor(remap[*p], remap[*q]),
                    Gate::Xor3(p, q, s) => Gate::Xor3(remap[*p], remap[*q], remap[*s]),
                    Gate::Maj3(p, q, s) => Gate::Maj3(remap[*p], remap[*q], remap[*s]),
                    Gate::Input(_) | Gate::Oracle { .. } | Gate::Answer(_) => unreachable!(),
                };
                gates.push(mapped);
                remap[i] = gates.len() - 1;
            }
        }
    }

    let dc = OrdinaryCircuit {
        n,
        ell,
        gates,
        output: remap[c.output],
    };
    (
        dc,
        sites
            .into_iter()
            .map(|s| s.expect("every occurrence has a site"))
            .collect(),
    )
}

pub fn bad_set<O: crate::oracle::Oracle>(
    dc: &OrdinaryCircuit,
    sites: &[OracleSite],
    x: &BitSlice,
    a: &BitSlice,
    o: &O,
) -> Vec<usize> {
    let w = dc.eval_all_wires(x, a);
    sites
        .iter()
        .filter(|s| {
            let u: Vec<bool> = s.query_wires.iter().map(|&q| w[q as usize]).collect();
            o.query(&u) != a[s.occurrence]
        })
        .map(|s| s.occurrence)
        .collect()
}
