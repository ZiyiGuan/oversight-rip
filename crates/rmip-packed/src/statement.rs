use std::collections::HashMap;

use rmip_frontend::bits::{BitSlice, BitVec};
use rmip_frontend::circuit::{Gate, OracleCircuit, WireId};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::Instance;

use crate::tree::Tree;

pub struct PackedStatement<'a> {
    pub inst: &'a Instance,
    pub circuit: &'a OracleCircuit,
    pub tree: Tree,
    pub threshold: u64,
    pub valid: bool,
    pub gate_index: Vec<u64>,
    pub plan: Option<QueryPlan>,
}

impl<'a> PackedStatement<'a> {
    pub fn new(
        inst: &'a Instance,
        circuit: &'a OracleCircuit,
        threshold: u64,
        valid: bool,
    ) -> Self {
        let views = inst.ell;
        let tree = Tree::new(views, 6, 6);
        let mut gate_index = vec![u64::MAX; views];
        for (i, g) in circuit.gates.iter().enumerate() {
            if let Gate::Oracle { occurrence, .. } = g {
                gate_index[*occurrence] = i as u64;
            }
        }
        assert!(
            gate_index.iter().all(|&i| i != u64::MAX),
            "every occurrence has a gate"
        );
        Self {
            inst,
            circuit,
            tree,
            threshold,
            valid,
            gate_index,
            plan: None,
        }
    }

    pub fn with_query_plan(mut self) -> Self {
        self.plan = Some(QueryPlan::new(self.circuit, &self.gate_index));
        self
    }

    pub fn views(&self) -> usize {
        self.inst.ell
    }
}

pub struct QueryPlan {
    pub width: usize,
    const_bits: Vec<u64>,
    patterns: Vec<Vec<(u8, u32)>>,
    runs: Vec<(u64, u32)>,
    var_wires: Vec<WireId>,
}

impl QueryPlan {
    pub fn new(circuit: &OracleCircuit, gate_index: &[u64]) -> Self {
        let mut width = 0usize;
        let mut narrowest = usize::MAX;
        let mut const_bits = Vec::with_capacity(gate_index.len());
        let mut patterns: Vec<Vec<(u8, u32)>> = Vec::new();
        let mut pattern_index: HashMap<Vec<(u8, u32)>, u32> = HashMap::new();
        let mut runs: Vec<(u64, u32)> = Vec::new();
        let mut var_wires: Vec<WireId> = Vec::new();
        let mut var_index: HashMap<WireId, u32> = HashMap::new();
        let mut vars: Vec<(u8, u32)> = Vec::new();
        for (j, &g) in gate_index.iter().enumerate() {
            let Gate::Oracle { query, .. } = &circuit.gates[g as usize] else {
                unreachable!("gate_index names oracle gates")
            };
            assert!(query.len() <= 64, "a query of at most 64 bits");
            width = width.max(query.len());
            narrowest = narrowest.min(query.len());
            let mut bits = 0u64;
            vars.clear();
            for (k, &w) in query.iter().enumerate() {
                match &circuit.gates[w] {
                    Gate::Const(true) => bits |= 1 << k,
                    Gate::Const(false) => {}
                    _ => {
                        let idx = *var_index.entry(w).or_insert_with(|| {
                            var_wires.push(w);
                            u32::try_from(var_wires.len() - 1).expect("variable wires fit u32")
                        });
                        vars.push((k as u8, idx));
                    }
                }
            }
            const_bits.push(bits);
            let pattern = match pattern_index.get(&vars) {
                Some(&p) => p,
                None => {
                    let p = u32::try_from(patterns.len()).expect("patterns fit u32");
                    patterns.push(vars.clone());
                    pattern_index.insert(vars.clone(), p);
                    p
                }
            };
            if runs.last().is_none_or(|&(_, p)| p != pattern) {
                runs.push((j as u64, pattern));
            }
        }
        runs.push((gate_index.len() as u64, u32::MAX));
        assert!(
            gate_index.is_empty() || narrowest == width,
            "every oracle query must have the same width ({narrowest} and {width} bits seen)"
        );
        Self {
            width,
            const_bits,
            patterns,
            runs,
            var_wires,
        }
    }

    fn var_values(&self, circuit: &OracleCircuit, x: &BitSlice) -> Vec<bool> {
        let mut memo: HashMap<WireId, bool> = HashMap::new();
        self.var_wires
            .iter()
            .map(|&w| wire_value(circuit, x, w, &mut memo))
            .collect()
    }

    fn pattern_bits(&self, values: &[bool]) -> Vec<u64> {
        self.patterns
            .iter()
            .map(|p| {
                p.iter().fold(0u64, |acc, &(k, idx)| {
                    acc | (u64::from(values[idx as usize]) << k)
                })
            })
            .collect()
    }

    pub fn word(&self, j: usize, values: &[bool]) -> u64 {
        let run = self.runs.partition_point(|&(start, _)| start <= j as u64) - 1;
        let pattern = &self.patterns[self.runs[run].1 as usize];
        pattern.iter().fold(self.const_bits[j], |acc, &(k, idx)| {
            acc | (u64::from(values[idx as usize]) << k)
        })
    }

    fn answer_words<O: Oracle>(
        &self,
        circuit: &OracleCircuit,
        x: &BitSlice,
        oracle: &O,
    ) -> Vec<u64> {
        let values = self.var_values(circuit, x);
        let bits = self.pattern_bits(&values);
        let gates = self.const_bits.len();
        let mut out = vec![0u64; gates.div_ceil(64)];
        for pair in self.runs.windows(2) {
            let (start, pattern) = (pair[0].0 as usize, pair[0].1 as usize);
            let end = pair[1].0 as usize;
            let var = bits[pattern];
            let mut j = start;
            while j < end {
                let stop = end.min((j | 63) + 1);
                let mut word = out[j / 64];
                for (jj, &c) in (j..stop).zip(&self.const_bits[j..stop]) {
                    word |= u64::from(oracle.query_word(c | var, self.width)) << (jj % 64);
                }
                out[j / 64] = word;
                j = stop;
            }
        }
        out
    }
}

pub trait WireMemo {
    fn get(&self, w: WireId) -> Option<bool>;
    fn set(&mut self, w: WireId, v: bool);
}

impl WireMemo for HashMap<WireId, bool> {
    fn get(&self, w: WireId) -> Option<bool> {
        HashMap::get(self, &w).copied()
    }
    fn set(&mut self, w: WireId, v: bool) {
        self.insert(w, v);
    }
}

pub fn query_bits(st: &PackedStatement<'_>, x: &BitSlice, j: usize) -> BitVec {
    let Gate::Oracle { query, .. } = &st.circuit.gates[st.gate_index[j] as usize] else {
        unreachable!("gate_index names oracle gates")
    };
    let mut memo: HashMap<WireId, bool> = HashMap::new();
    let bits: Vec<bool> = query
        .iter()
        .map(|&w| wire_value(st.circuit, x, w, &mut memo))
        .collect();
    BitVec::from_bools(bits)
}

pub fn packed_answer_words<O: Oracle>(
    st: &PackedStatement<'_>,
    x: &BitSlice,
    oracle: &O,
) -> Vec<u64> {
    match &st.plan {
        Some(plan) => plan.answer_words(st.circuit, x, oracle),
        None => {
            let (_, a) = st.circuit.eval_with_answers(x, oracle);
            let views = st.views();
            let mut out = vec![0u64; views.div_ceil(64)];
            for (j, w) in out.iter_mut().enumerate() {
                for r in 0..64.min(views - 64 * j) {
                    *w |= u64::from(a.get(64 * j + r)) << r;
                }
            }
            out
        }
    }
}

pub fn packed_answers<O: Oracle>(st: &PackedStatement<'_>, x: &BitSlice, oracle: &O) -> Vec<bool> {
    let words = packed_answer_words(st, x, oracle);
    (0..st.views())
        .map(|j| (words[j / 64] >> (j % 64)) & 1 == 1)
        .collect()
}

fn wire_value(c: &OracleCircuit, x: &BitSlice, w: WireId, memo: &mut impl WireMemo) -> bool {
    if let Some(v) = memo.get(w) {
        return v;
    }
    let v = match &c.gates[w] {
        Gate::Input(k) => x[*k],
        Gate::Const(b) => *b,
        Gate::Not(p) => !wire_value(c, x, *p, memo),
        Gate::And(p, q) => wire_value(c, x, *p, memo) && wire_value(c, x, *q, memo),
        Gate::Or(p, q) => wire_value(c, x, *p, memo) || wire_value(c, x, *q, memo),
        Gate::Xor(p, q) => wire_value(c, x, *p, memo) ^ wire_value(c, x, *q, memo),
        Gate::Xor3(p, q, s) => {
            wire_value(c, x, *p, memo) ^ wire_value(c, x, *q, memo) ^ wire_value(c, x, *s, memo)
        }
        Gate::Maj3(p, q, s) => {
            let (a, b, d) = (
                wire_value(c, x, *p, memo),
                wire_value(c, x, *q, memo),
                wire_value(c, x, *s, memo),
            );
            (a & b) | (a & d) | (b & d)
        }
        Gate::Answer(_) | Gate::Oracle { .. } => {
            panic!(
                "the query of an oracle gate depends on an oracle answer: the circuit is adaptive"
            )
        }
    };
    memo.set(w, v);
    v
}

pub fn relation_error(tree: &Tree, bits: usize) -> f64 {
    let kappa_log = bits.trailing_zeros() as usize;
    let f = 2f64.powi(bits as i32);
    let mut total = 0.0f64;
    let (_, total_log) = tree.layout(kappa_log);
    for (l, level) in tree.levels.iter().enumerate() {
        let n = (level.m() - 6) as f64;
        let k_log = level.block.k_log as f64;
        total += (3.0 * n + 127.0 + k_log + n + 2.0 * (level.m() - kappa_log) as f64 + 1.0) / f;
        if l >= 1 {
            total += tree.copy_coins(l) as f64 / f;
        }
    }
    total += (2.0 * total_log as f64 + 1.0) / f;
    total
}
