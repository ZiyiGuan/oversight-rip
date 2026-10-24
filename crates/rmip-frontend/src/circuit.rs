use crate::bits::{BitSlice, BitVec};
use crate::oracle::Oracle;

pub type WireId = usize;

pub type OccurrenceId = usize;

#[derive(Clone, Debug)]
pub enum Gate {
    Input(usize),
    Const(bool),
    Not(WireId),
    And(WireId, WireId),
    Or(WireId, WireId),
    Xor(WireId, WireId),
    Xor3(WireId, WireId, WireId),
    Maj3(WireId, WireId, WireId),
    Answer(usize),
    Oracle {
        occurrence: OccurrenceId,
        query: Vec<WireId>,
    },
}

#[derive(Clone, Debug)]
pub struct OracleCircuit {
    pub n: usize,
    pub gates: Vec<Gate>,
    pub output: WireId,
}

impl OracleCircuit {
    pub fn ell(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, Gate::Oracle { .. }))
            .count()
    }

    pub fn b_widths(&self) -> Vec<usize> {
        let mut w = vec![0; self.ell()];
        for g in &self.gates {
            if let Gate::Oracle { occurrence, query } = g {
                w[*occurrence] = query.len();
            }
        }
        w
    }

    pub fn b_max(&self) -> usize {
        self.b_widths().iter().copied().max().unwrap_or(0)
    }

    pub fn eval<O: Oracle>(&self, x: &BitSlice, o: &O) -> bool {
        self.eval_with_answers(x, o).0
    }

    pub fn eval_with_answers<O: Oracle>(&self, x: &BitSlice, o: &O) -> (bool, BitVec) {
        assert_eq!(x.len(), self.n, "input width");
        let mut w = vec![false; self.gates.len()];
        let mut a = BitVec::zeros(self.ell());
        for (i, g) in self.gates.iter().enumerate() {
            w[i] = match g {
                Gate::Input(k) => x[*k],
                Gate::Const(b) => *b,
                Gate::Answer(_) => panic!("Answer gate in an OracleCircuit"),
                Gate::Not(p) => !w[*p],
                Gate::And(p, q) => w[*p] && w[*q],
                Gate::Or(p, q) => w[*p] || w[*q],
                Gate::Xor(p, q) => w[*p] ^ w[*q],
                Gate::Xor3(p, q, s) => w[*p] ^ w[*q] ^ w[*s],
                Gate::Maj3(p, q, s) => (w[*p] & w[*q]) | (w[*p] & w[*s]) | (w[*q] & w[*s]),
                Gate::Oracle { occurrence, query } => {
                    let u: Vec<bool> = query.iter().map(|&q| w[q]).collect();
                    let ans = o.query(&u);
                    a.set(*occurrence, ans);
                    ans
                }
            };
        }
        (w[self.output], a)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut seen_occ = 0usize;
        for (i, g) in self.gates.iter().enumerate() {
            let ops: Vec<WireId> = match g {
                Gate::Input(k) => {
                    if *k >= self.n {
                        return Err(format!("gate {i}: Input({k}) exceeds n = {}", self.n));
                    }
                    vec![]
                }
                Gate::Const(_) => vec![],
                Gate::Answer(_) => return Err(format!("gate {i}: Answer in an OracleCircuit")),
                Gate::Not(p) => vec![*p],
                Gate::And(p, q) | Gate::Or(p, q) | Gate::Xor(p, q) => vec![*p, *q],
                Gate::Xor3(p, q, s) | Gate::Maj3(p, q, s) => vec![*p, *q, *s],
                Gate::Oracle { occurrence, query } => {
                    if *occurrence != seen_occ {
                        return Err(format!(
                            "gate {i}: occurrence {occurrence} out of order, expected {seen_occ}"
                        ));
                    }
                    seen_occ += 1;
                    if query.is_empty() {
                        return Err(format!("gate {i}: empty oracle query"));
                    }
                    query.clone()
                }
            };
            for p in ops {
                if p >= i {
                    return Err(format!("gate {i}: operand {p} is not strictly earlier"));
                }
            }
        }
        if self.output >= self.gates.len() {
            return Err(format!("output wire {} out of range", self.output));
        }
        if seen_occ == 0 {
            return Err("circuit has no oracle-gate occurrence (ell = 0)".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainShape {
    pub first_gate: WireId,
    pub log_inputs: usize,
    pub zero: WireId,
    pub inputs: Vec<WireId>,
}

impl ChainShape {
    pub fn gates(&self) -> usize {
        2 * ((1usize << self.log_inputs) - 1)
    }
}

pub struct CircuitBuilder {
    n: usize,
    gates: Vec<Gate>,
    next_occurrence: usize,
    zero: Option<WireId>,
    one: Option<WireId>,
}

impl CircuitBuilder {
    pub fn new(n: usize) -> Self {
        let gates = (0..n).map(Gate::Input).collect();
        Self {
            n,
            gates,
            next_occurrence: 0,
            zero: None,
            one: None,
        }
    }

    pub fn input(&self, i: usize) -> WireId {
        assert!(i < self.n);
        i
    }

    fn push(&mut self, g: Gate) -> WireId {
        self.gates.push(g);
        self.gates.len() - 1
    }

    pub fn constant(&mut self, b: bool) -> WireId {
        let slot = if b { self.one } else { self.zero };
        if let Some(w) = slot {
            return w;
        }
        let w = self.push(Gate::Const(b));
        if b {
            self.one = Some(w);
        } else {
            self.zero = Some(w);
        }
        w
    }

    pub fn not(&mut self, p: WireId) -> WireId {
        self.push(Gate::Not(p))
    }
    pub fn and(&mut self, p: WireId, q: WireId) -> WireId {
        self.push(Gate::And(p, q))
    }
    pub fn or(&mut self, p: WireId, q: WireId) -> WireId {
        self.push(Gate::Or(p, q))
    }
    pub fn xor(&mut self, p: WireId, q: WireId) -> WireId {
        self.push(Gate::Xor(p, q))
    }
    pub fn xor3(&mut self, p: WireId, q: WireId, s: WireId) -> WireId {
        self.push(Gate::Xor3(p, q, s))
    }
    pub fn maj3(&mut self, p: WireId, q: WireId, s: WireId) -> WireId {
        self.push(Gate::Maj3(p, q, s))
    }

    pub fn oracle(&mut self, query: Vec<WireId>) -> WireId {
        assert!(
            !query.is_empty(),
            "an oracle gate needs at least one query wire"
        );
        let occurrence = self.next_occurrence;
        self.next_occurrence += 1;
        self.push(Gate::Oracle { occurrence, query })
    }

    pub fn xor_mask(
        &mut self,
        bits: &[WireId],
        negated: &[WireId],
        mask: &crate::bits::BitVec,
    ) -> Vec<WireId> {
        (0..bits.len())
            .map(|k| if mask.get(k) { negated[k] } else { bits[k] })
            .collect()
    }

    pub fn full_adder(&mut self, a: WireId, b: WireId, c: WireId) -> (WireId, WireId) {
        let ab = self.xor(a, b);
        let sum = self.xor(ab, c);
        let and_ab = self.and(a, b);
        let and_abc = self.and(ab, c);
        let carry = self.or(and_ab, and_abc);
        (sum, carry)
    }

    pub fn half_adder(&mut self, a: WireId, b: WireId) -> (WireId, WireId) {
        let sum = self.xor(a, b);
        let carry = self.and(a, b);
        (sum, carry)
    }

    pub fn popcount(&mut self, bits: &[WireId]) -> Vec<WireId> {
        if bits.len() == 1 {
            return vec![bits[0]];
        }
        self.popcount_chain(bits).0
    }

    pub fn popcount_chain(&mut self, bits: &[WireId]) -> (Vec<WireId>, ChainShape) {
        assert!(
            bits.len() >= 2,
            "a popcount chain needs at least two inputs"
        );
        let zero = self.constant(false);
        let k = bits.len().next_power_of_two().trailing_zeros() as usize;
        let mut inputs = bits.to_vec();
        inputs.resize(1 << k, zero);
        let first_gate = self.gates.len();
        let mut out = Vec::with_capacity(k + 1);
        let mut level: Vec<WireId> = inputs.clone();
        loop {
            let kw = level.len();
            let mut prev = zero;
            let mut carries = Vec::with_capacity(kw / 2);
            for i in 0..kw / 2 {
                let (a, b) = (level[2 * i], level[2 * i + 1]);
                let s = self.xor3(prev, a, b);
                let cy = self.maj3(prev, a, b);
                prev = s;
                carries.push(cy);
            }
            out.push(prev);
            if kw == 2 {
                out.push(carries[0]);
                break;
            }
            level = carries;
        }
        (
            out,
            ChainShape {
                first_gate,
                log_inputs: k,
                zero,
                inputs,
            },
        )
    }

    pub fn ge_const(&mut self, s_lsb: &[WireId], t: usize) -> WireId {
        let width = s_lsb.len();
        if t == 0 {
            return self.constant(true);
        }
        if width < usize::BITS as usize && t >= (1usize << width) {
            return self.constant(false);
        }
        let mut gt = self.constant(false);
        let mut eq = self.constant(true);
        for i in (0..width).rev() {
            let s_i = s_lsb[i];
            let t_i = i < usize::BITS as usize && (t >> i) & 1 == 1;
            if t_i {
                eq = self.and(eq, s_i);
            } else {
                let inc = self.and(eq, s_i);
                gt = self.or(gt, inc);
                let ns = self.not(s_i);
                eq = self.and(eq, ns);
            }
        }
        self.or(gt, eq)
    }

    pub fn majority(&mut self, bits: &[WireId]) -> WireId {
        assert!(bits.len() % 2 == 1, "MAJ needs an odd arity to avoid ties");
        let pc = self.popcount(bits);
        self.ge_const(&pc, bits.len().div_ceil(2))
    }

    pub fn finish(self, output: WireId) -> OracleCircuit {
        let c = OracleCircuit {
            n: self.n,
            gates: self.gates,
            output,
        };
        c.validate()
            .expect("CircuitBuilder produced an invalid circuit");
        c
    }
}
