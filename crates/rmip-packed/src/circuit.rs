use crate::lincheck::SparseMatrix;

pub type Slot = usize;

pub type Form = Vec<Slot>;

pub fn form_add(a: &Form, b: &Form) -> Form {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

const INPUT_BASE: Slot = 1 << 40;

#[derive(Clone, Debug)]
enum Kind {
    One,
    Input,
    Zero,
    Product,
    Output,
}

#[derive(Clone, Debug)]
enum Node {
    Slot(Slot),
    Xor(usize, usize),
    Sum(Vec<Slot>),
}

#[derive(Clone, Debug)]
enum Step {
    Node(usize),
    Product { slot: Slot, a: usize, b: usize },
    Output { slot: Slot, a: usize },
}

#[derive(Clone, Debug)]
enum Op {
    Load { dst: u32, slot: u32 },
    Xor { dst: u32, a: u32, b: u32 },
    Sum { dst: u32, start: u32, len: u32 },
    Product { slot: u32, a: u32, b: u32 },
    Output { slot: u32, a: u32 },
}

#[derive(Clone, Copy, Debug)]
enum Rows {
    Slot,
    Nodes(u32, u32),
    Output(u32),
}

#[derive(Clone, Debug, Default)]
struct Program {
    ops: Vec<Op>,
    sum_slots: Vec<u32>,
    rows: Vec<Rows>,
    nodes: usize,
    input_slots: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct BlockBuilder {
    a_rows: Vec<Form>,
    b_rows: Vec<Form>,
    kinds: Vec<Kind>,
    inputs: usize,
    nodes: Vec<Node>,
    steps: Vec<Step>,
    node_of: std::collections::HashMap<Form, usize>,
    slot_nodes: Vec<Option<(usize, Option<usize>)>>,
    program: Program,
}

impl Default for BlockBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockBuilder {
    pub const ONE: Slot = 0;

    pub fn new() -> Self {
        Self {
            a_rows: vec![vec![0]],
            b_rows: vec![vec![0]],
            kinds: vec![Kind::One],
            inputs: 0,
            nodes: Vec::new(),
            steps: Vec::new(),
            node_of: std::collections::HashMap::new(),
            slot_nodes: vec![None],
            program: Program::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.a_rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.a_rows.is_empty()
    }

    fn push(&mut self, a: Form, b: Form, kind: Kind) -> Slot {
        let s = self.a_rows.len();
        self.a_rows.push(a);
        self.b_rows.push(b);
        self.kinds.push(kind);
        self.slot_nodes.push(None);
        s
    }

    fn node_of(&mut self, f: &Form) -> usize {
        if let Some(&n) = self.node_of.get(f) {
            return n;
        }
        let n = self.nodes.len();
        self.nodes.push(if f.len() == 1 {
            Node::Slot(f[0])
        } else {
            Node::Sum(f.clone())
        });
        self.steps.push(Step::Node(n));
        self.node_of.insert(f.clone(), n);
        n
    }

    pub fn xor(&mut self, a: &Form, b: &Form) -> Form {
        let f = form_add(a, b);
        if !self.node_of.contains_key(&f) {
            let (na, nb) = (self.node_of(a), self.node_of(b));
            let n = self.nodes.len();
            self.nodes.push(Node::Xor(na, nb));
            self.steps.push(Step::Node(n));
            self.node_of.insert(f.clone(), n);
        }
        f
    }

    pub fn one() -> Form {
        vec![Self::ONE]
    }

    fn skip_to(&mut self, at: usize) {
        assert!(at >= self.len(), "slot {at} is already allocated");
        while self.len() < at {
            self.push(Vec::new(), Self::one(), Kind::Zero);
        }
    }

    pub fn reserve_outputs(&mut self, stride: usize) -> Slot {
        assert!(stride.is_power_of_two() && self.len() == 1);
        self.skip_to(stride);
        for _ in 0..stride {
            self.push(Vec::new(), Self::one(), Kind::Output);
        }
        stride
    }

    pub fn set_output(&mut self, s: Slot, f: &Form) {
        assert!(matches!(self.kinds[s], Kind::Output));
        self.a_rows[s] = if f.is_empty() { Vec::new() } else { f.clone() };
        if f.is_empty() {
            self.kinds[s] = Kind::Zero;
            self.slot_nodes[s] = None;
        } else {
            let a = self.node_of(f);
            self.steps.push(Step::Output { slot: s, a });
            self.slot_nodes[s] = Some((a, None));
        }
    }

    pub fn input(&mut self) -> Form {
        let s = INPUT_BASE + self.inputs;
        self.inputs += 1;
        vec![s]
    }

    pub fn and(&mut self, a: &Form, b: &Form) -> Form {
        let (na, nb) = (self.node_of(a), self.node_of(b));
        let s = self.push(a.clone(), b.clone(), Kind::Product);
        self.steps.push(Step::Product {
            slot: s,
            a: na,
            b: nb,
        });
        self.slot_nodes[s] = Some((na, Some(nb)));
        vec![s]
    }

    pub fn full_adder(&mut self, a: &Form, b: &Form, c: &Form) -> (Form, Form) {
        let ab = self.xor(a, b);
        let bc = self.xor(b, c);
        let t = self.and(&ab, &bc);
        (self.xor(&ab, c), self.xor(&t, b))
    }

    pub fn half_adder(&mut self, a: &Form, b: &Form) -> (Form, Form) {
        (self.xor(a, b), self.and(a, b))
    }

    pub fn count(&mut self, mut bits: Vec<Vec<Form>>) -> Vec<Form> {
        let mut w = 0;
        while w < bits.len() {
            while bits[w].len() >= 2 {
                let (sum, carry) = if bits[w].len() >= 3 {
                    let (a, b, c) = (
                        bits[w].pop().unwrap(),
                        bits[w].pop().unwrap(),
                        bits[w].pop().unwrap(),
                    );
                    self.full_adder(&a, &b, &c)
                } else {
                    let (a, b) = (bits[w].pop().unwrap(), bits[w].pop().unwrap());
                    self.half_adder(&a, &b)
                };
                bits[w].insert(0, sum);
                if bits.len() == w + 1 {
                    bits.push(Vec::new());
                }
                bits[w + 1].push(carry);
            }
            w += 1;
        }
        bits.into_iter()
            .map(|mut b| b.pop().unwrap_or_default())
            .collect()
    }

    fn finish(mut self, count: usize, bits: usize) -> Finished {
        let in_stride = bits.next_power_of_two();
        let region = (in_stride * count.next_power_of_two()).max(1);
        assert_eq!(self.inputs, count * bits);
        let in_base = self.len().div_ceil(region) * region;
        self.skip_to(in_base);
        for i in 0..count.next_power_of_two() {
            for t in 0..in_stride {
                let s = self.len();
                if i < count && t < bits {
                    self.push(vec![s], Self::one(), Kind::Input);
                } else {
                    self.push(Vec::new(), Self::one(), Kind::Zero);
                }
            }
        }
        let k_log = self.len().next_power_of_two().trailing_zeros().max(7) as usize;
        self.skip_to(1 << k_log);
        let remap = |f: &mut Form| {
            for s in f.iter_mut() {
                if *s >= INPUT_BASE {
                    let n = *s - INPUT_BASE;
                    *s = in_base + (n / bits) * in_stride + n % bits;
                }
            }
            f.sort_unstable();
        };
        self.a_rows.iter_mut().for_each(remap);
        self.b_rows.iter_mut().for_each(remap);
        let remap_slot = |s: &mut Slot| {
            if *s >= INPUT_BASE {
                let n = *s - INPUT_BASE;
                *s = in_base + (n / bits) * in_stride + n % bits;
            }
        };
        for node in self.nodes.iter_mut() {
            match node {
                Node::Slot(s) => remap_slot(s),
                Node::Sum(slots) => {
                    slots.iter_mut().for_each(remap_slot);
                    slots.sort_unstable();
                }
                Node::Xor(..) => {}
            }
        }
        self.node_of = std::collections::HashMap::new();
        self.program = self.compile();
        Finished {
            a0: SparseMatrix {
                k: 1 << k_log,
                rows: self.a_rows.clone(),
            },
            b0: SparseMatrix {
                k: 1 << k_log,
                rows: self.b_rows.clone(),
            },
            k_log,
            in_base,
            in_stride,
            builder: self,
        }
    }

    fn witness(&self, inputs: &[bool]) -> Vec<bool> {
        let mut z = vec![false; self.len()];
        let mut next_input = 0;
        for (zs, kind) in z.iter_mut().zip(&self.kinds) {
            match kind {
                Kind::One => *zs = true,
                Kind::Input => {
                    *zs = inputs[next_input];
                    next_input += 1;
                }
                _ => {}
            }
        }
        assert_eq!(next_input, inputs.len(), "every input consumed");
        let eval = |z: &[bool], f: &Form| f.iter().fold(false, |acc, &i| acc ^ z[i]);
        for pass in [Kind::Product, Kind::Output] {
            for s in 0..self.len() {
                if std::mem::discriminant(&self.kinds[s]) == std::mem::discriminant(&pass) {
                    z[s] = eval(&z, &self.a_rows[s]) & eval(&z, &self.b_rows[s]);
                }
            }
        }
        z
    }
}

pub struct Sliced {
    pub z: Vec<u64>,
    pub a: Vec<u64>,
    pub b: Vec<u64>,
}

impl BlockBuilder {
    fn compile(&self) -> Program {
        let mut ops = Vec::with_capacity(self.steps.len());
        let mut sum_slots = Vec::new();
        let as_u32 = |x: usize| u32::try_from(x).expect("a block's program fits u32");
        for step in &self.steps {
            match *step {
                Step::Node(n) => ops.push(match &self.nodes[n] {
                    Node::Slot(s) => Op::Load {
                        dst: as_u32(n),
                        slot: as_u32(*s),
                    },
                    Node::Xor(p, q) => Op::Xor {
                        dst: as_u32(n),
                        a: as_u32(*p),
                        b: as_u32(*q),
                    },
                    Node::Sum(slots) => {
                        let start = as_u32(sum_slots.len());
                        sum_slots.extend(slots.iter().map(|&s| as_u32(s)));
                        Op::Sum {
                            dst: as_u32(n),
                            start,
                            len: as_u32(slots.len()),
                        }
                    }
                }),
                Step::Product { slot, a, b } => ops.push(Op::Product {
                    slot: as_u32(slot),
                    a: as_u32(a),
                    b: as_u32(b),
                }),
                Step::Output { .. } => {}
            }
        }
        for step in &self.steps {
            if let Step::Output { slot, a } = *step {
                ops.push(Op::Output {
                    slot: as_u32(slot),
                    a: as_u32(a),
                });
            }
        }
        let rows: Vec<Rows> = self
            .slot_nodes
            .iter()
            .map(|n| match *n {
                Some((a, Some(b))) => Rows::Nodes(as_u32(a), as_u32(b)),
                Some((a, None)) => Rows::Output(as_u32(a)),
                None => Rows::Slot,
            })
            .collect();
        let input_slots: Vec<u32> = self
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| matches!(k, Kind::Input))
            .map(|(s, _)| as_u32(s))
            .collect();
        let (k, nodes) = (self.len(), self.nodes.len());
        for op in &ops {
            match *op {
                Op::Load { dst, slot } => assert!((dst as usize) < nodes && (slot as usize) < k),
                Op::Xor { dst, a, b } => {
                    assert!((dst as usize) < nodes && (a as usize) < nodes && (b as usize) < nodes)
                }
                Op::Sum { dst, start, len } => {
                    assert!((dst as usize) < nodes && (start + len) as usize <= sum_slots.len());
                    assert!(sum_slots[start as usize..(start + len) as usize]
                        .iter()
                        .all(|&s| (s as usize) < k));
                }
                Op::Product { slot, a, b } => {
                    assert!((slot as usize) < k && (a as usize) < nodes && (b as usize) < nodes)
                }
                Op::Output { slot, a } => assert!((slot as usize) < k && (a as usize) < nodes),
            }
        }
        for r in &rows {
            match *r {
                Rows::Slot => {}
                Rows::Nodes(a, b) => assert!((a as usize) < nodes && (b as usize) < nodes),
                Rows::Output(a) => assert!((a as usize) < nodes),
            }
        }
        assert!(input_slots.iter().all(|&s| (s as usize) < k) && k >= 1);
        assert_eq!(rows.len(), k);
        Program {
            ops,
            sum_slots,
            rows,
            nodes: self.nodes.len(),
            input_slots,
        }
    }

    fn witness_sliced_into(&self, inputs: &[u64], out: &mut Sliced, scratch: &mut Vec<u64>) {
        let k = self.len();
        let p = &self.program;
        assert_eq!(inputs.len(), p.input_slots.len(), "every input consumed");
        out.z.clear();
        out.z.resize(k, 0);
        let z = &mut out.z;
        z[0] = u64::MAX;
        for (&s, &v) in p.input_slots.iter().zip(inputs) {
            z[s as usize] = v;
        }
        scratch.clear();
        scratch.resize(p.nodes, 0);
        let v = scratch.as_mut_slice();
        let z = z.as_mut_slice();
        unsafe {
            for op in &p.ops {
                match *op {
                    Op::Load { dst, slot } => {
                        *v.get_unchecked_mut(dst as usize) = *z.get_unchecked(slot as usize)
                    }
                    Op::Xor { dst, a, b } => {
                        *v.get_unchecked_mut(dst as usize) =
                            *v.get_unchecked(a as usize) ^ *v.get_unchecked(b as usize)
                    }
                    Op::Sum { dst, start, len } => {
                        *v.get_unchecked_mut(dst as usize) = p
                            .sum_slots
                            .get_unchecked(start as usize..(start + len) as usize)
                            .iter()
                            .fold(0u64, |acc, &s| acc ^ *z.get_unchecked(s as usize))
                    }
                    Op::Product { slot, a, b } => {
                        *z.get_unchecked_mut(slot as usize) =
                            *v.get_unchecked(a as usize) & *v.get_unchecked(b as usize)
                    }
                    Op::Output { slot, a } => {
                        *z.get_unchecked_mut(slot as usize) = *v.get_unchecked(a as usize) & z[0]
                    }
                }
            }
            out.a.clear();
            out.b.clear();
            out.a.resize(k, 0);
            out.b.resize(k, 0);
            let one = z[0];
            for (s, rows) in p.rows.iter().enumerate() {
                let (a, b) = match *rows {
                    Rows::Slot => (*z.get_unchecked(s), one),
                    Rows::Nodes(na, nb) => {
                        (*v.get_unchecked(na as usize), *v.get_unchecked(nb as usize))
                    }
                    Rows::Output(na) => (*v.get_unchecked(na as usize), one),
                };
                *out.a.get_unchecked_mut(s) = a;
                *out.b.get_unchecked_mut(s) = b;
            }
        }
    }

    fn witness_sliced(&self, inputs: &[u64]) -> Sliced {
        let mut out = Sliced {
            z: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        };
        let mut scratch = Vec::new();
        self.witness_sliced_into(inputs, &mut out, &mut scratch);
        out
    }

    pub fn witness_sliced_from_rows(&self, inputs: &[u64]) -> Sliced {
        let k = self.len();
        let mut z = vec![0u64; k];
        let mut next_input = 0;
        for (zs, kind) in z.iter_mut().zip(&self.kinds) {
            match kind {
                Kind::One => *zs = u64::MAX,
                Kind::Input => {
                    *zs = inputs[next_input];
                    next_input += 1;
                }
                _ => {}
            }
        }
        assert_eq!(next_input, inputs.len(), "every input consumed");
        let eval = |z: &[u64], f: &Form| f.iter().fold(0u64, |acc, &i| acc ^ z[i]);
        let mut a = vec![0u64; k];
        let mut b = vec![0u64; k];
        for pass in [Kind::Product, Kind::Output] {
            for s in 0..k {
                if std::mem::discriminant(&self.kinds[s]) == std::mem::discriminant(&pass) {
                    a[s] = eval(&z, &self.a_rows[s]);
                    b[s] = eval(&z, &self.b_rows[s]);
                    z[s] = a[s] & b[s];
                }
            }
        }
        for s in 0..k {
            if matches!(self.kinds[s], Kind::One | Kind::Input | Kind::Zero) {
                a[s] = eval(&z, &self.a_rows[s]);
                b[s] = eval(&z, &self.b_rows[s]);
                debug_assert_eq!(z[s], a[s] & b[s]);
            }
        }
        Sliced { z, a, b }
    }
}

pub fn transpose_64x64(m: &mut [u64; 64]) {
    macro_rules! round {
        ($j:literal, $mask:literal) => {
            for run in m.chunks_exact_mut(2 * $j) {
                let (lo, hi) = run.split_at_mut($j);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let t = ((*x >> $j) ^ *y) & $mask;
                    *x ^= t << $j;
                    *y ^= t;
                }
            }
        };
    }
    round!(32, 0x0000_0000_FFFF_FFFFu64);
    round!(16, 0x0000_FFFF_0000_FFFFu64);
    round!(8, 0x00FF_00FF_00FF_00FFu64);
    round!(4, 0x0F0F_0F0F_0F0F_0F0Fu64);
    round!(2, 0x3333_3333_3333_3333u64);
    round!(1, 0x5555_5555_5555_5555u64);
}

struct Finished {
    a0: SparseMatrix,
    b0: SparseMatrix,
    k_log: usize,
    in_base: usize,
    in_stride: usize,
    builder: BlockBuilder,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub a0: SparseMatrix,
    pub b0: SparseMatrix,
    pub k_log: usize,
    pub in_base: usize,
    pub in_count: usize,
    pub in_bits: usize,
    pub in_stride: usize,
    pub out_base: usize,
    pub out_bits: usize,
    pub out_stride: usize,
    builder: std::sync::Arc<BlockBuilder>,
}

impl Block {
    pub fn from_matrices(a0: SparseMatrix, b0: SparseMatrix, k_log: usize) -> Self {
        assert_eq!(a0.k, 1 << k_log, "A_0 is k by k");
        assert_eq!(b0.k, 1 << k_log, "B_0 is k by k");
        Block {
            a0,
            b0,
            k_log,
            in_base: 0,
            in_count: 0,
            in_bits: 0,
            in_stride: 0,
            out_base: 0,
            out_bits: 0,
            out_stride: 0,
            builder: std::sync::Arc::new(BlockBuilder::new()),
        }
    }

    pub fn witness(&self, inputs: &[bool]) -> Vec<bool> {
        self.builder.witness(inputs)
    }

    pub fn k(&self) -> usize {
        1 << self.k_log
    }

    pub fn witness_sliced(&self, inputs: &[u64]) -> Sliced {
        self.builder.witness_sliced(inputs)
    }

    pub fn witness_sliced_into(&self, inputs: &[u64], out: &mut Sliced, scratch: &mut Vec<u64>) {
        self.builder.witness_sliced_into(inputs, out, scratch)
    }

    pub fn witness_sliced_from_rows(&self, inputs: &[u64]) -> Sliced {
        self.builder.witness_sliced_from_rows(inputs)
    }

    pub fn program_size(&self) -> usize {
        self.builder.program.ops.len()
    }

    pub fn output(&self, z: &[bool]) -> u64 {
        (0..self.out_bits).fold(0u64, |acc, t| acc | (u64::from(z[self.out_base + t]) << t))
    }

    pub fn nnz(&self) -> usize {
        self.a0.nnz() + self.b0.nnz()
    }

    pub fn products(&self) -> usize {
        self.builder
            .kinds
            .iter()
            .filter(|k| matches!(k, Kind::Product))
            .count()
    }
}

pub fn sum_bits(count: usize, bits: usize) -> usize {
    let max = (count as u128) * ((1u128 << bits) - 1);
    (128 - max.leading_zeros()) as usize
}

pub fn adder_block(count: usize, bits: usize) -> Block {
    assert!(count >= 1 && bits >= 1);
    let out_bits = sum_bits(count, bits);
    let out_stride = out_bits.next_power_of_two();
    let mut b = BlockBuilder::new();
    let out_base = b.reserve_outputs(out_stride);
    let mut by_weight: Vec<Vec<Form>> = vec![Vec::new(); bits];
    for _ in 0..count {
        for weight in by_weight.iter_mut() {
            let f = b.input();
            weight.push(f);
        }
    }
    let counted = b.count(by_weight);
    assert_eq!(counted.len(), out_bits, "the count is {out_bits} bits wide");
    for (t, f) in counted.iter().enumerate() {
        b.set_output(out_base + t, f);
    }
    for t in out_bits..out_stride {
        b.set_output(out_base + t, &Vec::new());
    }
    let f = b.finish(count, bits);
    Block {
        a0: f.a0,
        b0: f.b0,
        k_log: f.k_log,
        in_base: f.in_base,
        in_count: count,
        in_bits: bits,
        in_stride: f.in_stride,
        out_base,
        out_bits,
        out_stride,
        builder: std::sync::Arc::new(f.builder),
    }
}
