use crate::circuit::{adder_block, transpose_64x64, Block, Sliced};
use crate::lincheck::BlockR1cs;
use crate::ringswitch::BitClaim;
use crate::zerocheck::{eq_table, PackedBits};
use crate::PackedField;

#[derive(Clone, Debug)]
pub struct Level {
    pub block: Block,
    pub n_log: usize,
    r1cs: std::sync::OnceLock<BlockR1cs>,
}

impl Level {
    pub fn new(block: Block, n_log: usize) -> Self {
        Self {
            block,
            n_log,
            r1cs: std::sync::OnceLock::new(),
        }
    }

    pub fn m(&self) -> usize {
        self.block.k_log + self.n_log
    }

    pub fn r1cs(&self) -> &BlockR1cs {
        self.r1cs.get_or_init(|| {
            BlockR1cs::new(
                self.m(),
                self.block.k_log,
                self.block.a0.clone(),
                self.block.b0.clone(),
            )
        })
    }

    pub fn words_log(&self, kappa_log: usize) -> usize {
        self.m() - kappa_log
    }
}

#[derive(Clone, Debug)]
pub struct Tree {
    pub views: usize,
    pub levels: Vec<Level>,
    pub v: usize,
    pub c: usize,
}

pub struct LevelWitness<F: PackedField> {
    pub z: PackedBits<F>,
    pub a: PackedBits<F>,
    pub b: PackedBits<F>,
}

impl<F: PackedField> LevelWitness<F> {
    pub fn from_z_bits(level: &Level, z: &[bool]) -> Self {
        let k = level.block.k();
        let m = level.m();
        let blocks = 1usize << level.n_log;
        assert_eq!(z.len(), 1 << m, "2^m bits");
        assert_eq!(k % 64, 0, "a block is whole lanes");
        let words = (blocks * k) / F::BITS;
        let (mut zv, mut av, mut bv) = (
            vec![F::ZERO; words],
            vec![F::ZERO; words],
            vec![F::ZERO; words],
        );
        let (mut lz, mut la, mut lb) = (vec![0u64; k], vec![0u64; k], vec![0u64; k]);
        for g in 0..blocks.div_ceil(64) {
            lz.iter_mut().for_each(|l| *l = 0);
            for j in 0..64.min(blocks - 64 * g) {
                let base = (64 * g + j) * k;
                for (s, lane) in lz.iter_mut().enumerate() {
                    if z[base + s] {
                        *lane |= 1 << j;
                    }
                }
            }
            for (rows, lanes) in [
                (&level.block.a0.rows, &mut la),
                (&level.block.b0.rows, &mut lb),
            ] {
                for (i, row) in rows.iter().enumerate() {
                    lanes[i] = row.iter().fold(0u64, |v, &c| v ^ lz[c]);
                }
            }
            scatter::<F>(&lz, &mut zv, g, k, blocks);
            scatter::<F>(&la, &mut av, g, k, blocks);
            scatter::<F>(&lb, &mut bv, g, k, blocks);
        }
        Self {
            z: PackedBits::new(m, zv),
            a: PackedBits::new(m, av),
            b: PackedBits::new(m, bv),
        }
    }
}

impl Tree {
    pub fn new(views: usize, v_log: usize, c_log: usize) -> Self {
        let v = 1 << v_log;
        assert!(
            views >= v && views.is_power_of_two(),
            "views must be a power of two of at least {v}"
        );
        assert!(c_log >= 1, "the fan-in must be at least two");
        let n0_log = views.trailing_zeros() as usize - v_log;
        let mut levels = vec![Level::new(adder_block(v, 1), n0_log)];
        while levels.last().unwrap().n_log > 0 {
            let prev = levels.last().unwrap();
            let fan_log = c_log.min(prev.n_log);
            let block = adder_block(1 << fan_log, prev.block.out_bits);
            levels.push(Level::new(block, prev.n_log - fan_log));
        }
        Self {
            views,
            levels,
            v,
            c: 1 << c_log,
        }
    }

    pub fn layout(&self, kappa_log: usize) -> (Vec<usize>, usize) {
        let mut offsets = Vec::with_capacity(self.levels.len());
        let mut next = 0usize;
        for level in &self.levels {
            let len = 1 << level.words_log(kappa_log);
            assert_eq!(next % len, 0, "each level at a multiple of its length");
            offsets.push(next);
            next += len;
        }
        (offsets, next.next_power_of_two().trailing_zeros() as usize)
    }

    pub fn fan_in(&self, level: usize) -> usize {
        self.levels[level].block.in_count
    }

    pub fn level_witness<F: PackedField>(
        &self,
        level: usize,
        sliced: &[Vec<u64>],
    ) -> LevelWitness<F> {
        let lv = &self.levels[level];
        let k = lv.block.k();
        let blocks = 1usize << lv.n_log;
        let groups = blocks.div_ceil(64);
        assert_eq!(sliced.len(), groups);
        let kappa = F::BITS;
        let words = (blocks * k) / kappa;
        let mut z = vec![F::ZERO; words];
        let mut a = vec![F::ZERO; words];
        let mut b = vec![F::ZERO; words];
        let mut s = Sliced {
            z: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        };
        let mut scratch = Vec::new();
        for (g, inputs) in sliced.iter().enumerate() {
            lv.block.witness_sliced_into(inputs, &mut s, &mut scratch);
            for (lanes, out) in [(&s.z, &mut z), (&s.a, &mut a), (&s.b, &mut b)] {
                scatter::<F>(lanes, out, g, k, blocks);
            }
        }
        let m = lv.m();
        LevelWitness {
            z: PackedBits::new(m, z),
            a: PackedBits::new(m, a),
            b: PackedBits::new(m, b),
        }
    }

    pub fn slice_inputs(&self, level: usize, bits: &[bool]) -> Vec<Vec<u64>> {
        let lv = &self.levels[level];
        let per_block = lv.block.in_count * lv.block.in_bits;
        let blocks = 1usize << lv.n_log;
        assert_eq!(bits.len(), blocks * per_block);
        (0..blocks.div_ceil(64))
            .map(|g| {
                (0..per_block)
                    .map(|n| {
                        (0..64)
                            .filter(|&j| 64 * g + j < blocks && bits[(64 * g + j) * per_block + n])
                            .fold(0u64, |acc, j| acc | (1 << j))
                    })
                    .collect()
            })
            .collect()
    }

    pub fn outputs_as_next_inputs<F: PackedField>(
        &self,
        level: usize,
        w: &LevelWitness<F>,
    ) -> Vec<bool> {
        let lv = &self.levels[level];
        let next_bits = self.levels[level + 1].block.in_bits;
        assert_eq!(next_bits, lv.block.out_bits);
        let k = lv.block.k();
        (0..1usize << lv.n_log)
            .flat_map(|j| (0..next_bits).map(move |t| w.z.bit(j * k + lv.block.out_base + t)))
            .collect()
    }

    pub fn slice_answer_words(&self, words: &[u64]) -> Vec<Vec<u64>> {
        let lv = &self.levels[0];
        assert_eq!(
            lv.block.in_count * lv.block.in_bits,
            64,
            "a level 0 block takes 64 one-bit inputs"
        );
        let blocks = 1usize << lv.n_log;
        assert_eq!(words.len(), blocks);
        (0..blocks.div_ceil(64))
            .map(|g| {
                let mut tile = [0u64; 64];
                let n = 64.min(blocks - 64 * g);
                tile[..n].copy_from_slice(&words[64 * g..64 * g + n]);
                transpose_64x64(&mut tile);
                tile.to_vec()
            })
            .collect()
    }

    pub fn witness_from_words<F: PackedField>(&self, answers: &[u64]) -> Vec<LevelWitness<F>> {
        assert_eq!(answers.len() * 64, self.views);
        let mut out: Vec<LevelWitness<F>> = Vec::with_capacity(self.levels.len());
        let mut inputs: Vec<bool> = Vec::new();
        for level in 0..self.levels.len() {
            let sliced = if level == 0 {
                self.slice_answer_words(answers)
            } else {
                self.slice_inputs(level, &inputs)
            };
            let w = self.level_witness::<F>(level, &sliced);
            if level + 1 < self.levels.len() {
                inputs = self.outputs_as_next_inputs(level, &w);
            }
            out.push(w);
        }
        out
    }

    pub fn witness<F: PackedField>(&self, answers: &[bool]) -> Vec<LevelWitness<F>> {
        assert_eq!(answers.len(), self.views);
        if self.v == 64 {
            let words: Vec<u64> = answers
                .as_chunks::<64>()
                .0
                .iter()
                .map(|chunk| {
                    chunk
                        .iter()
                        .enumerate()
                        .fold(0u64, |acc, (r, &a)| acc | (u64::from(a) << r))
                })
                .collect();
            return self.witness_from_words::<F>(&words);
        }
        let mut out: Vec<LevelWitness<F>> = Vec::with_capacity(self.levels.len());
        let mut inputs = answers.to_vec();
        for level in 0..self.levels.len() {
            let sliced = self.slice_inputs(level, &inputs);
            let w = self.level_witness::<F>(level, &sliced);
            if level + 1 < self.levels.len() {
                inputs = self.outputs_as_next_inputs(level, &w);
            }
            out.push(w);
        }
        out
    }

    pub fn count<F: PackedField>(&self, top: &LevelWitness<F>) -> u64 {
        let lv = self.levels.last().unwrap();
        (0..lv.block.out_bits).fold(0u64, |acc, t| {
            acc | (u64::from(top.z.bit(lv.block.out_base + t)) << t)
        })
    }

    pub fn answer_bit(&self, view: usize) -> usize {
        let b = &self.levels[0].block;
        (view / self.v) * b.k() + b.in_base + view % self.v
    }

    pub fn count_bit(&self) -> usize {
        self.levels.last().unwrap().block.out_base
    }

    pub fn count_from_word(&self, word: u128, kappa: usize) -> u64 {
        let lv = self.levels.last().unwrap();
        let shift = self.count_bit() % kappa;
        ((word >> shift) & ((1u128 << lv.block.out_bits) - 1)) as u64
    }

    pub fn copy_coins(&self, level: usize) -> usize {
        let lv = &self.levels[level];
        lv.block.in_stride.trailing_zeros() as usize
            + self.fan_in(level).trailing_zeros() as usize
            + lv.n_log
    }

    pub fn copy_points<F: PackedField>(&self, level: usize, rho: &[F]) -> (Vec<F>, Vec<F>) {
        assert_eq!(rho.len(), self.copy_coins(level));
        let lv = &self.levels[level];
        let prev = &self.levels[level - 1];
        let t_log = lv.block.in_stride.trailing_zeros() as usize;
        let c_log = self.fan_in(level).trailing_zeros() as usize;
        let (rho_t, rest) = rho.split_at(t_log);
        let (rho_i, rho_j) = rest.split_at(c_log);
        let boolean =
            |x: usize, bits: usize| (0..bits).map(move |b| F::from_bool((x >> b) & 1 == 1));
        let mut input = rho_t.to_vec();
        input.extend_from_slice(rho_i);
        input.extend(boolean(
            lv.block.in_base >> (t_log + c_log),
            lv.block.k_log - t_log - c_log,
        ));
        input.extend_from_slice(rho_j);
        assert_eq!(prev.block.out_stride, lv.block.in_stride);
        let mut output = rho_t.to_vec();
        output.extend(boolean(
            prev.block.out_base >> t_log,
            prev.block.k_log - t_log,
        ));
        output.extend_from_slice(rho_i);
        output.extend_from_slice(rho_j);
        debug_assert_eq!(input.len(), lv.m());
        debug_assert_eq!(output.len(), prev.m());
        (input, output)
    }
}

fn scatter<F: PackedField>(lanes: &[u64], out: &mut [F], g: usize, k: usize, blocks: usize) {
    let kappa = F::BITS;
    let mut tile = [0u64; 64];
    for s0 in (0..k).step_by(64) {
        tile.copy_from_slice(&lanes[s0..s0 + 64]);
        transpose_64x64(&mut tile);
        for (j, &bits) in tile.iter().enumerate() {
            let block = 64 * g + j;
            if block >= blocks {
                break;
            }
            let bit = block * k + s0;
            let word = bit / kappa;
            let shift = bit % kappa;
            let cur = out[word].to_bits();
            out[word] = F::from_bits(cur | (u128::from(bits) << shift));
        }
    }
}

pub fn bit_claim_at<F: PackedField>(point: &[F], value: F) -> BitClaim<F> {
    let kappa_log = F::BITS.trailing_zeros() as usize;
    BitClaim {
        coeffs: eq_table(&point[..kappa_log]),
        point: point[kappa_log..].to_vec(),
        value,
    }
}

pub fn eval_at<F: PackedField>(bits: &PackedBits<F>, point: &[F]) -> F {
    bit_claim_at(point, F::ZERO).evaluate(bits)
}

pub type SlicedWitness = Sliced;
