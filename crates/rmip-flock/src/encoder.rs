use flock_core::ntt::AdditiveNttF128;
use flock_field::F256Unreduced;
use flock_field::F128;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    InvalidShape(&'static str),
    SizeOverflow,
    ResourceLimit { requested: usize, limit: usize },
    Allocation,
    Length { expected: usize, actual: usize },
    EncoderMismatch,
    PositionOutOfDomain { position: usize, rows: usize },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "encoder: {self:?}")
    }
}
impl std::error::Error for EncodeError {}

fn product(a: usize, b: usize) -> Result<usize, EncodeError> {
    let n = a.checked_mul(b).ok_or(EncodeError::SizeOverflow)?;
    if n > isize::MAX as usize / std::mem::size_of::<F128>() {
        return Err(EncodeError::SizeOverflow);
    }
    Ok(n)
}

fn power(log: usize) -> Result<usize, EncodeError> {
    if log >= usize::BITS as usize {
        return Err(EncodeError::InvalidShape("domain does not fit an index"));
    }
    Ok(1usize << log)
}

fn length(actual: usize, expected: usize) -> Result<(), EncodeError> {
    if actual != expected {
        Err(EncodeError::Length { expected, actual })
    } else {
        Ok(())
    }
}

fn zeros(n: usize, limit: usize) -> Result<Vec<F128>, EncodeError> {
    if n > limit {
        return Err(EncodeError::ResourceLimit {
            requested: n,
            limit,
        });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| EncodeError::Allocation)?;
    values.resize(n, F128::ZERO);
    Ok(values)
}

fn evaluate_lanes(basis: &[F128], coefficients: &[F128], answer: &mut [F128]) {
    debug_assert_eq!(coefficients.len(), basis.len() * answer.len());
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { field64_kernels::evaluate(basis, coefficients, answer) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let lanes = answer.len();
        for first in (0..lanes).step_by(8) {
            let width = (lanes - first).min(8);
            let mut sums = [F256Unreduced::ZERO; 8];
            for (&factor, row) in basis.iter().zip(coefficients.chunks_exact(lanes)) {
                for (sum, &value) in sums[..width].iter_mut().zip(&row[first..first + width]) {
                    *sum ^= factor.mul_unreduced(value);
                }
            }
            for (value, sum) in answer[first..first + width].iter_mut().zip(sums) {
                *value = sum.reduce();
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod field64_kernels {
    use super::{F256Unreduced, F128};
    use core::arch::aarch64::{vdupq_n_u64, veorq_u64, vld1q_u64, vst1q_u64};
    use flock_field::f64_neon::{mul2_unreduced, reduce2};

    pub(super) unsafe fn evaluate(basis: &[F128], coefficients: &[F128], answer: &mut [F128]) {
        unsafe {
            let lanes = answer.len();
            let paired = lanes & !1;
            for first in (0..paired).step_by(8) {
                let pairs = ((paired - first).min(8)) / 2;
                match pairs {
                    4 => evaluate_pairs::<4>(basis, coefficients, answer, first),
                    3 => evaluate_pairs::<3>(basis, coefficients, answer, first),
                    2 => evaluate_pairs::<2>(basis, coefficients, answer, first),
                    1 => evaluate_pairs::<1>(basis, coefficients, answer, first),
                    _ => unreachable!(),
                }
            }
            if paired < lanes {
                let mut sum = F256Unreduced::ZERO;
                for (&factor, row) in basis.iter().zip(coefficients.chunks_exact(lanes)) {
                    sum ^= factor.mul_unreduced(row[paired]);
                }
                answer[paired] = sum.reduce();
            }
        }
    }

    #[inline]
    unsafe fn evaluate_pairs<const PAIRS: usize>(
        basis: &[F128],
        coefficients: &[F128],
        answer: &mut [F128],
        first: usize,
    ) {
        unsafe {
            let mut lows = [vdupq_n_u64(0); PAIRS];
            let mut highs = [vdupq_n_u64(0); PAIRS];
            for (&factor, row) in basis.iter().zip(coefficients.chunks_exact(answer.len())) {
                let factor = vdupq_n_u64(factor.lo);
                for pair in 0..PAIRS {
                    let values = vld1q_u64(row.as_ptr().add(first + 2 * pair).cast());
                    let (low, high) = mul2_unreduced(factor, values);
                    lows[pair] = veorq_u64(lows[pair], low);
                    highs[pair] = veorq_u64(highs[pair], high);
                }
            }
            for pair in 0..PAIRS {
                vst1q_u64(
                    answer.as_mut_ptr().add(first + 2 * pair).cast(),
                    reduce2(lows[pair], highs[pair]),
                );
            }
        }
    }
}

mod prefix_kernels {
    use super::F128;

    pub(super) fn fused3(groups: [&mut [F128]; 8], twiddles: &[F128; 7]) {
        let len = groups[0].len();
        debug_assert!(groups.iter().all(|group| group.len() == len));
        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "aes")]
        {
            let pointers = groups.map(|group| group.as_mut_ptr());
            unsafe { neon::fused3(pointers, len, twiddles) }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            let [g0, g1, g2, g3, g4, g5, g6, g7] = groups;
            for i in 0..len {
                let mut v = [g0[i], g1[i], g2[i], g3[i], g4[i], g5[i], g6[i], g7[i]];
                chain(&mut v, twiddles);
                [g0[i], g1[i], g2[i], g3[i], g4[i], g5[i], g6[i], g7[i]] = v;
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    fn chain(v: &mut [F128; 8], t: &[F128; 7]) {
        let mut butterfly = |a: usize, b: usize, twiddle: F128| {
            v[a] += v[b] * twiddle;
            v[b] += v[a];
        };
        for i in 0..4 {
            butterfly(i, i + 4, t[0]);
        }
        for s in 0..2 {
            for i in 0..2 {
                butterfly(4 * s + i, 4 * s + i + 2, t[1 + s]);
            }
        }
        for s in 0..4 {
            butterfly(2 * s, 2 * s + 1, t[3 + s]);
        }
    }

    pub(super) fn butterfly(top: &mut [F128], bottom: &mut [F128], twiddle: F128) {
        debug_assert_eq!(top.len(), bottom.len());
        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "aes")]
        {
            unsafe { neon::butterfly(top.as_mut_ptr(), bottom.as_mut_ptr(), top.len(), twiddle) }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        for (u, v) in top.iter_mut().zip(bottom) {
            *u += *v * twiddle;
            *v += *u;
        }
    }

    pub(super) fn accumulate(target: &mut [F128], source: &[F128], factor: F128) {
        debug_assert_eq!(target.len(), source.len());
        if factor == F128::ZERO {
            return;
        }
        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "aes")]
        {
            unsafe { neon::accumulate(target.as_mut_ptr(), source.as_ptr(), target.len(), factor) }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        for (u, &v) in target.iter_mut().zip(source) {
            *u += v * factor;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    mod neon {
        use super::F128;
        use core::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64, vld1q_u64, vst1q_u64};
        use flock_field::f64_neon::mul2;

        #[inline(always)]
        unsafe fn load(pointer: *const F128) -> uint64x2_t {
            unsafe { vld1q_u64(pointer.cast()) }
        }
        #[inline(always)]
        unsafe fn store(pointer: *mut F128, value: uint64x2_t) {
            unsafe { vst1q_u64(pointer.cast(), value) }
        }
        #[inline(always)]
        unsafe fn pair(u: uint64x2_t, v: uint64x2_t, t: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
            unsafe {
                let u = veorq_u64(u, mul2(v, t));
                (u, veorq_u64(v, u))
            }
        }

        pub(super) unsafe fn fused3(groups: [*mut F128; 8], len: usize, twiddles: &[F128; 7]) {
            unsafe {
                let t = twiddles.map(|twiddle| vdupq_n_u64(twiddle.lo));
                let mut i = 0;
                while i + 2 <= len {
                    let mut v = groups.map(|group| load(group.add(i)));
                    for k in 0..4 {
                        (v[k], v[k + 4]) = pair(v[k], v[k + 4], t[0]);
                    }
                    for s in 0..2 {
                        for k in 0..2 {
                            let (a, b) = (4 * s + k, 4 * s + k + 2);
                            (v[a], v[b]) = pair(v[a], v[b], t[1 + s]);
                        }
                    }
                    for s in 0..4 {
                        (v[2 * s], v[2 * s + 1]) = pair(v[2 * s], v[2 * s + 1], t[3 + s]);
                    }
                    for (group, value) in groups.iter().zip(v) {
                        store(group.add(i), value);
                    }
                    i += 2;
                }
                if i < len {
                    let mut v = groups.map(|group| *group.add(i));
                    let mut butterfly = |a: usize, b: usize, twiddle: F128| {
                        v[a] += v[b] * twiddle;
                        v[b] += v[a];
                    };
                    for k in 0..4 {
                        butterfly(k, k + 4, twiddles[0]);
                    }
                    for s in 0..2 {
                        for k in 0..2 {
                            butterfly(4 * s + k, 4 * s + k + 2, twiddles[1 + s]);
                        }
                    }
                    for s in 0..4 {
                        butterfly(2 * s, 2 * s + 1, twiddles[3 + s]);
                    }
                    for (group, value) in groups.iter().zip(v) {
                        *group.add(i) = value;
                    }
                }
            }
        }

        pub(super) unsafe fn butterfly(
            top: *mut F128,
            bottom: *mut F128,
            len: usize,
            twiddle: F128,
        ) {
            unsafe {
                let t = vdupq_n_u64(twiddle.lo);
                let mut i = 0;
                while i + 2 <= len {
                    let (u, v) = pair(load(top.add(i)), load(bottom.add(i)), t);
                    store(top.add(i), u);
                    store(bottom.add(i), v);
                    i += 2;
                }
                if i < len {
                    *top.add(i) += *bottom.add(i) * twiddle;
                    *bottom.add(i) += *top.add(i);
                }
            }
        }

        pub(super) unsafe fn accumulate(
            target: *mut F128,
            source: *const F128,
            len: usize,
            factor: F128,
        ) {
            unsafe {
                let t = vdupq_n_u64(factor.lo);
                let mut i = 0;
                while i + 4 <= len {
                    let p0 = mul2(load(source.add(i)), t);
                    let p1 = mul2(load(source.add(i + 2)), t);
                    store(target.add(i), veorq_u64(load(target.add(i)), p0));
                    store(target.add(i + 2), veorq_u64(load(target.add(i + 2)), p1));
                    i += 4;
                }
                while i + 2 <= len {
                    store(
                        target.add(i),
                        veorq_u64(load(target.add(i)), mul2(load(source.add(i)), t)),
                    );
                    i += 2;
                }
                if i < len {
                    *target.add(i) += *source.add(i) * factor;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableShape {
    native_domain_log: usize,
    depth: usize,
    coefficient_count: usize,
    lanes: usize,
}

impl TableShape {
    pub fn new(
        native_domain_log: usize,
        depth: usize,
        coefficient_count: usize,
        lanes: usize,
    ) -> Result<Self, EncodeError> {
        power(native_domain_log)?;
        if native_domain_log > F128::BITS || depth > native_domain_log {
            return Err(EncodeError::InvalidShape(
                "invalid native dimension or depth",
            ));
        }
        let rows = power(native_domain_log - depth)?;
        if !coefficient_count.is_power_of_two() || coefficient_count > rows || lanes == 0 {
            return Err(EncodeError::InvalidShape(
                "invalid coefficient count or lane count",
            ));
        }
        product(rows, lanes)?;
        Ok(Self {
            native_domain_log,
            depth,
            coefficient_count,
            lanes,
        })
    }

    pub fn native_domain_log(self) -> usize {
        self.native_domain_log
    }
    pub fn depth(self) -> usize {
        self.depth
    }
    pub fn domain_log(self) -> usize {
        self.native_domain_log - self.depth
    }
    pub fn rows(self) -> usize {
        1usize << self.domain_log()
    }
    pub fn coefficient_count(self) -> usize {
        self.coefficient_count
    }
    pub fn lanes(self) -> usize {
        self.lanes
    }
    pub fn log_inv_rate(self) -> usize {
        self.domain_log() - self.coefficient_count.trailing_zeros() as usize
    }
    pub fn coefficient_words(self) -> usize {
        self.coefficient_count * self.lanes
    }
    pub fn encoded_words(self) -> usize {
        self.rows() * self.lanes
    }

    fn check_position(self, position: usize) -> Result<(), EncodeError> {
        if position >= self.rows() {
            Err(EncodeError::PositionOutOfDomain {
                position,
                rows: self.rows(),
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coefficients {
    shape: TableShape,
    values: Vec<F128>,
}

impl Coefficients {
    pub fn new(shape: TableShape, values: Vec<F128>) -> Result<Self, EncodeError> {
        length(values.len(), shape.coefficient_words())?;
        Ok(Self { shape, values })
    }
    pub fn shape(&self) -> TableShape {
        self.shape
    }
    pub fn values(&self) -> &[F128] {
        &self.values
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTable {
    shape: TableShape,
    values: Vec<F128>,
}

impl EncodedTable {
    pub fn new(shape: TableShape, values: Vec<F128>) -> Result<Self, EncodeError> {
        length(values.len(), shape.encoded_words())?;
        Ok(Self { shape, values })
    }
    pub fn shape(&self) -> TableShape {
        self.shape
    }
    pub fn values(&self) -> &[F128] {
        &self.values
    }
    pub fn row(&self, position: usize) -> Result<&[F128], EncodeError> {
        self.shape.check_position(position)?;
        Ok(&self.values[position * self.shape.lanes..(position + 1) * self.shape.lanes])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialTable {
    shape: TableShape,
    layer: usize,
    values: Vec<F128>,
}

impl PartialTable {
    pub fn shape(&self) -> TableShape {
        self.shape
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn values(&self) -> &[F128] {
        &self.values
    }
}

const PREFIX_TILE_BYTES: usize = 1 << 19;
const PREFIX_MIN_TILE_WIDTH: usize = 16;
const PREFIX_PRODUCT_QUARTERS: u128 = 4;
const FINISH_PRODUCT_QUARTERS: u128 = 4;
const TABLE_WORD_QUARTERS: u128 = 2;

#[derive(Clone, Debug)]
pub struct ShiftedEncoder {
    native_domain_log: usize,
    ntt: Option<AdditiveNttF128>,
    max_words: usize,
}

impl ShiftedEncoder {
    pub fn new(native_domain_log: usize, max_words: usize) -> Result<Self, EncodeError> {
        power(native_domain_log)?;
        if native_domain_log > F128::BITS || max_words == 0 {
            return Err(EncodeError::InvalidShape(
                "invalid encoder dimension or cap",
            ));
        }
        Ok(Self {
            native_domain_log,
            ntt: (native_domain_log != 0).then(|| AdditiveNttF128::standard(native_domain_log)),
            max_words,
        })
    }

    fn check(&self, shape: TableShape) -> Result<(), EncodeError> {
        if shape.native_domain_log != self.native_domain_log {
            Err(EncodeError::EncoderMismatch)
        } else {
            Ok(())
        }
    }

    pub fn encode(&self, coefficients: &Coefficients) -> Result<EncodedTable, EncodeError> {
        self.check(coefficients.shape)?;
        let mut values = zeros(coefficients.shape.encoded_words(), self.max_words)?;
        values[..coefficients.values.len()].copy_from_slice(&coefficients.values);
        if coefficients.shape.rows() > 1 {
            self.ntt
                .as_ref()
                .ok_or(EncodeError::EncoderMismatch)?
                .forward_transform_interleaved(&mut values, coefficients.shape.lanes);
        }
        EncodedTable::new(coefficients.shape, values)
    }

    pub fn prefix_layer(
        &self,
        shape: TableShape,
        query_count: usize,
    ) -> Result<usize, EncodeError> {
        self.check(shape)?;
        let (domain, first) = (shape.domain_log(), shape.log_inv_rate());
        let rows = shape.rows() as u128;
        let queries = query_count as u128;
        let mut best = (u128::MAX, first);
        for layer in first..=domain {
            if layer > first && shape.encoded_words() > self.max_words {
                break;
            }
            let span = 1u128 << (domain - layer);
            let mut cost = 0u128;
            if layer > first {
                cost += PREFIX_PRODUCT_QUARTERS * ((layer - first) as u128) * (rows / 2)
                    + TABLE_WORD_QUARTERS * rows;
            }
            cost += queries * (FINISH_PRODUCT_QUARTERS * (span - 1) + span.div_ceil(2));
            if cost < best.0 {
                best = (cost, layer);
            }
        }
        Ok(best.1)
    }

    pub fn encode_prefix(
        &self,
        coefficients: &Coefficients,
        layer: usize,
    ) -> Result<PartialTable, EncodeError> {
        self.check(coefficients.shape)?;
        let shape = coefficients.shape;
        let first = shape.log_inv_rate();
        if layer < first || layer > shape.domain_log() {
            return Err(EncodeError::InvalidShape(
                "prefix layer outside the inverse-rate and domain bounds",
            ));
        }
        if layer == first {
            return Ok(PartialTable {
                shape,
                layer,
                values: Vec::new(),
            });
        }
        let ntt = self.ntt.as_ref().ok_or(EncodeError::EncoderMismatch)?;
        let mut values = zeros(shape.encoded_words(), self.max_words)?;
        let width = (shape.rows() >> layer) * shape.lanes;
        let period = 1usize << (layer - first);
        let view_rows = 1usize << layer;
        let tile = ((PREFIX_TILE_BYTES / F128::BYTES) >> layer).min(width);
        if tile < PREFIX_MIN_TILE_WIDTH.min(width) {
            for (row, chunk) in values.chunks_exact_mut(width).enumerate() {
                let source = (row % period) * width;
                chunk.copy_from_slice(&coefficients.values[source..source + width]);
            }
            ntt.forward_transform_interleaved_from_layer(&mut values, width, first);
            return Ok(PartialTable {
                shape,
                layer,
                values,
            });
        }
        let mut twiddles = Vec::new();
        twiddles
            .try_reserve_exact(view_rows)
            .map_err(|_| EncodeError::Allocation)?;
        let mut offsets = Vec::with_capacity(layer + 1);
        for at in 0..layer {
            offsets.push(twiddles.len());
            if at >= first {
                twiddles.extend((0..1usize << at).map(|block| ntt.twiddle(at, block)));
            }
        }
        let twiddle = |at: usize, block: usize| twiddles[offsets[at] + block];
        let segment = |row: usize, column: usize, columns: usize| {
            row * width + column..row * width + column + columns
        };
        for column in (0..width).step_by(tile) {
            let columns = tile.min(width - column);
            for row in 0..view_rows {
                let source = (row % period) * width + column;
                values[segment(row, column, columns)]
                    .copy_from_slice(&coefficients.values[source..source + columns]);
            }
            let mut at = first;
            while at < layer {
                let block_rows = view_rows >> at;
                if at + 3 <= layer {
                    let eighth = block_rows / 8;
                    for block in 0..1usize << at {
                        let group = [
                            twiddle(at, block),
                            twiddle(at + 1, 2 * block),
                            twiddle(at + 1, 2 * block + 1),
                            twiddle(at + 2, 4 * block),
                            twiddle(at + 2, 4 * block + 1),
                            twiddle(at + 2, 4 * block + 2),
                            twiddle(at + 2, 4 * block + 3),
                        ];
                        for row in block * block_rows..block * block_rows + eighth {
                            let rows = values
                                .get_disjoint_mut(std::array::from_fn::<_, 8, _>(|k| {
                                    segment(row + k * eighth, column, columns)
                                }))
                                .map_err(|_| EncodeError::InvalidShape("overlapping tile rows"))?;
                            prefix_kernels::fused3(rows, &group);
                        }
                    }
                    at += 3;
                } else {
                    let half = block_rows / 2;
                    for block in 0..1usize << at {
                        let factor = twiddle(at, block);
                        for row in block * block_rows..block * block_rows + half {
                            let [top, bottom] = values
                                .get_disjoint_mut([
                                    segment(row, column, columns),
                                    segment(row + half, column, columns),
                                ])
                                .map_err(|_| EncodeError::InvalidShape("overlapping tile rows"))?;
                            prefix_kernels::butterfly(top, bottom, factor);
                        }
                    }
                    at += 1;
                }
            }
        }
        Ok(PartialTable {
            shape,
            layer,
            values,
        })
    }

    pub fn finish_row(
        &self,
        coefficients: &Coefficients,
        partial: &PartialTable,
        position: usize,
    ) -> Result<Vec<F128>, EncodeError> {
        self.check(coefficients.shape)?;
        if partial.shape != coefficients.shape {
            return Err(EncodeError::EncoderMismatch);
        }
        let shape = partial.shape;
        shape.check_position(position)?;
        let (domain, lanes) = (shape.domain_log(), shape.lanes);
        let mut layer = partial.layer;
        let span = 1usize << (domain - layer);
        let mut block = position >> (domain - layer);
        let source = if layer == shape.log_inv_rate() {
            &coefficients.values[..]
        } else {
            &partial.values[block * span * lanes..(block + 1) * span * lanes]
        };
        let mut answer = zeros(lanes, self.max_words)?;
        if span == 1 {
            answer.copy_from_slice(source);
            return Ok(answer);
        }
        let ntt = self.ntt.as_ref().ok_or(EncodeError::EncoderMismatch)?;
        let factor = |layer: usize, block: usize| {
            let twiddle = ntt.twiddle(layer, block);
            if (position >> (domain - layer - 1)) & 1 == 1 {
                twiddle + F128::ONE
            } else {
                twiddle
            }
        };
        let mut len = span / 2;
        let (top, bottom) = source.split_at(len * lanes);
        let mut scratch = zeros(len * lanes, self.max_words)?;
        scratch.copy_from_slice(top);
        prefix_kernels::accumulate(&mut scratch, bottom, factor(layer, block));
        while len > 1 {
            block = 2 * block + ((position >> (domain - layer - 1)) & 1);
            layer += 1;
            len /= 2;
            let (top, bottom) = scratch[..2 * len * lanes].split_at_mut(len * lanes);
            prefix_kernels::accumulate(top, bottom, factor(layer, block));
        }
        answer.copy_from_slice(&scratch[..lanes]);
        Ok(answer)
    }

    pub fn basis_factor(&self, depth: usize, position: usize) -> Result<F128, EncodeError> {
        if depth >= self.native_domain_log {
            return Err(EncodeError::InvalidShape(
                "no split factor for constant domain",
            ));
        }
        let rows = power(self.native_domain_log - depth)?;
        if position >= rows {
            return Err(EncodeError::PositionOutOfDomain { position, rows });
        }
        Ok(F128::from_u64((position & 1) as u64)
            + self
                .ntt
                .as_ref()
                .ok_or(EncodeError::EncoderMismatch)?
                .twiddle(self.native_domain_log - depth - 1, position >> 1))
    }

    pub fn evaluate(
        &self,
        coefficients: &Coefficients,
        position: usize,
    ) -> Result<Vec<F128>, EncodeError> {
        self.check(coefficients.shape)?;
        let shape = coefficients.shape;
        shape.check_position(position)?;
        let mut basis = zeros(shape.coefficient_count, self.max_words)?;
        basis[0] = F128::ONE;
        let mut factors = Vec::with_capacity(shape.coefficient_count.trailing_zeros() as usize);
        for bit in 0..shape.coefficient_count.trailing_zeros() as usize {
            factors.push(self.basis_factor(shape.depth + bit, position >> bit)?);
        }
        for h in 1..basis.len() {
            basis[h] = basis[h & (h - 1)] * factors[h.trailing_zeros() as usize];
        }
        let mut answer = zeros(shape.lanes, self.max_words)?;
        evaluate_lanes(&basis, &coefficients.values, &mut answer);
        Ok(answer)
    }
}
