use rmip_backend_interleaved::code;
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{
    Alphabet, AuditMode, BackendParams, BooleanBinding, Claim, CommitmentBackend, Coord,
    LevelParams, LongOracle, OpeningVerifier, QueryPlan, Regime, Symbol,
};
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::transcript::{OracleId, RawChallenger, TranscriptPrefix};

pub const LEVEL0_ORACLE: OracleId = 0;
pub const LEVEL1_ORACLE: OracleId = 1;

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub nu_row: usize,
    pub nu_col: usize,
    pub n_enc: usize,
    pub lambda_bits: f64,
}

impl Shape {
    pub fn balanced(nu: usize, rate_inv: usize, lambda_bits: f64) -> Self {
        assert!(
            nu >= 3,
            "the ladder needs at least 4 rows and a message longer than one \
             element, so nu must be at least 3; got {nu}"
        );
        let nu_row = (nu / 2).max(2);
        let nu_col = nu - nu_row;
        Self {
            nu_row,
            nu_col,
            n_enc: (1 << nu_col) * rate_inv,
            lambda_bits,
        }
    }
    pub fn m(&self) -> usize {
        1 << self.nu_row
    }
    pub fn n(&self) -> usize {
        1 << self.nu_col
    }
    pub fn delta(&self) -> f64 {
        code::unique_decoding_radius(self.n(), self.n_enc)
    }
    pub fn queries(&self) -> usize {
        code::queries_for(self.lambda_bits, self.delta())
    }
}

pub fn fold_rows(rows: &[Vec<F128>], alpha: F128) -> Vec<Vec<F128>> {
    let half = rows.len() / 2;
    (0..half)
        .map(|i| {
            rows[i]
                .iter()
                .zip(&rows[i + half])
                .map(|(lo, hi)| lo.add(alpha.mul(lo.add(*hi))))
                .collect()
        })
        .collect()
}

fn fold_column(col: &[F128], alpha: F128) -> Vec<F128> {
    let half = col.len() / 2;
    (0..half)
        .map(|i| col[i].add(alpha.mul(col[i].add(col[i + half]))))
        .collect()
}

fn fold_challenge_from(h: &TranscriptPrefix) -> F128 {
    h.coins()
        .ok()
        .and_then(|c| c.last().copied())
        .unwrap_or(F128::ZERO)
}

pub struct FoldLadder {
    shape: Shape,
    rows: Vec<Vec<F128>>,
    oracles: Vec<LongOracle<F128>>,
    proximity: Vec<F128>,
    proximity_coeffs: Vec<F128>,
    opening: Vec<F128>,
    opening_coeffs: Vec<F128>,
    opened_point: Vec<F128>,
    alphas: Vec<F128>,
    residual: Vec<Vec<F128>>,
    ordinary_bytes: usize,
}

impl FoldLadder {
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            rows: Vec::new(),
            oracles: Vec::new(),
            proximity: Vec::new(),
            proximity_coeffs: Vec::new(),
            opening: Vec::new(),
            opening_coeffs: Vec::new(),
            opened_point: Vec::new(),
            alphas: Vec::new(),
            residual: Vec::new(),
            ordinary_bytes: 0,
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    fn combine(&self, c: &[F128]) -> Vec<F128> {
        let mut out = vec![F128::ZERO; self.shape.n()];
        for (i, row) in self.rows.iter().enumerate() {
            let ci = c[i];
            if ci == F128::ZERO {
                continue;
            }
            for (j, v) in row.iter().enumerate() {
                out[j] = out[j].add(ci.mul(*v));
            }
        }
        out
    }

    fn encode_rows(rows: &[Vec<F128>], n_enc: usize) -> Vec<Symbol<F128>> {
        let dom = code::domain(n_enc);
        let enc: Vec<Vec<F128>> = rows.iter().map(|r| code::encode(r, &dom)).collect();
        (0..n_enc)
            .map(|c| Symbol(enc.iter().map(|e| e[c]).collect()))
            .collect()
    }
}

struct Ladder {
    picked: Vec<usize>,
    next: usize,
}

impl QueryPlan<F128> for Ladder {
    fn next(&mut self, _prev: Option<&Symbol<F128>>) -> Option<(OracleId, Coord)> {
        let k = self.next;
        let c = *self.picked.get(k / 2)?;
        self.next += 1;
        Some((
            if k.is_multiple_of(2) {
                LEVEL0_ORACLE
            } else {
                LEVEL1_ORACLE
            },
            Coord::from(c),
        ))
    }
}

fn draw_column(ch: &mut dyn RawChallenger, n_enc: usize) -> usize {
    let bits = usize::BITS - (n_enc - 1).leading_zeros();
    let mask = (1usize << bits) - 1;
    loop {
        let c = (ch.sample().to_u128() as usize) & mask;
        if c < n_enc {
            return c;
        }
    }
}

pub struct LadderVerifier {
    shape: Shape,
    pub proximity: Vec<F128>,
    proximity_coeffs: Vec<F128>,
    pub opening: Vec<F128>,
    opening_coeffs: Vec<F128>,
    opened_point: Vec<F128>,
    alphas: Vec<F128>,
    residual: Vec<Vec<F128>>,
}

impl LadderVerifier {
    #[allow(clippy::too_many_arguments)]
    pub fn from_public_transcript(
        shape: Shape,
        proximity: Vec<F128>,
        proximity_coeffs: Vec<F128>,
        opening: Vec<F128>,
        opening_coeffs: Vec<F128>,
        opened_point: Vec<F128>,
        alphas: Vec<F128>,
        residual: Vec<Vec<F128>>,
    ) -> Self {
        Self {
            shape,
            proximity,
            proximity_coeffs,
            opening,
            opening_coeffs,
            opened_point,
            alphas,
            residual,
        }
    }
}

pub struct LadderReplay {
    probe: FoldLadder,
    proximity_coeffs: Vec<F128>,
    proximity: Vec<F128>,
    opening: Vec<F128>,
    opening_coeffs: Vec<F128>,
    opened_point: Vec<F128>,
    alphas: Vec<F128>,
    residual: Vec<Vec<F128>>,
}

impl LadderReplay {
    pub fn new(shape: Shape) -> Self {
        Self {
            probe: FoldLadder::new(shape),
            proximity_coeffs: Vec::new(),
            proximity: Vec::new(),
            opening: Vec::new(),
            opening_coeffs: Vec::new(),
            opened_point: Vec::new(),
            alphas: Vec::new(),
            residual: Vec::new(),
        }
    }

    fn shape(&self) -> Shape {
        self.probe.shape
    }
}

impl rmip_iop::replay::TranscriptVerifier for LadderReplay {
    type Opening = LadderVerifier;

    fn replay_commit(&mut self, tape: &mut dyn rmip_iop::replay::Replay) {
        match tape.take_post() {
            Some(id) if id == LEVEL0_ORACLE => {}
            _ => tape.fault_backend(),
        }
        let (m, n) = (self.shape().m(), self.shape().n());
        self.proximity_coeffs = (0..m).map(|_| tape.sample()).collect();
        match tape.take_ordinary() {
            Some(bytes) if bytes.len() == n * 16 => {
                self.proximity = rmip_iop::replay::decode_elements(&bytes);
            }
            _ => tape.fault_backend(),
        }
    }

    fn replay_open(&mut self, point: &[F128], tape: &mut dyn rmip_iop::replay::Replay) {
        self.opened_point = point.to_vec();
        self.opening_coeffs = eq_table(&point[..self.shape().nu_row]);
        let n = self.shape().n();
        match tape.take_ordinary() {
            Some(bytes) if bytes.len() == n * 16 => {
                self.opening = rmip_iop::replay::decode_elements(&bytes);
            }
            _ => tape.fault_backend(),
        }
        let a1 = tape.sample();
        match tape.take_post() {
            Some(id) if id == LEVEL1_ORACLE => {}
            _ => tape.fault_backend(),
        }
        let a2 = tape.sample();
        self.alphas = vec![a1, a2];
        let rows = self.shape().m() / 4;
        match tape.take_ordinary() {
            Some(bytes) if bytes.len() == rows * n * 16 => {
                self.residual = rmip_iop::replay::decode_elements(&bytes)
                    .chunks(n)
                    .map(|r| r.to_vec())
                    .collect();
            }
            _ => tape.fault_backend(),
        }
    }

    fn query_plan(&self, tape: &mut dyn rmip_iop::replay::Replay) -> Box<dyn QueryPlan<F128>> {
        self.probe.query_plan(tape)
    }

    fn opening_verifier(&self) -> LadderVerifier {
        LadderVerifier::from_public_transcript(
            self.shape(),
            self.proximity.clone(),
            self.proximity_coeffs.clone(),
            self.opening.clone(),
            self.opening_coeffs.clone(),
            self.opened_point.clone(),
            self.alphas.clone(),
            self.residual.clone(),
        )
    }

    fn q_long(&self) -> usize {
        2 * self.shape().queries()
    }
}

impl OpeningVerifier<F128> for LadderVerifier {
    fn check(&self, claims: &[Claim<F128>], answers: &[(OracleId, Coord, Symbol<F128>)]) -> bool {
        let nu = self.shape.nu_row + self.shape.nu_col;
        if answers.is_empty()
            || self.alphas.len() != 2
            || self.proximity.len() != self.shape.n()
            || self.proximity_coeffs.len() != self.shape.m()
            || self.opening.len() != self.shape.n()
            || self.opening_coeffs.len() != self.shape.m()
            || self.opened_point.len() != nu
            || self.residual.len() != self.shape.m() / 4
            || self.residual.iter().any(|row| row.len() != self.shape.n())
            || claims.iter().any(|claim| claim.point.len() != nu)
        {
            return false;
        }
        let at = |row: &[F128], col: Coord| {
            code::eval_at(
                row,
                F128::new(u128::try_from(col).expect("a codeword column fits u128")),
            )
        };

        if !answers.len().is_multiple_of(2) {
            return false;
        }
        for pair in answers.chunks(2) {
            let (id0, c0, s0) = &pair[0];
            let (id1, c1, s1) = &pair[1];
            if *id0 != LEVEL0_ORACLE || *id1 != LEVEL1_ORACLE || c0 != c1 {
                return false;
            }
            let col = *c0;
            if col >= Coord::from(self.shape.n_enc)
                || s0.0.len() != self.shape.m()
                || s1.0.len() != self.shape.m() / 2
            {
                return false;
            }
            let dot = |c: &[F128]| {
                c.iter()
                    .zip(&s0.0)
                    .map(|(a, b)| a.mul(*b))
                    .fold(F128::ZERO, |x, y| x.add(y))
            };
            if dot(&self.proximity_coeffs) != at(&self.proximity, col) {
                return false;
            }
            if dot(&self.opening_coeffs) != at(&self.opening, col) {
                return false;
            }
            if fold_column(&s0.0, self.alphas[0]) != s1.0 {
                return false;
            }
            let folded1 = fold_column(&s1.0, self.alphas[1]);
            if folded1.len() != self.residual.len() {
                return false;
            }
            if folded1
                .iter()
                .zip(&self.residual)
                .any(|(v, r)| *v != at(r, col))
            {
                return false;
            }
        }

        claims.iter().all(|c| {
            c.point == self.opened_point
                && eval_mle(&self.opening, &c.point[self.shape.nu_row..]) == c.value
        })
    }
}

impl CommitmentBackend for FoldLadder {
    type Field = F128;
    type Commitment = usize;
    type Verifier = LadderVerifier;

    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
        let (m, n) = (self.shape.m(), self.shape.n());
        assert_eq!(z.len(), m * n, "trace length must match the shape");
        assert!(m >= 4, "the ladder folds twice, so m must be at least 4");
        self.rows = z.chunks(n).map(|r| r.to_vec()).collect();

        self.oracles = vec![LongOracle::new(
            LEVEL0_ORACLE,
            Alphabet::Column { elements: m },
            Self::encode_rows(&self.rows, self.shape.n_enc),
        )];
        ch.post_long_oracle(&self.oracles[0]);

        self.proximity_coeffs = (0..m).map(|_| ch.sample()).collect();
        self.proximity = self.combine(&self.proximity_coeffs);
        let mut proximity_message = Vec::with_capacity(self.proximity.len() * 16);
        for value in &self.proximity {
            proximity_message.extend_from_slice(&value.to_be_bytes());
        }
        ch.record_ordinary(&proximity_message);
        self.ordinary_bytes = proximity_message.len();
        m * n
    }

    fn open(&mut self, claims: &[Claim<F128>], ch: &mut dyn RawChallenger) {
        assert_eq!(claims.len(), 1, "the ladder opens one batched claim");
        let p = &claims[0].point;
        self.opened_point = p.clone();
        self.opening_coeffs = eq_table(&p[..self.shape.nu_row]);
        self.opening = self.combine(&self.opening_coeffs);
        let mut opening_message = Vec::with_capacity(self.opening.len() * 16);
        for value in &self.opening {
            opening_message.extend_from_slice(&value.to_be_bytes());
        }
        ch.record_ordinary(&opening_message);
        self.ordinary_bytes += opening_message.len();

        let a1 = ch.sample();
        self.alphas = vec![a1];
        let m1 = fold_rows(&self.rows, a1);

        let symbols: Vec<Symbol<F128>> = self.oracles[0]
            .symbols
            .iter()
            .map(|s| Symbol(fold_column(&s.0, a1)))
            .collect();
        self.oracles.push(LongOracle::new(
            LEVEL1_ORACLE,
            Alphabet::Column { elements: m1.len() },
            symbols,
        ));
        ch.post_long_oracle(self.oracles.last().expect("level 1 was just appended"));

        let a2 = ch.sample();
        self.alphas.push(a2);
        self.residual = fold_rows(&m1, a2);
        let residual_bytes = self.residual.len() * self.shape.n() * 16;
        let mut residual_message = Vec::with_capacity(residual_bytes);
        for row in &self.residual {
            for value in row {
                residual_message.extend_from_slice(&value.to_be_bytes());
            }
        }
        ch.record_ordinary(&residual_message);
        self.ordinary_bytes += residual_message.len();
    }

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
        let picked = (0..self.shape.queries())
            .map(|_| draw_column(ch, self.shape.n_enc))
            .collect();
        Box::new(Ladder { picked, next: 0 })
    }

    fn oracles(&self) -> &[LongOracle<F128>] {
        &self.oracles
    }

    fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
        let col = usize::try_from(coord).expect("fold-ladder coordinate fits usize");
        let x = code::domain_point(self.shape.n_enc, col);
        match id {
            LEVEL0_ORACLE => Symbol(rmip_opt::encode::eval_column(&self.rows, x)),
            LEVEL1_ORACLE => {
                let alpha = fold_challenge_from(h);
                Symbol(fold_column(
                    &rmip_opt::encode::eval_column(&self.rows, x),
                    alpha,
                ))
            }
            other => panic!("no oracle {other} in this ladder"),
        }
    }

    fn verifier(&self) -> LadderVerifier {
        LadderVerifier::from_public_transcript(
            self.shape,
            self.proximity.clone(),
            self.proximity_coeffs.clone(),
            self.opening.clone(),
            self.opening_coeffs.clone(),
            self.opened_point.clone(),
            self.alphas.clone(),
            self.residual.clone(),
        )
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        assert_eq!(
            trace_len,
            self.shape.m() * self.shape.n(),
            "trace length must match the public backend shape"
        );
        self.params(claims)
    }

    fn params(&self, claims: usize) -> BackendParams {
        let q = self.shape.queries();
        let m = self.shape.m() as u64;
        let miss_base = rmip_iop::backend::div_error_upward(
            (self.shape.n_enc + self.shape.n()) as f64,
            (2 * self.shape.n_enc) as f64,
        );
        let flat = rmip_iop::backend::pow_error_upward(miss_base, q);
        BackendParams {
            name: "fold-ladder",
            q_long: 2 * q,
            eps_bind: 0.0,
            eps_open: rmip_iop::backend::composed_opening_error(
                rmip_iop::backend::sum_errors_upward(&[
                    rmip_iop::backend::mul_error_upward(2.0, flat),
                    rmip_iop::backend::mul_error_upward(
                        2.0,
                        rmip_iop::backend::div_error_upward(
                            self.shape.n_enc as f64,
                            2f64.powi(128),
                        ),
                    ),
                ]),
                claims,
                self.shape.nu_row + self.shape.nu_col,
            ),
            regime: Regime::UniqueDecoding,
            boolean_binding: BooleanBinding::ProvedByRelation,
            schedule: vec![
                LevelParams {
                    dim: self.shape.nu_col,
                    rate_inv: self.shape.n_enc / self.shape.n(),
                    queries: q,
                    lambda_bits: self.shape.lambda_bits,
                },
                LevelParams {
                    dim: self.shape.nu_col,
                    rate_inv: self.shape.n_enc / self.shape.n(),
                    queries: q,
                    lambda_bits: self.shape.lambda_bits,
                },
            ],
            l_long: Coord::from(self.shape.n_enc as u128 * u128::from(m + m / 2)),
            a_long: q as u64 * (m + m / 2),
            alphabet: vec![
                Alphabet::Column {
                    elements: self.shape.m(),
                },
                Alphabet::Column {
                    elements: self.shape.m() / 2,
                },
            ],
            ordinary_bytes: self.ordinary_bytes,
            r_open: 3,
            audit: AuditMode::OneOf,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rmip_iop::mle::eval_mle;

    fn z(nu: usize) -> Vec<F128> {
        (0..1usize << nu)
            .map(|i| F128::from_bool((i * 11 + 5) % 7 < 3))
            .collect()
    }
    fn claim(z: &[F128], nu: usize) -> Vec<Claim<F128>> {
        let p: Vec<F128> = (0..nu).map(|i| F128::new(i as u128 * 7919 + 13)).collect();
        vec![Claim {
            value: eval_mle(z, &p),
            point: p,
        }]
    }

    struct SelfReading(FoldLadder);

    impl CommitmentBackend for SelfReading {
        type Field = F128;
        type Commitment = usize;
        type Verifier = LadderVerifier;
        fn commit(&mut self, t: &[F128], ch: &mut dyn RawChallenger) -> usize {
            self.0.commit(t, ch)
        }
        fn open(&mut self, c: &[Claim<F128>], ch: &mut dyn RawChallenger) {
            self.0.open(c, ch)
        }
        fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
            self.0.query_plan(ch)
        }
        fn oracles(&self) -> &[LongOracle<F128>] {
            self.0.oracles()
        }
        fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
            if id == LEVEL1_ORACLE {
                let alpha = self.0.alphas.first().copied().unwrap_or(F128::ZERO);
                let col = usize::try_from(coord).expect("test coordinate fits usize");
                let x = code::domain_point(self.0.shape.n_enc, col);
                let folded = fold_rows(&self.0.rows, alpha);
                return Symbol(folded.iter().map(|r| code::eval_at(r, x)).collect());
            }
            self.0.reconstruct(id, h, coord)
        }
        fn verifier(&self) -> LadderVerifier {
            self.0.verifier()
        }
        fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
            self.0.public_params(trace_len, claims)
        }
        fn params(&self, k: usize) -> BackendParams {
            self.0.params(k)
        }
    }

    #[test]
    fn item_ten_catches_a_reconstruct_that_reads_self() {
        let nu = 10;
        let t = z(nu);
        let c = claim(&t, nu);
        let sh = Shape::balanced(nu, 4, 4.0);

        let mut b = SelfReading(FoldLadder::new(sh));
        let ordinary = rmip_backend_conformance::check(&mut b, &t, &c, false);
        assert!(
            ordinary.is_empty(),
            "the bug must be INVISIBLE to the ordinary suite, or item 10 proves nothing: {ordinary:#?}"
        );

        let v = rmip_backend_conformance::check_prefix_only_reconstruction(
            || SelfReading(FoldLadder::new(sh)),
            &t,
            &c,
        );
        assert!(
            !v.is_empty(),
            "item 10 did not catch a reconstruct that reads self"
        );
        assert!(v.iter().all(|x| x.item == 10), "{v:#?}");

        let ok = rmip_backend_conformance::check_prefix_only_reconstruction(
            || FoldLadder::new(sh),
            &t,
            &c,
        );
        assert!(ok.is_empty(), "{ok:#?}");
    }
}
