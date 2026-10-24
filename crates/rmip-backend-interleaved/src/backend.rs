use crate::code;
use rmip_field::F128;
use rmip_iop::backend::{
    Alphabet, AuditMode, BackendParams, BooleanBinding, Claim, CommitmentBackend, Coord,
    LevelParams, LongOracle, OpeningVerifier, QueryPlan, Regime, Symbol,
};
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::transcript::{OracleId, RawChallenger, TranscriptPrefix};
use rmip_opt::ntt::NttField;

pub const CODEWORD_ORACLE: OracleId = 0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CodeRegime {
    UniqueDecoding,
    Johnson { eta: f64 },
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub nu_row: usize,
    pub nu_col: usize,
    pub n_enc: usize,
    pub lambda_bits: f64,
    pub regime: CodeRegime,
}

impl Shape {
    pub fn balanced(nu: usize, rate_inv: usize, lambda_bits: f64) -> Self {
        let nu_row = nu / 2;
        let nu_col = nu - nu_row;
        Self {
            nu_row,
            nu_col,
            n_enc: (1 << nu_col) * rate_inv,
            lambda_bits,
            regime: CodeRegime::UniqueDecoding,
        }
    }
    pub fn with_johnson(mut self, eta: f64) -> Self {
        self.regime = CodeRegime::Johnson { eta };
        self
    }
    pub fn m(&self) -> usize {
        1 << self.nu_row
    }
    pub fn n(&self) -> usize {
        1 << self.nu_col
    }
    pub fn delta(&self) -> f64 {
        match self.regime {
            CodeRegime::UniqueDecoding => code::unique_decoding_radius(self.n(), self.n_enc),
            CodeRegime::Johnson { eta } => code::johnson_radius(self.n(), self.n_enc, eta),
        }
    }
    pub fn queries(&self) -> usize {
        code::queries_for(self.lambda_bits, self.delta())
    }
}

pub struct LigeroOf<F: NttField = F128> {
    shape: Shape,
    materialise: bool,
    rows: Vec<Vec<F>>,
    oracles: Vec<LongOracle<F128>>,
    proximity: Vec<F>,
    proximity_coeffs: Vec<F>,
    opening: Vec<F>,
    opening_coeffs: Vec<F>,
    opened_point: Vec<F>,
    ordinary_bytes: usize,
}

pub type Ligero = LigeroOf<F128>;
pub type Ligero64 = LigeroOf<rmip_field::F64>;

impl<F: NttField> LigeroOf<F> {
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            materialise: true,
            rows: Vec::new(),
            oracles: Vec::new(),
            proximity: Vec::new(),
            proximity_coeffs: Vec::new(),
            opening: Vec::new(),
            opening_coeffs: Vec::new(),
            opened_point: Vec::new(),
            ordinary_bytes: 0,
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn lazy(shape: Shape) -> Self {
        Self {
            materialise: false,
            ..Self::new(shape)
        }
    }

    pub fn oracle_shapes(&self) -> Vec<(OracleId, Alphabet, usize)> {
        if self.rows.is_empty() {
            return Vec::new();
        }
        vec![(
            CODEWORD_ORACLE,
            Alphabet::Column {
                elements: self.shape.m(),
            },
            self.shape.n_enc,
        )]
    }

    fn combine(&self, c: &[F]) -> Vec<F> {
        let mut acc = vec![F::Unreduced::default(); self.shape.n()];
        for (i, row) in self.rows.iter().enumerate() {
            let ci = c[i];
            if ci == F::ZERO {
                continue;
            }
            F::axpy_unreduced(&mut acc, ci, row);
        }
        F::reduce_all(&acc)
    }
}

pub fn draw_column(ch: &mut dyn RawChallenger, n_enc: usize) -> usize {
    assert!(
        n_enc > 0 && n_enc <= 1 << 32,
        "column index must fit a coin"
    );
    let bits = usize::BITS - (n_enc - 1).leading_zeros();
    let mask = (1usize << bits) - 1;
    loop {
        let c = (ch.sample().to_u128() as usize) & mask;
        if c < n_enc {
            return c;
        }
    }
}

struct Columns {
    picked: Vec<usize>,
    next: usize,
}

impl QueryPlan<F128> for Columns {
    fn next(&mut self, _prev: Option<&Symbol<F128>>) -> Option<(OracleId, Coord)> {
        if self.next >= self.picked.len() {
            return None;
        }
        let c = self.picked[self.next];
        self.next += 1;
        Some((CODEWORD_ORACLE, Coord::from(c)))
    }
}

pub struct LigeroVerifierOf<F: NttField = F128> {
    shape: Shape,
    pub proximity: Vec<F>,
    proximity_coeffs: Vec<F>,
    pub opening: Vec<F>,
    opening_coeffs: Vec<F>,
    opened_point: Vec<F>,
}

pub type LigeroVerifier = LigeroVerifierOf<F128>;

impl<F: NttField> LigeroVerifierOf<F> {
    pub fn from_public_transcript(
        shape: Shape,
        proximity: Vec<F>,
        proximity_coeffs: Vec<F>,
        opening: Vec<F>,
        opening_coeffs: Vec<F>,
        opened_point: Vec<F>,
    ) -> Self {
        Self {
            shape,
            proximity,
            proximity_coeffs,
            opening,
            opening_coeffs,
            opened_point,
        }
    }
}

pub struct LigeroReplay {
    probe: Ligero,
    proximity_coeffs: Vec<F128>,
    proximity: Vec<F128>,
    opening: Vec<F128>,
    opening_coeffs: Vec<F128>,
    opened_point: Vec<F128>,
}

impl LigeroReplay {
    pub fn new(shape: Shape) -> Self {
        Self {
            probe: Ligero::new(shape),
            proximity_coeffs: Vec::new(),
            proximity: Vec::new(),
            opening: Vec::new(),
            opening_coeffs: Vec::new(),
            opened_point: Vec::new(),
        }
    }

    fn shape(&self) -> &Shape {
        &self.probe.shape
    }
}

impl rmip_iop::replay::TranscriptVerifier for LigeroReplay {
    type Opening = LigeroVerifier;

    fn replay_commit(&mut self, tape: &mut dyn rmip_iop::replay::Replay) {
        match tape.take_post() {
            Some(id) if id == CODEWORD_ORACLE => {}
            _ => tape.fault_backend(),
        }
        let m = self.shape().m();
        let n = self.shape().n();
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
    }

    fn query_plan(&self, tape: &mut dyn rmip_iop::replay::Replay) -> Box<dyn QueryPlan<F128>> {
        self.probe.query_plan(tape)
    }

    fn opening_verifier(&self) -> LigeroVerifier {
        LigeroVerifier::from_public_transcript(
            self.probe.shape,
            self.proximity.clone(),
            self.proximity_coeffs.clone(),
            self.opening.clone(),
            self.opening_coeffs.clone(),
            self.opened_point.clone(),
        )
    }

    fn q_long(&self) -> usize {
        self.shape().queries()
    }
}

impl<F: NttField> OpeningVerifier<F128> for LigeroVerifierOf<F> {
    fn check(&self, claims: &[Claim<F128>], answers: &[(OracleId, Coord, Symbol<F128>)]) -> bool {
        let Some(claims) = claims
            .iter()
            .map(|c| {
                Some(Claim {
                    point: F::from_words(&c.point)?,
                    value: F::from_word(c.value)?,
                })
            })
            .collect::<Option<Vec<Claim<F>>>>()
        else {
            return false;
        };
        let Some(answers) = answers
            .iter()
            .map(|(id, col, sym)| Some((*id, *col, Symbol(F::from_words(&sym.0)?))))
            .collect::<Option<Vec<(OracleId, Coord, Symbol<F>)>>>()
        else {
            return false;
        };
        let (claims, answers) = (claims.as_slice(), answers.as_slice());
        let nu = self.shape.nu_row + self.shape.nu_col;
        if answers.is_empty()
            || self.proximity.len() != self.shape.n()
            || self.proximity_coeffs.len() != self.shape.m()
            || self.opening.len() != self.shape.n()
            || self.opening_coeffs.len() != self.shape.m()
            || self.opened_point.len() != nu
            || claims.iter().any(|claim| claim.point.len() != nu)
        {
            return false;
        }
        let at = |col: Coord| {
            code::eval_rows_at(
                &[&self.proximity, &self.opening],
                F::from_bits(u128::try_from(col).expect("a codeword column fits u128")),
            )
        };

        for (id, col, sym) in answers {
            if *id != CODEWORD_ORACLE || *col >= Coord::from(self.shape.n_enc) {
                return false;
            }
            if sym.0.len() != self.shape.m() {
                return false;
            }
            let dot = |c: &[F]| F::dot(c, &sym.0);
            let rhs = at(*col);
            if dot(&self.proximity_coeffs) != rhs[0] {
                return false;
            }
            if dot(&self.opening_coeffs) != rhs[1] {
                return false;
            }
        }

        claims.iter().all(|c| {
            c.point == self.opened_point
                && eval_mle(&self.opening, &c.point[self.shape.nu_row..]) == c.value
        })
    }
}

impl<F: NttField> CommitmentBackend for LigeroOf<F> {
    type Field = F128;
    type Commitment = usize;
    type Verifier = LigeroVerifierOf<F>;

    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
        let (m, n) = (self.shape.m(), self.shape.n());
        assert_eq!(z.len(), m * n, "trace length must match the shape");
        self.rows = z
            .chunks(n)
            .map(|r| {
                F::from_words(r).expect("the committed words carry elements of the backend's field")
            })
            .collect();

        let dom = code::domain::<F>(self.shape.n_enc);
        let symbols: Vec<Symbol<F128>> = if self.materialise {
            rmip_opt::encode::encode_columns(&self.rows, &dom)
                .into_iter()
                .map(|col| Symbol(F::to_words(&col)))
                .collect()
        } else {
            Vec::new()
        };
        let alphabet = Alphabet::Column { elements: m };
        let oracle = if self.materialise {
            LongOracle::new(CODEWORD_ORACLE, alphabet, symbols)
        } else {
            LongOracle::new_lazy(CODEWORD_ORACLE, alphabet, self.shape.n_enc)
        };
        self.oracles = vec![oracle];
        ch.post_long_oracle(&self.oracles[0]);

        self.proximity_coeffs = ch.sample_many(m).into_iter().map(F::from_coin).collect();
        self.proximity = self.combine(&self.proximity_coeffs);
        let proximity_message = F::to_bytes(&self.proximity);
        ch.record_ordinary(&proximity_message);
        self.ordinary_bytes = proximity_message.len();
        m * n
    }

    fn open(&mut self, claims: &[Claim<F128>], ch: &mut dyn RawChallenger) {
        assert_eq!(claims.len(), 1, "Ligero opens one batched claim");
        let p = F::from_words(&claims[0].point)
            .expect("the opened point carries elements of the backend's field");
        self.opened_point = p.clone();
        self.opening_coeffs = eq_table(&p[..self.shape.nu_row]);
        self.opening = self.combine(&self.opening_coeffs);
        let opening_message = F::to_bytes(&self.opening);
        ch.record_ordinary(&opening_message);
        self.ordinary_bytes += opening_message.len();
    }

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
        let q = self.shape.queries();
        let picked: Vec<usize> = (0..q).map(|_| draw_column(ch, self.shape.n_enc)).collect();
        Box::new(Columns { picked, next: 0 })
    }

    fn oracles(&self) -> &[LongOracle<F128>] {
        assert!(
            self.materialise,
            "Ligero::lazy() materialises no codeword: drive it with \
             rmip-mip-prover, which reads coordinates through `reconstruct`"
        );
        &self.oracles
    }

    fn reconstruct(&self, id: OracleId, _h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
        assert_eq!(id, CODEWORD_ORACLE);
        let col = usize::try_from(coord).expect("flat coordinate fits usize");
        let x = code::domain_point::<F>(self.shape.n_enc, col);
        Symbol(F::to_words(&rmip_opt::encode::eval_column(&self.rows, x)))
    }

    fn verifier(&self) -> LigeroVerifierOf<F> {
        LigeroVerifierOf::from_public_transcript(
            self.shape,
            self.proximity.clone(),
            self.proximity_coeffs.clone(),
            self.opening.clone(),
            self.opening_coeffs.clone(),
            self.opened_point.clone(),
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
        let escape = rmip_iop::backend::pow_error_upward(miss_base, q);
        BackendParams {
            name: "interleaved-rs",
            q_long: q,
            eps_bind: 0.0,
            eps_open: rmip_iop::backend::composed_opening_error(
                rmip_iop::backend::sum_errors_upward(&[
                    rmip_iop::backend::mul_error_upward(2.0, escape),
                    rmip_iop::backend::div_error_upward(
                        self.shape.n_enc as f64,
                        2f64.powi(F::KAPPA as i32),
                    ),
                ]),
                claims,
                self.shape.nu_row + self.shape.nu_col,
            ),
            regime: match self.shape.regime {
                CodeRegime::UniqueDecoding => Regime::UniqueDecoding,
                CodeRegime::Johnson { .. } => Regime::JohnsonOod { ood_samples: 0 },
            },
            boolean_binding: BooleanBinding::ProvedByRelation,
            schedule: vec![LevelParams {
                dim: self.shape.nu_col,
                rate_inv: self.shape.n_enc / self.shape.n(),
                queries: q,
                lambda_bits: self.shape.lambda_bits,
            }],
            l_long: Coord::from(self.shape.n_enc as u128 * u128::from(m)),
            a_long: q as u64 * m,
            alphabet: vec![Alphabet::Column {
                elements: self.shape.m(),
            }],
            ordinary_bytes: self.ordinary_bytes,
            r_open: 1,
            audit: AuditMode::OneOf,
        }
    }
}

pub fn measure_check_at_shape(shape: Shape) -> f64 {
    use rmip_field::BinaryField;

    fn xorshift(s: &mut u64) -> F128 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        F128::new(((*s as u128) << 64) | s.wrapping_mul(0x9e37_79b9_7f4a_7c15) as u128)
    }

    let (m, n, q) = (shape.m(), shape.n(), shape.queries());
    let nu = shape.nu_row + shape.nu_col;
    let mut s = 0x243f_6a88_85a3_08d3u64;

    let proximity: Vec<F128> = (0..n).map(|_| xorshift(&mut s)).collect();
    let opening: Vec<F128> = (0..n).map(|_| xorshift(&mut s)).collect();
    let mut proximity_coeffs: Vec<F128> = (0..m).map(|_| xorshift(&mut s)).collect();
    let mut opening_coeffs: Vec<F128> = (0..m).map(|_| xorshift(&mut s)).collect();
    proximity_coeffs[0] = F128::ONE;
    proximity_coeffs[1] = F128::ZERO;
    opening_coeffs[0] = F128::ZERO;
    opening_coeffs[1] = F128::ONE;
    let opened_point: Vec<F128> = (0..nu).map(|_| xorshift(&mut s)).collect();

    let cols: Vec<usize> = (0..q).map(|k| (k * 7919 + 13) % shape.n_enc).collect();
    let answers: Vec<(rmip_iop::transcript::OracleId, Coord, Symbol<F128>)> = cols
        .iter()
        .map(|&col| {
            let mut sym: Vec<F128> = (0..m).map(|_| xorshift(&mut s)).collect();
            let x = F128::new(col as u128);
            let tail_p = (2..m).fold(F128::ZERO, |a, k| a.add(proximity_coeffs[k].mul(sym[k])));
            let tail_o = (2..m).fold(F128::ZERO, |a, k| a.add(opening_coeffs[k].mul(sym[k])));
            sym[0] = code::eval_at(&proximity, x).add(tail_p);
            sym[1] = code::eval_at(&opening, x).add(tail_o);
            (CODEWORD_ORACLE, Coord::from(col), Symbol(sym))
        })
        .collect();

    let value = rmip_iop::mle::eval_mle(&opening, &opened_point[shape.nu_row..]);
    let claims = vec![Claim {
        point: opened_point.clone(),
        value,
    }];

    let v = LigeroVerifier {
        shape,
        proximity,
        proximity_coeffs,
        opening,
        opening_coeffs,
        opened_point,
    };
    assert!(
        v.check(&claims, &answers),
        "synthetic data must PASS: `check` returns on the first mismatching \
         column, so a failing input times a fraction of the real work"
    );

    for _ in 0..20 {
        std::hint::black_box(v.check(&claims, &answers));
    }
    let mut each = f64::INFINITY;
    for _ in 0..7 {
        let budget = std::time::Duration::from_millis(60);
        let t0 = std::time::Instant::now();
        let mut k: u64 = 0;
        while t0.elapsed() < budget {
            std::hint::black_box(v.check(&claims, &answers));
            k += 1;
        }
        each = each.min(t0.elapsed().as_secs_f64() / k as f64);
    }
    each
}

#[cfg(test)]
#[test]
#[ignore = "scaling probe; run with --ignored --nocapture"]
fn verifier_cost_at_sizes_the_prover_cannot_reach() {
    const COMPUTE_NS_PER_CELL: f64 = 3.18;
    const SUMCHECK_US: f64 = 6.0;
    const R: f64 = 100.0;

    println!(
        "{:>5} {:>13} {:>7} {:>6} {:>9} {:>11} {:>12} {:>12} {:>9}",
        "nu", "N", "m", "q", "state", "check 1x", "verifier xR", "compute", "ratio"
    );

    let mut prev = f64::INFINITY;
    for nu in [16usize, 18, 20, 21, 22, 24, 26, 28, 30, 32, 36] {
        let shape = Shape::balanced(nu, 4, 4.0);
        let (m, n, q) = (shape.m(), shape.n(), shape.queries());
        let each = measure_check_at_shape(shape);

        let state = (2 * n + 2 * m + q * m) * 16;
        let n_cells = (1u64 << nu) as f64;
        let verifier = (each * 1e6 + SUMCHECK_US) * R / 1e6;
        let compute = n_cells * COMPUTE_NS_PER_CELL / 1e9;
        let ratio = verifier / compute;
        println!(
            "{:>5} {:>13} {:>7} {:>6} {:>8.2}MB {:>9.1}us {:>10.1}ms {:>10.2}s {:>8.4}x",
            nu,
            1u64 << nu,
            m,
            q,
            state as f64 / 1e6,
            each * 1e6,
            verifier * 1e3,
            compute,
            ratio
        );

        assert!(ratio < prev, "verifier/compute must fall with N");
        prev = ratio;
        let _ = prev;
    }
}
