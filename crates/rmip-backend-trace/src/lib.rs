use rmip_field::F128;
use rmip_iop::backend::{
    Alphabet, AuditMode, BackendParams, BooleanBinding, Claim, CommitmentBackend, Coord,
    LongOracle, OpeningVerifier, QueryPlan, Regime, Symbol,
};
use rmip_iop::mle::eval_mle;
use rmip_iop::transcript::{OracleId, RawChallenger, TranscriptPrefix};

pub const TRACE_ORACLE: OracleId = 0;

pub struct RawTrace {
    values: Vec<F128>,
    oracles: Vec<LongOracle<F128>>,
    posting_prefix: Option<TranscriptPrefix>,
}

impl Default for RawTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl RawTrace {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            oracles: Vec::new(),
            posting_prefix: None,
        }
    }

    pub fn n(&self) -> usize {
        self.values.len()
    }

    pub fn posting_prefix(&self) -> Option<&TranscriptPrefix> {
        self.posting_prefix.as_ref()
    }
}

struct EveryCoordinate {
    next: usize,
    n: usize,
}

impl QueryPlan<F128> for EveryCoordinate {
    fn next(&mut self, _prev: Option<&Symbol<F128>>) -> Option<(OracleId, Coord)> {
        if self.next >= self.n {
            return None;
        }
        let c = self.next;
        self.next += 1;
        Some((TRACE_ORACLE, Coord::from(c)))
    }
}

pub struct RawTraceVerifier {
    n: usize,
}

impl RawTraceVerifier {
    pub fn from_public_length(n: usize) -> Self {
        Self { n }
    }
}

impl OpeningVerifier<F128> for RawTraceVerifier {
    fn check(
        &self,
        claims: &[Claim<F128>],
        answers: &[(rmip_iop::transcript::OracleId, Coord, Symbol<F128>)],
    ) -> bool {
        if self.n == 0 || !self.n.is_power_of_two() {
            return false;
        }
        let nu = self.n.trailing_zeros() as usize;
        if claims.iter().any(|claim| claim.point.len() != nu) {
            return false;
        }
        let mut table = vec![None; self.n];
        for (id, coord, sym) in answers {
            let Some(ix) = usize::try_from(*coord).ok().filter(|&i| i < self.n) else {
                return false;
            };
            if *id != TRACE_ORACLE || sym.0.len() != 1 || table[ix].is_some() {
                return false;
            }
            table[ix] = Some(sym.0[0]);
        }
        let Some(table) = table.into_iter().collect::<Option<Vec<F128>>>() else {
            return false;
        };
        if table.iter().any(|v| *v != F128::ZERO && *v != F128::ONE) {
            return false;
        }
        claims.iter().all(|c| eval_mle(&table, &c.point) == c.value)
    }
}

fn raw_params(trace_len: usize, claims: usize) -> BackendParams {
    let n = trace_len as u64;
    let nu = trace_len.checked_ilog2().unwrap_or(0) as usize;
    BackendParams {
        name: "raw-trace",
        q_long: trace_len,
        eps_bind: 0.0,
        eps_open: rmip_iop::backend::composed_opening_error(0.0, claims, nu),
        regime: Regime::UniqueDecoding,
        boolean_binding: BooleanBinding::Native,
        schedule: Vec::new(),
        l_long: Coord::from(n),
        a_long: n,
        alphabet: vec![Alphabet::Field],
        ordinary_bytes: 0,
        r_open: 0,
        audit: AuditMode::OneOf,
    }
}

impl CommitmentBackend for RawTrace {
    type Field = F128;
    type Commitment = usize;
    type Verifier = RawTraceVerifier;

    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
        assert!(!z.is_empty(), "empty trace");
        assert!(z.len().is_power_of_two(), "N must be 2^nu");
        assert!(
            z.iter().all(|v| *v == F128::ZERO || *v == F128::ONE),
            "RawTrace declares BooleanBinding::Native; a field-valued table is a \
             different backend -- see the module header"
        );
        self.values = z.to_vec();
        self.oracles = vec![LongOracle::new(
            TRACE_ORACLE,
            Alphabet::Field,
            z.iter().map(|&v| Symbol(vec![v])).collect(),
        )];
        ch.post_long_oracle(&self.oracles[0]);
        self.posting_prefix = Some(ch.transcript().prefix_at(TRACE_ORACLE));
        self.values.len()
    }

    fn open(&mut self, _claims: &[Claim<F128>], _ch: &mut dyn RawChallenger) {}

    fn query_plan(&self, _ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
        Box::new(EveryCoordinate {
            next: 0,
            n: self.values.len(),
        })
    }

    fn oracles(&self) -> &[LongOracle<F128>] {
        &self.oracles
    }

    fn reconstruct(&self, id: OracleId, _h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
        assert_eq!(id, TRACE_ORACLE, "this backend posts exactly one oracle");
        Symbol(vec![
            self.values[usize::try_from(coord).expect("raw-trace coordinate fits usize")],
        ])
    }

    fn verifier(&self) -> RawTraceVerifier {
        RawTraceVerifier::from_public_length(self.values.len())
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        assert!(
            trace_len != 0 && trace_len.is_power_of_two(),
            "RawTrace commits only a non-empty power-of-two trace; \
             public_params was handed {trace_len}"
        );
        raw_params(trace_len, claims)
    }

    fn params(&self, claims: usize) -> BackendParams {
        raw_params(self.values.len(), claims)
    }
}
