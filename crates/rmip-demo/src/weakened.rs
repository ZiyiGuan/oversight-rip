use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{
    Alphabet, AuditMode, BackendParams, BooleanBinding, Claim, CommitmentBackend, Coord,
    LongOracle, OpeningVerifier, QueryPlan, Regime, Symbol,
};
use rmip_iop::mle::eval_mle;
use rmip_iop::transcript::{OracleId, RawChallenger, TranscriptPrefix};

const TRACE_ORACLE: OracleId = 0;

#[derive(Default)]
pub struct WeakenedTrace {
    values: Vec<F128>,
    oracles: Vec<LongOracle<F128>>,
    pub honest_declaration: bool,
}

impl WeakenedTrace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn honest_about_itself() -> Self {
        Self {
            honest_declaration: true,
            ..Default::default()
        }
    }

    fn declared_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        let n = trace_len as u64;
        let nu = trace_len.checked_ilog2().unwrap_or(0) as usize;
        BackendParams {
            name: "weakened-trace",
            q_long: trace_len,
            eps_bind: 0.0,
            eps_open: rmip_iop::backend::composed_opening_error(0.0, claims, nu),
            regime: Regime::UniqueDecoding,
            boolean_binding: if self.honest_declaration {
                BooleanBinding::RequiresExplicitRows
            } else {
                BooleanBinding::Native
            },
            schedule: Vec::new(),
            l_long: Coord::from(n),
            a_long: n,
            alphabet: vec![Alphabet::Field],
            ordinary_bytes: 0,
            r_open: 0,
            audit: AuditMode::OneOf,
        }
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

pub struct WeakenedVerifier {
    n: usize,
}

impl OpeningVerifier<F128> for WeakenedVerifier {
    fn check(&self, claims: &[Claim<F128>], answers: &[(OracleId, Coord, Symbol<F128>)]) -> bool {
        let mut table = vec![None; self.n];
        for (id, coord, sym) in answers {
            let Some(ix) = usize::try_from(*coord).ok().filter(|&i| i < self.n) else {
                return false;
            };
            if *id != TRACE_ORACLE || table[ix].is_some() {
                return false;
            }
            table[ix] = Some(sym.0[0]);
        }
        let Some(table) = table.into_iter().collect::<Option<Vec<F128>>>() else {
            return false;
        };
        claims.iter().all(|c| eval_mle(&table, &c.point) == c.value)
    }
}

impl CommitmentBackend for WeakenedTrace {
    type Field = F128;
    type Commitment = usize;
    type Verifier = WeakenedVerifier;

    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
        assert!(z.len().is_power_of_two(), "N must be 2^nu");
        self.values = z.to_vec();
        self.oracles = vec![LongOracle::new(
            TRACE_ORACLE,
            Alphabet::Field,
            z.iter().map(|&v| Symbol(vec![v])).collect(),
        )];
        ch.post_long_oracle(&self.oracles[0]);
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
        assert_eq!(id, TRACE_ORACLE);
        Symbol(vec![
            self.values[usize::try_from(coord).expect("weakened-trace coordinate fits usize")],
        ])
    }

    fn verifier(&self) -> WeakenedVerifier {
        WeakenedVerifier {
            n: self.values.len(),
        }
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        self.declared_params(trace_len, claims)
    }

    fn params(&self, claims: usize) -> BackendParams {
        self.declared_params(self.values.len(), claims)
    }
}

pub fn non_boolean() -> F128 {
    let v = F128::new(0xdead_beef_cafe);
    debug_assert_ne!(v.mul(v), v, "must not be idempotent");
    v
}
