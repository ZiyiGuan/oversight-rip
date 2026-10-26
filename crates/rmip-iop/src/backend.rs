use crate::transcript::{
    OracleId, RawChallenger, Transcript, TranscriptPrefix, RESERVED_ORACLE_ID,
};
use rmip_field::{BinaryField, F128};
use std::sync::Arc;

pub type Coord = crate::wide::U1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim<F: BinaryField> {
    pub point: Vec<F>,
    pub value: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alphabet {
    Field,
    Column { elements: usize },
}

impl Alphabet {
    pub fn elements(&self) -> usize {
        match self {
            Alphabet::Field => 1,
            Alphabet::Column { elements } => *elements,
        }
    }

    pub fn admits<F: BinaryField>(&self, symbol: &Symbol<F>) -> bool {
        symbol.0.len() == self.elements()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol<F: BinaryField>(pub Vec<F>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LongOracle<F: BinaryField> {
    pub id: OracleId,
    pub alphabet: Alphabet,
    pub symbols: Arc<[Symbol<F>]>,
    logical_len: Coord,
}

impl<F: BinaryField> LongOracle<F> {
    pub fn new(id: OracleId, alphabet: Alphabet, symbols: Vec<Symbol<F>>) -> Self {
        let logical_len = Coord::from(symbols.len());
        Self {
            id,
            alphabet,
            symbols: symbols.into(),
            logical_len,
        }
    }

    pub fn new_lazy(id: OracleId, alphabet: Alphabet, logical_len: usize) -> Self {
        Self::new_lazy_wide(id, alphabet, Coord::from(logical_len))
    }

    pub fn new_lazy_wide(id: OracleId, alphabet: Alphabet, logical_len: Coord) -> Self {
        Self {
            id,
            alphabet,
            symbols: Arc::from([]),
            logical_len,
        }
    }

    pub fn len(&self) -> Coord {
        self.logical_len
    }

    pub fn is_empty(&self) -> bool {
        self.logical_len == Coord::ZERO
    }

    pub fn raw_len(&self) -> Option<Coord> {
        let elements = u64::try_from(self.alphabet.elements()).ok()?;
        self.logical_len.checked_mul_u64(elements)
    }
}

#[derive(Clone, Debug)]
pub struct FrozenOracleSet<F: BinaryField> {
    oracles: Vec<LongOracle<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FreezeError {
    DuplicateId(OracleId),
    RemovedOrReordered(OracleId),
    Replaced(OracleId),
    ContentlessPost(OracleId),
    Unregistered(OracleId),
    Missing(OracleId),
    ReservedId(OracleId),
    SymbolWidth {
        oracle: OracleId,
        declared: usize,
        found: usize,
    },
}

fn check_symbol_widths<F: BinaryField>(oracle: &LongOracle<F>) -> Result<(), FreezeError> {
    let declared = oracle.alphabet.elements();
    match oracle.symbols.iter().find(|s| !oracle.alphabet.admits(s)) {
        Some(bad) => Err(FreezeError::SymbolWidth {
            oracle: oracle.id,
            declared,
            found: bad.0.len(),
        }),
        None => Ok(()),
    }
}

impl<F: BinaryField> FrozenOracleSet<F> {
    pub fn empty() -> Self {
        Self {
            oracles: Vec::new(),
        }
    }

    pub fn capture(oracles: &[LongOracle<F>]) -> Result<Self, FreezeError> {
        let mut frozen = Self {
            oracles: Vec::new(),
        };
        frozen.update(oracles)?;
        Ok(frozen)
    }

    pub fn update(&mut self, current: &[LongOracle<F>]) -> Result<(), FreezeError> {
        if current.len() < self.oracles.len() {
            return Err(FreezeError::RemovedOrReordered(
                self.oracles[current.len()].id,
            ));
        }
        for (old, now) in self.oracles.iter().zip(current) {
            if old.id != now.id {
                return Err(FreezeError::RemovedOrReordered(old.id));
            }
            if old != now {
                return Err(FreezeError::Replaced(old.id));
            }
        }
        for oracle in &current[self.oracles.len()..] {
            if self.oracles.iter().any(|old| old.id == oracle.id) {
                return Err(FreezeError::DuplicateId(oracle.id));
            }
            check_symbol_widths(oracle)?;
            self.oracles.push(oracle.clone());
        }
        Ok(())
    }

    pub fn get(&self, id: OracleId) -> Option<&LongOracle<F>> {
        self.oracles.iter().find(|oracle| oracle.id == id)
    }

    pub fn as_slice(&self) -> &[LongOracle<F>] {
        &self.oracles
    }

    pub fn record_post(&mut self, oracle: &LongOracle<F>) -> Result<(), FreezeError> {
        if self.oracles.iter().any(|old| old.id == oracle.id) {
            return Err(FreezeError::DuplicateId(oracle.id));
        }
        check_symbol_widths(oracle)?;
        self.oracles.push(oracle.clone());
        Ok(())
    }

    pub fn validate_current(&self, current: &[LongOracle<F>]) -> Result<(), FreezeError> {
        for (posted, now) in self.oracles.iter().zip(current) {
            if posted.id != now.id {
                return Err(FreezeError::RemovedOrReordered(posted.id));
            }
            if posted != now {
                return Err(FreezeError::Replaced(posted.id));
            }
        }
        if let Some(extra) = current.get(self.oracles.len()) {
            return Err(FreezeError::Unregistered(extra.id));
        }
        if let Some(missing) = self.oracles.get(current.len()) {
            return Err(FreezeError::Missing(missing.id));
        }
        Ok(())
    }
}

pub struct OracleRecordingChallenger<'a> {
    inner: &'a mut dyn RawChallenger,
    frozen: FrozenOracleSet<F128>,
    announced: Vec<OracleId>,
    error: Option<FreezeError>,
}

impl<'a> OracleRecordingChallenger<'a> {
    pub fn new(inner: &'a mut dyn RawChallenger) -> Self {
        Self {
            inner,
            frozen: FrozenOracleSet::empty(),
            announced: Vec::new(),
            error: None,
        }
    }

    pub fn frozen(&self) -> &FrozenOracleSet<F128> {
        &self.frozen
    }

    pub fn error(&self) -> Option<&FreezeError> {
        self.error.as_ref()
    }

    pub fn post_public_oracle(&mut self, id: OracleId) {
        if self.announce(id) {
            self.inner.post_oracle(id);
        }
    }

    fn reserve(&mut self, id: OracleId) -> bool {
        if id == RESERVED_ORACLE_ID {
            if self.error.is_none() {
                self.error = Some(FreezeError::ReservedId(id));
            }
            false
        } else {
            true
        }
    }

    fn announce(&mut self, id: OracleId) -> bool {
        if self.announced.contains(&id) {
            if self.error.is_none() {
                self.error = Some(FreezeError::DuplicateId(id));
            }
            false
        } else {
            self.announced.push(id);
            true
        }
    }
}

impl RawChallenger for OracleRecordingChallenger<'_> {
    fn sample(&mut self) -> F128 {
        self.inner.sample()
    }

    fn sample_many(&mut self, n: usize) -> Vec<F128> {
        self.inner.sample_many(n)
    }

    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.inner.record_ordinary(bytes);
    }

    fn post_oracle(&mut self, id: OracleId) {
        if self.reserve(id) && self.announce(id) {
            if self.error.is_none() {
                self.error = Some(FreezeError::ContentlessPost(id));
            }
            self.inner.post_oracle(id);
        }
    }

    fn post_long_oracle(&mut self, oracle: &LongOracle<F128>) {
        if !self.reserve(oracle.id) {
            return;
        }
        if !self.announce(oracle.id) {
            return;
        }
        if let Err(error) = self.frozen.record_post(oracle) {
            if self.error.is_none() {
                self.error = Some(error);
            }
            return;
        }
        self.inner.post_long_oracle(oracle);
    }

    fn transcript(&self) -> &Transcript {
        self.inner.transcript()
    }
}

pub trait QueryPlan<F: BinaryField> {
    fn next(&mut self, prev: Option<&Symbol<F>>) -> Option<(OracleId, Coord)>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    UniqueDecoding,
    JohnsonOod { ood_samples: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanBinding {
    Native,
    RequiresExplicitRows,
    ProvedByRelation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LevelParams {
    pub dim: usize,
    pub rate_inv: usize,
    pub queries: usize,
    pub lambda_bits: f64,
}

#[derive(Clone, Debug)]
pub struct BackendParams {
    pub name: &'static str,
    pub q_long: usize,
    pub eps_bind: f64,
    pub eps_open: f64,
    pub regime: Regime,
    pub boolean_binding: BooleanBinding,
    pub schedule: Vec<LevelParams>,
    pub l_long: Coord,
    pub a_long: u64,
    pub alphabet: Vec<Alphabet>,
    pub ordinary_bytes: usize,
    pub r_open: usize,
    pub audit: AuditMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AuditMode {
    #[default]
    OneOf,
    Sequential,
    Rewind,
}

impl AuditMode {
    pub fn refuses_padding(self) -> bool {
        matches!(self, Self::Sequential | Self::Rewind)
    }
}

pub fn relation_error(mu: usize, nu: usize) -> f64 {
    div_error_upward((3 * mu + 2 * nu + 1) as f64, 2f64.powi(128))
}

pub fn sum_errors_upward(terms: &[f64]) -> f64 {
    terms.iter().copied().fold(0.0, |acc, term| {
        assert!(term.is_finite() && term >= 0.0, "invalid error term");
        if term == 0.0 {
            return acc;
        }
        let rounded = acc + term;
        let stepped = f64::from_bits(rounded.to_bits() + 1);
        assert!(stepped.is_finite(), "error sum overflowed");
        stepped
    })
}

pub fn mul_error_upward(lhs: f64, rhs: f64) -> f64 {
    assert!(lhs.is_finite() && lhs >= 0.0, "invalid error factor");
    assert!(rhs.is_finite() && rhs >= 0.0, "invalid error factor");
    if lhs == 0.0 || rhs == 0.0 {
        return 0.0;
    }
    let rounded = lhs * rhs;
    assert!(rounded.is_finite(), "error product overflowed");
    let up = f64::from_bits(rounded.to_bits() + 1);
    assert!(
        up.is_finite(),
        "error product overflowed in the outward step"
    );
    up
}

pub fn div_error_upward(numerator: f64, denominator: f64) -> f64 {
    assert!(
        numerator.is_finite() && numerator >= 0.0,
        "invalid error numerator"
    );
    assert!(
        denominator.is_finite() && denominator > 0.0,
        "invalid error denominator"
    );
    if numerator == 0.0 {
        return 0.0;
    }
    let rounded = numerator / denominator;
    let stepped = f64::from_bits(rounded.to_bits() + 1);
    assert!(stepped.is_finite(), "error quotient overflowed");
    stepped
}

pub fn pow_error_upward(mut base: f64, mut exponent: usize) -> f64 {
    assert!(base.is_finite() && base >= 0.0, "invalid error factor");
    let mut result = 1.0;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul_error_upward(result, base);
        }
        exponent >>= 1;
        if exponent > 0 {
            base = mul_error_upward(base, base);
        }
    }
    result
}

pub fn composed_opening_error(backend_error: f64, claims: usize, nu: usize) -> f64 {
    assert!(
        backend_error.is_finite() && backend_error >= 0.0,
        "invalid backend error"
    );
    if claims == 0 {
        backend_error
    } else {
        sum_errors_upward(&[backend_error, crate::batch::error(claims, nu)])
    }
}

impl BackendParams {
    pub fn same_fixed_schedule(&self, other: &Self) -> bool {
        self.name == other.name
            && self.q_long == other.q_long
            && self.eps_bind.to_bits() == other.eps_bind.to_bits()
            && self.eps_open.to_bits() == other.eps_open.to_bits()
            && self.regime == other.regime
            && self.boolean_binding == other.boolean_binding
            && self.schedule == other.schedule
            && self.l_long == other.l_long
            && self.a_long == other.a_long
            && self.alphabet == other.alphabet
            && self.r_open == other.r_open
            && self.audit == other.audit
    }

    pub fn check_admissible(&self, frontend_has_booleanity_rows: bool) -> Result<(), String> {
        for (label, error) in [("eps_bind", self.eps_bind), ("eps_open", self.eps_open)] {
            if !error.is_finite() || !(0.0..=1.0).contains(&error) {
                return Err(format!(
                    "{}: {label} = {error:?} is not a probability in [0, 1]",
                    self.name
                ));
            }
        }
        if self.q_long == 0 {
            return Err(format!(
                "{}: q_long = 0 is invalid; Step 3 requires at least one auditable occurrence",
                self.name
            ));
        }
        if let Regime::JohnsonOod { ood_samples } = self.regime {
            if ood_samples == 0 {
                return Err(format!(
                    "{}: JohnsonOod with ood_samples = 0 is list decoding with no \
                     candidate selection, which CT1 forbids",
                    self.name
                ));
            }
        }
        if self.boolean_binding == BooleanBinding::RequiresExplicitRows
            && !frontend_has_booleanity_rows
        {
            return Err(format!(
                "{}: binds only to a field-valued table, so the R1CS must carry the N \
                 Booleanity rows",
                self.name
            ));
        }
        if self.alphabet.is_empty() {
            return Err(format!(
                "{}: no answer alphabet declared, so q_long = {} is undefined",
                self.name, self.q_long
            ));
        }
        let q_long = u64::try_from(self.q_long).map_err(|_| {
            format!(
                "{}: q_long = {} does not fit the u64 answer-volume accounting",
                self.name, self.q_long
            )
        })?;
        if self.a_long < q_long {
            return Err(format!(
                "{}: A_long = {} is below q_long = {}; every occurrence returns at \
                 least one element",
                self.name, self.a_long, self.q_long
            ));
        }
        if Coord::from(self.a_long) > self.l_long {
            return Err(format!(
                "{}: A_long = {} exceeds L_long = {}; a query phase cannot return \
                 more volume than the oracles it reads",
                self.name, self.a_long, self.l_long
            ));
        }
        Ok(())
    }
}

pub trait OpeningVerifier<F: BinaryField> {
    fn check(&self, claims: &[Claim<F>], answers: &[(OracleId, Coord, Symbol<F>)]) -> bool;
}

pub trait CommitmentBackend {
    type Field: BinaryField;
    type Commitment: Clone;
    type Verifier: OpeningVerifier<Self::Field>;

    fn commit(&mut self, z: &[Self::Field], ch: &mut dyn RawChallenger) -> Self::Commitment;

    fn open(&mut self, claims: &[Claim<Self::Field>], ch: &mut dyn RawChallenger);

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<Self::Field>>;

    fn oracles(&self) -> &[LongOracle<Self::Field>];

    fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<Self::Field>;

    fn verifier(&self) -> Self::Verifier;

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams;

    fn params(&self, claims: usize) -> BackendParams;
}

#[cfg(test)]
mod error_accounting_tests {
    use super::*;

    #[test]
    fn contentless_backend_posts_are_rejected_but_public_posts_are_explicit() {
        let mut inner = crate::transcript::SeededChallenger::new(1);
        let mut ch = OracleRecordingChallenger::new(&mut inner);
        ch.post_public_oracle(9);
        assert_eq!(ch.error(), None);

        RawChallenger::post_oracle(&mut ch, 10);
        assert_eq!(ch.error(), Some(&FreezeError::ContentlessPost(10)));
        assert!(
            ch.transcript().try_prefix_at(10).is_some(),
            "bad post remains auditable"
        );
    }

    #[test]
    fn a_symbol_wider_than_the_declared_alphabet_is_refused_at_the_posting() {
        let declared = Alphabet::Column { elements: 2 };
        let good = LongOracle::<F128>::new(3, declared, vec![Symbol(vec![F128::ZERO, F128::ONE])]);
        let bad = LongOracle::<F128>::new(
            4,
            declared,
            vec![Symbol(vec![F128::ZERO, F128::ONE]), Symbol(vec![F128::ONE])],
        );

        let mut frozen = FrozenOracleSet::<F128>::empty();
        assert_eq!(frozen.record_post(&good), Ok(()));
        assert_eq!(
            frozen.record_post(&bad),
            Err(FreezeError::SymbolWidth {
                oracle: 4,
                declared: 2,
                found: 1
            })
        );
        assert_eq!(
            FrozenOracleSet::<F128>::capture(std::slice::from_ref(&bad)).err(),
            Some(FreezeError::SymbolWidth {
                oracle: 4,
                declared: 2,
                found: 1
            })
        );

        let lazy = LongOracle::<F128>::new_lazy(5, declared, 16);
        assert_eq!(frozen.record_post(&lazy), Ok(()));
    }

    #[test]
    fn a_backend_cannot_post_the_reserved_padding_identifier() {
        let mut inner = crate::transcript::SeededChallenger::new(3);
        let mut ch = OracleRecordingChallenger::new(&mut inner);
        let oracle = LongOracle::<F128>::new(
            RESERVED_ORACLE_ID,
            Alphabet::Field,
            vec![Symbol(vec![F128::ZERO])],
        );
        ch.post_long_oracle(&oracle);
        assert_eq!(
            ch.error(),
            Some(&FreezeError::ReservedId(RESERVED_ORACLE_ID))
        );
        assert!(
            ch.transcript().try_prefix_at(RESERVED_ORACLE_ID).is_none(),
            "the reserved posting must not reach the transcript"
        );
        assert!(ch.frozen().get(RESERVED_ORACLE_ID).is_none());

        let mut inner = crate::transcript::SeededChallenger::new(3);
        let mut ch = OracleRecordingChallenger::new(&mut inner);
        RawChallenger::post_oracle(&mut ch, RESERVED_ORACLE_ID);
        assert_eq!(
            ch.error(),
            Some(&FreezeError::ReservedId(RESERVED_ORACLE_ID))
        );
        assert!(ch.transcript().try_prefix_at(RESERVED_ORACLE_ID).is_none());

        let mut inner = crate::transcript::SeededChallenger::new(3);
        let mut ch = OracleRecordingChallenger::new(&mut inner);
        ch.post_public_oracle(RESERVED_ORACLE_ID);
        assert_eq!(ch.error(), None);
        assert!(ch.transcript().try_prefix_at(RESERVED_ORACLE_ID).is_some());
    }

    #[test]
    #[should_panic(expected = "error sum overflowed")]
    fn an_overflowing_error_sum_panics_instead_of_poisoning_the_bound() {
        let _ = sum_errors_upward(&[f64::MAX]);
    }

    #[test]
    #[should_panic(expected = "error quotient overflowed")]
    fn an_overflowing_error_quotient_panics_instead_of_poisoning_the_bound() {
        let _ = div_error_upward(f64::MAX, f64::MIN_POSITIVE);
    }

    #[test]
    fn a_positive_cryptographic_term_is_not_rounded_out_of_the_bound() {
        let dominant = 0.125;
        let tiny = 2f64.powi(-128);
        assert_eq!(
            dominant + tiny,
            dominant,
            "fixture must exercise absorption"
        );
        assert!(sum_errors_upward(&[dominant, tiny]) > dominant);
    }

    #[test]
    fn exact_backend_checks_still_charge_batching() {
        assert_eq!(composed_opening_error(0.0, 0, 13), 0.0);
        assert!(composed_opening_error(0.0, 718, 13) >= crate::batch::error(718, 13));
    }

    #[test]
    fn products_and_powers_are_rounded_outward() {
        let base = 0.75;
        assert!(mul_error_upward(base, base) >= base * base);
        assert!(div_error_upward(1.0, 3.0) >= 1.0 / 3.0);
        assert!(pow_error_upward(base, 17) >= base.powi(17));
        assert_eq!(pow_error_upward(base, 0), 1.0);
    }

    #[test]
    fn zero_query_backends_are_inadmissible() {
        let params = BackendParams {
            name: "zero-query",
            q_long: 0,
            eps_bind: 0.0,
            eps_open: 0.0,
            regime: Regime::UniqueDecoding,
            boolean_binding: BooleanBinding::ProvedByRelation,
            schedule: Vec::new(),
            l_long: Coord::from(1u64),
            a_long: 0,
            alphabet: vec![Alphabet::Field],
            ordinary_bytes: 0,
            r_open: 0,
            audit: AuditMode::OneOf,
        };
        assert!(params
            .check_admissible(false)
            .unwrap_err()
            .contains("q_long = 0"));
    }

    #[test]
    fn an_audit_cannot_return_more_volume_than_the_oracles_hold() {
        let sane = BackendParams {
            name: "inflated-audit",
            q_long: 4,
            eps_bind: 0.0,
            eps_open: 0.0,
            regime: Regime::UniqueDecoding,
            boolean_binding: BooleanBinding::ProvedByRelation,
            schedule: Vec::new(),
            l_long: Coord::from(16u64),
            a_long: 4,
            alphabet: vec![Alphabet::Field],
            ordinary_bytes: 0,
            r_open: 0,
            audit: AuditMode::OneOf,
        };
        assert!(
            sane.check_admissible(false).is_ok(),
            "the control must pass"
        );

        let mut boundary = sane.clone();
        boundary.a_long = u64::try_from(boundary.l_long).expect("test l_long fits u64");
        boundary.q_long = usize::try_from(boundary.l_long).expect("test l_long fits usize");
        assert!(
            boundary.check_admissible(false).is_ok(),
            "A_long == L_long is the raw-trace case and must stay admissible"
        );

        let mut inflated = sane.clone();
        inflated.a_long = u64::try_from(sane.l_long).expect("test l_long fits u64") + 1;
        let err = inflated.check_admissible(false).unwrap_err();
        assert!(err.contains("A_long"), "{err}");
        assert!(err.contains("L_long"), "{err}");
    }

    #[test]
    fn backend_error_parameters_must_be_probabilities() {
        let valid = BackendParams {
            name: "bad-error",
            q_long: 1,
            eps_bind: 0.0,
            eps_open: 0.0,
            regime: Regime::UniqueDecoding,
            boolean_binding: BooleanBinding::ProvedByRelation,
            schedule: Vec::new(),
            l_long: Coord::from(1u64),
            a_long: 1,
            alphabet: vec![Alphabet::Field],
            ordinary_bytes: 0,
            r_open: 0,
            audit: AuditMode::OneOf,
        };

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.25, 1.25] {
            let mut params = valid.clone();
            params.eps_bind = bad;
            assert!(params
                .check_admissible(false)
                .unwrap_err()
                .contains("eps_bind"));

            let mut params = valid.clone();
            params.eps_open = bad;
            assert!(params
                .check_admissible(false)
                .unwrap_err()
                .contains("eps_open"));
        }

        assert!(valid.check_admissible(false).is_ok());
    }

    #[test]
    fn lazy_raw_length_overflow_is_rejected_without_wrapping() {
        let oracle =
            LongOracle::<F128>::new_lazy_wide(7, Alphabet::Column { elements: 2 }, Coord::MAX);
        assert_eq!(oracle.raw_len(), None);

        let small = LongOracle::<F128>::new_lazy(8, Alphabet::Column { elements: 3 }, 2);
        assert_eq!(small.raw_len(), Some(Coord::from(6u64)));
    }
}
