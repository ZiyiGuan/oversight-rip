use rmip_iop::backend::{
    Alphabet, BackendParams, Claim, CommitmentBackend, Coord, LongOracle, QueryPlan, Symbol,
};
use rmip_iop::transcript::{OracleId, RawChallenger, TranscriptPrefix};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OracleMeta {
    pub id: OracleId,
    pub alphabet: Alphabet,
    pub len: Coord,
}

impl OracleMeta {
    pub fn of<F: rmip_field::BinaryField>(o: &LongOracle<F>) -> Self {
        Self {
            id: o.id,
            alphabet: o.alphabet,
            len: o.len(),
        }
    }
}

pub trait LazyBackend: CommitmentBackend {
    fn oracle_meta(&self) -> Vec<OracleMeta>;

    fn symbol_at(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<Self::Field> {
        self.reconstruct(id, h, coord)
    }
}

pub struct Eager<B>(pub B);

impl<B: CommitmentBackend> CommitmentBackend for Eager<B> {
    type Field = B::Field;
    type Commitment = B::Commitment;
    type Verifier = B::Verifier;

    fn commit(&mut self, z: &[Self::Field], ch: &mut dyn RawChallenger) -> Self::Commitment {
        self.0.commit(z, ch)
    }
    fn open(&mut self, claims: &[Claim<Self::Field>], ch: &mut dyn RawChallenger) {
        self.0.open(claims, ch)
    }
    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<Self::Field>> {
        self.0.query_plan(ch)
    }
    fn oracles(&self) -> &[LongOracle<Self::Field>] {
        self.0.oracles()
    }
    fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<Self::Field> {
        self.0.reconstruct(id, h, coord)
    }
    fn verifier(&self) -> Self::Verifier {
        self.0.verifier()
    }
    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        self.0.public_params(trace_len, claims)
    }
    fn params(&self, claims: usize) -> BackendParams {
        self.0.params(claims)
    }
}

impl<B: CommitmentBackend> LazyBackend for Eager<B> {
    fn oracle_meta(&self) -> Vec<OracleMeta> {
        self.0.oracles().iter().map(OracleMeta::of).collect()
    }
}

impl<F: rmip_backend_interleaved::NttField> LazyBackend for rmip_backend_interleaved::LigeroOf<F> {
    fn oracle_meta(&self) -> Vec<OracleMeta> {
        self.oracle_shapes()
            .into_iter()
            .map(|(id, alphabet, len)| OracleMeta {
                id,
                alphabet,
                len: Coord::from(len),
            })
            .collect()
    }
}
