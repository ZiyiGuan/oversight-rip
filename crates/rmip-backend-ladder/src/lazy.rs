use crate::backend::LadderOf;
use rmip_mip_prover::{LazyBackend, OracleMeta};
use rmip_opt::ntt::NttField;

impl<F: NttField> LazyBackend for LadderOf<F> {
    fn oracle_meta(&self) -> Vec<OracleMeta> {
        self.posted()
            .iter()
            .map(|&id| OracleMeta {
                id,
                alphabet: self.alphabet_of(id),
                len: self.domain_of(id),
            })
            .collect()
    }
}
