use rmip_field::F128;
use rmip_iop::backend::{Alphabet, FreezeError, FrozenOracleSet, LongOracle, Symbol};
use std::sync::Arc;

fn oracle(id: usize, value: F128) -> LongOracle<F128> {
    LongOracle::new(id, Alphabet::Field, vec![Symbol(vec![value])])
}

#[test]
fn a_frozen_posting_cannot_be_changed_through_the_backends_handle() {
    let mut current = vec![oracle(0, F128::ONE)];
    let mut frozen = FrozenOracleSet::capture(&current).unwrap();

    Arc::make_mut(&mut current[0].symbols)[0] = Symbol(vec![F128::ZERO]);
    assert_eq!(frozen.get(0).unwrap().symbols[0], Symbol(vec![F128::ONE]));
    assert_eq!(frozen.update(&current), Err(FreezeError::Replaced(0)));
}

#[test]
fn later_phases_may_only_append_new_postings() {
    let mut current = vec![oracle(0, F128::ONE)];
    let mut frozen = FrozenOracleSet::capture(&current).unwrap();
    current.push(oracle(1, F128::ZERO));
    frozen.update(&current).unwrap();
    assert_eq!(frozen.as_slice().len(), 2);

    current.swap(0, 1);
    assert_eq!(
        frozen.update(&current),
        Err(FreezeError::RemovedOrReordered(0))
    );
}
