use rmip_field::F128;
use rmip_holo::pack::{orbit, pack, unpack};
use rmip_iop::mle::eval_mle;

fn boolean_column(len: usize, seed: u64) -> Vec<F128> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            }
        })
        .collect()
}

fn point(n: usize, seed: u64) -> Vec<F128> {
    (0..n)
        .map(|k| F128::new((seed as u128) * 7919 + k as u128 * 104_729 + 3))
        .collect()
}

#[test]
fn the_orbit_recovers_every_slot_exactly() {
    for &log_lambda in &[1usize, 2, 3, 7] {
        let lambda = 1 << log_lambda;
        for &nv in &[3usize, 6] {
            let len = 1usize << nv;
            let cols: Vec<Vec<F128>> = (0..lambda)
                .map(|u| boolean_column(len, 1000 + u as u64))
                .collect();
            let packed = pack(&cols);

            let rho = point(nv, 7);
            let evals: Vec<F128> = orbit(&rho, lambda)
                .iter()
                .map(|p| eval_mle(&packed, p))
                .collect();
            let got = unpack(&evals, lambda).expect("a full orbit chunk recovers");

            for (u, col) in cols.iter().enumerate() {
                let want = eval_mle(col, &rho);
                assert_eq!(
                    got[u], want,
                    "lambda={lambda} nv={nv} slot {u}: orbit recovery must be exact"
                );
            }
        }
    }
}

#[test]
fn packing_is_a_bit_or() {
    let cols: Vec<Vec<F128>> = (0..8).map(|u| boolean_column(16, 50 + u)).collect();
    let packed = pack(&cols);
    for z in 0..16 {
        let mut want = 0u128;
        for (u, col) in cols.iter().enumerate() {
            if col[z] == F128::ONE {
                want |= 1u128 << u;
            }
        }
        assert_eq!(
            packed[z].to_u128(),
            want,
            "entry {z} must be the OR of its slot bits"
        );
    }
}

#[test]
fn packing_shrinks_the_committed_table() {
    let lambda = 128usize;
    let len = 1usize << 10;
    let cols: Vec<Vec<F128>> = (0..lambda)
        .map(|u| boolean_column(len, u as u64 + 1))
        .collect();
    let unpacked_elements: usize = cols.iter().map(Vec::len).sum();
    let packed = pack(&cols);
    assert_eq!(
        unpacked_elements / packed.len(),
        lambda,
        "must divide by exactly lambda"
    );
}

#[test]
fn the_forward_map_inverts_unpack() {
    use rmip_holo::pack::evals_from_values;
    for &log_lambda in &[1usize, 2, 4, 7] {
        let lambda = 1usize << log_lambda;
        let values: Vec<F128> = (0..lambda)
            .map(|u| F128::new(u as u128 * 104_729 + 12_345))
            .collect();
        let evals = evals_from_values(&values);
        assert_eq!(
            unpack(&evals, lambda).expect("a full chunk recovers"),
            values,
            "lambda={lambda}: unpack ∘ forward = id"
        );
    }
}

#[test]
fn a_short_or_long_chunk_is_refused_by_unpack() {
    use rmip_holo::pack::{evals_from_values, LAMBDA};
    let values: Vec<F128> = (0..LAMBDA)
        .map(|u| F128::new(u as u128 * 7919 + 1))
        .collect();
    let evals = evals_from_values(&values);
    assert_eq!(evals.len(), LAMBDA, "the forward map is width-preserving");

    assert!(unpack(&evals, LAMBDA).is_some(), "a full chunk is accepted");
    assert!(
        unpack(&evals[..LAMBDA - 1], LAMBDA).is_none(),
        "a short chunk must be refused"
    );
    assert!(
        unpack(&evals[..1], LAMBDA).is_none(),
        "a one-element chunk must be refused"
    );
    let mut long = evals.clone();
    long.push(F128::ZERO);
    assert!(
        unpack(&long, LAMBDA).is_none(),
        "an over-long chunk must be refused"
    );
    assert!(
        unpack(&evals[..LAMBDA - 1], LAMBDA - 1).is_some(),
        "the width is the caller's"
    );
}
