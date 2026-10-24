use crate::zerocheck::Zerocheck;
use rmip_field::{BinaryField, F128};

pub fn summand(tau: &[F128], table: &[F128]) -> Zerocheck {
    Zerocheck::new(tau, table.to_vec(), table.to_vec(), table.to_vec())
}

pub fn check_endpoint(v: F128, final_value: F128) -> bool {
    v.mul(v).add(v) == final_value
}

pub fn sqrt(x: F128) -> F128 {
    static TABLES: std::sync::OnceLock<Vec<[u128; 256]>> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        let cols: Vec<u128> = (0..128)
            .map(|i| sqrt_ladder(F128::new(1u128 << i)).to_u128())
            .collect();
        cols.chunks(8)
            .map(|byte_cols| {
                let mut t = [0u128; 256];
                for (i, &c) in byte_cols.iter().enumerate() {
                    let (lo, hi) = t.split_at_mut(1 << i);
                    for (l, h) in lo.iter().zip(hi.iter_mut()) {
                        *h = *l ^ c;
                    }
                }
                t
            })
            .collect()
    });
    let bits = x.to_u128();
    let mut acc = 0u128;
    for (i, table) in tables.iter().enumerate() {
        let byte = ((bits >> (8 * i)) & 0xff) as usize;
        acc ^= table[byte];
    }
    F128::new(acc)
}

pub fn sqrt_ladder(x: F128) -> F128 {
    let mut r = x;
    for _ in 0..(F128::KAPPA - 1) {
        r = r.square();
    }
    r
}

pub fn claim_from_lincheck_endpoint(r_col: &[F128], v_z: F128) -> (Vec<F128>, F128) {
    (r_col.iter().map(|&t| sqrt(t)).collect(), sqrt(v_z))
}

pub fn residual_at(table: &[F128], tau: &[F128]) -> F128 {
    use crate::mle::eval_mle;
    let sigma: Vec<F128> = tau.iter().map(|&t| sqrt(t)).collect();
    eval_mle(table, &sigma).square().add(eval_mle(table, tau))
}
