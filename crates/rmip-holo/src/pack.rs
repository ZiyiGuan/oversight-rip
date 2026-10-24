use rmip_field::{BinaryField, F128};
use rmip_iop::booleanity::sqrt;

pub const LOG_LAMBDA: usize = 4;
pub const LAMBDA: usize = 1 << LOG_LAMBDA;

fn beta(u: usize) -> F128 {
    debug_assert!(
        u < F128::KAPPA,
        "beta must stay inside the polynomial basis"
    );
    F128::new(1u128 << u)
}

pub fn pack(columns: &[Vec<F128>]) -> Vec<F128> {
    assert!(
        !columns.is_empty() && columns.len() <= F128::KAPPA,
        "pack takes 1..=KAPPA columns, got {}",
        columns.len()
    );
    let len = columns[0].len();
    let mut out = vec![0u128; len];
    for (u, col) in columns.iter().enumerate() {
        assert_eq!(col.len(), len, "columns must share the entry cube");
        for (z, &v) in col.iter().enumerate() {
            debug_assert!(
                v == F128::ZERO || v == F128::ONE,
                "index columns are Boolean"
            );
            if v == F128::ONE {
                out[z] |= 1u128 << u;
            }
        }
    }
    out.into_iter().map(F128::new).collect()
}

pub fn orbit(rho: &[F128], lambda: usize) -> Vec<Vec<F128>> {
    let mut pts = Vec::with_capacity(lambda);
    let mut cur = rho.to_vec();
    for _ in 0..lambda {
        pts.push(cur.clone());
        for c in cur.iter_mut() {
            *c = c.square();
        }
    }
    pts
}

fn inverse_moore(lambda: usize) -> Vec<Vec<F128>> {
    let mut m: Vec<Vec<F128>> = Vec::with_capacity(lambda);
    let mut roots: Vec<F128> = (0..lambda).map(beta).collect();
    for _ in 0..lambda {
        m.push(roots.clone());
        for r in roots.iter_mut() {
            *r = sqrt(*r);
        }
    }
    let mut inv: Vec<Vec<F128>> = (0..lambda)
        .map(|i| {
            (0..lambda)
                .map(|j| if i == j { F128::ONE } else { F128::ZERO })
                .collect()
        })
        .collect();
    for col in 0..lambda {
        let piv = (col..lambda)
            .find(|&r| m[r][col] != F128::ZERO)
            .expect("the Moore matrix is nonsingular for F_2-independent beta");
        m.swap(col, piv);
        inv.swap(col, piv);
        let s = m[col][col].inv().expect("pivot is nonzero");
        for j in 0..lambda {
            m[col][j] = m[col][j].mul(s);
            inv[col][j] = inv[col][j].mul(s);
        }
        for r in 0..lambda {
            if r == col || m[r][col] == F128::ZERO {
                continue;
            }
            let f = m[r][col];
            for j in 0..lambda {
                let (a, b) = (m[col][j].mul(f), inv[col][j].mul(f));
                m[r][j] = m[r][j].add(a);
                inv[r][j] = inv[r][j].add(b);
            }
        }
    }
    inv
}

type Inverse = &'static Vec<Vec<F128>>;

fn cached_inverse(lambda: usize) -> Inverse {
    use std::sync::OnceLock;
    static CACHE: OnceLock<std::sync::Mutex<Vec<(usize, Inverse)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut g = cache.lock().expect("pack cache poisoned");
    if let Some(&(_, v)) = g.iter().find(|&&(l, _)| l == lambda) {
        return v;
    }
    let leaked: Inverse = Box::leak(Box::new(inverse_moore(lambda)));
    g.push((lambda, leaked));
    leaked
}

pub fn unpack(evals: &[F128], lambda: usize) -> Option<Vec<F128>> {
    if evals.len() != lambda {
        return None;
    }
    let mut f = Vec::with_capacity(lambda);
    for (v, &e) in evals.iter().enumerate() {
        let mut cur = e;
        for _ in 0..v {
            cur = sqrt(cur);
        }
        f.push(cur);
    }
    let inv = cached_inverse(lambda);
    Some(
        (0..lambda)
            .map(|u| {
                let mut acc = [F128::ZERO; 4];
                for (i, (a, b)) in inv[u].iter().zip(&f).enumerate() {
                    acc[i & 3] = acc[i & 3].add(a.mul(*b));
                }
                acc[0].add(acc[1]).add(acc[2]).add(acc[3])
            })
            .collect(),
    )
}

pub fn evals_from_values(values: &[F128]) -> Vec<F128> {
    let lambda = values.len();
    (0..lambda)
        .map(|v| {
            values
                .iter()
                .enumerate()
                .map(|(u, &y)| {
                    let mut p = y;
                    for _ in 0..v {
                        p = p.square();
                    }
                    beta(u).mul(p)
                })
                .fold(F128::ZERO, |a, b| a.add(b))
        })
        .collect()
}
