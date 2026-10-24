use rmip_field::{BinaryField, F128};

pub fn interpolation_points(degree: usize) -> Vec<F128> {
    (0..=degree).map(|i| F128::new(i as u128)).collect()
}

pub fn eq<F: BinaryField>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len(), "eq needs two points of the same arity");
    a.iter()
        .zip(b)
        .fold(F::ONE, |acc, (x, y)| acc.mul(F::ONE.add(*x).add(*y)))
}

pub fn eq_table<F: BinaryField>(r: &[F]) -> Vec<F> {
    let mut table = vec![F::ONE];
    scaled_eq_table_into(r, F::ONE, &mut table);
    table
}

pub fn scaled_eq_table_into<F: BinaryField>(r: &[F], scale: F, out: &mut Vec<F>) {
    out.clear();
    out.reserve(1usize << r.len());
    out.push(scale);
    for &rk in r.iter().rev() {
        let n = out.len();
        out.resize(2 * n, F::ZERO);
        for i in 0..n {
            let v = out[i];
            let p = v.mul(rk);
            out[i] = v.add(p);
            out[n + i] = p;
        }
    }
}

pub fn scaled_eq_table<F: BinaryField>(r: &[F], scale: F) -> Vec<F> {
    let mut out = Vec::new();
    scaled_eq_table_into(r, scale, &mut out);
    out
}

pub fn eval_mle<F: BinaryField>(table: &[F], r: &[F]) -> F {
    assert_eq!(table.len(), 1 << r.len(), "table length must be 2^|r|");
    let mut cur = table.to_vec();
    for &rk in r {
        fold_in_place(&mut cur, rk);
    }
    cur[0]
}

pub fn fold_in_place<F: BinaryField>(table: &mut Vec<F>, r: F) {
    let half = table.len() / 2;
    for h in 0..half {
        let lo = table[h];
        let hi = table[half + h];
        table[h] = lo.add(r.mul(lo.add(hi)));
    }
    table.truncate(half);
}

pub fn lift(bits: &[bool]) -> Vec<F128> {
    bits.iter().map(|&b| F128::from_bool(b)).collect()
}

#[derive(Clone, Debug)]
pub struct InterpolationDomain {
    points: Vec<F128>,
    weights: Vec<F128>,
    binius: binius_math::univariate::EvaluationDomain<crate::binius_channel::B128>,
}

impl InterpolationDomain {
    pub fn from_points(points: Vec<F128>) -> Self {
        let weights = points
            .iter()
            .enumerate()
            .map(|(j, &pj)| {
                let den = points
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != j)
                    .fold(F128::ONE, |a, (_, &pk)| a.mul(pj.add(pk)));
                den.inv().expect("interpolation points must be distinct")
            })
            .collect();
        let binius = binius_math::univariate::EvaluationDomain::from_points(
            points
                .iter()
                .copied()
                .map(crate::binius_channel::up)
                .collect(),
        );
        Self {
            points,
            weights,
            binius,
        }
    }

    pub fn points(&self) -> &[F128] {
        &self.points
    }

    pub fn weights(&self) -> &[F128] {
        &self.weights
    }

    pub fn extrapolate(&self, values: &[F128], r: F128) -> F128 {
        assert_eq!(values.len(), self.points.len());
        use crate::binius_channel::{down, up};
        let vals: Vec<_> = values.iter().copied().map(up).collect();
        down(self.binius.extrapolate(&vals, up(r)))
    }
}

pub fn monomial_from_evals(evals: &[F128]) -> Vec<F128> {
    let degree = evals.len() - 1;
    static MATRICES: std::sync::OnceLock<Vec<Vec<Vec<F128>>>> = std::sync::OnceLock::new();
    let ms = MATRICES.get_or_init(|| {
        (0..=MAX_CACHED_DEGREE)
            .map(|d| {
                let dom = InterpolationDomain::from_points(interpolation_points(d));
                (0..=d)
                    .map(|j| {
                        let mut c = vec![dom.weights()[j]];
                        for (k, &pk) in dom.points().iter().enumerate() {
                            if k == j {
                                continue;
                            }
                            let mut next = vec![F128::ZERO; c.len() + 1];
                            for (i, &ci) in c.iter().enumerate() {
                                next[i] = next[i].add(ci.mul(pk));
                                next[i + 1] = next[i + 1].add(ci);
                            }
                            c = next;
                        }
                        c
                    })
                    .collect()
            })
            .collect()
    });
    if degree > MAX_CACHED_DEGREE {
        let dom = InterpolationDomain::from_points(interpolation_points(degree));
        let mut out = vec![F128::ZERO; degree + 1];
        for (j, &ev) in evals.iter().enumerate() {
            let mut c = vec![dom.weights()[j]];
            for (k, &pk) in dom.points().iter().enumerate() {
                if k == j {
                    continue;
                }
                let mut next = vec![F128::ZERO; c.len() + 1];
                for (i, &ci) in c.iter().enumerate() {
                    next[i] = next[i].add(ci.mul(pk));
                    next[i + 1] = next[i + 1].add(ci);
                }
                c = next;
            }
            for (i, &cij) in c.iter().enumerate() {
                out[i] = out[i].add(cij.mul(ev));
            }
        }
        return out;
    }
    let cols = &ms[degree];
    let mut out = vec![F128::ZERO; degree + 1];
    for (j, col) in cols.iter().enumerate() {
        for (i, &cij) in col.iter().enumerate() {
            out[i] = out[i].add(cij.mul(evals[j]));
        }
    }
    out
}

pub fn horner(coeffs: &[F128], x: F128) -> F128 {
    use crate::binius_channel::{down, up};
    let cb: Vec<_> = coeffs.iter().copied().map(up).collect();
    down(binius_math::univariate::evaluate_univariate(&cb, &up(x)))
}

const MAX_CACHED_DEGREE: usize = 8;

pub fn canonical_domain(degree: usize) -> &'static InterpolationDomain {
    assert!(
        degree <= MAX_CACHED_DEGREE,
        "canonical_domain: degree {degree} is past the cached maximum {MAX_CACHED_DEGREE}; \
         build an InterpolationDomain from the points instead"
    );
    static DOMAINS: std::sync::OnceLock<Vec<InterpolationDomain>> = std::sync::OnceLock::new();
    let domains = DOMAINS.get_or_init(|| {
        (0..=MAX_CACHED_DEGREE)
            .map(|d| InterpolationDomain::from_points(interpolation_points(d)))
            .collect()
    });
    &domains[degree]
}

fn is_canonical(points: &[F128]) -> bool {
    !points.is_empty()
        && points.len() <= MAX_CACHED_DEGREE + 1
        && points
            .iter()
            .enumerate()
            .all(|(i, p)| p.to_u128() == i as u128)
}

pub fn interpolate(points: &[F128], values: &[F128], r: F128) -> F128 {
    assert_eq!(points.len(), values.len());
    if is_canonical(points) {
        return canonical_domain(points.len() - 1).extrapolate(values, r);
    }
    InterpolationDomain::from_points(points.to_vec()).extrapolate(values, r)
}
