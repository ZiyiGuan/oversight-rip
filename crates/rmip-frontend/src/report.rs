use crate::Ratio;

fn next_up_nonnegative(value: f64) -> f64 {
    assert!(
        value.is_finite() && value >= 0.0,
        "expected a finite nonnegative value"
    );
    f64::from_bits(value.to_bits() + 1)
}

fn next_down_positive(value: f64) -> f64 {
    assert!(
        value.is_finite() && value > 0.0,
        "expected a finite positive value"
    );
    f64::from_bits(value.to_bits() - 1)
}

fn u64_upward(value: u64) -> f64 {
    let rounded = value as f64;
    if rounded as u128 >= value as u128 {
        rounded
    } else {
        next_up_nonnegative(rounded)
    }
}

fn u64_downward(value: u64) -> f64 {
    let rounded = value as f64;
    if rounded as u128 <= value as u128 {
        rounded
    } else {
        next_down_positive(rounded)
    }
}

fn div_upward(numerator: f64, denominator: f64) -> f64 {
    assert!(numerator.is_finite() && numerator > 0.0);
    assert!(denominator.is_finite() && denominator > 0.0);
    let rounded = numerator / denominator;
    assert!(
        rounded.is_finite() && rounded > 0.0,
        "local miss probability is not representable"
    );
    if rounded.is_subnormal() {
        return next_up_nonnegative(rounded);
    }
    let residual = (-rounded).mul_add(denominator, numerator);
    if residual > 0.0 {
        next_up_nonnegative(rounded)
    } else {
        rounded
    }
}

fn mul_upward(lhs: f64, rhs: f64) -> f64 {
    assert!(lhs.is_finite() && lhs >= 0.0);
    assert!(rhs.is_finite() && rhs >= 0.0);
    if lhs == 0.0 || rhs == 0.0 {
        return 0.0;
    }
    let rounded = lhs * rhs;
    assert!(rounded.is_finite(), "local miss bound overflowed");
    if rounded == 0.0 || rounded.is_subnormal() {
        return next_up_nonnegative(rounded);
    }
    let residual = lhs.mul_add(rhs, -rounded);
    if residual > 0.0 {
        next_up_nonnegative(rounded)
    } else {
        rounded
    }
}

fn pow_upward(mut base: f64, mut exponent: usize) -> f64 {
    let mut result = 1.0;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul_upward(result, base);
        }
        exponent >>= 1;
        if exponent > 0 {
            base = mul_upward(base, base);
        }
    }
    result
}

pub fn local_sampling_repetitions(eps_rob: Ratio, sigma_loc: f64) -> usize {
    assert!(
        sigma_loc.is_finite() && sigma_loc > 0.0 && sigma_loc < 1.0,
        "local error target must be finite and lie in (0,1)"
    );
    assert!(eps_rob.den > 0, "robustness denominator must be positive");
    assert!(
        eps_rob.num > 0 && eps_rob.num <= eps_rob.den,
        "robustness gap must lie in (0,1]"
    );
    if eps_rob.num == eps_rob.den {
        return 1;
    }

    let miss = div_upward(
        u64_upward(eps_rob.den - eps_rob.num),
        u64_downward(eps_rob.den),
    );
    assert!(
        miss < 1.0,
        "robustness gap is below the f64 accounting precision"
    );

    let estimate = sigma_loc.ln() / miss.ln();
    assert!(
        estimate.is_finite() && estimate > 0.0 && estimate < usize::MAX as f64,
        "local sampling count does not fit usize"
    );
    let t = (estimate.ceil() as usize).max(1);
    first_passing_from(t, |t| pow_upward(miss, t) <= sigma_loc)
}

fn first_passing_from(start: usize, passes: impl Fn(usize) -> bool) -> usize {
    if passes(start) {
        return start;
    }
    let mut lo = start;
    let mut step = 1usize;
    let mut hi = lo.checked_add(step).expect("count overflow");
    while !passes(hi) {
        lo = hi;
        step = step.checked_mul(2).expect("count overflow");
        hi = lo.checked_add(step).expect("count overflow");
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if passes(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

#[derive(Clone, Debug)]
pub struct ParamReport {
    pub n_trace: usize,
    pub nu: usize,
    pub m: usize,
    pub mu: usize,
    pub z: usize,
    pub ell: usize,
    pub b_max: usize,
    pub k: usize,
    pub depth: usize,
    pub b_tr: usize,
    pub eps_rob: Ratio,
    pub booleanity: crate::r1cs::Booleanity,
}

impl ParamReport {
    pub fn t_for(&self, sigma_loc: f64) -> usize {
        local_sampling_repetitions(self.eps_rob, sigma_loc)
    }

    pub fn k_max(&self, t: usize) -> usize {
        t * (self.b_max + 1)
    }
}

#[cfg(test)]
mod walk_tests {
    use super::*;

    fn by_unit_walk(eps_rob: Ratio, sigma_loc: f64) -> usize {
        if eps_rob.num == eps_rob.den {
            return 1;
        }
        let miss = div_upward(
            u64_upward(eps_rob.den - eps_rob.num),
            u64_downward(eps_rob.den),
        );
        let mut t = ((sigma_loc.ln() / miss.ln()).ceil() as usize).max(1);
        while pow_upward(miss, t) > sigma_loc {
            t += 1;
        }
        t
    }

    #[test]
    fn the_sampling_count_bisection_agrees_with_the_unit_walk() {
        let targets = [0.125_f64, 0.23019025, 1.0 / 3.0, 2f64.powi(-20)];
        for sigma in targets {
            for den in 1..=64u64 {
                for num in 1..=den {
                    let r = Ratio { num, den };
                    assert_eq!(
                        local_sampling_repetitions(r, sigma),
                        by_unit_walk(r, sigma),
                        "{num}/{den} at {sigma}"
                    );
                }
            }
            for num in (1..16384u64).step_by(97) {
                let r = Ratio { num, den: 16384 };
                assert_eq!(local_sampling_repetitions(r, sigma), by_unit_walk(r, sigma));
            }
        }
        let tiny = Ratio {
            num: 1,
            den: 1 << 40,
        };
        let sigma = 2f64.powi(-20);
        let t = local_sampling_repetitions(tiny, sigma);
        let miss = div_upward(u64_upward(tiny.den - 1), u64_downward(tiny.den));
        assert!(pow_upward(miss, t) <= sigma);
        assert!(pow_upward(miss, t - 1) > sigma);
    }
}
