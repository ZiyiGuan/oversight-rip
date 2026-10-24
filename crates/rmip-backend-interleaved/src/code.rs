use rmip_opt::ntt::NttField;

pub fn domain<F: NttField>(n_enc: usize) -> Vec<F> {
    (0..n_enc).map(|i| F::from_bits(i as u128)).collect()
}

#[inline]
pub fn domain_point<F: NttField>(n_enc: usize, c: usize) -> F {
    debug_assert!(c < n_enc, "coordinate {c} outside a domain of {n_enc}");
    F::from_bits(c as u128)
}

pub fn eval_at<F: NttField>(row: &[F], x: F) -> F {
    let idx = x.to_bits() as usize;
    let log_n = row.len().next_power_of_two().trailing_zeros() as usize;
    debug_assert!(
        row.len().is_power_of_two(),
        "message length must be a power of two"
    );
    let basis = F::ntt_tables().novel_basis_at(idx, log_n);
    F::dot(row, &basis)
}

pub fn eval_rows_at<F: NttField>(rows: &[&[F]], x: F) -> Vec<F> {
    let Some(first) = rows.first() else {
        return Vec::new();
    };
    debug_assert!(rows.iter().all(|r| r.len() == first.len()), "ragged rows");
    let idx = x.to_bits() as usize;
    let log_n = first.len().next_power_of_two().trailing_zeros() as usize;
    let basis = F::ntt_tables().novel_basis_at(idx, log_n);
    rows.iter().map(|row| F::dot(row, &basis)).collect()
}

pub fn encode<F: NttField>(row: &[F], dom: &[F]) -> Vec<F> {
    dom.iter().map(|&x| eval_at(row, x)).collect()
}

pub fn unique_decoding_radius(n: usize, n_enc: usize) -> f64 {
    assert!(
        n >= 1 && n <= n_enc,
        "an [n_enc, n] code needs 1 <= n <= n_enc"
    );
    assert!(
        n_enc < (1usize << 52),
        "lengths must be exact in f64 for the directed quotient"
    );
    let num = (n_enc - n) as f64;
    let den = 2.0 * n_enc as f64;
    let rounded = num / den;
    if rounded.mul_add(den, -num) > 0.0 {
        f64::from_bits(rounded.to_bits() - 1)
    } else {
        rounded
    }
}

pub fn johnson_radius(n: usize, n_enc: usize, eta: f64) -> f64 {
    assert!(eta > 0.0 && eta < 1.0, "the Johnson slack lies in (0, 1)");
    let rho = n as f64 / n_enc as f64;
    let radius = 1.0 - rho.sqrt() - eta;
    assert!(
        radius > unique_decoding_radius(n, n_enc),
        "a Johnson radius below the unique-decoding one"
    );
    radius
}

pub fn johnson_list_log2(n: usize, n_enc: usize, eta: f64) -> f64 {
    let rho = n as f64 / n_enc as f64;
    (1.0 / (2.0 * eta * rho.sqrt())).log2()
}

pub fn johnson_exceptional_log2(n: usize, n_enc: usize, eta: f64, rows: usize) -> f64 {
    let rho = n as f64 / n_enc as f64;
    let sqrt_rho = rho.sqrt();
    let gamma = 1.0 - sqrt_rho - eta;
    let m = ((sqrt_rho / (2.0 * eta)).ceil() as usize).max(3) as f64;
    let half = m + 0.5;
    let a = (2.0 * half.powi(5) + 3.0 * half * gamma * rho) / (3.0 * rho.powf(1.5)) * n_enc as f64
        + half / sqrt_rho;
    a.log2() + ((rows as f64).log2() - 1.0).max(0.0)
}

pub fn johnson_selection_error(list_log2: f64, pins: usize, kappa: usize) -> f64 {
    assert!(pins >= 1);
    let pairs = {
        let l = list_log2.exp2();
        (l * (l - 1.0) / 2.0).max(1.0)
    };
    let per_pin = 2f64.powi(-(kappa as i32));
    rmip_iop::backend::mul_error_upward(pairs, rmip_iop::backend::pow_error_upward(per_pin, pins))
}

pub fn queries_for(lambda_bits: f64, delta: f64) -> usize {
    fn next_up_nonnegative(value: f64) -> f64 {
        assert!(value.is_finite() && value >= 0.0);
        f64::from_bits(value.to_bits() + 1)
    }

    fn next_down_positive(value: f64) -> f64 {
        assert!(value.is_finite() && value > 0.0);
        f64::from_bits(value.to_bits() - 1)
    }

    fn one_minus_upward(rhs: f64) -> f64 {
        let lhs = 1.0;
        let neg_rhs = -rhs;
        let rounded = lhs + neg_rhs;
        let tail = rounded - lhs;
        let residual = (lhs - (rounded - tail)) + (neg_rhs - tail);
        if residual > 0.0 {
            next_up_nonnegative(rounded)
        } else {
            rounded
        }
    }

    fn mul_upward(lhs: f64, rhs: f64) -> f64 {
        if lhs == 0.0 || rhs == 0.0 {
            return 0.0;
        }
        let rounded = lhs * rhs;
        assert!(rounded.is_finite(), "query escape bound overflowed");
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

    assert!(
        lambda_bits.is_finite() && lambda_bits > 0.0,
        "soundness target must be a finite positive bit count"
    );
    assert!(
        delta.is_finite() && delta > 0.0 && delta < 1.0,
        "relative distance must be finite and lie in (0,1)"
    );

    let miss_base = one_minus_upward(delta);
    assert!(
        miss_base < 1.0,
        "distance is below the f64 accounting precision"
    );
    let denominator = -(1.0 - delta).log2();
    let estimate = lambda_bits / denominator;
    assert!(
        denominator.is_finite()
            && denominator > 0.0
            && estimate.is_finite()
            && estimate > 0.0
            && estimate < usize::MAX as f64,
        "query count does not fit usize"
    );
    let mut q = (estimate.ceil() as usize).max(1);

    let rounded_target = 2.0f64.powf(-lambda_bits);
    assert!(
        rounded_target.is_finite() && rounded_target > 0.0,
        "soundness target is below the f64 accounting precision"
    );
    let target_lower = if lambda_bits.fract() == 0.0 {
        rounded_target
    } else {
        next_down_positive(rounded_target)
    };
    assert!(
        target_lower > 0.0,
        "soundness target is below the f64 accounting precision"
    );
    while pow_upward(miss_base, q) > target_lower {
        q = q.checked_add(1).expect("query count overflow");
    }
    q
}

#[cfg(test)]
mod tests {

    use super::*;
    use rmip_field::{BinaryField, F128};

    #[test]
    fn encoding_is_f_linear() {
        let dom = domain(32);
        let a: Vec<F128> = (0..8).map(|i| F128::new(i * 31 + 1)).collect();
        let b: Vec<F128> = (0..8).map(|i| F128::new(i * 17 + 5)).collect();
        let c = F128::new(0xabc);
        let combo: Vec<F128> = a.iter().zip(&b).map(|(x, y)| x.add(c.mul(*y))).collect();

        let (ea, eb, ec) = (encode(&a, &dom), encode(&b, &dom), encode(&combo, &dom));
        for i in 0..dom.len() {
            assert_eq!(ec[i], ea[i].add(c.mul(eb[i])), "position {i}");
        }
    }

    #[test]
    fn the_decoding_radius_never_exceeds_the_rational() {
        fn cmp_with_rational(v: f64, num: u128, den: u128) -> std::cmp::Ordering {
            let bits = v.to_bits();
            let exp_field = ((bits >> 52) & 0x7ff) as i32;
            assert!(exp_field > 0, "normal floats only");
            let mant = ((bits & ((1u64 << 52) - 1)) | (1u64 << 52)) as u128;
            let exp = exp_field - 1075;
            assert!(exp < 0 && -exp < 64);
            (mant * den).cmp(&(num << (-exp) as u32))
        }
        for rate_inv in 2usize..=9 {
            for n in (1usize..=200).chain([255, 256, 257, 1 << 13, (1 << 13) + 1, 1 << 20]) {
                let n_enc = n * rate_inv;
                let v = unique_decoding_radius(n, n_enc);
                let (num, den) = ((n_enc - n) as u128, 2 * n_enc as u128);
                assert_ne!(
                    cmp_with_rational(v, num, den),
                    std::cmp::Ordering::Greater,
                    "radius above the rational at n={n}, rate 1/{rate_inv}"
                );
                let up = f64::from_bits(v.to_bits() + 1);
                assert_eq!(
                    cmp_with_rational(up, num, den),
                    std::cmp::Ordering::Greater,
                    "radius more than one ulp below the rational at n={n}, rate 1/{rate_inv}"
                );
            }
        }
        for n in [8usize, 32, 128, 1024, 1 << 17] {
            assert_eq!(unique_decoding_radius(n, 4 * n), 0.375);
        }
    }

    #[test]
    fn minimum_distance_holds() {
        let (n, n_enc) = (8usize, 32usize);
        let dom = domain(n_enc);
        let zero = encode(&vec![F128::ZERO; n], &dom);
        assert!(zero.iter().all(|v| *v == F128::ZERO));
        for k in 0..n {
            let mut m = vec![F128::ZERO; n];
            m[k] = F128::ONE;
            let w = encode(&m, &dom);
            let nonzero = w.iter().filter(|v| **v != F128::ZERO).count();
            assert!(
                nonzero > n_enc - n,
                "degree-{k} monomial vanished at too many points: {nonzero}"
            );
        }
    }

    #[test]
    fn query_count_matches_the_formula() {
        let d = unique_decoding_radius(64, 256);
        assert!((d - 0.375).abs() < 1e-12);
        let q = queries_for(4.0, d);
        assert_eq!(q, 6);
        assert_eq!(queries_for(40.0, d), 59);
    }

    #[test]
    fn query_count_postcondition_closes_a_logarithm_rounding_boundary() {
        let lambda = f64::from_bits(0x4000_99fb_e6f0_a67b);
        assert_eq!((lambda / -0.75f64.log2()).ceil() as usize, 5);
        assert_eq!(queries_for(lambda, 0.25), 6);
    }

    #[test]
    fn query_count_rejects_invalid_probability_inputs() {
        assert!(std::panic::catch_unwind(|| queries_for(f64::NAN, 0.25)).is_err());
        assert!(std::panic::catch_unwind(|| queries_for(4.0, f64::INFINITY)).is_err());
        assert!(std::panic::catch_unwind(|| queries_for(0.0, 0.25)).is_err());
    }
}
