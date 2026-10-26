use rmip_iop::backend::{
    div_error_upward, mul_error_upward, pow_error_upward, sum_errors_upward, AuditMode,
};

pub fn audit_denominator(audit: AuditMode, q_long: usize) -> usize {
    match audit {
        AuditMode::OneOf => q_long,
        AuditMode::Sequential | AuditMode::Rewind => 1,
    }
}

fn exact_count(count: usize) -> f64 {
    assert!(
        count as u128 <= (1u128 << 53),
        "query count exceeds exact f64 integer range"
    );
    count as f64
}

fn positive_difference_down(lhs: f64, rhs: f64) -> f64 {
    let rounded = lhs - rhs;
    assert!(
        rounded.is_finite() && rounded > 0.0,
        "soundness gap must be positive"
    );
    let down = f64::from_bits(rounded.to_bits() - 1);
    assert!(
        down > 0.0,
        "soundness gap is below the accounting precision"
    );
    down
}

pub fn one_shot_bound(s_iop: f64, q_long: usize) -> f64 {
    assert!(q_long >= 1);
    assert!(s_iop.is_finite() && (0.0..1.0).contains(&s_iop));
    let q = exact_count(q_long);
    div_error_upward(sum_errors_upward(&[q - 1.0, s_iop]), q).min(1.0)
}

pub fn repetitions_for(sigma: f64, s_iop: f64, q_long: usize) -> usize {
    assert!(sigma > 0.0 && sigma < 1.0);
    assert!(s_iop < 1.0, "the numerator must be positive");
    assert!(
        one_shot_bound(s_iop, q_long) < 1.0,
        "f64 accounting cannot certify a positive Step-3 catch probability"
    );
    let gap_lower = positive_difference_down(1.0, s_iop);
    let scale = div_error_upward(exact_count(q_long), gap_lower);
    let log_recip = sum_errors_upward(&[(1.0 / sigma).ln()]);
    let estimate = mul_error_upward(scale, log_recip);
    assert!(
        estimate <= usize::MAX as f64,
        "repetition count does not fit usize"
    );
    let closed_form = (estimate.ceil() as usize).max(1);
    let passes = |r| repeated_bound(s_iop, q_long, r) <= sigma;
    let upper = first_passing_from(closed_form, passes);
    if passes(1) {
        return 1;
    }
    let (mut lo, mut hi) = (1usize, upper);
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

pub fn repeated_bound(s_iop: f64, q_long: usize, r: usize) -> f64 {
    pow_error_upward(one_shot_bound(s_iop, q_long), r).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repetitions_by_unit_walk(sigma: f64, s_iop: f64, q_long: usize) -> usize {
        let mut r = 1;
        while repeated_bound(s_iop, q_long, r) > sigma {
            r += 1;
        }
        r
    }

    fn closed_form(sigma: f64, s_iop: f64, q_long: usize) -> f64 {
        q_long as f64 * (1.0 / sigma).ln() / (1.0 - s_iop)
    }

    #[test]
    fn bisection_agrees_with_the_unit_walk() {
        let sigma20 = 2f64.powi(-20);
        for (s, q, r) in [
            (0.1877778957132255_f64, 1usize, 9),
            (0.1836854221764957, 1, 9),
            (0.23260714176424716, 1, 10),
            (0.24004498300200852, 1, 10),
        ] {
            assert_eq!(
                repetitions_for(sigma20, s, q),
                r,
                "ledgered R moved: {s} {q}"
            );
            assert_eq!(r, repetitions_by_unit_walk(sigma20, s, q));
        }
        for sigma in [2f64.powi(-1), 2f64.powi(-20), 2f64.powi(-40), 1e-9] {
            for q in [1usize, 2, 3, 6, 12, 64, 100, 1000, 8192] {
                for i in 0..400u32 {
                    let s = 1.0 - (-(i as f64) * 12.0 * std::f64::consts::LN_2 / 399.0).exp();
                    if one_shot_bound(s, q) >= 1.0 {
                        continue;
                    }
                    let r = repetitions_for(sigma, s, q);
                    assert!(
                        repeated_bound(s, q, r) <= sigma,
                        "sigma={sigma} s={s} q={q}"
                    );
                    assert!(
                        r == 1 || repeated_bound(s, q, r - 1) > sigma,
                        "sigma={sigma} s={s} q={q}"
                    );
                    if closed_form(sigma, s, q) < 20_000.0 {
                        assert_eq!(
                            r,
                            repetitions_by_unit_walk(sigma, s, q),
                            "sigma={sigma} s={s} q={q}"
                        );
                    }
                }
            }
        }
        for (s, q) in [
            (1.0 - 2f64.powi(-20), 8192usize),
            (1.0 - 2f64.powi(-30), 12),
        ] {
            let r = repetitions_for(sigma20, s, q);
            assert!(repeated_bound(s, q, r) <= sigma20);
            assert!(repeated_bound(s, q, r - 1) > sigma20);
        }
    }

    #[test]
    fn equation_d_is_an_outward_rounded_power() {
        for (s_iop, q) in [
            (0.1877778957132255_f64, 1usize),
            (0.18777789571322548, 6),
            (0.24004498300200852, 6),
            (0.1877778957132255, 2048),
        ] {
            let r = repetitions_for(2f64.powi(-20), s_iop, q);
            let one = one_shot_bound(s_iop, q);
            assert!(repeated_bound(s_iop, q, r) >= one.powi(r as i32));
            assert!(
                repeated_bound(s_iop, q, r) <= 2f64.powi(-20),
                "R = {r} must achieve sigma in (D) at s_IOP = {s_iop}, q = {q}"
            );
        }
    }

    #[test]
    fn the_256_view_ledger_counts_are_stable() {
        let sigma = 2f64.powi(-20);
        let s_bundled = 0.1877778957132255;
        let r_bundled = repetitions_for(sigma, s_bundled, 1);
        assert_eq!(r_bundled, 9, "the bundled 256-view R moved");
        assert_eq!(r_bundled * 4, 36);

        let s_flat = 0.18777789571322548;
        let closed = closed_form(sigma, s_flat, 6);
        assert_eq!(
            closed.ceil() as usize,
            103,
            "the flat-vintage closed form moved"
        );
        let r_flat = repetitions_for(sigma, s_flat, 6);
        assert_eq!(r_flat, 96, "the flat 256-view exact minimum moved");
        assert!(
            r_flat <= 103,
            "the exact minimum is at most the closed form"
        );
    }

    #[test]
    fn bound_behaves() {
        assert!((one_shot_bound(0.25, 1) - 0.25).abs() < 1e-12);
        assert!(one_shot_bound(0.25, 30) > one_shot_bound(0.25, 10));
        assert!(one_shot_bound(0.5, 10) > one_shot_bound(0.25, 10));
        let a = repetitions_for(2f64.powi(-20), 0.25, 10);
        let b = repetitions_for(2f64.powi(-20), 0.25, 20);
        let ratio = b as f64 / a as f64;
        assert!((2.02..=2.05).contains(&ratio), "ratio {ratio}");
        let half = repetitions_for(2f64.powi(-20), 0.5, 30);
        let tiny = repetitions_for(2f64.powi(-20), 2f64.powi(-80), 30);
        let ratio = half as f64 / tiny as f64;
        assert!((2.01..=2.03).contains(&ratio), "ratio {ratio}");
    }
}
