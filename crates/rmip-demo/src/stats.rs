#[derive(Clone, Copy, Debug)]
pub struct Rate {
    pub successes: usize,
    pub trials: usize,
    pub lo: f64,
    pub hi: f64,
}

impl Rate {
    pub fn new(successes: usize, trials: usize) -> Self {
        assert!(trials > 0);
        let z = 1.959_963_984_540_054_f64;
        let n = trials as f64;
        let p = successes as f64 / n;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denom;
        let half = z / denom * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
        let lo = if successes == 0 {
            0.0
        } else {
            (center - half).max(0.0)
        };
        let hi = if successes == trials {
            1.0
        } else {
            (center + half).min(1.0)
        };
        Self {
            successes,
            trials,
            lo,
            hi,
        }
    }

    fn point(&self) -> f64 {
        self.successes as f64 / self.trials as f64
    }

    pub fn consistent_with(&self, predicted: f64) -> bool {
        predicted >= self.lo && predicted <= self.hi
    }

    fn half_width(&self) -> f64 {
        (self.hi - self.lo) / 2.0
    }

    pub fn resolves(&self, predicted: f64) -> bool {
        if predicted <= 0.0 || predicted >= 1.0 {
            return true;
        }
        self.half_width() < predicted.min(1.0 - predicted)
    }

    pub fn distinguishes(&self, a: f64, b: f64) -> bool {
        self.consistent_with(a) != self.consistent_with(b)
    }

    pub fn trials_needed_to_separate(a: f64, b: f64) -> usize {
        let d = (a - b).abs();
        if d <= 0.0 {
            return usize::MAX;
        }
        let p = (a + b) / 2.0;
        let z = 1.96;
        ((2.0 * z / d).powi(2) * p * (1.0 - p)).ceil() as usize
    }
}

impl std::fmt::Display for Rate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.4} [{:.4}, {:.4}] ({}/{})",
            self.point(),
            self.lo,
            self.hi,
            self.successes,
            self.trials
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_stays_in_range_at_small_p() {
        let r = Rate::new(62, 4000);
        assert!(r.lo > 0.0, "lower bound must stay positive");
        assert!(r.consistent_with(1.0 / 64.0));
        let z = Rate::new(0, 100);
        assert_eq!(z.lo, 0.0);
        assert!(
            z.hi > 0.0,
            "upper bound must be positive even with no successes"
        );
        assert!(
            z.consistent_with(0.0),
            "a rate of 0 must be consistent with 0"
        );
        let o = Rate::new(100, 100);
        assert_eq!(o.hi, 1.0);
        assert!(o.lo < 1.0);
        assert!(
            o.consistent_with(1.0),
            "a rate of 1 must be consistent with 1"
        );
    }

    #[test]
    fn exact_endpoints_hold_at_every_sample_size() {
        for n in [
            1usize, 2, 7, 10, 100, 333, 500, 1000, 2000, 4000, 9999, 10_000, 65_536,
        ] {
            let none = Rate::new(0, n);
            assert_eq!(none.lo, 0.0, "n = {n}");
            assert!(
                none.consistent_with(0.0),
                "n = {n}: 0/{n} must be consistent with 0.0"
            );

            let all = Rate::new(n, n);
            assert_eq!(all.hi, 1.0, "n = {n}");
            assert!(
                all.consistent_with(1.0),
                "n = {n}: {n}/{n} must be consistent with 1.0"
            );

            assert!(none.hi > 0.0 && all.lo < 1.0, "n = {n}");
        }
    }

    fn exact_and_bound(r: usize) -> (f64, f64) {
        let inst = rmip_frontend::mnist::MnistInstance::new(r);
        let t = rmip_frontend::report::local_sampling_repetitions(inst.eps_rob, 0.125) as i32;
        let ell = inst.ell() as f64;
        let exact = (1.0 - inst.minimal_bad_set_size() as f64 / ell).powi(t);
        let bound = (1.0 - inst.eps_rob.num as f64 / inst.eps_rob.den as f64).powi(t);
        (exact, bound)
    }

    #[test]
    fn the_256_view_case_cannot_separate_exact_from_bound() {
        let (exact, bound) = exact_and_bound(256);
        assert!(exact < bound, "the exact escape sits below the bound");
        let demo_trials = 4_000usize;
        let need = Rate::trials_needed_to_separate(exact, bound);
        assert!(
            need > 10 * demo_trials,
            "would need {need} trials, far past the demo's {demo_trials}"
        );
        let r = Rate::new((exact * demo_trials as f64).round() as usize, demo_trials);
        assert!(r.consistent_with(exact) && r.consistent_with(bound));
        assert!(!r.distinguishes(exact, bound));
    }

    #[test]
    fn the_16_view_case_can_separate_exact_from_bound() {
        let (exact, bound) = exact_and_bound(16);
        assert!(exact < bound, "the exact escape sits below the bound");
        let need = Rate::trials_needed_to_separate(exact, bound);
        assert!(need < 4_000, "only {need} trials needed");
    }
}
