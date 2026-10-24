use rmip_field::{BinaryField, F128};
use rmip_iop::mle::{
    canonical_domain, eq, eq_table, eval_mle, fold_in_place, interpolate, interpolation_points,
    lift, scaled_eq_table, InterpolationDomain,
};

fn pt(v: &[u128]) -> Vec<F128> {
    v.iter().map(|&x| F128::new(x)).collect()
}

#[test]
fn eq_table_is_the_mle_kernel() {
    for m in 1..=7 {
        let n = 1usize << m;
        let table: Vec<F128> = (0..n).map(|i| F128::new((i as u128) * 31 + 7)).collect();
        let r = pt(&(0..m as u128).map(|k| k * 101 + 3).collect::<Vec<_>>());
        let folded = eval_mle(&table, &r);
        let summed = eq_table(&r)
            .iter()
            .zip(&table)
            .map(|(e, t)| e.mul(*t))
            .fold(F128::ZERO, |a, b| a.add(b));
        assert_eq!(folded, summed, "m = {m}");
    }
}

#[test]
fn first_variable_is_most_significant() {
    let r = pt(&[5, 9]);
    let t = eq_table(&r);
    let (r0, r1) = (r[0], r[1]);
    let (n0, n1) = (F128::ONE.add(r0), F128::ONE.add(r1));
    assert_eq!(t[0], n0.mul(n1), "index 0 = (0,0)");
    assert_eq!(t[1], n0.mul(r1), "index 1 = (0,1)");
    assert_eq!(t[2], r0.mul(n1), "index 2 = (1,0) -- r[0] is the HIGH bit");
    assert_eq!(t[3], r0.mul(r1), "index 3 = (1,1)");
}

#[test]
fn mle_interpolates_the_table_on_the_cube() {
    let m = 5;
    let n = 1usize << m;
    let table: Vec<F128> = (0..n).map(|i| F128::new((i as u128) * 17 + 2)).collect();
    for s in 0..n {
        let r: Vec<F128> = (0..m)
            .rev()
            .map(|k| F128::from_bool((s >> k) & 1 == 1))
            .collect();
        assert_eq!(eval_mle(&table, &r), table[s], "s = {s}");
    }
}

#[test]
fn eq_is_a_partition_of_unity() {
    for m in 1..=6 {
        let r = pt(&(0..m as u128).map(|k| k * 7 + 11).collect::<Vec<_>>());
        let total = eq_table(&r).iter().fold(F128::ZERO, |a, b| a.add(*b));
        assert_eq!(total, F128::ONE, "m = {m}");
    }
}

#[test]
fn interpolation_points_are_distinct() {
    for d in 0..=8 {
        let p = interpolation_points(d);
        assert_eq!(p.len(), d + 1);
        for i in 0..p.len() {
            for j in (i + 1)..p.len() {
                assert_ne!(p[i], p[j], "points {i} and {j} collide at degree {d}");
            }
        }
    }
    assert_eq!(F128::ONE.add(F128::ONE), F128::ZERO);
    assert_ne!(interpolation_points(2)[2], F128::ZERO);
    assert_eq!(interpolation_points(2)[2], F128::new(2));
}

#[test]
fn interpolate_recovers_the_polynomial() {
    let coeffs = pt(&[3, 5, 7]);
    let eval_at = |x: F128| {
        let mut acc = F128::ZERO;
        let mut pow = F128::ONE;
        for c in &coeffs {
            acc = acc.add(c.mul(pow));
            pow = pow.mul(x);
        }
        acc
    };
    let points = interpolation_points(2);
    let values: Vec<F128> = points.iter().map(|&p| eval_at(p)).collect();
    for probe in [0u128, 1, 2, 3, 12345, u128::MAX] {
        let r = F128::new(probe);
        assert_eq!(
            interpolate(&points, &values, r),
            eval_at(r),
            "probe {probe:#x}"
        );
    }
}

#[test]
fn folding_agrees_with_evaluation() {
    let m = 6;
    let table: Vec<F128> = (0..1usize << m)
        .map(|i| F128::new((i as u128) * 13 + 1))
        .collect();
    let r = pt(&[2, 3, 5, 7, 11, 13]);
    let mut cur = table.clone();
    for (k, &rk) in r.iter().enumerate() {
        fold_in_place(&mut cur, rk);
        assert_eq!(cur.len(), 1 << (m - k - 1));
    }
    assert_eq!(cur[0], eval_mle(&table, &r));
}

#[test]
fn lift_embeds_the_prime_subfield() {
    assert_eq!(lift(&[false, true]), vec![F128::ZERO, F128::ONE]);
}

#[test]
fn eq_in_closed_form_matches_the_tables() {
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        F128::new(((s as u128) << 64) | s.wrapping_mul(0x9e37_79b9_7f4a_7c15) as u128)
    };
    for nu in 1usize..=8 {
        for _ in 0..40 {
            let a: Vec<F128> = (0..nu).map(|_| next()).collect();
            let b: Vec<F128> = (0..nu).map(|_| next()).collect();
            assert_eq!(eq(&a, &b), eval_mle(&eq_table(&a), &b), "nu={nu}");
        }
        let a: Vec<F128> = (0..nu).map(|_| next()).collect();
        assert_eq!(eq(&a, &a), F128::ONE, "eq(a,a) must be 1, nu={nu}");
        let zeros = vec![F128::ZERO; nu];
        let mut ones = zeros.clone();
        ones[0] = F128::ONE;
        assert_eq!(
            eq(&zeros, &ones),
            F128::ZERO,
            "distinct Boolean points, nu={nu}"
        );
    }
}

#[test]
fn scaled_eq_table_matches_the_obvious_form() {
    let mut s: u64 = 0xbb67_ae85_84ca_a73b;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        F128::new(((s as u128) << 64) | s.wrapping_mul(0x9e37_79b9_7f4a_7c15) as u128)
    };
    for nu in 0usize..=9 {
        let r: Vec<F128> = (0..nu).map(|_| next()).collect();
        let scale = next();
        let got = scaled_eq_table(&r, scale);
        let want: Vec<F128> = eq_table(&r).iter().map(|v| v.mul(scale)).collect();
        assert_eq!(got, want, "nu={nu}");
        assert_eq!(got.len(), 1usize << nu);
    }
    for nu in 0usize..=9 {
        let r: Vec<F128> = (0..nu).map(|_| next()).collect();
        assert_eq!(
            scaled_eq_table(&r, F128::ONE),
            eq_table(&r),
            "nu={nu}, scale=1"
        );
    }
}

#[test]
fn cached_and_uncached_interpolation_agree() {
    for degree in 1..=4usize {
        let points = interpolation_points(degree);
        let shifted: Vec<F128> = points.iter().map(|p| p.add(F128::new(0x1234))).collect();
        assert!(shifted
            .iter()
            .enumerate()
            .any(|(i, p)| p.to_u128() != i as u128));

        for seed in 0..8u128 {
            let values: Vec<F128> = (0..=degree)
                .map(|i| F128::new(seed * 7919 + i as u128 * 104_729 + 1))
                .collect();

            let domain = InterpolationDomain::from_points(points.clone());
            for r in [
                F128::ZERO,
                F128::ONE,
                F128::new(2),
                F128::new(0xdead_beef),
                points[degree],
            ] {
                assert_eq!(
                    interpolate(&points, &values, r),
                    domain.extrapolate(&values, r)
                );
                for (i, &p) in points.iter().enumerate() {
                    assert_eq!(interpolate(&points, &values, p), values[i]);
                }
                let _ = r;
            }

            let vals_shift: Vec<F128> = shifted
                .iter()
                .map(|&x| interpolate(&points, &values, x))
                .collect();
            for r in [F128::ZERO, F128::new(3), F128::new(0xfeed)] {
                assert_eq!(
                    interpolate(&shifted, &vals_shift, r),
                    interpolate(&points, &values, r),
                    "degree {degree} seed {seed}"
                );
            }
        }
    }
}

#[test]
fn barycentric_weights_are_the_inverse_products() {
    for degree in 1..=4usize {
        let points = interpolation_points(degree);
        let d = InterpolationDomain::from_points(points.clone());
        for (j, &pj) in points.iter().enumerate() {
            let den = points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .fold(F128::ONE, |a, (_, &pk)| a.mul(pj.add(pk)));
            assert_eq!(d.weights()[j].mul(den), F128::ONE);
        }
        assert_eq!(d.points(), &points[..]);
    }
}

#[test]
fn monomial_conversion_agrees_with_interpolation() {
    use rmip_iop::mle::{horner, monomial_from_evals};
    for degree in 1..=4usize {
        let points = interpolation_points(degree);
        for seed in 0..6u128 {
            let evals: Vec<F128> = (0..=degree)
                .map(|i| F128::new(seed * 7919 + i as u128 * 104_729 + 3))
                .collect();
            let coeffs = monomial_from_evals(&evals);
            assert_eq!(coeffs.len(), degree + 1);
            for r in [F128::ZERO, F128::ONE, F128::new(2), F128::new(0xdead_beef)] {
                assert_eq!(horner(&coeffs, r), interpolate(&points, &evals, r));
            }
            for (j, &p) in points.iter().enumerate() {
                assert_eq!(horner(&coeffs, p), evals[j]);
            }
        }
    }
}

#[test]
#[should_panic(expected = "past the cached maximum")]
fn canonical_domain_states_its_own_bound() {
    let _ = canonical_domain(9);
}

#[test]
fn interpolate_past_the_cached_degree_takes_the_uncached_path() {
    for degree in 0..=8usize {
        assert_eq!(
            canonical_domain(degree).points(),
            &interpolation_points(degree)[..]
        );
    }

    let degree = 9usize;
    let points = interpolation_points(degree);
    let values: Vec<F128> = (0..=degree)
        .map(|i| F128::new(i as u128 * 104_729 + 17))
        .collect();
    let built = InterpolationDomain::from_points(points.clone());
    for r in [F128::ZERO, F128::ONE, F128::new(2), F128::new(0xdead_beef)] {
        assert_eq!(
            interpolate(&points, &values, r),
            built.extrapolate(&values, r)
        );
    }
    for (i, &p) in points.iter().enumerate() {
        assert_eq!(interpolate(&points, &values, p), values[i], "node {i}");
    }
}
