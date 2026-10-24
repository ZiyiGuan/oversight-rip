use rmip_field::{F128, F64};
use rmip_packed::subfield::{lagrange_weights, SkipTables, Subfield8};
use rmip_packed::PackedField;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn the_byte_field_is_the_subfield<F: PackedField>() {
    let t = Subfield8::<F>::new();
    for a in 0..=255u8 {
        assert_eq!(t.decode(t.embed[a as usize]), a);
        assert_eq!(
            t.embed[a as usize].pow(256),
            t.embed[a as usize],
            "in the subfield"
        );
        if a != 0 {
            assert_eq!(t.mul(a, t.inv(a)), 1);
        }
        for b in [0u8, 1, 2, 3, 77, 128, 255] {
            assert_eq!(
                t.embed[t.mul(a, b) as usize],
                t.embed[a as usize] * t.embed[b as usize]
            );
            assert_eq!(
                t.embed[(a ^ b) as usize],
                t.embed[a as usize] + t.embed[b as usize]
            );
        }
    }
    assert_eq!(t.embed[0], F::ZERO);
    assert_eq!(t.embed[1], F::ONE);
    assert_eq!(t.embed[2], F::subfield_8_root());
    assert_eq!(t.mul(0x80, 0x02), 0x1B);
    assert_eq!(t.mul(0x53, 0xCA), 0x01, "the AES inverse pair");
}

fn the_lane_multiply_agrees<F: PackedField>() {
    let t = Subfield8::<F>::new();
    let mut s = 0x55_u64;
    for _ in 0..200 {
        let a: [u8; 64] = std::array::from_fn(|_| lcg(&mut s) as u8);
        let b: [u8; 64] = std::array::from_fn(|_| lcg(&mut s) as u8);
        let lanes = t.mul64(&a, &b);
        for i in 0..64 {
            assert_eq!(lanes[i], t.mul(a[i], b[i]), "lane {i}");
        }
    }
}

fn direct_eval<F: PackedField>(tables: &SkipTables<F>, bits: u64, z: F) -> F {
    let w = lagrange_weights(&tables.s, z);
    let mut acc = F::ZERO;
    for (j, wj) in w.iter().enumerate() {
        if (bits >> j) & 1 == 1 {
            acc += *wj;
        }
    }
    acc
}

fn the_skip_tables_interpolate<F: PackedField>() {
    let tables = SkipTables::<F>::new();
    assert_eq!(tables.s.len(), 64);
    assert_eq!(tables.lambda.len(), 64);
    assert_eq!(tables.s[0], F::ZERO);
    let mut s = 0x5C1F_u64;
    for _ in 0..200 {
        let bits = lcg(&mut s) ^ (lcg(&mut s) << 53);
        let on_lambda = tables.on_lambda(bits);
        for (li, &l) in tables.lambda.iter().enumerate() {
            assert_eq!(
                tables.subfield.embed[on_lambda[li] as usize],
                direct_eval(&tables, bits, l)
            );
        }
        for (j, &sj) in tables.s.iter().enumerate() {
            assert_eq!(
                direct_eval(&tables, bits, sj),
                F::from_bool((bits >> j) & 1 == 1)
            );
        }
        let z = F::from_bits(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128);
        let w = tables.fold_weights(z);
        let folded = w
            .iter()
            .enumerate()
            .filter(|(j, _)| (bits >> j) & 1 == 1)
            .fold(F::ZERO, |acc, (_, wj)| acc + *wj);
        assert_eq!(folded, direct_eval(&tables, bits, z));
        let values: Vec<F> = tables
            .lambda
            .iter()
            .map(|&l| direct_eval(&tables, bits, l))
            .collect();
        let lw = tables.lambda_weights(z);
        let interp = values
            .iter()
            .zip(&lw)
            .fold(F::ZERO, |acc, (v, w)| acc + *v * *w);
        assert_eq!(interp, direct_eval(&tables, bits, z));
        let vanish = |x: F| tables.s.iter().fold(F::ONE, |acc, &sj| acc * (x + sj));
        let p_on_lambda: Vec<F> = tables
            .lambda
            .iter()
            .map(|&l| direct_eval(&tables, bits, l) * vanish(l))
            .collect();
        let cw = tables.combined_lambda_weights(z);
        let p_z = p_on_lambda
            .iter()
            .zip(&cw)
            .fold(F::ZERO, |acc, (v, w)| acc + *v * *w);
        assert_eq!(p_z, direct_eval(&tables, bits, z) * vanish(z));
    }
}

#[test]
fn the_byte_field_is_the_subfield_at_both_fields() {
    the_byte_field_is_the_subfield::<F64>();
    the_byte_field_is_the_subfield::<F128>();
}

#[test]
fn the_lane_multiply_agrees_at_both_fields() {
    the_lane_multiply_agrees::<F64>();
    the_lane_multiply_agrees::<F128>();
}

#[test]
fn the_skip_tables_interpolate_at_both_fields() {
    the_skip_tables_interpolate::<F64>();
    the_skip_tables_interpolate::<F128>();
}

fn the_kernels_match_their_references<F: PackedField>() {
    let tables = SkipTables::<F>::new();
    let mut state = 0x5eed_0000_dead_beefu64;
    for round in 0..300 {
        let mut a = [0u64; 8];
        let mut b = [0u64; 8];
        for k in 0..8 {
            a[k] = lcg(&mut state);
            b[k] = lcg(&mut state);
            if round % 3 == 1 && k % 2 == 0 {
                a[k] = 0;
            }
            if round % 5 == 2 && k == 3 {
                b[k] = 0;
            }
            if round % 7 == 3 {
                b[k] = u64::MAX;
            }
        }
        for &w in a.iter().chain(&b) {
            assert_eq!(tables.on_lambda(w), tables.on_lambda_portable(w));
        }
        assert_eq!(tables.products8(&a, &b), tables.products8_portable(&a, &b));
        assert_eq!(
            rmip_packed::subfield::transpose_8x64(&a),
            rmip_packed::subfield::transpose_8x64_portable(&a)
        );
        let xi = F::subfield_8_root();
        let folded = tables.products8(&a, &b);
        for l in [0usize, 7, 31, 63] {
            let mut expect = F::ZERO;
            let mut xk = F::ONE;
            for k in 0..8 {
                let av = tables.subfield.embed[tables.on_lambda(a[k])[l] as usize];
                let bv = tables.subfield.embed[tables.on_lambda(b[k])[l] as usize];
                expect += xk * av * bv;
                xk *= xi;
            }
            assert_eq!(
                tables.subfield.embed[folded[l] as usize], expect,
                "lane {l}"
            );
        }
    }
}

#[test]
fn the_kernels_match_their_references_at_both_fields() {
    the_kernels_match_their_references::<F64>();
    the_kernels_match_their_references::<F128>();
}
