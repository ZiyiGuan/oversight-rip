use rmip_backend_interleaved::{code, Ligero, LigeroVerifier, Shape, CODEWORD_ORACLE};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{Claim, CommitmentBackend, Coord, OpeningVerifier, Symbol};
use rmip_iop::batch;
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::transcript::{Entry, OsChallenger, RawChallenger, SeededChallenger};

fn shape(nu: usize) -> Shape {
    Shape::balanced(nu, 4, 4.0)
}

fn table(nu: usize) -> Vec<F128> {
    (0..1usize << nu)
        .map(|i| F128::from_bool((i * 7 + 3) % 5 < 2))
        .collect()
}

fn one_claim(z: &[F128], nu: usize) -> Vec<Claim<F128>> {
    let raw: Vec<Claim<F128>> = (0..6)
        .map(|k| {
            let p: Vec<F128> = (0..nu)
                .map(|i| F128::new((k + 1) * 7919 + i as u128))
                .collect();
            Claim {
                value: eval_mle(z, &p),
                point: p,
            }
        })
        .collect();
    let mut ch = SeededChallenger::new(1);
    vec![batch::reduce(&raw, z, &mut ch).expect("batching verifies")]
}

fn honest(nu: usize, seed: u64) -> (Ligero, Vec<Claim<F128>>) {
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(seed);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);
    (b, claims)
}

fn five_sigma(p: f64, trials: usize) -> f64 {
    5.0 * (p * (1.0 - p) / trials as f64).sqrt()
}

fn victims(n_enc: usize, bad: usize) -> Vec<bool> {
    let mut mark = vec![false; n_enc];
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut placed = 0usize;
    while placed < bad {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let c = (s as usize) % n_enc;
        if !mark[c] {
            mark[c] = true;
            placed += 1;
        }
    }
    mark
}

fn corrupt_rate(b: &Ligero, claims: &[Claim<F128>], bad: usize, trials: usize) -> f64 {
    let sh = b.shape();
    let mark = victims(sh.n_enc, bad);
    let v = b.verifier();
    let mut random = OsChallenger::new();
    let mut accepted = 0usize;
    for trial in 0..trials {
        let mut plan = b.query_plan(&mut random);
        let mut answers = Vec::new();
        let mut hit = false;
        while let Some((id, coord)) = plan.next(None) {
            let ix = usize::try_from(coord).expect("coord fits usize");
            let mut sym = b.oracles()[0].symbols[ix].clone();
            if mark[ix] {
                sym.0[0] = sym.0[0].add(F128::ONE);
                hit = true;
            }
            answers.push((id, coord, sym));
        }
        let ok = v.check(claims, &answers);
        assert_eq!(
            ok, !hit,
            "trial {trial}: acceptance must be exactly `no queried column was \
             overwritten` (bad = {bad})"
        );
        accepted += ok as usize;
    }
    accepted as f64 / trials as f64
}

#[test]
fn corrupting_a_fraction_of_the_columns_never_beats_the_declared_escape_term() {
    let nu = 10;
    let (b, claims) = honest(nu, 0xA11);
    let sh = b.shape();
    let q = sh.queries();
    assert_eq!((sh.n(), sh.n_enc, q), (32, 128, 6));
    let declared = (1.0 - sh.delta()).powi(q as i32);
    let radius = (sh.delta() * sh.n_enc as f64).round() as usize;
    assert_eq!(radius, 48);

    let trials = 3_000usize;
    for bad in [8usize, 32, radius, radius + 1, 64, 96] {
        let f = bad as f64 / sh.n_enc as f64;
        let ideal = (1.0 - f).powi(q as i32);
        let observed = corrupt_rate(&b, &claims, bad, trials);
        assert!(
            (observed - ideal).abs() < five_sigma(ideal, trials),
            "bad = {bad}: observed {observed:.5} against the ideal (1-f)^q = \
             {ideal:.5} (5 sigma = {:.5})",
            five_sigma(ideal, trials)
        );
        if bad >= radius {
            assert!(
                observed <= declared + five_sigma(declared, trials),
                "bad = {bad}: observed {observed:.5} ABOVE the declared \
                 per-event escape term {declared:.5}"
            );
        }
    }
}

#[test]
#[ignore = "measurement; run with --release --ignored --nocapture"]
fn corrupting_a_fraction_of_the_columns_heavy() {
    let nu = 10;
    let (b, claims) = honest(nu, 0xA11);
    let sh = b.shape();
    let q = sh.queries();
    let declared = (1.0 - sh.delta()).powi(q as i32);
    let radius = (sh.delta() * sh.n_enc as f64).round() as usize;

    let trials = 60_000usize;
    println!(
        "{:>5} {:>8} {:>10} {:>10} {:>10}",
        "bad", "f", "observed", "(1-f)^q", "5 sigma"
    );
    for bad in [8usize, 32, radius, radius + 1, 64, 96] {
        let f = bad as f64 / sh.n_enc as f64;
        let ideal = (1.0 - f).powi(q as i32);
        let observed = corrupt_rate(&b, &claims, bad, trials);
        let band = five_sigma(ideal, trials);
        println!("{bad:>5} {f:>8.4} {observed:>10.6} {ideal:>10.6} {band:>10.6}");
        assert!((observed - ideal).abs() < band, "bad = {bad}");
        if bad >= radius {
            assert!(
                observed <= declared + five_sigma(declared, trials),
                "bad = {bad}: observed {observed:.6} ABOVE declared {declared:.6}"
            );
        }
    }
}

#[test]
fn the_unique_decoding_radius_is_the_largest_one_that_decodes() {
    for nu in 4..=26usize {
        let sh = shape(nu);
        let (n, n_enc) = (sh.n(), sh.n_enc);
        let d = n_enc - n + 1;
        let e_float = sh.delta() * n_enc as f64;
        assert_eq!(
            e_float,
            e_float.trunc(),
            "nu = {nu}: radius is not integral"
        );
        let e = e_float as usize;
        assert_eq!(2 * e, n_enc - n, "nu = {nu}");
        assert!(2 * e < d, "nu = {nu}: 2e = {} is not below d = {d}", 2 * e);
        assert!(
            2 * (e + 1) >= d,
            "nu = {nu}: the declared radius is not maximal"
        );
    }
}

#[test]
fn the_two_posted_rows_precede_every_column_coin() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(0xC01);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);
    let rows_end = ch.transcript().entries().len();
    let ordinary = ch
        .transcript()
        .entries()
        .iter()
        .filter(|e| matches!(e, Entry::Ordinary(_)))
        .count();
    assert_eq!(ordinary, 2, "commit and open post exactly one row each");
    let _ = b.query_plan(&mut ch);
    let after = ch.transcript().entries();
    assert!(after.len() > rows_end, "query_plan drew no coin");
    for e in &after[rows_end..] {
        assert!(
            matches!(e, Entry::Coin(_)),
            "the query phase may only draw coins"
        );
    }
}

#[test]
fn a_row_longer_than_the_message_slot_is_refused() {
    let nu = 10;
    let (b, claims) = honest(nu, 0xB0B);
    let sh = b.shape();
    let v = b.verifier();
    let answers: Vec<(usize, Coord, Symbol<F128>)> = (0..sh.queries())
        .map(|k| {
            let c = (k * 17 + 3) % sh.n_enc;
            (
                CODEWORD_ORACLE,
                Coord::from(c),
                b.oracles()[0].symbols[c].clone(),
            )
        })
        .collect();
    assert!(v.check(&claims, &answers), "the honest control must accept");

    let mut long = v.opening.clone();
    long.push(F128::ONE);
    let mut short = v.opening.clone();
    short.pop();
    for opening in [long, short] {
        let forged = LigeroVerifier::from_public_transcript(
            sh,
            v.proximity.clone(),
            replay_proximity_coeffs(sh, 0xB0B),
            opening,
            eq_table(&claims[0].point[..sh.nu_row]),
            claims[0].point.clone(),
        );
        assert!(
            !forged.check(&claims, &answers),
            "a row of the wrong length must be refused"
        );
    }
}

fn replay_proximity_coeffs(sh: Shape, seed: u64) -> Vec<F128> {
    let mut ch = SeededChallenger::new(seed);
    (0..sh.m()).map(|_| ch.sample()).collect()
}

fn kernel_vector(mut rows: Vec<Vec<F128>>, n: usize) -> Option<Vec<F128>> {
    let mut pivots: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for col in 0..n {
        if r == rows.len() {
            break;
        }
        let Some(p) = (r..rows.len()).find(|&i| rows[i][col] != F128::ZERO) else {
            continue;
        };
        rows.swap(r, p);
        let inv = rows[r][col].inv().expect("pivot is nonzero");
        for entry in rows[r].iter_mut().take(n) {
            *entry = entry.mul(inv);
        }
        let pivot_row = rows[r].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != r && row[col] != F128::ZERO {
                let f = row[col];
                for (entry, p) in row.iter_mut().take(n).zip(&pivot_row) {
                    *entry = entry.add(f.mul(*p));
                }
            }
        }
        pivots.push(col);
        r += 1;
    }
    let free = (0..n).find(|c| !pivots.contains(c))?;
    let mut v = vec![F128::ZERO; n];
    v[free] = F128::ONE;
    for (i, &pc) in pivots.iter().enumerate() {
        v[pc] = rows[i][free];
    }
    Some(v)
}

fn basis_rows(n: usize, pts: &[usize]) -> Vec<Vec<F128>> {
    let units: Vec<Vec<F128>> = (0..n)
        .map(|j| {
            let mut u = vec![F128::ZERO; n];
            u[j] = F128::ONE;
            u
        })
        .collect();
    pts.iter()
        .map(|&t| {
            let x = F128::new(t as u128);
            units.iter().map(|u| code::eval_at(u, x)).collect()
        })
        .collect()
}

#[test]
fn the_best_forgery_row_agrees_on_exactly_n_minus_one_columns() {
    let sh = shape(10);
    let (n, n_enc) = (sh.n(), sh.n_enc);
    let target: Vec<usize> = (0..n - 1).map(|k| (k * 3 + 1) % n_enc).collect();
    assert_eq!(target.len(), n - 1);

    let e = kernel_vector(basis_rows(n, &target), n).expect("n-1 constraints leave a kernel");
    assert!(e.iter().any(|v| *v != F128::ZERO), "kernel vector is zero");

    let zeros: Vec<usize> = (0..n_enc)
        .filter(|&c| code::eval_at(&e, F128::new(c as u128)) == F128::ZERO)
        .collect();
    assert_eq!(
        zeros, target,
        "the extremal error message must vanish on exactly the chosen columns"
    );
    assert_eq!(
        n_enc - zeros.len(),
        n_enc - n + 1,
        "its codeword weight must be the minimum distance"
    );

    let one_more: Vec<usize> = (0..n).map(|k| (k * 3 + 1) % n_enc).collect();
    assert!(
        kernel_vector(basis_rows(n, &one_more), n).is_none(),
        "no nonzero message may vanish on n columns"
    );
}

#[test]
fn a_false_evaluation_survives_only_where_the_error_codeword_vanishes() {
    let nu = 10;
    let (b, claims) = honest(nu, 0xF0E);
    let sh = b.shape();
    let (n, n_enc, q) = (sh.n(), sh.n_enc, sh.queries());
    let v = b.verifier();

    let target: Vec<usize> = (0..n - 1).map(|k| (k * 5 + 2) % n_enc).collect();
    let e = kernel_vector(basis_rows(n, &target), n).expect("kernel exists");
    let p_col = &claims[0].point[sh.nu_row..];
    let delta = eval_mle(&e, p_col);
    assert_ne!(
        delta,
        F128::ZERO,
        "the error message must move the claimed value, or the claim is not false"
    );

    let forged_row: Vec<F128> = v.opening.iter().zip(&e).map(|(a, b)| a.add(*b)).collect();
    let false_claims = vec![Claim {
        point: claims[0].point.clone(),
        value: claims[0].value.add(delta),
    }];
    assert_ne!(false_claims[0].value, claims[0].value);

    let forged = LigeroVerifier::from_public_transcript(
        sh,
        v.proximity.clone(),
        replay_proximity_coeffs(sh, 0xF0E),
        forged_row,
        eq_table(&claims[0].point[..sh.nu_row]),
        claims[0].point.clone(),
    );
    let control = LigeroVerifier::from_public_transcript(
        sh,
        v.proximity.clone(),
        replay_proximity_coeffs(sh, 0xF0E),
        v.opening.clone(),
        eq_table(&claims[0].point[..sh.nu_row]),
        claims[0].point.clone(),
    );

    let trials = 4_000usize;
    let mut random = OsChallenger::new();
    let mut accepted = 0usize;
    for trial in 0..trials {
        let mut plan = b.query_plan(&mut random);
        let mut answers = Vec::new();
        let mut all_safe = true;
        while let Some((id, coord)) = plan.next(None) {
            let ix = usize::try_from(coord).expect("coord fits usize");
            all_safe &= target.contains(&ix);
            answers.push((id, coord, b.oracles()[0].symbols[ix].clone()));
        }
        assert!(
            control.check(&claims, &answers),
            "trial {trial}: the honest control must accept"
        );
        let ok = forged.check(&false_claims, &answers);
        assert_eq!(
            ok, all_safe,
            "trial {trial}: the forgery survives exactly on the error codeword's zeros"
        );
        accepted += ok as usize;
    }

    let observed = accepted as f64 / trials as f64;
    let ideal = ((n - 1) as f64 / n_enc as f64).powi(q as i32);
    let declared = (1.0 - sh.delta()).powi(q as i32);
    assert!(
        observed <= ideal + five_sigma(ideal, trials),
        "observed {observed:.6} above the extremal forgery rate {ideal:.6}"
    );
    assert!(
        observed <= declared + five_sigma(declared, trials),
        "observed {observed:.6} ABOVE the declared per-event term {declared:.6}"
    );
    assert!(
        ideal * 100.0 < declared,
        "the declared term must dominate by a wide margin"
    );
}
