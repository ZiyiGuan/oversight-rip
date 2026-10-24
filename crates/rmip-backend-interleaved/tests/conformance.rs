use rmip_backend_interleaved::{Ligero, Shape, CODEWORD_ORACLE};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{
    composed_opening_error, div_error_upward, mul_error_upward, pow_error_upward,
    sum_errors_upward, Alphabet, BooleanBinding, Claim, CommitmentBackend, Coord, OpeningVerifier,
    Regime, Symbol,
};
use rmip_iop::batch;
use rmip_iop::mle::eval_mle;
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

#[test]
fn the_interleaved_rs_conforms() {
    let nu = 10;
    let z = table(nu);
    let mut b = Ligero::new(shape(nu));
    let v = rmip_backend_conformance::check(&mut b, &z, &one_claim(&z, nu), false);
    assert!(v.is_empty(), "conformance violations: {v:#?}");
}

#[test]
fn the_interleaved_rs_is_deterministic() {
    let nu = 8;
    let z = table(nu);
    let v = rmip_backend_conformance::check_deterministic(
        || Ligero::new(shape(nu)),
        &z,
        &one_claim(&z, nu),
    );
    assert!(v.is_empty(), "{v:#?}");
}

#[test]
fn proximity_and_opening_vectors_are_each_one_ordinary_message() {
    let nu = 10;
    let z = table(nu);
    let mut backend = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(0xF12A);

    backend.commit(&z, &mut ch);
    assert_eq!(
        ch.transcript()
            .entries()
            .iter()
            .filter(|entry| matches!(entry, Entry::Ordinary(_)))
            .count(),
        1,
        "the commitment-phase proximity vector is one prover message"
    );

    backend.open(&one_claim(&z, nu), &mut ch);
    let ordinary: Vec<&Vec<u8>> = ch
        .transcript()
        .entries()
        .iter()
        .filter_map(|entry| match entry {
            Entry::Ordinary(bytes) => Some(bytes),
            _ => None,
        })
        .collect();
    assert_eq!(
        ordinary.len(),
        2,
        "opening adds exactly one ordinary message"
    );
    assert_eq!(
        ordinary.iter().map(|bytes| bytes.len()).sum::<usize>(),
        backend.params(1).ordinary_bytes
    );
    assert_eq!(backend.params(1).r_open, 1);
}

#[test]
fn q_long_is_independent_of_n() {
    let mut seen = Vec::new();
    for nu in [8usize, 10, 12, 14] {
        let z = table(nu);
        let mut b = Ligero::new(shape(nu));
        let mut ch = SeededChallenger::new(1);
        b.commit(&z, &mut ch);
        let p = b.params(1);
        seen.push((1usize << nu, p.q_long));
        assert_eq!(p.a_long, p.q_long as u64 * b.shape().m() as u64);
    }
    let qs: Vec<usize> = seen.iter().map(|(_, q)| *q).collect();
    assert!(
        qs.iter().all(|q| *q == qs[0]),
        "q_long varied with N: {seen:?}"
    );
    let (n, q) = seen[seen.len() - 1];
    assert!(q * 100 < n, "q_long = {q} is not << N = {n}");
}

#[test]
fn an_inconsistent_column_is_caught() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(3);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let v = b.verifier();
    let mut plan = b.query_plan(&mut ch);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        answers.push((
            id,
            coord,
            b.oracles()[0].symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert!(v.check(&claims, &answers), "honest columns must verify");

    for victim in 0..answers.len() {
        let mut bad = answers.clone();
        bad[victim].2 .0[0] = bad[victim].2 .0[0].add(F128::ONE);
        assert!(!v.check(&claims, &bad), "corrupt column {victim} accepted");
    }
}

#[test]
fn a_column_of_the_wrong_height_is_refused() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(3);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let v = b.verifier();
    let mut plan = b.query_plan(&mut ch);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        answers.push((
            id,
            coord,
            b.oracles()[0].symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert!(v.check(&claims, &answers), "honest columns must verify");
    let m = b.shape().m();
    assert!(m >= 2, "a column of one element has no proper half");

    for victim in 0..answers.len() {
        let mut half = answers.clone();
        half[victim].2 .0.truncate(m / 2);
        assert!(!v.check(&claims, &half), "half of column {victim} accepted");

        let mut long = answers.clone();
        long[victim].2 .0.push(F128::ZERO);
        assert!(
            !v.check(&claims, &long),
            "column {victim} plus one accepted"
        );
    }
}

#[test]
fn parameters_are_declared_honestly() {
    let nu = 12;
    let z = table(nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(5);
    b.commit(&z, &mut ch);
    let p = b.params(718);

    assert_eq!(p.name, "interleaved-rs");
    assert_eq!(p.regime, Regime::UniqueDecoding);
    assert_eq!(p.boolean_binding, BooleanBinding::ProvedByRelation);
    assert_eq!(
        p.alphabet,
        vec![Alphabet::Column {
            elements: b.shape().m()
        }]
    );
    assert_eq!(
        p.schedule.len(),
        1,
        "one level; this backend is flat, not recursive"
    );
    assert_eq!(
        p.eps_bind, 0.0,
        "unique decoding makes |S_cm| > 1 impossible"
    );
    assert!(
        p.eps_open > 0.0 && p.eps_open < 0.5,
        "eps_open = {}",
        p.eps_open
    );
    let per_event = (1.0 - b.shape().delta()).powi(b.shape().queries() as i32);
    assert!(
        p.eps_open > 2.0 * per_event - 1e-12,
        "eps_open must cover the proximity escape and the opening escape"
    );
    assert!(
        p.check_admissible(false).is_ok(),
        "ProvedByRelation needs no R1CS rows"
    );
    assert!(Coord::from(p.a_long) < p.l_long);
}

#[test]
fn eps_open_rounds_its_correlated_agreement_term_outward() {
    for nu in [8usize, 10, 12, 13, 20] {
        let sh = shape(nu);
        let exact = sh.n_enc as f64 / 2f64.powi(128);
        let rounded = div_error_upward(sh.n_enc as f64, 2f64.powi(128));
        assert!(
            rounded >= exact,
            "the correlated-agreement term is below the exact quotient at nu = {nu}"
        );
        let miss_base = div_error_upward((sh.n_enc + sh.n()) as f64, (2 * sh.n_enc) as f64);
        let escape = pow_error_upward(miss_base, sh.queries());
        let claims = 718;
        let old = composed_opening_error(
            sum_errors_upward(&[mul_error_upward(2.0, escape), exact]),
            claims,
            sh.nu_row + sh.nu_col,
        );
        let declared = Ligero::new(sh).params(claims).eps_open;
        assert_eq!(
            declared.to_bits(),
            old.to_bits(),
            "eps_open moved at nu = {nu}: {declared} against {old}"
        );
    }
}

#[test]
fn reconstruct_recomputes_and_agrees() {
    let nu = 10;
    let z = table(nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(7);
    b.commit(&z, &mut ch);
    let h = ch.transcript().prefix_at(CODEWORD_ORACLE);
    for coord in [0usize, 1, 37, b.shape().n_enc - 1] {
        assert_eq!(
            b.reconstruct(CODEWORD_ORACLE, &h, Coord::from(coord)),
            b.oracles()[0].symbols[coord]
        );
    }
}

#[test]
fn optimised_encoding_matches_the_reference() {
    use rmip_backend_interleaved::code;

    for nu in [4usize, 8, 10, 12] {
        let z = table(nu);
        let sh = shape(nu);
        let mut b = Ligero::new(sh);
        let mut ch = SeededChallenger::new(2);
        b.commit(&z, &mut ch);

        let dom = code::domain(sh.n_enc);
        let rows: Vec<Vec<F128>> = z.chunks(sh.n()).map(|r| r.to_vec()).collect();
        let reference: Vec<Vec<F128>> = rows.iter().map(|r| code::encode(r, &dom)).collect();

        for (c, posted) in b.oracles()[0].symbols.iter().enumerate() {
            assert_eq!(posted.0.len(), sh.m(), "nu = {nu}, column {c}");
            for (i, r) in reference.iter().enumerate() {
                assert_eq!(posted.0[i], r[c], "nu = {nu}, column {c}, row {i}");
            }
        }

        let h = ch.transcript().prefix_at(CODEWORD_ORACLE);
        for c in [0usize, 1, sh.n_enc / 2, sh.n_enc - 1] {
            let recon = b.reconstruct(CODEWORD_ORACLE, &h, Coord::from(c));
            for (i, r) in reference.iter().enumerate() {
                assert_eq!(recon.0[i], r[c], "reconstruct nu = {nu}, col {c}, row {i}");
            }
        }
    }
}

#[test]
fn query_columns_are_drawn_from_verifier_coins() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(7);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let picked = |seed: u64| -> Vec<Coord> {
        let mut ch = SeededChallenger::new(seed);
        let mut plan = b.query_plan(&mut ch);
        let mut out = Vec::new();
        while let Some((_, c)) = plan.next(None) {
            out.push(c);
        }
        out
    };

    assert_eq!(picked(101), picked(101));
    let distinct: std::collections::HashSet<Vec<Coord>> =
        (0..64u64).map(|s| picked(0xA11CE + s)).collect();
    assert!(
        distinct.len() > 32,
        "only {} distinct plans in 64 draws",
        distinct.len()
    );

    let mut ch = SeededChallenger::new(5);
    let before = ch.transcript().entries().len();
    let _ = b.query_plan(&mut ch);
    let coins = ch.transcript().entries().len() - before;
    assert_eq!(coins, b.shape().queries(), "one coin per queried column");
}

#[test]
fn escape_frequency_matches_the_proximity_term_of_eps_open() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(11);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let sh = b.shape();
    let bad = (sh.delta() * sh.n_enc as f64).round() as usize;
    let predicted = (1.0 - bad as f64 / sh.n_enc as f64).powi(sh.queries() as i32);
    assert!(
        b.params(1).eps_open > 2.0 * predicted - 1e-12,
        "eps_open must dominate two copies of the experiment run below"
    );

    let trials = 20_000usize;
    let mut random = OsChallenger::new();
    let escapes = (0..trials)
        .filter(|_| {
            let mut plan = b.query_plan(&mut random);
            let mut hit = false;
            while let Some((_, c)) = plan.next(None) {
                hit |= c < Coord::from(bad);
            }
            !hit
        })
        .count();

    let observed = escapes as f64 / trials as f64;
    let sigma = (predicted * (1.0 - predicted) / trials as f64).sqrt();
    assert!(
        (observed - predicted).abs() < 5.0 * sigma,
        "OS-CSPRNG escape rate {observed:.5} vs the ideal per-event proximity \
         term {predicted:.5} (5σ = {:.5})",
        5.0 * sigma
    );
}

#[test]
fn a_corrupted_column_outside_the_old_fixed_progression_is_caught() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(13);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let sh = b.shape();
    let q = sh.queries();
    let old_plan: Vec<usize> = (0..q).map(|k| (k * (sh.n_enc / q)) % sh.n_enc).collect();
    let victim = (0..sh.n_enc)
        .find(|c| !old_plan.contains(c))
        .expect("some column was unreachable");

    let v = b.verifier();
    let trials = 2_000usize;
    let mut queried = 0usize;
    let mut random = OsChallenger::new();
    for s in 0..trials {
        let mut plan = b.query_plan(&mut random);
        let mut answers = Vec::new();
        let mut hit = false;
        while let Some((id, coord)) = plan.next(None) {
            let mut sym =
                b.oracles()[0].symbols[usize::try_from(coord).expect("coord fits usize")].clone();
            if coord == Coord::from(victim) {
                sym.0[0] = sym.0[0].add(F128::ONE);
                hit = true;
            }
            answers.push((id, coord, sym));
        }
        assert_eq!(
            v.check(&claims, &answers),
            !hit,
            "execution {s}: victim queried = {hit}"
        );
        queried += hit as usize;
    }
    assert!(
        queried > 0,
        "the corrupted column was never queried in {trials} executions"
    );
    let rate = queried as f64 / trials as f64;
    let expected = 1.0 - (1.0 - 1.0 / sh.n_enc as f64).powi(q as i32);
    assert!(
        (rate - expected).abs() < 0.02,
        "hit rate {rate:.4} vs expected {expected:.4}"
    );
}

#[test]
fn a_claim_value_that_the_opened_vector_does_not_support_is_rejected() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(61);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let v = b.verifier();
    let mut ch2 = SeededChallenger::new(62);
    let mut plan = b.query_plan(&mut ch2);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        answers.push((
            id,
            coord,
            b.oracles()[0].symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert!(v.check(&claims, &answers), "honest baseline");

    let mut bad = claims.clone();
    bad[0].value = bad[0].value.add(F128::ONE);
    assert!(
        !v.check(&bad, &answers),
        "a wrong claimed VALUE was accepted"
    );

    let mut moved = claims.clone();
    moved[0].point[0] = moved[0].point[0].add(F128::ONE);
    assert!(
        !v.check(&moved, &answers),
        "a claim at a DIFFERENT point was accepted"
    );
}

#[test]
fn the_proximity_and_opening_checks_each_have_an_isolating_adversary() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = Ligero::new(shape(nu));
    let mut ch = SeededChallenger::new(63);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let mut ch2 = SeededChallenger::new(64);
    let mut plan = b.query_plan(&mut ch2);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        answers.push((
            id,
            coord,
            b.oracles()[0].symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert!(b.verifier().check(&claims, &answers), "honest baseline");

    let mut vp = b.verifier();
    vp.proximity[0] = vp.proximity[0].add(F128::ONE);
    assert!(
        !vp.check(&claims, &answers),
        "a corrupted proximity response was accepted"
    );

    let mut vo = b.verifier();
    vo.opening[0] = vo.opening[0].add(F128::ONE);
    let mut consistent = claims.clone();
    consistent[0].value = eval_mle(&vo.opening, &consistent[0].point[b.shape().nu_row..]);
    assert!(
        !vo.check(&consistent, &answers),
        "a corrupted opening vector was accepted"
    );
}

#[test]
fn a_commit_only_instance_reconstructs_every_posted_oracle() {
    let nu = 10;
    let z = table(nu);
    let v = rmip_backend_conformance::check_prefix_only_reconstruction(
        || Ligero::new(shape(nu)),
        &z,
        &one_claim(&z, nu),
    );
    assert!(v.is_empty(), "prefix-only reconstruction failed: {v:#?}");
}

#[test]
fn malformed_public_opening_messages_are_rejected_without_panicking() {
    let shape = shape(6);
    let verifier = rmip_backend_interleaved::LigeroVerifier::from_public_transcript(
        shape,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let claim = Claim {
        point: vec![F128::ZERO; 6],
        value: F128::ZERO,
    };
    let answer = (
        CODEWORD_ORACLE,
        Coord::ZERO,
        Symbol(vec![F128::ZERO; shape.m()]),
    );
    assert!(!verifier.check(&[claim], &[answer]));
}
