use rmip_backend_recursive::{fold_rows, FoldLadder, Shape, LEVEL0_ORACLE, LEVEL1_ORACLE};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{
    Alphabet, Claim, CommitmentBackend, Coord, OpeningVerifier, Regime, Symbol,
};
use rmip_iop::batch;
use rmip_iop::mle::eval_mle;
use rmip_iop::transcript::{Entry, OsChallenger, RawChallenger, SeededChallenger};

fn shape(nu: usize) -> Shape {
    Shape::balanced(nu, 4, 4.0)
}

fn table(nu: usize) -> Vec<F128> {
    (0..1usize << nu)
        .map(|i| F128::from_bool((i * 11 + 5) % 7 < 3))
        .collect()
}

fn one_claim(z: &[F128], nu: usize) -> Vec<Claim<F128>> {
    let raw: Vec<Claim<F128>> = (0..5)
        .map(|k| {
            let p: Vec<F128> = (0..nu)
                .map(|i| F128::new((k + 2) * 6151 + i as u128))
                .collect();
            Claim {
                value: eval_mle(z, &p),
                point: p,
            }
        })
        .collect();
    let mut ch = SeededChallenger::new(4);
    vec![batch::reduce(&raw, z, &mut ch).expect("batching verifies")]
}

#[test]
fn the_ladder_conforms() {
    let nu = 10;
    let z = table(nu);
    let mut b = FoldLadder::new(shape(nu));
    let v = rmip_backend_conformance::check(&mut b, &z, &one_claim(&z, nu), false);
    assert!(v.is_empty(), "conformance violations: {v:#?}");
}

#[test]
fn the_ladder_is_deterministic() {
    let nu = 10;
    let z = table(nu);
    let v = rmip_backend_conformance::check_deterministic(
        || FoldLadder::new(shape(nu)),
        &z,
        &one_claim(&z, nu),
    );
    assert!(v.is_empty(), "{v:#?}");
}

#[test]
fn ladder_vector_framing_matches_its_declared_opening_rounds() {
    let nu = 10;
    let z = table(nu);
    let mut backend = FoldLadder::new(shape(nu));
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

    let opening_start = ch.transcript().entries().len();
    backend.open(&one_claim(&z, nu), &mut ch);
    assert!(matches!(
        ch.transcript().entries()[opening_start],
        Entry::Ordinary(_)
    ));
    assert!(matches!(
        ch.transcript().entries()[opening_start + 1],
        Entry::Coin(_)
    ));
    assert_eq!(
        ch.transcript().entries()[opening_start + 2],
        Entry::OraclePost(LEVEL1_ORACLE)
    );
    assert!(matches!(
        ch.transcript().entries()[opening_start + 3],
        Entry::Coin(_)
    ));
    assert!(matches!(
        ch.transcript().entries()[opening_start + 4],
        Entry::Ordinary(_)
    ));
    assert_eq!(ch.transcript().entries().len(), opening_start + 5);

    let ordinary_bytes = ch
        .transcript()
        .entries()
        .iter()
        .filter_map(|entry| match entry {
            Entry::Ordinary(bytes) => Some(bytes.len()),
            _ => None,
        })
        .sum::<usize>();
    assert_eq!(ordinary_bytes, backend.params(1).ordinary_bytes);
    assert_eq!(backend.params(1).r_open, 3);
}

#[test]
fn a_commit_only_instance_reconstructs_the_folding_oracle() {
    let nu = 10;
    let z = table(nu);
    let v = rmip_backend_conformance::check_prefix_only_reconstruction(
        || FoldLadder::new(shape(nu)),
        &z,
        &one_claim(&z, nu),
    );
    assert!(v.is_empty(), "prefix-only reconstruction failed: {v:#?}");
}

#[test]
fn the_folding_oracle_is_load_bearing_in_its_prefix() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);

    let run = |seed: u64| {
        let mut b = FoldLadder::new(shape(nu));
        let mut ch = SeededChallenger::new(seed);
        b.commit(&z, &mut ch);
        b.open(&claims, &mut ch);
        let h0 = ch.transcript().prefix_at(LEVEL0_ORACLE);
        let h1 = ch.transcript().prefix_at(LEVEL1_ORACLE);
        (b, h0, h1)
    };
    let (b_a, h0_a, h1_a) = run(21);
    let (_b_b, h0_b, h1_b) = run(22);

    assert_ne!(
        h1_a, h1_b,
        "distinct executions must give distinct level-1 prefixes"
    );

    for _ in 0..4 {
        assert_eq!(
            b_a.reconstruct(LEVEL1_ORACLE, &h1_a, Coord::from(5u64)),
            b_a.reconstruct(LEVEL1_ORACLE, &h1_a, Coord::from(5u64))
        );
    }

    assert_eq!(
        h0_a.as_bytes(),
        h0_b.as_bytes(),
        "both executions post level 0 first, so both prefixes are empty"
    );
    let mut och = SeededChallenger::new(23);
    och.record_ordinary(b"a message no ladder execution sends");
    och.sample();
    och.sample();
    och.post_oracle(LEVEL0_ORACLE);
    let h0_other = och.transcript().prefix_at(LEVEL0_ORACLE);
    assert_ne!(
        h0_a.as_bytes(),
        h0_other.as_bytes(),
        "the hand-built level-0 prefix must really differ"
    );

    let mut differed = 0;
    for c in 0..32u64 {
        let coord = Coord::from(c);
        if b_a.reconstruct(LEVEL1_ORACLE, &h1_a, coord)
            != b_a.reconstruct(LEVEL1_ORACLE, &h1_b, coord)
        {
            differed += 1;
        }
        assert_eq!(
            b_a.reconstruct(LEVEL0_ORACLE, &h0_a, coord),
            b_a.reconstruct(LEVEL0_ORACLE, &h0_other, coord),
            "the committed level depends on no coin, so its prefix cannot matter"
        );
    }
    assert!(
        differed > 28,
        "only {differed}/32 level-1 coordinates moved with the prefix"
    );
}

#[test]
fn every_posted_oracle_is_reconstructible() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(31);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    assert_eq!(b.oracles().len(), 2, "the ladder must post two oracles");
    for o in b.oracles() {
        let h = ch.transcript().prefix_at(o.id);
        for coord in 0..o.symbols.len() {
            assert_eq!(
                b.reconstruct(o.id, &h, Coord::from(coord)),
                o.symbols[coord],
                "oracle {}",
                o.id
            );
        }
    }
}

#[test]
fn folding_and_encoding_commute() {
    let nu = 10;
    let z = table(nu);
    let sh = shape(nu);
    let rows: Vec<Vec<F128>> = z.chunks(sh.n()).map(|r| r.to_vec()).collect();
    let alpha = F128::new(0x1234_5678);

    let dom = rmip_backend_interleaved::code::domain(sh.n_enc);
    let enc_then_fold: Vec<Vec<F128>> = {
        let enc: Vec<Vec<F128>> = rows
            .iter()
            .map(|r| rmip_backend_interleaved::code::encode(r, &dom))
            .collect();
        let half = enc.len() / 2;
        (0..half)
            .map(|i| {
                enc[i]
                    .iter()
                    .zip(&enc[i + half])
                    .map(|(lo, hi)| lo.add(alpha.mul(lo.add(*hi))))
                    .collect()
            })
            .collect()
    };
    let fold_then_enc: Vec<Vec<F128>> = fold_rows(&rows, alpha)
        .iter()
        .map(|r| rmip_backend_interleaved::code::encode(r, &dom))
        .collect();
    assert_eq!(enc_then_fold, fold_then_enc);
}

#[test]
fn a_bad_fold_is_caught() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(41);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let v = b.verifier();
    let mut ch2 = SeededChallenger::new(42);
    let mut plan = b.query_plan(&mut ch2);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        let o = b.oracles().iter().find(|o| o.id == id).unwrap();
        answers.push((
            id,
            coord,
            o.symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert!(v.check(&claims, &answers), "honest answers must verify");

    for victim in (1..answers.len()).step_by(2) {
        let mut bad = answers.clone();
        bad[victim].2 .0[0] = bad[victim].2 .0[0].add(F128::ONE);
        assert!(
            !v.check(&claims, &bad),
            "corrupt fold at occurrence {victim} accepted"
        );
    }
}

#[test]
fn parameters_are_declared_honestly() {
    let nu = 12;
    let z = table(nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(51);
    b.commit(&z, &mut ch);
    b.open(&one_claim(&z, nu), &mut ch);
    let p = b.params(1);

    assert_eq!(p.name, "fold-ladder");
    assert_eq!(p.regime, Regime::UniqueDecoding);
    assert_eq!(p.schedule.len(), 2, "two levels");
    assert_eq!(p.alphabet.len(), 2, "one alphabet per posted oracle");
    assert_eq!(p.q_long, 2 * b.shape().queries());
    assert_eq!(p.r_open, 3);
    match (&p.alphabet[0], &p.alphabet[1]) {
        (Alphabet::Column { elements: a }, Alphabet::Column { elements: c }) => {
            assert_eq!(*a, b.shape().m());
            assert_eq!(*c, b.shape().m() / 2);
        }
        other => panic!("unexpected alphabets: {other:?}"),
    }

    let flat = rmip_backend_interleaved::Ligero::new(rmip_backend_interleaved::Shape::balanced(
        nu, 4, 4.0,
    ));
    assert!(
        p.q_long > flat.params(1).q_long,
        "this crate is the R9 instrument, not a competitive backend"
    );
}

#[test]
fn the_rounded_correlated_agreement_term_moves_no_declared_number() {
    for nu in [6, 8, 10, 12, 14, 16] {
        let z = table(nu);
        let mut b = FoldLadder::new(shape(nu));
        let mut ch = SeededChallenger::new(53);
        b.commit(&z, &mut ch);
        b.open(&one_claim(&z, nu), &mut ch);
        let sh = b.shape();

        let miss_base =
            rmip_iop::backend::div_error_upward((sh.n_enc + sh.n()) as f64, (2 * sh.n_enc) as f64);
        let flat = rmip_iop::backend::pow_error_upward(miss_base, sh.queries());
        for claims in [1usize, 3, 17] {
            let old = rmip_iop::backend::composed_opening_error(
                rmip_iop::backend::sum_errors_upward(&[
                    rmip_iop::backend::mul_error_upward(2.0, flat),
                    2.0 * sh.n_enc as f64 / 2f64.powi(128),
                ]),
                claims,
                sh.nu_row + sh.nu_col,
            );
            assert_eq!(
                b.params(claims).eps_open.to_bits(),
                old.to_bits(),
                "eps_open moved at nu = {nu}, claims = {claims}"
            );
        }
    }
}

#[test]
fn a_bad_level_zero_column_is_caught() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(71);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let v = b.verifier();
    let mut ch2 = SeededChallenger::new(72);
    let mut plan = b.query_plan(&mut ch2);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        let o = b.oracles().iter().find(|o| o.id == id).unwrap();
        answers.push((
            id,
            coord,
            o.symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert!(v.check(&claims, &answers), "honest baseline");

    for victim in (0..answers.len()).step_by(2) {
        let mut bad = answers.clone();
        bad[victim].2 .0[0] = bad[victim].2 .0[0].add(F128::ONE);
        assert!(
            !v.check(&claims, &bad),
            "corrupt level-0 column at {victim} accepted"
        );
    }
}

#[test]
fn each_ladder_check_has_an_isolating_adversary() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(73);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let mut ch2 = SeededChallenger::new(74);
    let mut plan = b.query_plan(&mut ch2);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        let o = b.oracles().iter().find(|o| o.id == id).unwrap();
        answers.push((
            id,
            coord,
            o.symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
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

    let mut bad = answers.clone();
    bad[1].2 .0[0] = bad[1].2 .0[0].add(F128::ONE);
    assert!(
        !b.verifier().check(&claims, &bad),
        "a broken fold was accepted"
    );
}

#[test]
fn the_ladder_rejects_a_claim_its_opened_vector_does_not_support() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(75);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let v = b.verifier();
    let mut ch2 = SeededChallenger::new(76);
    let mut plan = b.query_plan(&mut ch2);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        let o = b.oracles().iter().find(|o| o.id == id).unwrap();
        answers.push((
            id,
            coord,
            o.symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
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
fn the_ladder_escape_frequency_matches_the_flat_bound_it_declares() {
    let nu = 10;
    let z = table(nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(11);
    b.commit(&z, &mut ch);
    b.open(&one_claim(&z, nu), &mut ch);

    let sh = b.shape();
    let q = sh.queries();
    let bad = (sh.delta() * sh.n_enc as f64).round() as usize;
    let predicted = (1.0 - bad as f64 / sh.n_enc as f64).powi(q as i32);

    assert!(
        b.params(1).eps_open > 2.0 * predicted - 1e-12,
        "declared eps_open {} must dominate 2 x {predicted}",
        b.params(1).eps_open
    );
    assert_eq!(b.params(1).q_long, 2 * q);

    let trials = 20_000usize;
    let mut random = OsChallenger::new();
    let escapes = (0..trials)
        .filter(|_| {
            let mut plan = b.query_plan(&mut random);
            let mut hit = false;
            let mut occurrences = 0usize;
            while let Some((_, c)) = plan.next(None) {
                hit |= c < Coord::from(bad);
                occurrences += 1;
            }
            assert_eq!(occurrences, 2 * q);
            !hit
        })
        .count();

    let observed = escapes as f64 / trials as f64;
    let sigma = (predicted * (1.0 - predicted) / trials as f64).sqrt();
    assert!(
        (observed - predicted).abs() < 5.0 * sigma,
        "OS-CSPRNG escape rate {observed:.5} vs the ideal per-event term \
         {predicted:.5} (5σ = {:.5}) -- the flat bound does NOT describe this \
         ladder and `params` is overclaiming",
        5.0 * sigma
    );

    assert!(
        (observed - (1.0 - bad as f64 / sh.n_enc as f64).powi(2 * q as i32)).abs() > 5.0 * sigma,
        "escape rate is consistent with 2q independent columns, so this test \
         is not distinguishing the ladder's actual sampling from a flat one"
    );
}

#[test]
fn malformed_public_opening_messages_are_rejected_without_panicking() {
    let shape = shape(6);
    let verifier = rmip_backend_recursive::LadderVerifier::from_public_transcript(
        shape,
        Vec::new(),
        Vec::new(),
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
    let answers = [
        (
            LEVEL0_ORACLE,
            Coord::from(0u64),
            Symbol(vec![F128::ZERO; shape.m()]),
        ),
        (
            LEVEL1_ORACLE,
            Coord::from(0u64),
            Symbol(vec![F128::ZERO; shape.m() / 2]),
        ),
    ];
    assert!(!verifier.check(&[claim], &answers));
}

#[test]
#[should_panic(expected = "nu must be at least 3")]
fn balanced_rejects_a_degenerate_column_side() {
    let _ = Shape::balanced(2, 4, 4.0);
}

#[test]
#[should_panic(expected = "nu must be at least 3")]
fn balanced_rejects_a_nu_that_would_underflow_the_column_side() {
    let _ = Shape::balanced(1, 4, 4.0);
}

#[test]
fn the_smallest_accepted_shape_is_not_degenerate() {
    let sh = Shape::balanced(3, 4, 4.0);
    assert_eq!((sh.nu_row, sh.nu_col), (2, 1));
    assert_eq!(sh.m(), 4);
    assert_eq!(sh.n(), 2, "a one-element message would divide by itself");
    assert_eq!(sh.n_enc, 8);
}

#[test]
fn a_malformed_prefix_cannot_reach_the_fold_challenge() {
    use rmip_iop::transcript::TranscriptPrefix;

    assert!(TranscriptPrefix::from_bytes(&[0x7F]).is_err());
    let mut truncated_coin = vec![2u8];
    truncated_coin.extend_from_slice(&[0u8; 15]);
    assert!(TranscriptPrefix::from_bytes(&truncated_coin).is_err());
    let mut lying_length = vec![1u8];
    lying_length.extend_from_slice(&4u64.to_le_bytes());
    lying_length.push(0xAA);
    assert!(TranscriptPrefix::from_bytes(&lying_length).is_err());

    let nu = 8;
    let z = table(nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(61);
    b.commit(&z, &mut ch);
    let empty = TranscriptPrefix::from_bytes(&[]).expect("the empty prefix decodes");
    assert_eq!(empty.entry_count(), 0);
    for coord in [0usize, 3, 17] {
        let level0 = b.reconstruct(LEVEL0_ORACLE, &empty, Coord::from(coord));
        let folded = b.reconstruct(LEVEL1_ORACLE, &empty, Coord::from(coord));
        let half = level0.0.len() / 2;
        assert_eq!(
            folded.0,
            level0.0[..half].to_vec(),
            "a coin-free prefix must fold with zero, i.e. take the top half"
        );
    }
}

#[test]
fn the_plan_and_the_verifier_share_one_pairing_convention() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut b = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(63);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let mut ch2 = SeededChallenger::new(64);
    let mut plan = b.query_plan(&mut ch2);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        let o = b.oracles().iter().find(|o| o.id == id).unwrap();
        answers.push((
            id,
            coord,
            o.symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert_eq!(answers.len(), b.params(1).q_long);
    assert!(
        answers.len() >= 4,
        "the rearrangements below need two pairs"
    );
    assert!(answers.len().is_multiple_of(2));

    for pair in answers.chunks(2) {
        assert_eq!(pair[0].0, LEVEL0_ORACLE, "level 0 comes first in a pair");
        assert_eq!(pair[1].0, LEVEL1_ORACLE, "level 1 comes second in a pair");
        assert_eq!(pair[0].1, pair[1].1, "both halves are the same column");
    }

    let v = b.verifier();
    assert!(v.check(&claims, &answers), "honest answers must verify");

    let mut swapped_levels = answers.clone();
    swapped_levels.swap(0, 1);
    assert!(
        !v.check(&claims, &swapped_levels),
        "a pair emitted level 1 first was accepted"
    );

    let mut interleaved = answers.clone();
    interleaved.swap(1, 2);
    assert!(
        !v.check(&claims, &interleaved),
        "columns interleaved across two pairs were accepted"
    );
}
