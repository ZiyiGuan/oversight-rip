use rmip_backend_recursive::{fold_rows, FoldLadder, Shape, LEVEL0_ORACLE, LEVEL1_ORACLE};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{Claim, CommitmentBackend, Coord, OpeningVerifier, Symbol};
use rmip_iop::batch;
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::transcript::{Entry, OracleId, RawChallenger, SeededChallenger};

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

fn fold_one_column(col: &[F128], alpha: F128) -> Vec<F128> {
    let rows: Vec<Vec<F128>> = col.iter().map(|v| vec![*v]).collect();
    fold_rows(&rows, alpha).into_iter().map(|r| r[0]).collect()
}

fn fold_coin(entries: &[Entry]) -> F128 {
    let post = entries
        .iter()
        .position(|e| *e == Entry::OraclePost(LEVEL1_ORACLE))
        .expect("the ladder posts a level-1 oracle");
    match entries[post - 1] {
        Entry::Coin(c) => c,
        ref other => panic!("alpha_1 must be the entry before the posting, found {other:?}"),
    }
}

fn proximity_coeffs(entries: &[Entry], m: usize) -> Vec<F128> {
    assert_eq!(entries[0], Entry::OraclePost(LEVEL0_ORACLE));
    let coins: Vec<F128> = entries[1..=m]
        .iter()
        .map(|e| match e {
            Entry::Coin(c) => *c,
            other => panic!("expected a commitment-phase coin, found {other:?}"),
        })
        .collect();
    assert!(matches!(entries[m + 1], Entry::Ordinary(_)));
    coins
}

fn dot(c: &[F128], sym: &[F128]) -> F128 {
    c.iter()
        .zip(sym)
        .map(|(a, b)| a.mul(*b))
        .fold(F128::ZERO, |x, y| x.add(y))
}

struct Run {
    backend: FoldLadder,
    claims: Vec<Claim<F128>>,
    answers: Vec<(OracleId, Coord, Symbol<F128>)>,
    alpha1: F128,
    proximity_coeffs: Vec<F128>,
    opening_coeffs: Vec<F128>,
}

fn honest(nu: usize, seed: u64) -> Run {
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut backend = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(seed);
    backend.commit(&z, &mut ch);
    backend.open(&claims, &mut ch);

    let entries = ch.transcript().entries().to_vec();
    let sh = backend.shape();
    let alpha1 = fold_coin(&entries);
    let proximity_coeffs = proximity_coeffs(&entries, sh.m());
    let opening_coeffs = eq_table(&claims[0].point[..sh.nu_row]);

    let mut ch2 = SeededChallenger::new(seed ^ 0x5151);
    let mut plan = backend.query_plan(&mut ch2);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        let o = backend
            .oracles()
            .iter()
            .find(|o| o.id == id)
            .expect("the plan names a posted oracle");
        answers.push((
            id,
            coord,
            o.symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    Run {
        backend,
        claims,
        answers,
        alpha1,
        proximity_coeffs,
        opening_coeffs,
    }
}

#[test]
fn a_relabelled_level_one_half_is_rejected_by_the_label_clause_alone() {
    let nu = 10;
    let run = honest(nu, 0x9A11);
    let sh = run.backend.shape();
    let v = run.backend.verifier();
    let at = |row: &[F128], col: Coord| {
        rmip_backend_interleaved::code::eval_at(
            row,
            F128::new(u128::try_from(col).expect("a codeword column fits u128")),
        )
    };

    assert!(
        v.check(&run.claims, &run.answers),
        "honest answers must verify"
    );
    assert!(run.answers.len() >= 2 && run.answers.len().is_multiple_of(2));

    for pair in 0..run.answers.len() / 2 {
        let (lo, hi) = (2 * pair, 2 * pair + 1);
        let c0 = run.answers[lo].1;
        let s0 = run.answers[lo].2 .0.clone();
        let s1 = run.answers[hi].2 .0.clone();

        let mut bad = run.answers.clone();
        bad[hi].1 = Coord::from((usize::try_from(c0).unwrap() + 1) % sh.n_enc);

        assert_ne!(bad[hi].1, run.answers[hi].1, "the label must really move");
        assert_eq!(bad[hi].0, run.answers[hi].0, "the id is untouched");
        assert_eq!(bad[hi].2, run.answers[hi].2, "the symbol is untouched");
        assert_eq!(
            bad.iter().zip(&run.answers).filter(|(a, b)| a != b).count(),
            1,
            "exactly one occurrence differs"
        );
        assert!(bad[hi].1 < Coord::from(sh.n_enc));

        assert_eq!(
            dot(&run.proximity_coeffs, &s0),
            at(&v.proximity, c0),
            "pair {pair}: the proximity identity must still hold"
        );
        assert_eq!(
            dot(&run.opening_coeffs, &s0),
            at(&v.opening, c0),
            "pair {pair}: the opening identity must still hold"
        );
        assert_eq!(
            fold_one_column(&s0, run.alpha1),
            s1,
            "pair {pair}: the fold equation must still hold"
        );

        assert!(
            !v.check(&run.claims, &bad),
            "pair {pair}: a mislabelled level-1 half was accepted"
        );

        bad[hi].1 = c0;
        assert_eq!(bad, run.answers, "pair {pair}: the lists must be identical");
        assert!(
            v.check(&run.claims, &bad),
            "pair {pair}: the restored list must verify"
        );
    }
}

#[test]
fn a_level_one_symbol_from_another_column_is_caught_by_the_fold_equation() {
    let nu = 10;
    let run = honest(nu, 0x9A22);
    let sh = run.backend.shape();
    let v = run.backend.verifier();
    let at = |row: &[F128], col: Coord| {
        rmip_backend_interleaved::code::eval_at(
            row,
            F128::new(u128::try_from(col).expect("a codeword column fits u128")),
        )
    };
    let level1 = run
        .backend
        .oracles()
        .iter()
        .find(|o| o.id == LEVEL1_ORACLE)
        .expect("the ladder posts a level-1 oracle")
        .symbols
        .clone();

    assert!(
        v.check(&run.claims, &run.answers),
        "honest answers must verify"
    );

    for pair in 0..run.answers.len() / 2 {
        let (lo, hi) = (2 * pair, 2 * pair + 1);
        let c0 = run.answers[lo].1;
        let foreign_col = Coord::from((usize::try_from(c0).unwrap() + 1) % sh.n_enc);
        let s0 = run.answers[lo].2 .0.clone();
        let foreign = level1[usize::try_from(foreign_col).expect("coord fits usize")].clone();

        assert_ne!(foreign_col, c0);
        assert_ne!(
            foreign, run.answers[hi].2,
            "the foreign symbol must differ from the honest one"
        );
        assert_eq!(foreign.0.len(), sh.m() / 2, "and have the declared width");

        let mut bad = run.answers.clone();
        bad[hi].2 = foreign.clone();
        assert_eq!(bad[lo].1, bad[hi].1);
        assert_eq!(bad[lo].0, LEVEL0_ORACLE);
        assert_eq!(bad[hi].0, LEVEL1_ORACLE);

        assert_eq!(dot(&run.proximity_coeffs, &s0), at(&v.proximity, c0));
        assert_eq!(dot(&run.opening_coeffs, &s0), at(&v.opening, c0));
        assert_ne!(
            fold_one_column(&s0, run.alpha1),
            foreign.0,
            "pair {pair}: the fold equation must be violated, or this test pins nothing"
        );

        assert!(
            !v.check(&run.claims, &bad),
            "pair {pair}: a level-1 symbol from column {foreign_col} was accepted at column {c0}"
        );
    }
}
