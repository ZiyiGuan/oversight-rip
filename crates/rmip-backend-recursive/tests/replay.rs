use rmip_backend_recursive::{
    FoldLadder, LadderReplay, LadderVerifier, Shape, LEVEL0_ORACLE, LEVEL1_ORACLE,
};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{Claim, CommitmentBackend, Coord, OpeningVerifier, Symbol};
use rmip_iop::batch;
use rmip_iop::mle::eval_mle;
use rmip_iop::replay::{CoinDeriver, ReplayError, Tape, TranscriptVerifier};
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

struct SeededDeriver(SeededChallenger);

impl CoinDeriver for SeededDeriver {
    fn absorb_ordinary(&mut self, _bytes: &[u8]) {}
    fn absorb_post(&mut self, _id: OracleId, _root: Option<&[u8; 32]>) {}
    fn derive(&mut self) -> F128 {
        self.0.sample()
    }
}

struct Recorded {
    prover: FoldLadder,
    claims: Vec<Claim<F128>>,
    entries: Vec<Entry>,
    occurrences: Vec<(OracleId, Coord)>,
    seed: u64,
}

fn record(nu: usize, seed: u64) -> Recorded {
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let mut prover = FoldLadder::new(shape(nu));
    let mut ch = SeededChallenger::new(seed);
    prover.commit(&z, &mut ch);
    prover.open(&claims, &mut ch);
    let mut plan = prover.query_plan(&mut ch);
    let mut occurrences = Vec::new();
    while let Some(x) = plan.next(None) {
        occurrences.push(x);
    }
    let entries = ch.transcript().entries().to_vec();
    Recorded {
        prover,
        claims,
        entries,
        occurrences,
        seed,
    }
}

fn answers(
    prover: &FoldLadder,
    occurrences: &[(OracleId, Coord)],
) -> Vec<(OracleId, Coord, Symbol<F128>)> {
    occurrences
        .iter()
        .map(|(id, coord)| {
            let o = prover
                .oracles()
                .iter()
                .find(|o| o.id == *id)
                .expect("the plan names a posted oracle");
            (
                *id,
                *coord,
                o.symbols[usize::try_from(*coord).expect("coord fits usize")].clone(),
            )
        })
        .collect()
}

fn replay(
    nu: usize,
    point: &[F128],
    entries: &[Entry],
    seed: u64,
) -> (Option<ReplayError>, Vec<(OracleId, Coord)>, LadderVerifier) {
    let mut tape = Tape::new(
        entries,
        Vec::new(),
        SeededDeriver(SeededChallenger::new(seed)),
    );
    let mut r = LadderReplay::new(shape(nu));
    r.replay_commit(&mut tape);
    r.replay_open(point, &mut tape);
    let mut plan = r.query_plan(&mut tape);
    let mut occurrences = Vec::new();
    while let Some(x) = plan.next(None) {
        occurrences.push(x);
    }
    tape.finish();
    let verifier = r.opening_verifier();
    (tape.into_fault(), occurrences, verifier)
}

fn fold_coin_index(entries: &[Entry]) -> usize {
    let post = entries
        .iter()
        .position(|e| *e == Entry::OraclePost(LEVEL1_ORACLE))
        .expect("the ladder posts a level-1 oracle");
    assert!(
        matches!(entries[post - 1], Entry::Coin(_)),
        "alpha_1 must be the entry before the level-1 posting"
    );
    post - 1
}

fn residual_index(entries: &[Entry]) -> usize {
    entries
        .iter()
        .rposition(|e| matches!(e, Entry::Ordinary(_)))
        .expect("the ladder records a residual")
}

#[test]
fn an_honest_ladder_transcript_replays_and_verifies() {
    let nu = 10;
    let rec = record(nu, 0x5EED);
    let point = rec.claims[0].point.clone();
    let (fault, occurrences, verifier) = replay(nu, &point, &rec.entries, rec.seed);

    assert_eq!(fault, None, "an honest transcript must replay cleanly");
    assert_eq!(
        occurrences, rec.occurrences,
        "the replayed plan must be the prover's plan"
    );
    assert!(
        verifier.check(&rec.claims, &answers(&rec.prover, &occurrences)),
        "a replayed honest ladder must verify"
    );
}

#[test]
fn a_forged_fold_coin_is_caught_by_the_replay() {
    let nu = 10;
    let rec = record(nu, 0x5EED + 1);
    let point = rec.claims[0].point.clone();

    let index = fold_coin_index(&rec.entries);
    let mut forged = rec.entries.clone();
    forged[index] = match &forged[index] {
        Entry::Coin(c) => Entry::Coin(c.add(F128::ONE)),
        _ => unreachable!("checked by fold_coin_index"),
    };

    let (fault, occurrences, verifier) = replay(nu, &point, &forged, rec.seed);
    assert_eq!(fault, Some(ReplayError::ForgedCoin { index }));
    assert!(
        !verifier.check(&rec.claims, &answers(&rec.prover, &occurrences)),
        "a ladder folded at a coin the transcript does not carry was accepted"
    );
}

#[test]
fn a_tampered_residual_message_fails_the_terminal_fold() {
    let nu = 10;
    let rec = record(nu, 0x5EED + 2);
    let point = rec.claims[0].point.clone();

    let (fault, occurrences, honest) = replay(nu, &point, &rec.entries, rec.seed);
    assert_eq!(fault, None);
    let posted = answers(&rec.prover, &occurrences);
    assert!(honest.check(&rec.claims, &posted));

    let index = residual_index(&rec.entries);
    let mut tampered = rec.entries.clone();
    tampered[index] = match &tampered[index] {
        Entry::Ordinary(bytes) => {
            let mut bytes = bytes.clone();
            bytes[15] ^= 1;
            Entry::Ordinary(bytes)
        }
        _ => unreachable!("checked by residual_index"),
    };

    let (fault, occurrences, verifier) = replay(nu, &point, &tampered, rec.seed);
    assert_eq!(
        fault, None,
        "a message the deriver does not predict is not a replay fault; the \
         opening verifier is what must catch it"
    );
    assert_eq!(occurrences, rec.occurrences);
    assert!(
        !verifier.check(&rec.claims, &posted),
        "a residual that the twice-folded columns do not support was accepted"
    );
}

#[test]
fn a_symbol_of_the_wrong_width_is_refused_at_both_levels() {
    let nu = 10;
    let rec = record(nu, 0x5EED + 3);
    let point = rec.claims[0].point.clone();
    let (fault, occurrences, honest) = replay(nu, &point, &rec.entries, rec.seed);
    assert_eq!(fault, None);
    let posted = answers(&rec.prover, &occurrences);
    assert!(honest.check(&rec.claims, &posted));

    for victim in 0..posted.len() {
        let width = posted[victim].2 .0.len();
        assert!(width >= 2, "a one-element symbol has no proper half");

        let mut half = posted.clone();
        half[victim].2 .0.truncate(width / 2);
        assert!(
            !honest.check(&rec.claims, &half),
            "half of symbol {victim} was accepted"
        );

        let mut long = posted.clone();
        long[victim].2 .0.push(rmip_field::F128::ZERO);
        assert!(
            !honest.check(&rec.claims, &long),
            "symbol {victim} plus one element was accepted"
        );
    }
}

#[test]
fn the_replayed_verifier_matches_the_provers_own() {
    let nu = 10;
    let rec = record(nu, 0x5EED + 3);
    let point = rec.claims[0].point.clone();
    let (fault, occurrences, replayed) = replay(nu, &point, &rec.entries, rec.seed);
    assert_eq!(fault, None);
    let own = rec.prover.verifier();

    assert_eq!(
        replayed.proximity, own.proximity,
        "the proximity response must be the recorded one"
    );
    assert_eq!(
        replayed.opening, own.opening,
        "the opening vector must be the recorded one"
    );

    let posted = answers(&rec.prover, &occurrences);
    let mut bad_level0 = posted.clone();
    bad_level0[0].2 .0[0] = bad_level0[0].2 .0[0].add(F128::ONE);
    let mut bad_level1 = posted.clone();
    bad_level1[1].2 .0[0] = bad_level1[1].2 .0[0].add(F128::ONE);
    let mut short = posted.clone();
    short.pop();

    let mut wrong_value = rec.claims.clone();
    wrong_value[0].value = wrong_value[0].value.add(F128::ONE);
    let mut moved_point = rec.claims.clone();
    moved_point[0].point[0] = moved_point[0].point[0].add(F128::ONE);

    type Row<'a> = (
        &'a str,
        &'a [Claim<F128>],
        &'a [(OracleId, Coord, Symbol<F128>)],
    );
    let battery: [Row<'_>; 6] = [
        ("honest", &rec.claims, &posted),
        ("level-0 symbol", &rec.claims, &bad_level0),
        ("level-1 symbol", &rec.claims, &bad_level1),
        ("wrong value", &wrong_value, &posted),
        ("moved point", &moved_point, &posted),
        ("odd answers", &rec.claims, &short),
    ];
    for (name, claims, opened) in battery {
        assert_eq!(
            replayed.check(claims, opened),
            own.check(claims, opened),
            "the replayed verifier disagrees with the prover's at `{name}`"
        );
    }
    assert!(
        replayed.check(&rec.claims, &posted),
        "the honest row of the battery must be an ACCEPT, or the agreement \
         above is agreement on rejecting everything"
    );
}

#[test]
fn the_replay_declares_the_ladders_own_query_count() {
    let nu = 10;
    let rec = record(nu, 0x5EED + 4);
    let replay_side = LadderReplay::new(shape(nu));
    assert_eq!(replay_side.q_long(), rec.prover.params(1).q_long);
    assert_eq!(replay_side.q_long(), 2 * shape(nu).queries());
    assert_eq!(replay_side.q_long(), rec.occurrences.len());
    assert_eq!(rec.occurrences[0].0, LEVEL0_ORACLE);
    assert_eq!(rec.occurrences[1].0, LEVEL1_ORACLE);
}
