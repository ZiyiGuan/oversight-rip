use flock_core::challenger::Challenger;
use flock_prover::challenger::FsChallenger;
use rmip_flock::interactive::{Interactive, OracleStore, Recording, Replay};
use rmip_iop::transcript::SeededChallenger;

const DOMAIN: &[u8] = b"rmip-recording-fork-test";

#[test]
fn recording_an_interactive_challenger_leaves_its_fork_schedule_alone() {
    let run = |wrap: bool| -> (
        Vec<flock_field::F128>,
        Option<rmip_flock::interactive::CoinLog>,
    ) {
        let inner = Interactive::new(SeededChallenger::new(7), OracleStore::new());
        let mut coins = Vec::new();
        if wrap {
            let mut rec = Recording::new(inner);
            coins.push(rec.sample_f128());
            let mut child = rec.fork(b"child");
            coins.push(child.sample_f128());
            coins.push(rec.sample_f128());
            rec.merge_child(child);
            coins.push(rec.sample_f128());
            let log = rec.log();
            (coins, Some(log))
        } else {
            let mut ch = inner;
            coins.push(ch.sample_f128());
            let mut child = ch.fork(b"child");
            coins.push(child.sample_f128());
            coins.push(ch.sample_f128());
            ch.merge_child(child);
            coins.push(ch.sample_f128());
            (coins, None)
        }
    };
    let (plain, _) = run(false);
    let (recorded, log) = run(true);
    assert_eq!(plain, recorded, "recording changed the interactive coins");
    let log = log.unwrap();
    let streams = log.lock().unwrap();
    assert_eq!(streams.len(), 2);
    assert!(
        !streams[1].seeded,
        "an interactive fork records as unseeded"
    );
    assert_eq!(streams[1].offset, 1);
    assert_eq!(
        streams[0].coins.len(),
        3,
        "the parent's own coins, no seed words"
    );
    assert_eq!(streams[1].coins.len(), 1);
}

#[test]
fn recording_a_fiat_shamir_challenger_keeps_its_seeded_fork() {
    let mut rec = Recording::new(FsChallenger::new(DOMAIN));
    rec.observe_bytes(b"statement");
    let a = rec.sample_f128();
    let mut child = rec.fork(b"child");
    let b = child.sample_f128();
    rec.merge_child(child);
    let c = rec.sample_f128();
    let log = rec.log();
    let streams = log.lock().unwrap();
    assert_eq!(streams.len(), 2);
    assert!(streams[1].seeded);
    assert_eq!(
        streams[1].offset, 3,
        "one coin, then the two seed words, then the fork"
    );
    assert_eq!(streams[0].coins.len(), 4, "a, two seed words, c");
    assert_eq!(streams[0].coins[0], a);
    assert_eq!(streams[0].coins[3], c);
    assert_eq!(streams[1].coins, vec![b]);
}

#[test]
fn replay_reproduces_unseeded_and_seeded_forks() {
    let mut rec = Recording::new(Interactive::new(
        SeededChallenger::new(9),
        OracleStore::new(),
    ));
    let a = rec.sample_f128();
    let mut child = rec.fork(b"child");
    let b = child.sample_f128();
    rec.merge_child(child);
    let c = rec.sample_f128();
    let kind = rec.hash_kind();
    let mut replay = Replay::new(rec.log(), kind);
    assert_eq!(replay.sample_f128(), a);
    let mut child = replay.fork(b"child");
    assert_eq!(child.sample_f128(), b);
    replay.merge_child(child);
    assert_eq!(replay.sample_f128(), c);
    replay.assert_exhausted();

    let mut rec = Recording::new(FsChallenger::new(DOMAIN));
    let a = rec.sample_f128();
    let mut child = rec.fork(b"child");
    let b = child.sample_f128();
    rec.merge_child(child);
    let c = rec.sample_f128();
    let kind = rec.hash_kind();
    let mut replay = Replay::new(rec.log(), kind);
    assert_eq!(replay.sample_f128(), a);
    let mut child = replay.fork(b"child");
    assert_eq!(child.sample_f128(), b);
    replay.merge_child(child);
    assert_eq!(replay.sample_f128(), c);
    replay.assert_exhausted();
}
