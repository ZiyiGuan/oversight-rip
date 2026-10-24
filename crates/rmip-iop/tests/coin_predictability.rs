use rmip_field::F128;
use rmip_iop::transcript::{Entry, OsChallenger, RawChallenger, SeededChallenger};

fn step(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn next_coin_after(c: F128) -> F128 {
    let lo = (c.to_u128() & u64::MAX as u128) as u64;
    let hi2 = step(lo);
    let lo2 = step(hi2);
    F128::new(((hi2 as u128) << 64) | lo2 as u128)
}

fn coins_of(ch: &dyn RawChallenger) -> Vec<F128> {
    ch.transcript()
        .entries()
        .iter()
        .filter_map(|e| match e {
            Entry::Coin(c) => Some(*c),
            _ => None,
        })
        .collect()
}

#[test]
fn a_single_seeded_coin_reveals_the_whole_future_stream() {
    for seed in 1..=200u64 {
        let mut ch = SeededChallenger::new(seed);
        for _ in 0..12 {
            ch.sample();
        }
        let coins = coins_of(&ch);
        for k in 0..coins.len() - 1 {
            assert_eq!(
                next_coin_after(coins[k]),
                coins[k + 1],
                "seed {seed}: coin {k} did not determine coin {}",
                k + 1
            );
        }
    }
}

#[test]
fn ordinary_messages_and_oracle_posts_do_not_break_the_prediction() {
    let mut ch = SeededChallenger::new(0xBEEF);
    let a = ch.sample();
    ch.record_ordinary(b"a prover message");
    ch.post_oracle(0);
    ch.record_ordinary(&[7u8; 64]);
    let b = ch.sample();
    assert_eq!(next_coin_after(a), b);
}

#[test]
fn os_coins_do_not_follow_the_seeded_xorshift_predictor() {
    let mut ch = OsChallenger::new();
    for _ in 0..2000 {
        ch.sample();
    }
    let coins = coins_of(&ch);
    let hits = (0..coins.len() - 1)
        .filter(|&k| next_coin_after(coins[k]) == coins[k + 1])
        .count();
    assert_eq!(
        hits,
        0,
        "{hits} of {} OS coins were xorshift-predictable",
        coins.len() - 1
    );

    let distinct: std::collections::HashSet<u128> = coins.iter().map(|c| c.to_u128()).collect();
    assert_eq!(distinct.len(), coins.len(), "OS coins repeated");
    let nonzero_high = coins.iter().filter(|c| (c.to_u128() >> 64) != 0).count();
    assert!(
        nonzero_high > coins.len() - 10,
        "high halves look degenerate"
    );
}

#[test]
fn two_os_challengers_do_not_agree() {
    let mut a = OsChallenger::new();
    let mut b = OsChallenger::new();
    let sa: Vec<F128> = (0..64).map(|_| a.sample()).collect();
    let sb: Vec<F128> = (0..64).map(|_| b.sample()).collect();
    assert_ne!(sa, sb);
    let mut c = SeededChallenger::new(9);
    let mut d = SeededChallenger::new(9);
    let sc: Vec<F128> = (0..64).map(|_| c.sample()).collect();
    let sd: Vec<F128> = (0..64).map(|_| d.sample()).collect();
    assert_eq!(sc, sd);
}

#[test]
fn a_legal_prefix_leaks_the_future_to_whoever_holds_it() {
    let mut ch = SeededChallenger::new(0x5AFE);
    ch.post_oracle(0);
    let a1 = ch.sample();
    ch.post_oracle(1);
    let h = ch.transcript().prefix_at(1);

    let future: Vec<F128> = (0..8).map(|_| ch.sample()).collect();

    let mut c = *h.coins().unwrap().last().unwrap();
    assert_eq!(c, a1);
    for (k, actual) in future.iter().enumerate() {
        c = next_coin_after(c);
        assert_eq!(
            c, *actual,
            "predicted future coin {k} from the prefix alone"
        );
    }
}
