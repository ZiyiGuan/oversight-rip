#![cfg(target_arch = "aarch64")]

use flock_core::pcs::ligerito::LigeritoProfile;
use flock_field::{F128, F256};
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};
use rmip_flock::interactive::{Interactive, OracleStore};
use rmip_iop::replay::CoinReplay;
use rmip_iop::transcript::{Entry, OsChallenger, SeededChallenger};
use std::time::Instant;

fn inputs(n: usize) -> Vec<Compression> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as u32
    };
    (0..n)
        .map(|_| {
            let cv = std::array::from_fn(|_| next());
            let msg = std::array::from_fn(|_| next());
            (cv, msg, u64::from(next()), 64, next() & 15)
        })
        .collect()
}

#[test]
fn target40_roundtrip_and_adversarial_checks() {
    assert_eq!(std::mem::size_of::<F128>() * 8, F128::BITS);
    assert_eq!(std::mem::size_of::<F256>() * 8, F256::BITS);
    for n in [256, 257] {
        let setup = Blake3Setup::with_interactive_target(n, 40).unwrap();
        let report = setup.interactive_security_report().unwrap();
        assert_eq!(report.target_bits, 40);
        assert_eq!(report.base_field_bits, F128::BITS);
        assert_eq!(report.fold_field_bits, F256::BITS);
        assert!(report.native_error < 2f64.powi(-41));
        let store = OracleStore::new();
        let mut prover = Interactive::new(SeededChallenger::new(41), store.clone());
        let (proof, commitment, claim) = setup.prove_fast_union_ag(&inputs(n), &mut prover);
        assert_eq!(commitment.params.profile, LigeritoProfile::Interactive40);
        assert!(commitment.cap.is_empty());
        let mut verifier = Interactive::new(SeededChallenger::new(41), store.clone());
        assert!(
            matches!(setup.verify_union_ag(&commitment, &proof, &mut verifier), Ok(c) if c == claim)
        );
        assert_eq!(
            prover.with_transcript(|t| t.canonical()),
            verifier.with_transcript(|t| t.canonical())
        );

        let mut wrong_coins = Interactive::new(SeededChallenger::new(42), store.clone());
        assert!(setup
            .verify_union_ag(&commitment, &proof, &mut wrong_coins)
            .is_err());
        let mut wrong_row = proof.clone();
        wrong_row.pcs_open.inner.ligerito.initial_proof.opened_rows[0][0] += F128::ONE;
        let mut verifier = Interactive::new(SeededChallenger::new(41), store.clone());
        assert!(setup
            .verify_union_ag(&commitment, &wrong_row, &mut verifier)
            .is_err());
        let mut wrong_algebra = proof.clone();
        wrong_algebra.boolean.ag.final_a_eval += F128::ONE;
        let mut verifier = Interactive::new(SeededChallenger::new(41), store.clone());
        assert!(setup
            .verify_union_ag(&commitment, &wrong_algebra, &mut verifier)
            .is_err());
        let mut nonzero_nonce = proof.clone();
        nonzero_nonce.boolean.ag.r1_nonce = 1;
        let mut verifier = Interactive::new(SeededChallenger::new(41), store.clone());
        assert!(setup
            .verify_union_ag(&commitment, &nonzero_nonce, &mut verifier)
            .is_err());
        let mut wrong_target = commitment.clone();
        wrong_target.params.profile = LigeritoProfile::Fast;
        let mut verifier = Interactive::new(SeededChallenger::new(41), store);
        assert!(setup
            .verify_union_ag(&wrong_target, &proof, &mut verifier)
            .is_err());
    }
}

#[test]
fn target40_os_coins_replay_exactly() {
    let setup = Blake3Setup::with_interactive_target(256, 40).unwrap();
    let store = OracleStore::new();
    let mut prover = Interactive::new(OsChallenger::new(), store.clone());
    let (proof, commitment, claim) = setup.prove_fast_union_ag(&inputs(256), &mut prover);
    let coins: Vec<_> = prover.with_transcript(|t| {
        t.entries()
            .iter()
            .filter_map(|e| {
                if let Entry::Coin(c) = e {
                    Some(*c)
                } else {
                    None
                }
            })
            .collect()
    });
    let mut verifier = Interactive::new(CoinReplay::new(&coins), store.clone());
    assert!(
        matches!(setup.verify_union_ag(&commitment, &proof, &mut verifier), Ok(c) if c == claim)
    );
    assert!(verifier.with_source(|r| r.replayed_exactly()));
    assert_eq!(
        prover.with_transcript(|t| t.canonical()),
        verifier.with_transcript(|t| t.canonical())
    );
    let mut short = Interactive::new(CoinReplay::new(&coins[..coins.len() - 1]), store);
    let _ = setup.verify_union_ag(&commitment, &proof, &mut short);
    assert!(short.with_source(|r| r.overrun() > 0 && !r.replayed_exactly()));
}

#[test]
#[ignore = "measurement"]
fn interactive_target40_cost() {
    let sizes: Vec<usize> = std::env::var("RMIP_BLAKE3_LOG2S")
        .unwrap_or_else(|_| "12".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    for log_blocks in sizes {
        let n = 1usize << log_blocks;
        let input = inputs(n);
        let setup = Blake3Setup::with_interactive_target(n, 40).unwrap();
        let report = setup.interactive_security_report().unwrap();
        let mut times = Vec::new();
        let mut verify_times = Vec::new();
        let mut bytes = 0;
        let mut oracle_bytes = 0;
        for run in 0..4 {
            let store = OracleStore::new();
            let mut prover = Interactive::new(SeededChallenger::new(41), store.clone());
            let start = Instant::now();
            let (proof, commitment, claim) = setup.prove_fast_union_ag(&input, &mut prover);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let mut verifier = Interactive::new(SeededChallenger::new(41), store.clone());
            let start = Instant::now();
            assert!(
                matches!(setup.verify_union_ag(&commitment, &proof, &mut verifier), Ok(c) if c == claim)
            );
            let verification = start.elapsed().as_secs_f64() * 1000.0;
            if run != 0 {
                times.push(elapsed);
                verify_times.push(verification);
            }
            bytes = bincode::serialize(&proof).unwrap().len();
            oracle_bytes = (0..store.len())
                .map(|i| {
                    let (width, rows) = store.shape(i).unwrap();
                    width * rows * F128::BYTES
                })
                .sum::<usize>();
        }
        times.sort_by(f64::total_cmp);
        verify_times.sort_by(f64::total_cmp);
        println!("target40 base={} fold={} witness_bits=2^{} queries={:?} conditional_native_bits={:.6} prover_ms={:.3} verifier_ms={:.3} proof_bytes={} oracle_bytes={}",
            F128::BITS, F256::BITS, log_blocks + 14, report.queries, report.native_error_bits(), times[1], verify_times[1], bytes, oracle_bytes);
    }
}
