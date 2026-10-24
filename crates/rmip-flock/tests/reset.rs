#![cfg(target_arch = "aarch64")]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use flock_core::{pcs::Commitment, proof::R1csProofMergedLigeritoAg};
use flock_field::F128;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};
use rmip_flock::{
    interactive::{Interactive, OracleStore},
    reset::{
        checker::{EncodingPlan, SnapshotChecker},
        geometry, hydrate_proof, ResetChallenger, ResetSchedule,
    },
};
use rmip_iop::{
    replay::CoinReplay,
    transcript::{Entry, OsChallenger, RawChallenger, Transcript, TranscriptPrefix},
};

const MAX_WORDS: usize = 1 << 30;

struct RecordedOs {
    source: OsChallenger,
    log: Arc<Mutex<Vec<rmip_field::F128>>>,
}
impl RawChallenger for RecordedOs {
    fn sample(&mut self) -> rmip_field::F128 {
        let value = self.source.sample();
        self.log.lock().unwrap().push(value);
        value
    }
    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.source.record_ordinary(bytes);
    }
    fn post_oracle(&mut self, id: usize) {
        self.source.post_oracle(id);
    }
    fn transcript(&self) -> &Transcript {
        self.source.transcript()
    }
}

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
            (
                std::array::from_fn(|_| next()),
                std::array::from_fn(|_| next()),
                u64::from(next()),
                64,
                next() & 15,
            )
        })
        .collect()
}

fn transcript_coins(transcript: &Transcript) -> Vec<rmip_field::F128> {
    transcript
        .entries()
        .iter()
        .filter_map(|e| match e {
            Entry::Coin(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn accepts(
    setup: &Blake3Setup,
    commitment: &Commitment,
    placeholder: &R1csProofMergedLigeritoAg,
    schedule: &ResetSchedule,
    answers: Vec<Vec<F128>>,
    coins: &[rmip_field::F128],
) -> bool {
    let mut proof = placeholder.clone();
    if hydrate_proof(&mut proof, schedule, &answers).is_err() {
        return false;
    }
    let mut verifier = ResetChallenger::verifier(
        CoinReplay::new(coins),
        geometry(&setup.pcs_params, MAX_WORDS).unwrap(),
        schedule.clone(),
        answers,
    )
    .unwrap();
    setup
        .verify_union_ag(commitment, &proof, &mut verifier)
        .is_ok()
        && verifier.finish_verifier().is_ok()
        && verifier.with_source(CoinReplay::replayed_exactly)
}

#[test]
fn reset_matches_native_messages_and_all_rows_with_legal_snapshots() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            for n in [197, 256, 4096] {
                let setup = Blake3Setup::with_interactive_target(n, 40).unwrap();
                let log = Arc::new(Mutex::new(Vec::new()));
                let mut main = ResetChallenger::rewinding_main(
                    RecordedOs {
                        source: OsChallenger::new(),
                        log: log.clone(),
                    },
                    geometry(&setup.pcs_params, MAX_WORDS).unwrap(),
                    MAX_WORDS,
                )
                .unwrap();
                let (placeholder, commitment, claim) =
                    setup.prove_fast_union_ag(&inputs(n), &mut main);
                let schedule = main.finish_main().unwrap();
                let (rewound, _) = main.take_snapshots().unwrap();
                assert!(
                    main.take_snapshots().is_err(),
                    "the snapshots are taken once"
                );
                assert_eq!(rewound.len(), schedule.postings().len());
                let coins = log.lock().unwrap().clone();
                assert_eq!(transcript_coins(schedule.transcript()), coins);
                let mut checker = SnapshotChecker::new(
                    Blake3Setup::with_interactive_target(n, 40).unwrap(),
                    inputs(n),
                    MAX_WORDS,
                )
                .unwrap();
                let snapshots: Vec<_> = schedule
                    .postings()
                    .iter()
                    .map(|post| {
                        assert_eq!(
                            post.prefix(),
                            &schedule.transcript().prefix_at(post.oracle())
                        );
                        checker.prepare(post.oracle(), post.prefix()).unwrap()
                    })
                    .collect();
                let independent: Vec<_> = schedule
                    .queries()
                    .iter()
                    .map(|query| snapshots[query.oracle].answer(query.position).unwrap())
                    .collect();
                let answers: Vec<_> = schedule
                    .queries()
                    .iter()
                    .map(|query| rewound[query.oracle].answer(query.position).unwrap())
                    .collect();
                assert_eq!(answers, independent);
                for (own, other) in rewound.iter().zip(&snapshots) {
                    assert_eq!(
                        (own.oracle(), own.shape(), own.prefix_layer()),
                        (other.oracle(), other.shape(), other.prefix_layer())
                    );
                    let rows = own.shape().rows();
                    for position in (0..rows).step_by((rows / 256).max(1)).chain([rows - 1]) {
                        assert_eq!(
                            own.answer(position).unwrap(),
                            other.answer(position).unwrap()
                        );
                    }
                }
                let mut complete = SnapshotChecker::new(
                    Blake3Setup::with_interactive_target(n, 40).unwrap(),
                    inputs(n),
                    MAX_WORDS,
                )
                .unwrap();
                complete.set_encoding_plan(EncodingPlan::Complete).unwrap();
                for (post, snapshot) in schedule.postings().iter().zip(&snapshots) {
                    let full = complete.prepare(post.oracle(), post.prefix()).unwrap();
                    assert_eq!(full.prefix_layer(), post.shape().domain_log());
                    assert_eq!(full.table_words(), post.shape().encoded_words());
                    assert!(snapshot.prefix_layer() >= post.shape().log_inv_rate());
                    let rows = post.shape().rows();
                    for position in (0..rows).step_by((rows / 512).max(1)).chain([rows - 1]) {
                        assert_eq!(
                            snapshot.answer(position).unwrap(),
                            full.answer(position).unwrap()
                        );
                    }
                    for query in schedule
                        .queries()
                        .iter()
                        .filter(|q| q.oracle == post.oracle())
                    {
                        assert_eq!(
                            snapshot.answer(query.position).unwrap(),
                            full.answer(query.position).unwrap()
                        );
                    }
                }
                assert!(complete.set_encoding_plan(EncodingPlan::Prefix).is_err());
                assert_eq!(checker.encoding_plan(), EncodingPlan::Prefix);
                assert_eq!(
                    answers.len(),
                    setup
                        .interactive_security_report()
                        .unwrap()
                        .queries
                        .iter()
                        .sum::<usize>()
                );
                let store = OracleStore::new();
                let mut reference = Interactive::new(CoinReplay::new(&coins), store.clone());
                let (reference_proof, reference_commitment, reference_claim) =
                    setup.prove_fast_union_ag(&inputs(n), &mut reference);
                assert!(reference.with_source(CoinReplay::replayed_exactly));
                assert_eq!(claim, reference_claim);
                assert_eq!(
                    bincode::serialize(&commitment).unwrap(),
                    bincode::serialize(&reference_commitment).unwrap()
                );
                assert_eq!(
                    schedule.transcript().canonical(),
                    reference.with_transcript(Transcript::canonical)
                );
                for (query, answer) in schedule.queries().iter().zip(&answers) {
                    assert_eq!(store.read(query.oracle, query.position).unwrap(), *answer);
                }
                let mut hydrated = placeholder.clone();
                hydrate_proof(&mut hydrated, &schedule, &answers).unwrap();
                assert_eq!(
                    hydrated, reference_proof,
                    "every algebraic message and row must match"
                );
                assert!(accepts(
                    &setup,
                    &commitment,
                    &placeholder,
                    &schedule,
                    answers.clone(),
                    &coins
                ));
                let mut extra_coins = coins.clone();
                extra_coins.push(rmip_field::F128::ZERO);
                assert!(!accepts(
                    &setup,
                    &commitment,
                    &placeholder,
                    &schedule,
                    answers.clone(),
                    &extra_coins
                ));
                let mut changed_coins = coins.clone();
                *changed_coins.last_mut().unwrap() += rmip_field::F128::new(1);
                assert!(!accepts(
                    &setup,
                    &commitment,
                    &placeholder,
                    &schedule,
                    answers.clone(),
                    &changed_coins
                ));
                let mut bad_placeholder = placeholder.clone();
                bad_placeholder
                    .pcs_open
                    .inner
                    .ligerito
                    .initial_proof
                    .opened_rows[0][0] = F128::ONE;
                assert!(hydrate_proof(&mut bad_placeholder, &schedule, &answers).is_err());
                let mut bad_placeholder = placeholder.clone();
                bad_placeholder
                    .pcs_open
                    .inner
                    .ligerito
                    .initial_proof
                    .opened_rows[0]
                    .pop();
                assert!(hydrate_proof(&mut bad_placeholder, &schedule, &answers).is_err());

                for post in schedule.postings() {
                    let snapshot = &snapshots[post.oracle()];
                    assert!(Arc::ptr_eq(
                        snapshot,
                        &checker.prepare(post.oracle(), post.prefix()).unwrap()
                    ));
                    let first = snapshot.answer(0).unwrap();
                    assert_eq!(snapshot.answer_direct(0).unwrap(), first);
                    snapshot.answer(post.shape().rows() - 1).unwrap();
                    assert_eq!(snapshot.answer(0).unwrap(), first);
                    assert!(snapshot.answer(post.shape().rows()).is_err());
                }
                for post in schedule.postings() {
                    let index = schedule
                        .queries()
                        .iter()
                        .position(|q| q.oracle == post.oracle())
                        .unwrap();
                    let mut bad = answers.clone();
                    bad[index][0] += F128::ONE;
                    assert!(!accepts(
                        &setup,
                        &commitment,
                        &placeholder,
                        &schedule,
                        bad,
                        &coins
                    ));
                }
                let mut bad_message = placeholder.clone();
                bad_message.pcs_open.q_eval += F128::ONE;
                assert!(!accepts(
                    &setup,
                    &commitment,
                    &bad_message,
                    &schedule,
                    answers.clone(),
                    &coins
                ));
                let mut missing = answers.clone();
                missing.pop();
                assert!(hydrate_proof(&mut placeholder.clone(), &schedule, &missing).is_err());
                let mut extra = answers;
                extra.push(vec![F128::ZERO]);
                assert!(hydrate_proof(&mut placeholder.clone(), &schedule, &extra).is_err());

                let post = &schedule.postings()[0];
                let mut future = post.prefix().as_bytes().to_vec();
                Entry::Coin(rmip_field::F128::ZERO).encode_into(&mut future);
                let mut fresh = SnapshotChecker::new(
                    Blake3Setup::with_interactive_target(n, 40).unwrap(),
                    inputs(n),
                    MAX_WORDS,
                )
                .unwrap();
                assert!(fresh
                    .prepare(0, &TranscriptPrefix::from_bytes(&future).unwrap())
                    .is_err());
                let mut fresh = SnapshotChecker::new(
                    Blake3Setup::with_interactive_target(n, 40).unwrap(),
                    inputs(n),
                    MAX_WORDS,
                )
                .unwrap();
                let next = &schedule.postings()[1];
                assert!(fresh.prepare(next.oracle(), next.prefix()).is_err());
                let mut fresh = SnapshotChecker::new(
                    Blake3Setup::with_interactive_target(n, 40).unwrap(),
                    inputs(n),
                    MAX_WORDS,
                )
                .unwrap();
                fresh.prepare(post.oracle(), post.prefix()).unwrap();
                let mut future = next.prefix().as_bytes().to_vec();
                Entry::Coin(rmip_field::F128::ZERO).encode_into(&mut future);
                assert!(fresh
                    .prepare(
                        next.oracle(),
                        &TranscriptPrefix::from_bytes(&future).unwrap()
                    )
                    .is_err());
            }
        });
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[test]
#[ignore = "single rewinding prover, end-to-end measurement"]
fn reset_cost() {
    let log_blocks: usize = std::env::var("RMIP_MIP_LOG_BLOCKS")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .unwrap();
    assert!([8, 10, 12, 14, 16, 18, 20].contains(&log_blocks));
    rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| {
        let report = Blake3Setup::with_interactive_target(1 << log_blocks, 40).unwrap().interactive_security_report().unwrap();
        println!("reset_cost_v2 field={} fold_field={} witness_bits=2^{} native_target={} repetitions=1 warmup=1 measured=3 setup_excluded=true witness_included=true serialization_included=true trusted_reset=true rewinding_single_prover=true pins=0", report.base_field_bits, report.fold_field_bits, log_blocks + 14, report.target_bits);
        for run in 0..4 {
            let n = 1 << log_blocks;
            let setup = Blake3Setup::with_interactive_target(n, 40).unwrap();
            let witness = inputs(n);
            let shapes = geometry(&setup.pcs_params, MAX_WORDS).unwrap();
            let source_log = Arc::new(Mutex::new(Vec::new()));
            eprintln!("reset_phase_boundary run={run} role=prover begin");
            let start = Instant::now();
            let mut main = ResetChallenger::rewinding_main(RecordedOs { source: OsChallenger::new(), log: source_log.clone() }, shapes.clone(), MAX_WORDS).unwrap();
            let (proof, commitment, claim) = setup.prove_fast_union_ag(&witness, &mut main);
            let prove_ms = ms(start);
            let start = Instant::now();
            let schedule = main.finish_main().unwrap();
            let (snapshots, snapshot_time) = main.take_snapshots().unwrap();
            let wire = bincode::serialize(&(&proof, &commitment)).unwrap();
            drop((main, proof, commitment));
            let finish_ms = ms(start);
            let main_ms = prove_ms + finish_ms;
            let snapshot_ms = snapshot_time.as_secs_f64() * 1000.0;
            let coins = source_log.lock().unwrap().clone();
            assert_eq!(transcript_coins(schedule.transcript()), coins);
            let start = Instant::now();
            let replies: Vec<_> = schedule.queries().iter().map(|query| {
                let answer = snapshots[query.oracle].answer(query.position).unwrap();
                bincode::serialize(&answer).unwrap()
            }).collect();
            let answer_ms = ms(start);
            let (coefficient_words, table_words): (usize, usize) = snapshots.iter().map(|s| (s.coefficient_words(), s.table_words())).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            let layers: Vec<_> = snapshots.iter().map(|s| (s.shape(), s.prefix_layer(), s.table_words())).collect();
            let start = Instant::now();
            drop(snapshots);
            let cleanup_ms = ms(start);
            let checker_ms = answer_ms + cleanup_ms;
            eprintln!("reset_phase_boundary run={run} role=prover end");

            let start = Instant::now();
            let (mut proof, commitment): (R1csProofMergedLigeritoAg, Commitment) = bincode::deserialize(&wire).unwrap();
            let answers: Vec<Vec<F128>> = replies.iter().map(|reply| bincode::deserialize(reply).unwrap()).collect();
            hydrate_proof(&mut proof, &schedule, &answers).unwrap();
            let mut verifier = ResetChallenger::verifier(CoinReplay::new(&coins), shapes, schedule.clone(), answers).unwrap();
            assert_eq!(setup.verify_union_ag(&commitment, &proof, &mut verifier).unwrap(), claim);
            verifier.finish_verifier().unwrap();
            assert!(verifier.with_source(CoinReplay::replayed_exactly));
            assert_eq!(verifier.transcript().canonical(), schedule.transcript().canonical());
            drop((proof, commitment, verifier));
            let verifier_ms = ms(start);
            if run > 0 {
                for (oracle, (shape, layer, words)) in layers.iter().enumerate() {
                    println!("reset_snapshot run={run} oracle={oracle} encoded_words={} domain_log={} log_inv_rate={} lanes={} prefix_layer={layer} table_words={words}", shape.encoded_words(), shape.domain_log(), shape.log_inv_rate(), shape.lanes());
                }
                println!("reset run={run} main_ms={main_ms:.3} prove_ms={prove_ms:.3} finish_ms={finish_ms:.3} snapshot_ms={snapshot_ms:.3} checker_ms={checker_ms:.3} answer_ms={answer_ms:.3} cleanup_ms={cleanup_ms:.3} verifier_ms={verifier_ms:.3} q={} oracles={} proof_bytes={} reply_bytes={} coefficient_bytes={} table_bytes={} native_error_bits={:.6}", schedule.queries().len(), schedule.postings().len(), wire.len(), replies.iter().map(Vec::len).sum::<usize>(), coefficient_words * F128::BYTES, table_words * F128::BYTES, setup.interactive_security_report().unwrap().native_error_bits());
            }
        }
    });
}

#[test]
#[ignore = "per-oracle encoding schedule calibration on the native shapes"]
fn reset_prefix_calibration() {
    use rmip_flock::encoder::{Coefficients, ShiftedEncoder};
    let log_blocks: usize = std::env::var("RMIP_MIP_LOG_BLOCKS")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .unwrap();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut word = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| {
        let setup = Blake3Setup::with_interactive_target(1 << log_blocks, 40).unwrap();
        for (oracle, geometry) in geometry(&setup.pcs_params, MAX_WORDS).unwrap().into_iter().enumerate() {
            let shape = geometry.shape;
            let values = (0..shape.coefficient_words())
                .map(|_| {
                    let mut bytes = [0u8; F128::BYTES];
                    for limb in bytes.chunks_mut(8) {
                        limb.copy_from_slice(&word().to_le_bytes()[..limb.len()]);
                    }
                    F128::from_le_bytes(bytes)
                })
                .collect();
            let coefficients = Coefficients::new(shape, values).unwrap();
            let encoder = ShiftedEncoder::new(shape.native_domain_log(), MAX_WORDS).unwrap();
            let positions: Vec<_> = (0..geometry.queries).map(|_| word() as usize % shape.rows()).collect();
            let (domain, first, lanes) = (shape.domain_log(), shape.log_inv_rate(), shape.lanes());
            let half_products = (shape.rows() / 2 * lanes) as f64;
            let mut full_ms = f64::MAX;
            for _ in 0..3 {
                let start = Instant::now();
                std::hint::black_box(encoder.encode(&coefficients).unwrap());
                full_ms = full_ms.min(ms(start));
            }
            let start = Instant::now();
            for &position in &positions {
                std::hint::black_box(encoder.evaluate(&coefficients, position).unwrap());
            }
            let direct_ms = ms(start);
            let chosen = encoder.prefix_layer(shape, geometry.queries).unwrap();
            {
                let start = Instant::now();
                let mut table = vec![F128::ZERO; shape.encoded_words()];
                for chunk in table.chunks_exact_mut(shape.coefficient_words()) {
                    chunk.copy_from_slice(coefficients.values());
                }
                let fill_ms = ms(start);
                let ntt = flock_core::ntt::AdditiveNttF128::standard(domain);
                let start = Instant::now();
                ntt.forward_transform_interleaved_from_layer(&mut table, lanes, first);
                let transform_ms = ms(start);
                let mut again = vec![F128::ZERO; shape.encoded_words()];
                for chunk in again.chunks_exact_mut(shape.coefficient_words()) {
                    chunk.copy_from_slice(coefficients.values());
                }
                let start = Instant::now();
                ntt.forward_transform_interleaved_scalar_from_layer(&mut again, lanes, first);
                let scalar_ms = ms(start);
                assert!(again == table);
                let zero_twiddles: usize = (first..domain).map(|layer| (0..1usize << layer).filter(|&block| ntt.twiddle(layer, block) == F128::ZERO).count()).sum();
                let narrow: usize = (first..domain).map(|layer| (0..1usize << layer).filter(|&block| ntt.twiddle(layer, block).to_le_bytes()[8..].iter().all(|&b| b == 0)).count()).sum();
                let per_layer: Vec<String> = (first..domain).map(|layer| format!("{layer}:{:.2}", (0..1usize << layer).filter(|&block| ntt.twiddle(layer, block).to_le_bytes()[8..].iter().all(|&b| b == 0)).count() as f64 / (1usize << layer) as f64)).collect();
                println!("calibration_narrow m={} oracle={oracle} {}", log_blocks + 14, per_layer.join(" "));
                println!("calibration_split m={} oracle={oracle} fill_ms={fill_ms:.3} flock_transform_ms={transform_ms:.3} flock_ns_per_product={:.3} flock_scalar_ms={scalar_ms:.3} zero_twiddle_blocks={zero_twiddles} narrow_twiddle_blocks={narrow} blocks={}", log_blocks + 14, transform_ms * 1e6 / (half_products * (domain - first) as f64), (first..domain).map(|layer| 1usize << layer).sum::<usize>());
            }
            println!(
                "calibration m={} oracle={oracle} domain={domain} rate={first} lanes={lanes} q={} full_ms={full_ms:.3} full_ns_per_product={:.3} direct_ms={direct_ms:.3} direct_ns_per_product={:.3} chosen_layer={chosen}",
                log_blocks + 14,
                geometry.queries,
                full_ms * 1e6 / (half_products * (domain - first) as f64),
                direct_ms * 1e6 / (geometry.queries * shape.coefficient_words()) as f64,
            );
            for layer in first..=domain {
                let finish_products = geometry.queries as f64 * ((1usize << (domain - layer)) - 1) as f64 * lanes as f64;
                if finish_products > 4.0 * half_products * (domain - first) as f64 {
                    continue;
                }
                let mut prefix_ms = f64::MAX;
                let mut partial = None;
                for _ in 0..3 {
                    drop(partial.take());
                    let start = Instant::now();
                    partial = Some(encoder.encode_prefix(&coefficients, layer).unwrap());
                    prefix_ms = prefix_ms.min(ms(start));
                }
                let partial = partial.unwrap();
                let mut finish_ms = f64::MAX;
                for _ in 0..3 {
                    let start = Instant::now();
                    for &position in &positions {
                        std::hint::black_box(encoder.finish_row(&coefficients, &partial, position).unwrap());
                    }
                    finish_ms = finish_ms.min(ms(start));
                }
                let prefix_products = half_products * (layer - first) as f64;
                println!(
                    "calibration_layer m={} oracle={oracle} layer={layer} prefix_ms={prefix_ms:.3} finish_ms={finish_ms:.3} total_ms={:.3} prefix_ns_per_product={:.3} finish_ns_per_product={:.3}{}",
                    log_blocks + 14,
                    prefix_ms + finish_ms,
                    if prefix_products > 0.0 { prefix_ms * 1e6 / prefix_products } else { 0.0 },
                    if finish_products > 0.0 { finish_ms * 1e6 / finish_products } else { 0.0 },
                    if layer == chosen { " chosen" } else { "" },
                );
            }
        }
    });
}
