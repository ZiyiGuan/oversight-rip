use rmip_demo::wire::{
    decode, measure_execution, measure_execution_with_domains, write_execution,
    write_execution_with_domains, CommunicationBytes, Kind, VERSION, VERSION_WIDE,
};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{Coord, Symbol};
use rmip_iop::driver::{Execution, Occurrence, DUMMY_ORACLE};
use rmip_iop::transcript::Transcript;
use std::io::{self, Write};
use std::time::Duration;

fn fixture() -> (Transcript, Execution, Symbol<F128>) {
    let mut transcript = Transcript::new();
    transcript.post_oracle(DUMMY_ORACLE);
    transcript.record_ordinary(b"ordinary payload".to_vec());
    transcript.record_coin(F128::new(123));
    transcript.post_oracle(7);
    transcript.record_ordinary(Vec::new());
    transcript.record_coin(F128::new(456));
    let occurrence = Occurrence {
        oracle: 7,
        prefix: transcript.prefix_at(7),
        coord: Coord::from(9u64),
        answer: Symbol(vec![F128::ONE, F128::new(42)]),
    };
    let execution = Execution {
        iop_accepted: true,
        occurrences: vec![occurrence.clone(), occurrence],
        oracle_calls: 1,
        claims_submitted: 1,
        pad_claims: 0,
        rounds: 2,
        coins: 2,
        short_bytes: 16,
        transcript_before_sampling: transcript.entries().len() - 1,
        transcript_at_query_start: transcript.entries().len(),
        transcript_at_query_end: transcript.entries().len(),
        t_mat: Duration::ZERO,
        t_verify: Duration::ZERO,
        reduced_claim: None,
        t_open: Duration::ZERO,
        verifier_phases: Vec::new(),
        prover_phases: Vec::new(),
        prover_piop_phases: Vec::new(),
    };
    let checker_answer = Symbol(vec![F128::new(987), F128::ZERO, F128::ONE]);
    (transcript, execution, checker_answer)
}

fn wire() -> Vec<u8> {
    let (transcript, execution, answer) = fixture();
    let mut bytes = Vec::new();
    write_execution(
        &mut bytes,
        &transcript,
        &execution,
        &execution.occurrences[0],
        &answer,
    )
    .unwrap();
    bytes
}

#[test]
fn measured_directional_counts_equal_materialized_messages() {
    let (transcript, execution, answer) = fixture();
    let mut bytes = Vec::new();
    let materialized = write_execution(
        &mut bytes,
        &transcript,
        &execution,
        &execution.occurrences[0],
        &answer,
    )
    .unwrap();
    let counted =
        measure_execution(&transcript, &execution, &execution.occurrences[0], &answer).unwrap();
    assert_eq!(materialized, counted);
    assert_eq!(counted.total_bytes(), bytes.len() as u64);
    assert_eq!(
        counted.payload_bytes + counted.framing_bytes,
        bytes.len() as u64
    );
    let frames = decode(&bytes).unwrap();
    assert_eq!(frames.len() as u64, counted.frames);
    assert_eq!(frames[0].kind, Kind::PublicOraclePost);
    assert_eq!(frames[1].payload, b"ordinary payload");
    assert_eq!(frames[2].payload, F128::new(123).to_be_bytes());
    assert!(frames[4].payload.is_empty());
    let mut directions = [0u64; 4];
    for frame in &frames {
        let direction = match frame.kind {
            Kind::MainOrdinary | Kind::MainOraclePost | Kind::MainAnswer => 0,
            Kind::PublicCoin | Kind::PublicOraclePost | Kind::MainQuery => 1,
            Kind::CheckerQuery => 2,
            Kind::CheckerAnswer => 3,
        };
        directions[direction] += 14 + frame.payload.len() as u64;
    }
    assert_eq!(
        directions,
        [
            counted.main_to_verifier_bytes,
            counted.verifier_to_main_bytes,
            counted.verifier_to_checker_bytes,
            counted.checker_to_verifier_bytes,
        ]
    );
    assert_eq!(directions.iter().sum::<u64>(), counted.total_bytes());
    let mut total = CommunicationBytes::default();
    total += counted;
    total += counted;
    assert_eq!(total.total_bytes(), 2 * bytes.len() as u64);
}

#[test]
fn checker_question_is_exactly_the_protocol_triple_without_an_occurrence_index() {
    let (transcript, execution, answer) = fixture();
    let mut first = Vec::new();
    let mut duplicate = Vec::new();
    write_execution(
        &mut first,
        &transcript,
        &execution,
        &execution.occurrences[0],
        &answer,
    )
    .unwrap();
    write_execution(
        &mut duplicate,
        &transcript,
        &execution,
        &execution.occurrences[1],
        &answer,
    )
    .unwrap();
    assert_eq!(
        first, duplicate,
        "duplicate tagged queries must expose no selection index"
    );
    let frames = decode(&first).unwrap();
    let question = frames
        .iter()
        .find(|frame| frame.kind == Kind::CheckerQuery)
        .unwrap();
    let (oracle, prefix, coord) = question.checker_question().unwrap();
    assert_eq!((oracle, coord), (7, Coord::from(9u64)));
    assert_eq!(prefix, execution.occurrences[0].prefix);
    assert_eq!(question.payload.len(), 24 + prefix.len());
    let reply = frames.last().unwrap();
    assert_eq!(reply.kind, Kind::CheckerAnswer);
    assert_eq!(&reply.payload[..8], &3u64.to_be_bytes());
    assert_eq!(&reply.payload[8..24], &F128::new(987).to_be_bytes());
    assert_eq!(reply.payload.len(), 8 + 3 * 16);
}

struct ShortWriter(Vec<u8>);
impl Write for ShortWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = bytes.len().min(3);
        self.0.extend_from_slice(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn counted_writes_handle_partial_writes_and_propagate_sink_errors() {
    let (transcript, execution, answer) = fixture();
    let mut short = ShortWriter(Vec::new());
    let counts = write_execution(
        &mut short,
        &transcript,
        &execution,
        &execution.occurrences[0],
        &answer,
    )
    .unwrap();
    assert_eq!(short.0, wire());
    assert_eq!(counts.total_bytes(), short.0.len() as u64);
    let mut too_small = [0u8; 5];
    let error = write_execution(
        too_small.as_mut_slice(),
        &transcript,
        &execution,
        &execution.occurrences[0],
        &answer,
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::WriteZero);
}

#[test]
fn decoder_rejects_truncation_bad_versions_tags_lengths_and_trailing_payload() {
    let bytes = wire();
    for cut in 1..14 {
        assert!(decode(&bytes[..cut]).is_err());
    }
    let last_len = 14 + 8 + 3 * 16;
    for omitted in 1..last_len {
        assert!(decode(&bytes[..bytes.len() - omitted]).is_err());
    }
    for offset in [0, 4, 5] {
        let mut bad = bytes.clone();
        bad[offset] = 255;
        assert!(decode(&bad).is_err());
    }
    let mut length_overflow = bytes.clone();
    length_overflow[6..14].copy_from_slice(&u64::MAX.to_be_bytes());
    assert!(decode(&length_overflow).is_err());
    let mut field_count_overflow = bytes.clone();
    let count_at = bytes.len() - last_len + 14;
    field_count_overflow[count_at..count_at + 8].copy_from_slice(&u64::MAX.to_be_bytes());
    assert!(decode(&field_count_overflow).is_err());
    let frames = decode(&bytes).unwrap();
    let question = frames
        .iter()
        .find(|frame| frame.kind == Kind::CheckerQuery)
        .unwrap();
    let mut trailing = Vec::from(&b"RMLW\x01\x07"[..]);
    trailing.extend_from_slice(&(question.payload.len() as u64 + 8).to_be_bytes());
    trailing.extend_from_slice(question.payload);
    trailing.extend_from_slice(&123u64.to_be_bytes());
    assert!(
        decode(&trailing).is_err(),
        "a private audit index cannot be appended to the triple"
    );
    let mut bad_prefix_length = trailing;
    bad_prefix_length[22..30].copy_from_slice(&u64::MAX.to_be_bytes());
    assert!(decode(&bad_prefix_length).is_err());
}

#[test]
fn serialization_checks_query_chronology_and_posting_prefixes() {
    let (transcript, mut execution, answer) = fixture();
    execution.transcript_at_query_start -= 1;
    assert!(
        measure_execution(&transcript, &execution, &execution.occurrences[0], &answer).is_err()
    );
    execution.transcript_at_query_start += 1;
    execution.occurrences[0].prefix = transcript.prefix_at(DUMMY_ORACLE);
    assert!(
        measure_execution(&transcript, &execution, &execution.occurrences[0], &answer).is_err()
    );
}

#[test]
fn actual_two_prover_messages_round_trip() {
    use rmip_backend_interleaved::{Ligero, Shape};
    use rmip_demo::measured_execution::statement;
    use rmip_demo::mnist_bench::sparse_mnist_case;
    use rmip_iop::backend::CommitmentBackend;
    use rmip_iop::driver::{Cheat, Config};
    use rmip_iop::transcript::{RawChallenger, SeededChallenger};

    let (case, oracle) = sparse_mnist_case(256, "sparse-mnist");
    let shape = Shape::balanced(case.layout.nu, 4, 4.0);
    let mut checker_backend = Ligero::lazy(shape);
    checker_backend.commit(&case.honest_trace, &mut SeededChallenger::new(0));
    let checker = rmip_mip::PCheck::new(checker_backend);
    let mut main = Ligero::lazy(shape);
    let mut coins = SeededChallenger::new(123);
    let (execution, verdict) = rmip_mip_prover::execute_recording(
        &statement(&case),
        &mut main,
        &oracle,
        &Config::for_instance(&case.inst, 0.125),
        &Cheat::default(),
        &mut coins,
    );
    assert!(verdict.is_ok());
    let (audit, answer) =
        rmip_mip::checker::audit_with_answer(&execution, &checker, &mut SeededChallenger::new(999));
    let audited = match audit {
        rmip_mip::Audit::Accepted { audited } => audited,
        _ => panic!("honest two-prover audit failed"),
    };
    let mut bytes = Vec::new();
    let measured = write_execution(
        &mut bytes,
        coins.transcript(),
        &execution,
        &execution.occurrences[audited],
        &answer,
    )
    .unwrap();
    assert_eq!(measured.total_bytes(), bytes.len() as u64);
    let frames = decode(&bytes).unwrap();
    assert_eq!(
        frames.iter().filter(|f| f.kind == Kind::MainQuery).count(),
        execution.q_long()
    );
    assert_eq!(
        frames.iter().filter(|f| f.kind == Kind::MainAnswer).count(),
        execution.q_long()
    );
    assert_eq!(
        frames.iter().filter(|f| f.kind == Kind::PublicCoin).count(),
        execution.coins
    );
    assert_eq!(
        frames
            .iter()
            .filter(|f| f.kind == Kind::MainOrdinary)
            .map(|f| f.payload.len())
            .sum::<usize>(),
        execution.short_bytes
    );
    assert_eq!(checker.questions_asked(), 1);
}

#[test]
fn serialization_refuses_an_audited_occurrence_absent_from_the_execution() {
    let (transcript, execution, answer) = fixture();
    let listed = execution.occurrences[0].clone();
    let copy = Occurrence {
        oracle: listed.oracle,
        prefix: listed.prefix.clone(),
        coord: listed.coord,
        answer: listed.answer.clone(),
    };
    let mut bytes = Vec::new();
    write_execution(&mut bytes, &transcript, &execution, &copy, &answer)
        .expect("a by-value copy of a listed occurrence is a member");
    let variants = [
        Occurrence {
            coord: listed.coord.checked_add_u64(1).unwrap(),
            ..copy.clone()
        },
        Occurrence {
            answer: Symbol(vec![F128::ONE, F128::new(43)]),
            ..copy.clone()
        },
        Occurrence {
            answer: Symbol(vec![F128::ONE]),
            ..copy.clone()
        },
        Occurrence {
            oracle: DUMMY_ORACLE,
            ..copy.clone()
        },
        Occurrence {
            prefix: transcript.prefix_at(DUMMY_ORACLE),
            ..copy.clone()
        },
    ];
    for absent in variants {
        let error = write_execution(&mut Vec::new(), &transcript, &execution, &absent, &answer)
            .expect_err("an occurrence absent from the execution must be refused");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("absent"), "{error}");
    }
}

#[test]
fn wide_coordinates_round_trip_in_a_domain_of_2_64_plus_1() {
    let (transcript, mut execution, answer) = fixture();
    let domain = Coord::from((1u128 << 64) + 1);
    let domain_of = |oracle| (oracle == 7).then_some(domain);
    let top = Coord::from(1u128 << 64);
    for occurrence in &mut execution.occurrences {
        occurrence.coord = top;
    }
    let audited = execution.occurrences[0].clone();
    let mut bytes = Vec::new();
    let counts = write_execution_with_domains(
        &mut bytes,
        &transcript,
        &execution,
        &audited,
        &answer,
        domain_of,
    )
    .unwrap();
    assert_eq!(
        counts,
        measure_execution_with_domains(&transcript, &execution, &audited, &answer, domain_of)
            .unwrap()
    );
    assert_eq!(counts.total_bytes(), bytes.len() as u64);
    let frames = decode(&bytes).unwrap();
    let prefix_len = audited.prefix.as_bytes().len();
    let mut main_queries = 0;
    for frame in &frames {
        match frame.kind {
            Kind::MainQuery => {
                main_queries += 1;
                assert_eq!(frame.version, VERSION_WIDE);
                assert_eq!(frame.payload.len(), 24);
                assert_eq!(frame.payload[..8], 7u64.to_be_bytes());
                assert_eq!(
                    frame.payload[8..],
                    u128::try_from(top).unwrap().to_be_bytes()
                );
            }
            Kind::CheckerQuery => {
                assert_eq!(frame.version, VERSION_WIDE);
                assert_eq!(frame.payload.len(), 32 + prefix_len);
                let (oracle, prefix, coord) = frame.checker_question().unwrap();
                assert_eq!((oracle, &prefix, coord), (7, &audited.prefix, top));
            }
            _ => assert_eq!(frame.version, VERSION, "{:?} stays narrow", frame.kind),
        }
    }
    assert_eq!(main_queries, execution.occurrences.len());
    assert_eq!(counts.wide_frames, 3);
    assert_eq!(counts.to_json()["serialization"], "rmip-wire-v2");
    assert_eq!(counts.to_json()["version"], VERSION_WIDE);
    let (t, e, a) = fixture();
    let narrow = measure_execution(&t, &e, &e.occurrences[0], &a).unwrap();
    assert_eq!(counts.total_bytes(), narrow.total_bytes() + 3 * 8);
    assert_eq!(narrow.wide_frames, 0);
    assert_eq!(narrow.to_json()["serialization"], "rmip-wire-v1");
    assert!(write_execution(Vec::new(), &transcript, &execution, &audited, &answer).is_err());

    for occurrence in &mut execution.occurrences {
        occurrence.coord = Coord::from(9u64);
    }
    let audited = execution.occurrences[0].clone();
    let mut small = Vec::new();
    let small_counts = write_execution_with_domains(
        &mut small,
        &transcript,
        &execution,
        &audited,
        &answer,
        domain_of,
    )
    .unwrap();
    assert_eq!(small_counts.total_bytes(), counts.total_bytes());
    let frames = decode(&small).unwrap();
    let checker = frames
        .iter()
        .find(|f| f.kind == Kind::CheckerQuery)
        .unwrap();
    assert_eq!(checker.version, VERSION_WIDE);
    assert_eq!(checker.checker_question().unwrap().2, Coord::from(9u64));
}
