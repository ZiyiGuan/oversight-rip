use rmip_field::{BinaryField, F128};
use rmip_iop::transcript::{PrefixDecodeError, RawChallenger, SeededChallenger, TranscriptPrefix};
use rmip_iop::{Entry, Transcript};

fn sample_run() -> Transcript {
    let mut t = Transcript::new();
    t.record_ordinary(b"instance".to_vec());
    t.post_oracle(0);
    t.record_coin(F128::new(7));
    t.record_ordinary(b"sumcheck round".to_vec());
    t.post_oracle(1);
    t.record_coin(F128::new(9));
    t.post_oracle(2);
    t
}

#[test]
fn prefix_is_strictly_before_the_post() {
    let t = sample_run();
    assert_eq!(t.prefix_at(0).entry_count(), 1, "only the instance message");
    assert_eq!(t.prefix_at(1).entry_count(), 4);
    assert_eq!(t.prefix_at(2).entry_count(), 6);
}

#[test]
fn later_events_cannot_reach_an_earlier_prefix() {
    let mut t = sample_run();
    let before = t.prefix_at(0);
    t.record_coin(F128::new(0xdead));
    t.record_ordinary(b"more".to_vec());
    t.post_oracle(3);
    assert_eq!(
        t.prefix_at(0),
        before,
        "oracle 0's prefix must be frozen at its post"
    );
    assert_eq!(t.prefix_at(1), t.prefix_at(1));
}

#[test]
fn encoding_is_canonical_and_injective() {
    assert_eq!(sample_run().prefix_at(1), sample_run().prefix_at(1));

    let mut a = Transcript::new();
    a.record_coin(F128::new(1));
    a.post_oracle(0);
    let mut b = Transcript::new();
    b.record_coin(F128::new(2));
    b.post_oracle(0);
    assert_ne!(a.prefix_at(0), b.prefix_at(0));

    let mut c = Transcript::new();
    c.record_ordinary(b"ab".to_vec());
    c.post_oracle(0);
    let mut d = Transcript::new();
    d.record_ordinary(b"a".to_vec());
    d.record_ordinary(b"b".to_vec());
    d.post_oracle(0);
    assert_ne!(c.prefix_at(0), d.prefix_at(0));

    let mut e = Transcript::new();
    e.record_ordinary(F128::new(7).to_be_bytes().to_vec());
    e.post_oracle(0);
    let mut f = Transcript::new();
    f.record_coin(F128::new(7));
    f.post_oracle(0);
    assert_ne!(e.prefix_at(0), f.prefix_at(0));
}

#[test]
fn posts_carry_identifiers_not_contents() {
    let t = sample_run();
    for e in t.entries() {
        if let Entry::OraclePost(id) = e {
            assert!(*id < 3);
        }
    }
    assert!(matches!(t.entries()[1], Entry::OraclePost(0)));
}

#[test]
fn ordering_is_part_of_the_prefix() {
    let mut a = Transcript::new();
    a.record_ordinary(b"m".to_vec());
    a.record_coin(F128::new(3));
    a.post_oracle(0);
    let mut b = Transcript::new();
    b.record_coin(F128::new(3));
    b.record_ordinary(b"m".to_vec());
    b.post_oracle(0);
    assert_ne!(a.prefix_at(0), b.prefix_at(0));
}

#[test]
fn an_immediate_post_has_an_empty_prefix() {
    let mut t = Transcript::new();
    t.post_oracle(0);
    let p = t.prefix_at(0);
    assert!(p.is_empty());
    assert_eq!(p.entry_count(), 0);
}

#[test]
#[should_panic(expected = "posted twice")]
fn duplicate_oracle_ids_are_rejected() {
    let mut t = Transcript::new();
    t.post_oracle(0);
    t.post_oracle(0);
}

#[test]
#[should_panic(expected = "never posted")]
fn querying_an_unposted_oracle_is_a_bug() {
    Transcript::new().prefix_at(4);
}

#[test]
fn prefix_length_is_reportable() {
    let t = sample_run();
    let c_pref = (0..3).map(|i| t.prefix_at(i).len()).max().unwrap();
    assert!(c_pref > 0);
    assert!(
        t.prefix_at(0).len() < t.prefix_at(2).len(),
        "prefixes grow monotonically"
    );
}

#[test]
fn seeded_challenger_records_what_it_emits() {
    let mut ch = SeededChallenger::new(42);
    ch.record_ordinary(b"hello");
    ch.post_oracle(0);
    let c = ch.sample();
    assert_eq!(ch.transcript().entries().len(), 3);
    assert_eq!(ch.transcript().entries()[2], Entry::Coin(c));
    let mut ch2 = SeededChallenger::new(42);
    ch2.record_ordinary(b"hello");
    ch2.post_oracle(0);
    assert_eq!(ch2.sample(), c);
    assert_eq!(ch2.transcript().canonical(), ch.transcript().canonical());
}

#[test]
fn a_prefix_decodes_to_exactly_what_was_recorded() {
    let mut ch = SeededChallenger::new(0xDEC0DE);
    let mut expected = Vec::new();

    ch.record_ordinary(b"");
    expected.push(Entry::Ordinary(Vec::new()));
    let c1 = ch.sample();
    expected.push(Entry::Coin(c1));
    ch.record_ordinary(&[0xFFu8; 300]);
    expected.push(Entry::Ordinary(vec![0xFF; 300]));
    ch.post_oracle(7);
    expected.push(Entry::OraclePost(7));
    let c2 = ch.sample();
    expected.push(Entry::Coin(c2));
    ch.record_ordinary(&[1u8, 2, 3, 0, 0, 0, 0, 0, 0, 0, 99]);
    expected.push(Entry::Ordinary(vec![1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 99]));
    ch.post_oracle(9);

    let h = ch.transcript().prefix_at(9);
    assert_eq!(
        h.decode().expect("a prefix we produced must decode"),
        expected
    );
    assert_eq!(h.entry_count(), expected.len());
    assert_eq!(
        h.coins().unwrap(),
        vec![c1, c2],
        "coins in transcript order"
    );

    let h7 = ch.transcript().prefix_at(7);
    assert_eq!(h7.decode().unwrap(), expected[..3].to_vec());
}

#[test]
fn the_empty_prefix_decodes_to_nothing() {
    let mut ch = SeededChallenger::new(1);
    ch.post_oracle(0);
    let h = ch.transcript().prefix_at(0);
    assert!(h.is_empty());
    assert_eq!(h.decode().unwrap(), Vec::<Entry>::new());
    assert_eq!(h.coins().unwrap(), Vec::<rmip_field::F128>::new());
}

#[test]
fn different_coins_give_different_decoded_prefixes() {
    let read = |seed: u64| {
        let mut ch = SeededChallenger::new(seed);
        let _ = ch.sample();
        let _ = ch.sample();
        ch.post_oracle(3);
        let h = ch.transcript().prefix_at(3);
        (h.clone(), h.coins().unwrap())
    };
    let (h1, c1) = read(11);
    let (h2, c2) = read(12);
    assert_ne!(c1, c2, "distinct seeds must give distinct fold challenges");
    assert_ne!(h1, h2);
    let (h1b, c1b) = read(11);
    assert_eq!((h1, c1), (h1b, c1b));
}

#[test]
fn truncated_prefix_bytes_are_refused() {
    let t = sample_run();
    let h = t.prefix_at(2);
    let bytes = h.as_bytes().to_vec();
    assert_eq!(TranscriptPrefix::from_bytes(&bytes), Ok(h.clone()));

    for cut in [1usize, 2, 9, 16, bytes.len() - 1] {
        assert_eq!(
            TranscriptPrefix::from_bytes(&bytes[..cut]),
            Err(PrefixDecodeError::Truncated),
            "a prefix cut at {cut} bytes must be refused"
        );
    }
}

#[test]
fn an_unknown_entry_tag_is_refused() {
    for tag in [0u8, 4, 9, 0xFF] {
        assert_eq!(
            TranscriptPrefix::from_bytes(&[tag]),
            Err(PrefixDecodeError::BadTag(tag))
        );
    }
    for tag in [1u8, 2, 3] {
        assert_eq!(
            TranscriptPrefix::from_bytes(&[tag]),
            Err(PrefixDecodeError::Truncated)
        );
    }
}

#[test]
fn an_ordinary_length_past_the_end_is_refused() {
    let mut bytes = vec![1u8];
    bytes.extend_from_slice(&4u64.to_be_bytes());
    bytes.extend_from_slice(b"abc");
    assert_eq!(
        TranscriptPrefix::from_bytes(&bytes),
        Err(PrefixDecodeError::Truncated)
    );

    let mut huge = vec![1u8];
    huge.extend_from_slice(&u64::MAX.to_be_bytes());
    huge.extend_from_slice(b"abc");
    assert_eq!(
        TranscriptPrefix::from_bytes(&huge),
        Err(PrefixDecodeError::Truncated)
    );

    let mut tail = Vec::new();
    Entry::OraclePost(5).encode_into(&mut tail);
    tail.extend_from_slice(&bytes);
    assert_eq!(
        TranscriptPrefix::from_bytes(&tail),
        Err(PrefixDecodeError::Truncated)
    );
}

#[test]
fn the_entry_count_is_derived_by_decoding_not_declared() {
    let t = sample_run();
    for id in 0..3 {
        let h = t.prefix_at(id);
        let rebuilt = TranscriptPrefix::from_bytes(h.as_bytes()).expect("our own bytes decode");
        assert_eq!(rebuilt, h);
        assert_eq!(rebuilt.entry_count(), rebuilt.decode().unwrap().len());
        assert_eq!(rebuilt.entry_count(), h.entry_count());
    }
}
