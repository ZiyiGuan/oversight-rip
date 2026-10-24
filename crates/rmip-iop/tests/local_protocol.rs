use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::oracle::CountingOracle;
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, trace, Instance};
use rmip_iop::local_protocol::{self, HonestLocalProver, LocalError, LocalMessageProver};
use rmip_iop::mle::{eval_mle, lift};
use rmip_iop::transcript::{Entry, RawChallenger, SeededChallenger};

const VIEWS: usize = 16;

fn fixture() -> (Instance, Vec<F128>, SparseTableOracle) {
    let mnist = MnistInstance::new(VIEWS);
    let circuit = mnist.build();
    let oracle = mnist.oracle();
    let (ordinary, sites) = deoracle::deoraclize(&circuit);
    let layout = layout::TraceLayout::new(&ordinary);
    let _system = r1cs::emit(&ordinary, &layout, &R1csConfig::default());
    let locations = loc::build(&sites, &layout);
    let x = mnist.yes();
    assert!(
        circuit.eval(x.as_slice(), &oracle),
        "the true label must be accepted"
    );
    let (_, answers) = circuit.eval_with_answers(x.as_slice(), &oracle);
    let table =
        lift(trace::witness_trace(&ordinary, &layout, x.as_slice(), answers.as_slice()).as_slice());
    let instance = Instance {
        layout,
        loc: rmip_frontend::loc::Loc::from_rows(locations),
        ell: mnist.ell(),
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: mnist.eps_rob,
    };
    (instance, table, oracle)
}

#[test]
fn local_verifier_never_receives_the_trace() {
    let (instance, table, oracle) = fixture();
    let mut prover = SeededChallenger::new(17);
    let proof = local_protocol::prove(&instance, &table, 5, &mut prover);
    let mut verifier = SeededChallenger::new(17);
    let out = local_protocol::verify(&instance, &oracle, 5, &proof, &mut verifier);
    assert_eq!(out.verdict, Ok(()));
    assert_eq!(out.oracle_calls, 5);
    assert!(out
        .claims
        .iter()
        .all(|claim| eval_mle(&table, &claim.point) == claim.value));
    assert_eq!(
        prover.transcript().entries(),
        verifier.transcript().entries()
    );
}

#[test]
fn each_answer_block_is_one_whole_transcript_entry() {
    let (instance, table, oracle) = fixture();
    let t = 5;
    let mut prover = SeededChallenger::new(19);
    let proof = local_protocol::prove(&instance, &table, t, &mut prover);
    let mut verifier = SeededChallenger::new(19);
    let checked = local_protocol::verify(&instance, &oracle, t, &proof, &mut verifier);
    assert_eq!(checked.verdict, Ok(()));
    assert_eq!(
        prover.transcript().entries(),
        verifier.transcript().entries()
    );
    let entries = prover.transcript().entries();

    let coin_entries = entries
        .iter()
        .take_while(|entry| matches!(entry, Entry::Coin(_)))
        .count();
    assert!(
        coin_entries >= t,
        "exact rejection sampling consumes at least one coin per occurrence"
    );
    assert_eq!(
        entries.len(),
        coin_entries + t,
        "a variable coin prefix must be followed by exactly t answer blocks"
    );
    assert!(entries[..coin_entries]
        .iter()
        .all(|entry| matches!(entry, Entry::Coin(_))));
    for (entry, block) in entries[coin_entries..].iter().zip(&proof.blocks) {
        assert_eq!(block.len(), instance.b_max() + 1);
        let mut expected = Vec::with_capacity(block.len() * 16);
        for value in block {
            expected.extend_from_slice(&value.to_be_bytes());
        }
        assert_eq!(
            entry,
            &Entry::Ordinary(expected),
            "a whole answer block must be framed as one ordinary message"
        );
    }
}

#[test]
fn non_boolean_and_malformed_blocks_still_make_exactly_t_oracle_calls() {
    let (instance, table, table_oracle) = fixture();
    let t = 4;
    let mut prover = SeededChallenger::new(23);
    let proof = local_protocol::prove(&instance, &table, t, &mut prover);

    let mut non_boolean = proof.clone();
    non_boolean.blocks[0][0] = F128::new(2);
    let oracle = CountingOracle::new(&table_oracle);
    let mut verifier = SeededChallenger::new(23);
    let checked = local_protocol::verify(&instance, &oracle, t, &non_boolean, &mut verifier);
    assert_eq!(
        checked.verdict,
        Err(LocalError::NonBooleanBlock { occurrence: 0 })
    );
    assert_eq!(checked.oracle_calls, t);
    assert_eq!(
        oracle.occurrences(),
        t,
        "the real oracle, not only the report, must see t calls"
    );

    let mut wrong_width = proof.clone();
    wrong_width.blocks[1].pop();
    let oracle = CountingOracle::new(&table_oracle);
    let mut verifier = SeededChallenger::new(23);
    let checked = local_protocol::verify(&instance, &oracle, t, &wrong_width, &mut verifier);
    assert!(matches!(
        checked.verdict,
        Err(LocalError::WrongBlockWidth { occurrence: 1, .. })
    ));
    assert_eq!(checked.oracle_calls, t);
    assert_eq!(oracle.occurrences(), t);

    let mut missing = proof.clone();
    missing.blocks.pop();
    let oracle = CountingOracle::new(&table_oracle);
    let mut verifier = SeededChallenger::new(23);
    let checked = local_protocol::verify(&instance, &oracle, t, &missing, &mut verifier);
    assert!(matches!(
        checked.verdict,
        Err(LocalError::WrongBlockCount { .. })
    ));
    assert_eq!(checked.oracle_calls, t);
    assert_eq!(oracle.occurrences(), t);
    assert_eq!(
        verifier
            .transcript()
            .entries()
            .iter()
            .filter(|entry| matches!(entry, Entry::Ordinary(_)))
            .count(),
        t,
        "a missing block gets one empty continuation frame without shortening the schedule"
    );

    let mut surplus = proof.clone();
    surplus.blocks.push(vec![F128::ONE]);
    let oracle = CountingOracle::new(&table_oracle);
    let mut verifier = SeededChallenger::new(23);
    let checked = local_protocol::verify(&instance, &oracle, t, &surplus, &mut verifier);
    assert!(matches!(
        checked.verdict,
        Err(LocalError::WrongBlockCount { .. })
    ));
    assert_eq!(checked.oracle_calls, t);
    assert_eq!(oracle.occurrences(), t);
    assert_eq!(
        verifier
            .transcript()
            .entries()
            .iter()
            .filter(|entry| matches!(entry, Entry::Ordinary(_)))
            .count(),
        t,
        "surplus blocks are rejected without adding prover-selected message positions"
    );
}

struct ObservingLocalProver<'a> {
    inner: HonestLocalProver<'a>,
    sampling_coins: Vec<F128>,
}

impl LocalMessageProver for ObservingLocalProver<'_> {
    fn answer_blocks(&mut self, sampling_coins: &[F128], sampled: &[usize]) -> Vec<Vec<F128>> {
        self.sampling_coins = sampling_coins.to_vec();
        self.inner.answer_blocks(sampling_coins, sampled)
    }
}

#[test]
fn live_local_prover_observes_every_public_sampling_coin() {
    let (instance, table, oracle) = fixture();
    let mut prover = ObservingLocalProver {
        inner: HonestLocalProver::new(&instance, &table),
        sampling_coins: Vec::new(),
    };
    let mut verifier = SeededChallenger::new(0xC01A);
    let (_proof, checked) =
        local_protocol::verify_interactive(&instance, &oracle, 5, &mut prover, &mut verifier);
    assert_eq!(checked.verdict, Ok(()));
    let transcript_coins: Vec<F128> = verifier
        .transcript()
        .entries()
        .iter()
        .filter_map(|entry| match entry {
            Entry::Coin(coin) => Some(*coin),
            _ => None,
        })
        .collect();
    assert_eq!(prover.sampling_coins, transcript_coins);
    assert!(prover.sampling_coins.len() >= 5);
}
