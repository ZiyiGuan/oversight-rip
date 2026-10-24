use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::MnistInstance;
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, trace, Instance};
use rmip_iop::driver::Statement;
use rmip_iop::lincheck::Coefficients;
use rmip_iop::mle::{eval_mle, lift};
use rmip_iop::relation_protocol::{
    self, HonestRelationProver, RelationError, RelationMessageProver,
};
use rmip_iop::transcript::{Entry, OsChallenger, RawChallenger, SeededChallenger};

const VIEWS: usize = 24;

struct Fixture {
    inst: Instance,
    sys: r1cs::R1cs,
    layout: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    z: Vec<F128>,
}

fn fixture() -> Fixture {
    let mnist = MnistInstance::new(VIEWS);
    let circuit = mnist.build();
    let oracle = mnist.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let layout = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &layout, &R1csConfig::default());
    let locations = loc::build(&sites, &layout);
    let x = mnist.yes();
    assert!(circuit.eval(x.as_slice(), &oracle));
    let (_, answers) = circuit.eval_with_answers(x.as_slice(), &oracle);
    let z = lift(trace::witness_trace(&dc, &layout, x.as_slice(), answers.as_slice()).as_slice());
    let inst = Instance {
        layout: layout.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(locations),
        ell: mnist.ell(),
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: mnist.eps_rob,
    };
    Fixture {
        inst,
        sys,
        layout,
        circuit,
        dc,
        z,
    }
}

fn statement(f: &Fixture) -> Statement<'_> {
    Statement {
        inst: &f.inst,
        sys: &f.sys,
        layout: &f.layout,
        circuit: &f.circuit,
        dc: &f.dc,
        holo: None,
    }
}

fn coins(ch: &SeededChallenger) -> usize {
    ch.transcript()
        .entries()
        .iter()
        .filter(|e| matches!(e, Entry::Coin(_)))
        .count()
}

fn ordinary_messages(ch: &SeededChallenger) -> usize {
    ch.transcript()
        .entries()
        .iter()
        .filter(|entry| matches!(entry, Entry::Ordinary(_)))
        .count()
}

fn schedule_shape(ch: &SeededChallenger) -> Vec<&'static str> {
    ch.transcript()
        .entries()
        .iter()
        .map(|entry| match entry {
            Entry::Coin(_) => "coin",
            Entry::Ordinary(_) => "message",
            Entry::OraclePost(_) => "post",
        })
        .collect()
}

#[test]
fn verifier_accepts_an_honest_proof_without_receiving_the_trace() {
    let f = fixture();
    let mut prover = SeededChallenger::new(0x5151);
    let proof = relation_protocol::prove(&statement(&f), &f.z, &mut prover);

    let mut verifier = SeededChallenger::new(0x5151);
    let out = relation_protocol::verify(&statement(&f), &proof, &mut verifier);
    assert_eq!(out.verdict, Ok(()));
    assert_eq!(out.claims.len(), 3);
    assert!(out
        .claims
        .iter()
        .all(|c| eval_mle(&f.z, &c.point) == c.value));
    assert_eq!(
        prover.transcript().entries(),
        verifier.transcript().entries()
    );
}

#[test]
fn arbitrary_round_and_endpoint_messages_are_rejected() {
    let f = fixture();
    let mut prover = SeededChallenger::new(7);
    let proof = relation_protocol::prove(&statement(&f), &f.z, &mut prover);

    let mut bad_round = proof.clone();
    bad_round.zerocheck.round_polys[0][0] = bad_round.zerocheck.round_polys[0][0].add(F128::ONE);
    let mut verifier = SeededChallenger::new(7);
    assert_eq!(
        relation_protocol::verify(&statement(&f), &bad_round, &mut verifier).verdict,
        Err(RelationError::ZerocheckEndpoint)
    );

    let mut bad_endpoint = proof;
    bad_endpoint.lincheck_endpoint[0] = bad_endpoint.lincheck_endpoint[0].add(F128::ONE);
    let mut verifier = SeededChallenger::new(7);
    assert_eq!(
        relation_protocol::verify(&statement(&f), &bad_endpoint, &mut verifier).verdict,
        Err(RelationError::LincheckEndpoint)
    );
}

#[test]
fn malformed_proofs_still_consume_the_complete_relation_coin_schedule() {
    let f = fixture();
    let mut prover = SeededChallenger::new(9);
    let proof = relation_protocol::prove(&statement(&f), &f.z, &mut prover);

    let mut malformed = proof.clone();
    malformed.zerocheck.round_polys.clear();
    malformed.lincheck.round_polys[0].truncate(1);
    malformed.zerocheck_endpoint.clear();
    malformed.lincheck_endpoint.push(F128::ONE);

    let mut good_ch = SeededChallenger::new(9);
    let good = relation_protocol::verify(&statement(&f), &proof, &mut good_ch);
    assert_eq!(good.verdict, Ok(()));
    let mut bad_ch = SeededChallenger::new(9);
    let bad = relation_protocol::verify(&statement(&f), &malformed, &mut bad_ch);
    assert!(bad.verdict.is_err());
    assert_eq!(
        schedule_shape(&bad_ch),
        schedule_shape(&good_ch),
        "rejection must not reorder or reshape the verifier schedule"
    );
    assert_eq!(
        coins(&bad_ch),
        coins(&good_ch),
        "rejection must not shorten the verifier schedule"
    );
    assert_eq!(
        ordinary_messages(&bad_ch),
        ordinary_messages(&good_ch),
        "empty malformed messages must retain their transcript frames"
    );
    let shape = schedule_shape(&good_ch);
    assert!(shape.contains(&"coin") && shape.contains(&"message"));
}

#[test]
fn live_relation_verifier_owns_the_only_coin_source() {
    let f = fixture();
    let st = statement(&f);
    let mut prover = HonestRelationProver::new(&st, &f.z);
    let mut verifier = OsChallenger::new();
    let (_proof, out) = relation_protocol::verify_interactive(&st, &mut prover, &mut verifier);
    assert_eq!(out.verdict, Ok(()));
    assert!(out
        .claims
        .iter()
        .all(|claim| eval_mle(&f.z, &claim.point) == claim.value));
}

struct ArbitraryMessages {
    public_input_coins: usize,
    zerocheck_challenges: usize,
    lincheck_challenges: usize,
}

impl RelationMessageProver for ArbitraryMessages {
    fn holo_frame(&mut self) -> Vec<F128> {
        unreachable!("these test provers run scan-shaped statements")
    }

    fn observe_public_input_point(&mut self, r_pub: &[F128]) {
        self.public_input_coins = r_pub.len();
    }

    fn begin_zerocheck(&mut self, _tau: &[F128]) {}

    fn zerocheck_round_message(&mut self, round: usize) -> Vec<F128> {
        vec![F128::new(round as u128); 4]
    }

    fn receive_zerocheck_challenge(&mut self, _round: usize, _challenge: F128) {
        self.zerocheck_challenges += 1;
    }

    fn zerocheck_endpoint(&mut self) -> Vec<F128> {
        vec![F128::ZERO; 3]
    }

    fn begin_lincheck(&mut self, _row_point: &[F128], _coefficients: Coefficients) {}

    fn lincheck_round_message(&mut self, round: usize) -> Vec<F128> {
        vec![F128::new((round + 1) as u128); 3]
    }

    fn receive_lincheck_challenge(&mut self, _round: usize, _challenge: F128) {
        self.lincheck_challenges += 1;
    }

    fn lincheck_endpoint(&mut self) -> Vec<F128> {
        Vec::new()
    }
}

#[test]
fn live_relation_verifier_rejects_arbitrary_messages_without_shortening_schedule() {
    let f = fixture();
    let st = statement(&f);
    let mut prover = ArbitraryMessages {
        public_input_coins: 0,
        zerocheck_challenges: 0,
        lincheck_challenges: 0,
    };
    let mut verifier = SeededChallenger::new(0xA11CE);
    let (_proof, out) = relation_protocol::verify_interactive(&st, &mut prover, &mut verifier);
    assert!(out.verdict.is_err());
    assert_ne!(
        f.sys.mu, f.layout.nu,
        "the fixture must separate the row width from the coordinate width"
    );
    assert_eq!(prover.public_input_coins, f.layout.k);
    assert_eq!(prover.zerocheck_challenges, f.sys.mu);
    assert_eq!(prover.lincheck_challenges, f.layout.nu);
}

#[test]
fn the_padding_claim_is_the_verifier_computed_public_input_claim() {
    let f = fixture();
    let st = statement(&f);

    let mut prover_ch = SeededChallenger::new(0x9001);
    let produced = relation_protocol::prove_with_claims(&st, &f.z, &mut prover_ch);

    let mut verifier_ch = SeededChallenger::new(0x9001);
    let out = relation_protocol::verify(&st, &produced.proof, &mut verifier_ch);
    assert_eq!(out.verdict, Ok(()));

    let r_pub: Vec<F128> = verifier_ch
        .transcript()
        .entries()
        .iter()
        .filter_map(|e| match e {
            Entry::Coin(c) => Some(*c),
            _ => None,
        })
        .take(f.layout.k)
        .collect();
    assert_eq!(
        r_pub.len(),
        f.layout.k,
        "the fixture must draw public-input coins"
    );

    let block = trace::public_block(&f.layout, f.inst.x.as_slice());
    let expected = eval_mle(&lift(block.as_slice()), &r_pub);
    assert_eq!(
        out.public_input_claim.value, expected,
        "the labelled claim must be the one the verifier computes from inst.x"
    );
    assert_eq!(out.public_input_claim, produced.public_input_claim);

    assert_eq!(
        eval_mle(&f.z, &out.public_input_claim.point),
        out.public_input_claim.value
    );
    assert_eq!(out.public_input_claim, out.claims[1]);
    assert_eq!(produced.public_input_claim, produced.claims[1]);
}
