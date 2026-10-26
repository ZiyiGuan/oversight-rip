use rmip_backend_ladder::{LadderOf, SingleShotTarget};
use rmip_field::{OpeningField, F128, F64};
use rmip_iop::backend::{CommitmentBackend, OpeningVerifier};
use rmip_iop::transcript::SeededChallenger;
use rmip_packed::majority::{combined_words, prove, verify_piop, MajorityProof};
use rmip_packed::tree::Tree;
use rmip_packed::PackedField;

const VIEWS: usize = 256;
const THRESHOLD: u64 = 129;
const SEED: u64 = 731;

fn honest() -> (Tree, Vec<rmip_packed::tree::LevelWitness<F64>>) {
    let tree = Tree::new(VIEWS, 6, 6);
    let answers: Vec<bool> = (0..VIEWS).map(|i| i % 8 != 0).collect();
    let witness = tree.witness::<F64>(&answers);
    (tree, witness)
}

fn measured_shape(tree: &Tree) -> rmip_backend_ladder::LadderShape {
    let (_, total_log) = tree.layout(6);
    let shape = SingleShotTarget { bits: 40 }
        .rewinding_shape(total_log, 64)
        .expect("the measured shape");
    assert_eq!(shape.depth(), 0, "the measured ladder has depth 0");
    assert_eq!(shape.pins, 0, "the rewinding compilation pins nothing");
    assert_eq!(shape.queries(), 64, "the measured opening reads 64 columns");
    shape
}

fn accepted_after(change: usize) -> bool {
    let (tree, witness) = honest();
    let shape = measured_shape(&tree);
    let tables = F64::skip_tables();
    let sampled = vec![0, 1, 63, 64, 127, 255];
    let mut backend = LadderOf::<F64>::lazy(shape);
    let mut prover_coins = SeededChallenger::new(SEED);
    let mut p: MajorityProof<F64> = prove(
        &mut backend,
        tables,
        &tree,
        &witness,
        &sampled,
        &mut prover_coins,
    );
    match change {
        0 => {}
        1 => p.piop.sampled_words.clear(),
        2 => p.piop.levels.clear(),
        3 => p.piop.levels[0].zerocheck.round1.clear(),
        4 => p.piop.levels[0].zerocheck.s_hat_c.clear(),
        5 => p.piop.levels[0].zerocheck.rounds.clear(),
        6 => p.piop.levels[0].lincheck.z_vec.clear(),
        7 => p.piop.levels[1].switches.clear(),
        8 => p.piop.levels[1].switches[0].s_hat.clear(),
        9 => p.piop.levels[0].batch.rounds.clear(),
        10 => p.piop.cross.rounds.clear(),
        11 => p.answers.clear(),
        12 => p.answers[0].2 .0.clear(),
        13 => p.answers[0].0 = 99,
        14 => p.piop.sampled_words[0] += F64::ONE,
        15 => p.piop.levels[0].zerocheck.round1[0] += F64::ONE,
        16 => p.piop.levels[0].zerocheck.s_hat_c[0] += F64::ONE,
        17 => p.piop.levels[0].zerocheck.rounds[0].0 += F64::ONE,
        18 => p.piop.levels[0].zerocheck.final_a += F64::ONE,
        19 => p.piop.levels[0].zerocheck.final_b += F64::ONE,
        20 => p.piop.levels[0].lincheck.z_vec[0] += F64::ONE,
        21 => p.piop.levels[1].switches[0].s_hat[0] += F64::ONE,
        22 => p.piop.levels[0].batch.rounds[0].0 += F64::ONE,
        23 => p.piop.levels[0].batch.rounds[0].1 += F64::ONE,
        24 => p.piop.levels[0].batch.rounds[0].2 += F64::ONE,
        25 => p.piop.levels[0].batch.final_value += F64::ONE,
        26 => p.piop.cross.final_value += F64::ONE,
        27 => p.answers[0].2 .0[0] += F128::ONE,
        28 => p.answers[0].2 .0[0] = F128::new(1 << 64),
        _ => unreachable!(),
    }
    let mut independent = LadderOf::<F64>::lazy(shape);
    let mut verifier_coins = SeededChallenger::new(SEED);
    independent.commit(
        &F64::to_words(&combined_words(&tree, &witness)),
        &mut verifier_coins,
    );
    match verify_piop(
        tables,
        &tree,
        THRESHOLD,
        &sampled,
        &p.piop,
        &mut verifier_coins,
    ) {
        Err(_) => false,
        Ok(claim) => {
            if claim != p.piop.batched {
                return false;
            }
            independent.open(std::slice::from_ref(&claim), &mut verifier_coins);
            let _ = independent.query_plan(&mut verifier_coins);
            independent.verifier().check(&[claim], &p.answers)
        }
    }
}

#[test]
fn the_measured_shape_accepts_the_honest_proof() {
    assert!(accepted_after(0));
}

#[test]
fn the_measured_shape_rejects_every_corrupted_proof_without_panicking() {
    for change in 1..=28 {
        let outcome = std::panic::catch_unwind(|| accepted_after(change));
        assert!(outcome.is_ok(), "corruption {change} panicked the verifier");
        assert!(!outcome.unwrap(), "corruption {change} was accepted");
    }
}
