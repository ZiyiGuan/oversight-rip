use rmip_field::{BinaryField, F128};
use rmip_frontend::SparseF2;
use rmip_holo::Embedded;
use rmip_iop::driver::{execute_recording, Cheat, Config, Reject, Statement};
use rmip_iop::mateval::{HoloMatEval, MatEvalFault};
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use std::sync::OnceLock;

const VIEWS: usize = 4096;
const SEED: u64 = 11;

type Case = rmip_demo::harness::Case;
type TestOracle = rmip_frontend::mnist::SparseTableOracle;

#[derive(Clone, Copy, Debug)]
enum Attack {
    Honest,
    WrongValue(usize),
    CorruptFrame(usize),
    DropFrame(usize),
    DropLast,
    SwapAdjacent(usize),
    TrailingFrame,
    NoFrames,
    ExtraProverCoin,
}

struct Dishonest<'a> {
    inner: &'a Embedded,
    attack: Attack,
}

impl HoloMatEval for Dishonest<'_> {
    fn prove_frames(
        &self,
        a: &SparseF2,
        b: &SparseF2,
        c: &SparseF2,
        r_row: &[F128],
        r_col: &[F128],
        ch: &mut dyn RawChallenger,
    ) -> ([F128; 3], Vec<Vec<F128>>) {
        let (mut t, mut frames) = self.inner.prove_frames(a, b, c, r_row, r_col, ch);
        match self.attack {
            Attack::Honest => {}
            Attack::WrongValue(m) => t[m] = t[m].add(F128::ONE),
            Attack::CorruptFrame(i) => {
                let f = &mut frames[i];
                assert!(!f.is_empty(), "frame {i} must have an element to flip");
                f[0] = f[0].add(F128::ONE);
            }
            Attack::DropFrame(i) => {
                frames.remove(i);
            }
            Attack::DropLast => {
                frames.pop();
            }
            Attack::SwapAdjacent(i) => frames.swap(i, i + 1),
            Attack::TrailingFrame => frames.push(vec![F128::ONE]),
            Attack::NoFrames => frames.clear(),
            Attack::ExtraProverCoin => {
                let _ = ch.sample();
            }
        }
        (t, frames)
    }

    fn verify_frames(
        &self,
        next: &mut dyn FnMut() -> Vec<F128>,
        r_row: &[F128],
        r_col: &[F128],
        ch: &mut dyn RawChallenger,
    ) -> Result<[F128; 3], MatEvalFault> {
        self.inner.verify_frames(next, r_row, r_col, ch)
    }
}

fn case() -> &'static (Case, TestOracle) {
    static CASE: OnceLock<(Case, TestOracle)> = OnceLock::new();
    CASE.get_or_init(|| {
        let (c, o) = rmip_demo::harness::mnist_case(VIEWS);
        let c = rmip_demo::harness::with_forced_index(c);
        assert!(c.holo.is_some(), "the forced index");
        (c, o)
    })
}

struct Row {
    verdict: Result<(), Reject>,
    occurrences: usize,
    coins: usize,
}

fn run(c: &Case, o: &TestOracle, attack: Attack) -> Row {
    let h = Dishonest {
        inner: c.holo.as_deref().expect("embedded by the shipped rule"),
        attack,
    };
    let st = Statement {
        inst: &c.inst,
        sys: &c.sys,
        layout: &c.layout,
        circuit: &c.circuit,
        dc: &c.dc,
        holo: Some(&h),
    };
    let mut backend = rmip_backend_interleaved::Ligero::new(
        rmip_backend_interleaved::Shape::balanced(c.layout.nu, 4, 4.0),
    );
    let mut ch = SeededChallenger::new(SEED);
    let (e, verdict) = execute_recording(
        &st,
        &mut backend,
        o,
        &Config {
            t: c.t,
            ..Config::default()
        },
        &Cheat::default(),
        &mut ch,
    );
    Row {
        verdict,
        occurrences: e.occurrences.len(),
        coins: e.coins,
    }
}

#[test]
fn the_driver_rejects_a_dishonest_holographic_evaluation() {
    let (c, o) = case();
    let index = &c.holo.as_deref().expect("embedded").index;
    let zeta = index.zeta;
    let frames = 3 * (zeta + 2) + index.v_pre + 1;

    let honest = run(c, o, Attack::Honest);
    println!("honest: {:?}", honest.verdict);
    assert_eq!(
        honest.verdict,
        Ok(()),
        "the honest evaluator must be accepted, or every row below is vacuous"
    );
    assert!(
        honest.occurrences > 0,
        "the audit surface must be nonempty, or invariant 2 is a statement about zero"
    );

    let attacks = [
        ("wrong_value(0)", Attack::WrongValue(0)),
        ("wrong_value(2)", Attack::WrongValue(2)),
        ("corrupt_frame(T_A)", Attack::CorruptFrame(0)),
        ("corrupt_frame(sumcheck)", Attack::CorruptFrame(1)),
        ("corrupt_frame(endpoint)", Attack::CorruptFrame(zeta + 1)),
        (
            "corrupt_frame(batching)",
            Attack::CorruptFrame(3 * (zeta + 2)),
        ),
        (
            "corrupt_frame(opening row)",
            Attack::CorruptFrame(frames - 1),
        ),
        ("drop_frame(mid)", Attack::DropFrame(frames / 2)),
        ("drop_last", Attack::DropLast),
        ("swap_adjacent(0)", Attack::SwapAdjacent(0)),
        ("trailing_frame", Attack::TrailingFrame),
        ("no_frames", Attack::NoFrames),
        ("extra_prover_coin", Attack::ExtraProverCoin),
    ];

    for (name, attack) in attacks {
        let row = run(c, o, attack);
        println!("{name:<28}: {:?}", row.verdict);
        assert_eq!(
            row.verdict,
            Err(Reject::HolographicFrames),
            "{name}: a dishonest holographic evaluation must be rejected at step 4, \
             not by a later identity that happened to notice"
        );
        assert_eq!(
            row.occurrences, honest.occurrences,
            "{name}: the occurrence count must not depend on the verdict"
        );
        let expected_coins = match attack {
            Attack::ExtraProverCoin => honest.coins + 1,
            _ => honest.coins,
        };
        assert_eq!(
            row.coins, expected_coins,
            "{name}: the public coin count moved"
        );
    }
}
