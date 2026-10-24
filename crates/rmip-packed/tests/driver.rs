use std::cell::Cell;

use rmip_backend_interleaved::{LigeroOf, Shape};
use rmip_field::{F128, F64};
use rmip_frontend::bits::BitSlice;
use rmip_frontend::oracle::Oracle;
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::driver::{Config, Reject};
use rmip_iop::transcript::SeededChallenger;
use rmip_packed::driver::{execute_recording_packed, PackedCheat};
use rmip_packed::statement::PackedStatement;
use rmip_packed::PackedField;

struct Counting<'a, O: Oracle> {
    inner: &'a O,
    calls: Cell<usize>,
}

impl<O: Oracle> Oracle for Counting<'_, O> {
    fn query(&self, u: &BitSlice) -> bool {
        self.calls.set(self.calls.get() + 1);
        self.inner.query(u)
    }
}

struct Fixture {
    case: rmip_demo::harness::Case,
    oracle: rmip_frontend::mnist::SparseTableOracle,
    threshold: usize,
}

fn fixture(r: usize) -> Fixture {
    let mut predictions = vec![7u8; r - r * 6 / 100];
    predictions.extend(vec![1u8; r * 6 / 100]);
    let certified = rmip_demo::mnist_survey::certify(0, 7, predictions).expect("certify");
    let threshold = certified.certificate.threshold;
    let (case, oracle) = certified.build_case("synthetic").expect("build the case");
    Fixture {
        case,
        oracle,
        threshold,
    }
}

fn label_valid(x: &BitSlice) -> bool {
    let y: usize = (0..4).map(|k| usize::from(x[k]) << k).sum();
    y < 10
}

fn run<F: PackedField>(
    f: &Fixture,
    threshold: u64,
    valid: bool,
    cheat: &PackedCheat,
    seed: u64,
) -> (rmip_iop::driver::Execution, Result<(), Reject>, usize) {
    let st = PackedStatement::new(&f.case.inst, &f.case.circuit, threshold, valid);
    let (_, total_log) = st.tree.layout(F::BITS.trailing_zeros() as usize);
    let shape = Shape::balanced(total_log, 4, 4.0);
    let mut backend = LigeroOf::<F>::new(shape);
    let cfg = Config::for_instance(&f.case.inst, 0.125);
    let q_long = backend.public_params(1 << total_log, 1).q_long;
    let counting = Counting {
        inner: &f.oracle,
        calls: Cell::new(0),
    };
    let mut ch = SeededChallenger::new(seed);
    let (execution, verdict) =
        execute_recording_packed::<F, _, _>(&st, &mut backend, &counting, &cfg, cheat, &mut ch);
    assert_eq!(
        execution.q_long(),
        q_long,
        "the query phase always runs to q_long"
    );
    (execution, verdict, counting.calls.get())
}

#[test]
fn the_packed_driver_accepts_the_honest_table() {
    let f = fixture(4096);
    let valid = label_valid(f.case.inst.x.as_slice());
    assert!(valid);
    let cfg = Config::for_instance(&f.case.inst, 0.125);
    for seed in [1u64, 2] {
        let (e, verdict, calls) =
            run::<F64>(&f, f.threshold as u64, valid, &PackedCheat::default(), seed);
        assert_eq!(verdict, Ok(()), "GF(2^64), seed {seed}");
        assert_eq!(e.oracle_calls, cfg.t, "t verifier calls");
        assert_eq!(calls, f.case.inst.ell + cfg.t, "ell prover calls then t");
        assert_eq!(e.claims_submitted, 1);
        assert_eq!(e.pad_claims, 0);
        println!(
            "GF(2^64) seed {seed}: rounds {}, coins {}, short bytes {}, t_verify {:.2} ms",
            e.rounds,
            e.coins,
            e.short_bytes,
            e.t_verify.as_secs_f64() * 1e3
        );
    }
    let (e, verdict, calls) =
        run::<F128>(&f, f.threshold as u64, valid, &PackedCheat::default(), 3);
    assert_eq!(verdict, Ok(()), "GF(2^128)");
    assert_eq!(calls, f.case.inst.ell + cfg.t);
    assert!(e.iop_accepted());
}

#[test]
fn the_packed_driver_rejects_cheats() {
    let f = fixture(4096);
    let valid = label_valid(f.case.inst.x.as_slice());
    let cfg = Config::for_instance(&f.case.inst, 0.125);
    let (e, verdict, calls) = run::<F64>(
        &f,
        f.threshold as u64,
        valid,
        &PackedCheat {
            public_input_bits: vec![0],
            ..Default::default()
        },
        4,
    );
    assert_eq!(verdict, Err(Reject::OracleMismatch { occurrence: 0 }));
    assert_eq!(calls, f.case.inst.ell + cfg.t, "every call is still made");
    let _ = e;
    let (e, verdict, _) = run::<F64>(
        &f,
        f.case.inst.ell as u64 + 1,
        valid,
        &PackedCheat::default(),
        5,
    );
    assert!(
        matches!(verdict, Err(Reject::BelowThreshold { .. })),
        "{verdict:?}"
    );
    let _ = e;
    let (_, verdict, _) = run::<F64>(&f, f.threshold as u64, false, &PackedCheat::default(), 6);
    assert_eq!(verdict, Err(Reject::InvalidPublicInput));
    let (e, verdict, _) = run::<F64>(
        &f,
        f.threshold as u64,
        valid,
        &PackedCheat {
            corrupt_word: Some((7, 0xDEAD_BEEF)),
            ..Default::default()
        },
        7,
    );
    assert!(
        matches!(
            verdict,
            Err(Reject::PackedRelation(_)) | Err(Reject::OpeningMismatch { .. })
        ),
        "{verdict:?}"
    );
    let _ = e;
}
