use std::cell::Cell;

use rmip_field::{OpeningField, F128, F64};
use rmip_frontend::bits::BitSlice;
use rmip_frontend::oracle::Oracle;
use rmip_iop::backend::{AuditMode, CommitmentBackend};
use rmip_iop::driver::{Config, Reject};
use rmip_iop::transcript::OsChallenger;
use rmip_mip::PCheck;
use rmip_mip_prover::{execute_recording_packed, LazyBackend};
use rmip_packed::driver::{packed_witness, PackedCheat};
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

fn fixture(
    r: usize,
) -> (
    rmip_demo::harness::Case,
    rmip_frontend::mnist::SparseTableOracle,
    usize,
) {
    let mut predictions = vec![7u8; r - r * 6 / 100];
    predictions.extend(vec![1u8; r * 6 / 100]);
    let certified = rmip_demo::mnist_survey::certify(0, 7, predictions).expect("certify");
    let threshold = certified.certificate.threshold;
    let (case, oracle) = certified.build_case("synthetic").expect("build the case");
    (case, oracle, threshold)
}

fn run<F: PackedField, B: LazyBackend<Field = F128>>(
    case: &rmip_demo::harness::Case,
    oracle: &rmip_frontend::mnist::SparseTableOracle,
    threshold: usize,
    make: &dyn Fn() -> B,
    cheat: &PackedCheat,
) -> (Result<(), Reject>, bool, usize, usize) {
    let st = PackedStatement::new(&case.inst, &case.circuit, threshold as u64, true);
    let cfg = Config::for_instance(&case.inst, 2f64.powi(-21));
    let (_, _, words) = packed_witness::<F, _>(&st, oracle, &PackedCheat::default());
    let mut checker_backend = make();
    checker_backend.commit(&F::to_words(&words), &mut OsChallenger::new());
    let checker = PCheck::new(checker_backend);
    let audit_mode = checker.backend().params(1).audit;
    let counting = Counting {
        inner: oracle,
        calls: Cell::new(0),
    };
    let mut main = make();
    let mut ch = OsChallenger::new();
    let (execution, verdict) =
        execute_recording_packed::<F, _, _>(&st, &mut main, &counting, &cfg, cheat, &mut ch);
    let audited = match audit_mode {
        AuditMode::OneOf => {
            let mut coins = OsChallenger::new();
            rmip_mip::checker::audit_with_answer(&execution, &checker, &mut coins)
                .0
                .accepted()
        }
        AuditMode::Sequential => rmip_mip::audit_all_with_answers(&execution, &checker)
            .0
            .accepted(),
    };
    assert_eq!(execution.oracle_calls, cfg.t);
    (verdict, audited, counting.calls.get(), execution.q_long())
}

#[test]
fn the_lazy_packed_port_runs_on_the_ladder_at_both_fields() {
    let (case, oracle, threshold) = fixture(4096);
    let cfg = Config::for_instance(&case.inst, 2f64.powi(-21));
    let ell = case.inst.ell;
    for bits in [64usize, 128] {
        let st = PackedStatement::new(&case.inst, &case.circuit, threshold as u64, true);
        let (_, total_log) = st.tree.layout(bits.trailing_zeros() as usize);
        let mut ls = rmip_backend_ladder::LadderShape::for_trace(total_log, 23.0);
        ls.pins = if bits == 64 { 8 } else { 4 };
        let (verdict, audited, calls, q) = if bits == 64 {
            let ladder = || rmip_backend_ladder::Ladder64::lazy(ls);
            run::<F64, _>(&case, &oracle, threshold, &ladder, &PackedCheat::default())
        } else {
            let ladder = || rmip_backend_ladder::Ladder::lazy(ls);
            run::<F128, _>(&case, &oracle, threshold, &ladder, &PackedCheat::default())
        };
        assert_eq!(verdict, Ok(()), "ladder at {bits} bits");
        assert!(audited, "the sequential audit accepts");
        assert_eq!(calls, ell + cfg.t);
        assert_eq!(q, ls.schedule().tables.len(), "one occurrence per table");
    }
}

#[test]
fn the_lazy_packed_port_rejects_a_wrong_label() {
    let (case, oracle, threshold) = fixture(4096);
    let st = PackedStatement::new(&case.inst, &case.circuit, threshold as u64, true);
    let (_, total_log) = st.tree.layout(6);
    let mut ls = rmip_backend_ladder::LadderShape::for_trace(total_log, 23.0);
    ls.pins = 8;
    let ladder = || rmip_backend_ladder::Ladder64::lazy(ls);
    let cheat = PackedCheat {
        public_input_bits: vec![0],
        ..Default::default()
    };
    let (verdict, _, calls, q) = run::<F64, _>(&case, &oracle, threshold, &ladder, &cheat);
    assert!(
        matches!(verdict, Err(Reject::OracleMismatch { .. })),
        "{verdict:?}"
    );
    assert_eq!(
        calls,
        case.inst.ell + Config::for_instance(&case.inst, 2f64.powi(-21)).t
    );
    assert_eq!(
        q,
        ls.schedule().tables.len(),
        "the query phase still runs on a rejection"
    );
}

fn measured_shape(st: &PackedStatement<'_>) -> rmip_backend_ladder::LadderShape {
    let (_, total_log) = st.tree.layout(6);
    let shape = rmip_backend_ladder::SingleShotTarget { bits: 40 }
        .ladder_shape(total_log, Some(0), 64)
        .expect("the measured shape");
    assert_eq!(shape.depth(), 0);
    shape
}

fn run_measured(
    case: &rmip_demo::harness::Case,
    oracle: &rmip_frontend::mnist::SparseTableOracle,
    threshold: usize,
    cheat: &PackedCheat,
    checker_cheat: &PackedCheat,
) -> (Result<(), Reject>, bool, usize, usize) {
    let st = PackedStatement::new(&case.inst, &case.circuit, threshold as u64, true);
    let shape = measured_shape(&st);
    let cfg = Config::for_instance(&case.inst, 2f64.powi(-41));
    let (_, _, words) = packed_witness::<F64, _>(&st, oracle, checker_cheat);
    let mut checker_backend = rmip_backend_ladder::Ladder64::lazy(shape);
    checker_backend.commit(&F64::to_words(&words), &mut OsChallenger::new());
    let checker = PCheck::new(checker_backend);
    assert_eq!(checker.backend().params(1).audit, AuditMode::Sequential);
    let counting = Counting {
        inner: oracle,
        calls: Cell::new(0),
    };
    let mut main = rmip_backend_ladder::Ladder64::lazy(shape);
    let mut ch = OsChallenger::new();
    let (execution, verdict) =
        execute_recording_packed::<F64, _, _>(&st, &mut main, &counting, &cfg, cheat, &mut ch);
    let audited = rmip_mip::audit_all_with_answers(&execution, &checker)
        .0
        .accepted();
    assert_eq!(execution.oracle_calls, cfg.t);
    (verdict, audited, counting.calls.get(), execution.q_long())
}

#[test]
fn the_measured_configuration_accepts_the_honest_provers() {
    let (case, oracle, threshold) = fixture(4096);
    let cfg = Config::for_instance(&case.inst, 2f64.powi(-41));
    let (verdict, audited, calls, q) = run_measured(
        &case,
        &oracle,
        threshold,
        &PackedCheat::default(),
        &PackedCheat::default(),
    );
    assert_eq!(verdict, Ok(()));
    assert!(
        audited,
        "the sequential audit accepts the honest answering role"
    );
    assert_eq!(calls, case.inst.ell + cfg.t);
    assert_eq!(
        q, 1,
        "depth 0 has one table, so one query to the answering role"
    );
}

#[test]
fn the_measured_configuration_rejects_a_wrong_label_at_the_oracle() {
    let (case, oracle, threshold) = fixture(4096);
    let cheat = PackedCheat {
        public_input_bits: vec![0],
        ..Default::default()
    };
    let (verdict, _, _, q) =
        run_measured(&case, &oracle, threshold, &cheat, &PackedCheat::default());
    assert!(
        matches!(verdict, Err(Reject::OracleMismatch { .. })),
        "{verdict:?}"
    );
    assert_eq!(q, 1, "the query phase still runs on a rejection");
}

#[test]
fn the_measured_configuration_rejects_a_corrupted_committed_word() {
    let (case, oracle, threshold) = fixture(4096);
    for word in [0usize, 7, 40] {
        let cheat = PackedCheat {
            corrupt_word: Some((word, 0x5555_5555_5555_5555)),
            ..Default::default()
        };
        let (verdict, _, _, _) =
            run_measured(&case, &oracle, threshold, &cheat, &PackedCheat::default());
        assert!(verdict.is_err(), "word {word}: {verdict:?}");
        assert!(
            !matches!(verdict, Err(Reject::OracleMismatch { .. })),
            "word {word}: rejected by the relation or the opening, not the oracle check: {verdict:?}"
        );
    }
}

#[test]
fn the_measured_configuration_rejects_an_answering_role_that_disagrees() {
    let (case, oracle, threshold) = fixture(4096);
    let checker_cheat = PackedCheat {
        corrupt_word: Some((3, 0xdead_beef_dead_beef)),
        ..Default::default()
    };
    let (verdict, audited, _, _) = run_measured(
        &case,
        &oracle,
        threshold,
        &PackedCheat::default(),
        &checker_cheat,
    );
    assert_eq!(verdict, Ok(()), "the main role alone is honest");
    assert!(!audited, "the audit rejects the disagreeing answering role");
}
