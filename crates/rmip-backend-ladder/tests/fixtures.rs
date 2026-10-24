use rmip_backend_ladder::{require_unpadded, Ladder, LadderShape};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{AuditMode, CommitmentBackend};
use rmip_iop::driver::{Cheat, Config, Statement};
use rmip_iop::transcript::SeededChallenger;

fn run_views(views: usize, lambda_bits: f64, seeds: u64) {
    let (c, oracle) = rmip_demo::harness::mnist_case(views);
    let st = Statement {
        inst: &c.inst,
        sys: &c.sys,
        layout: &c.layout,
        circuit: &c.circuit,
        dc: &c.dc,
        holo: None,
    };
    let cfg = Config {
        t: c.t,
        ..Config::default()
    };
    require_unpadded(&cfg);
    let shape = LadderShape::for_trace(c.layout.nu, lambda_bits);
    let checker = {
        let mut b = Ladder::lazy(shape);
        b.commit(&c.honest_trace, &mut SeededChallenger::new(0xC4EC));
        rmip_mip::PCheck::new(b)
    };
    let q_long = shape.schedule().oracle_count();
    for seed in 0..seeds {
        let mut backend = Ladder::lazy(shape);
        let (mut exec, verdict) = rmip_mip_prover::execute_recording(
            &st,
            &mut backend,
            &oracle,
            &cfg,
            &Cheat::default(),
            &mut SeededChallenger::new(0x5EED + seed),
        );
        assert!(
            verdict.is_ok(),
            "honest ladder execution at {views} views: {verdict:?}"
        );
        assert_eq!(exec.q_long(), q_long, "one occurrence per table");
        assert_eq!(backend.params(1).audit, AuditMode::Sequential);
        let honest = rmip_mip::audit_all(&exec, &checker);
        assert!(honest.accepted(), "honest sequential audit: {honest:?}");
        let (_, answers) = rmip_mip::audit_all_with_answers(&exec, &checker);
        assert_eq!(answers.len(), q_long);
        for (k, occ) in exec.occurrences.iter().enumerate() {
            assert_eq!(answers[k], occ.answer, "table {} agrees", occ.oracle);
        }
        let victim = (seed as usize * 7 + 3) % q_long;
        let keep = exec.occurrences[victim].answer.clone();
        exec.occurrences[victim].answer.0[0] = exec.occurrences[victim].answer.0[0].add(F128::ONE);
        let caught = rmip_mip::audit_all(&exec, &checker);
        assert!(
            matches!(caught, rmip_mip::Audit::Inconsistent { audited } if audited == victim),
            "a corrupted answer at occurrence {victim} is audited as inconsistent there: {caught:?}"
        );
        exec.occurrences[victim].answer = keep;
        assert!(rmip_mip::audit_all(&exec, &checker).accepted());
    }
}

#[test]
fn the_mip_path_accepts_honest_and_the_sequential_audit_catches_any_corruption() {
    run_views(16, 4.0, 3);
    run_views(1024, 4.0, 2);
}

#[test]
fn the_demonstration_shapes_are_as_documented() {
    for (nu, depth, oracles) in [(11usize, 2usize, 6usize), (13, 4, 12), (23, 14, 42)] {
        let s = LadderShape::for_trace(nu, 4.0).schedule();
        assert_eq!(s.depth, depth, "nu = {nu}");
        assert_eq!(s.oracle_count(), oracles, "nu = {nu}");
        assert_eq!(s.queries, 6);
        let reads: usize = s.tables.iter().map(|t| s.queries * t.rows.len()).sum();
        assert!(
            reads <= 192 + 120 * depth,
            "nu = {nu}: {reads} symbols read"
        );
    }
}
