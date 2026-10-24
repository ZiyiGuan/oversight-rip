use rmip_backend_interleaved::{Ligero, Shape};
use rmip_backend_recursive::{FoldLadder, Shape as LadderShape};
use rmip_backend_trace::RawTrace;
use rmip_demo::fixtures;
use rmip_demo::harness::{measure, measure_all, mnist_case, run_case, Case, Row};
use rmip_field::F128;
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::mle;
use rmip_iop::transcript::SeededChallenger;

const DETERMINISTIC_TRIALS: usize = 240;

const LOCAL_CHEAT_TRIALS: usize = 4_000;

fn demo_rows_for<B: CommitmentBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    local: &Case,
    lc: &fixtures::Fixture,
    oracle: &SparseTableOracle,
    make: &F,
) -> (Row, Row) {
    let (rows, reg) = run_case(
        c,
        oracle,
        &[fixtures::public_input_swap()],
        DETERMINISTIC_TRIALS,
        make,
    );
    let lc_rows = measure_all(
        local,
        oracle,
        std::slice::from_ref(lc),
        LOCAL_CHEAT_TRIALS,
        reg.backend,
        make,
    );
    (rows[0].clone(), lc_rows[0].clone())
}

fn five_sigma(p: f64, n: usize) -> f64 {
    5.0 * (p * (1.0 - p) / n as f64).sqrt()
}

fn closed_form_local_cheat(
    views: usize,
    local: &Case,
    oracle: &SparseTableOracle,
) -> fixtures::Fixture {
    let inst = MnistInstance::new(views);
    fixtures::local_cheat_majority(
        &inst,
        &local.dc,
        &local.sites,
        &local.inst.x,
        local.t,
        oracle,
    )
}

#[test]
fn the_closed_form_local_cheat_matches_its_own_prediction_at_sixteen_views() {
    let views = 16;
    let (c, oracle) = mnist_case(views);
    let local = c.clone_for_no_instance();
    let lc = closed_form_local_cheat(views, &local, &oracle);

    let inst = MnistInstance::new(views);
    assert_eq!(inst.minimal_bad_set_size(), 9, "T - S_no at 16 views");
    let predicted = lc.prediction.accept_rate;
    let exact = (1.0 - 9.0 / 16.0f64).powi(4);
    assert_eq!(
        local.t, 4,
        "the 16-view case draws four local-validity samples"
    );
    assert!(
        (predicted - exact).abs() < 1e-12,
        "the minimal bad set must be 9 of ell = 16: predicted {predicted} vs {exact}"
    );
    let bound = (1.0 - 7.0 / 16.0f64).powi(4);
    assert_eq!((c.eps_rob.num, c.eps_rob.den), (7, 16));
    assert!(
        (predicted - bound).abs() > five_sigma(predicted, LOCAL_CHEAT_TRIALS),
        "the exact rate and the bound must be separable at this trial count"
    );

    let nu = c.layout.nu;
    let raw = RawTrace::new;
    let lig = || Ligero::new(Shape::balanced(nu, 4, 4.0));
    let lad = || FoldLadder::new(LadderShape::balanced(nu, 4, 4.0));

    let rows = vec![
        demo_rows_for(&c, &local, &lc, &oracle, &raw),
        demo_rows_for(&c, &local, &lc, &oracle, &lig),
        demo_rows_for(&c, &local, &lc, &oracle, &lad),
    ];

    let band = five_sigma(predicted, LOCAL_CHEAT_TRIALS);
    println!(
        "MNIST at {views} views, predicted LocalCheat rate {predicted:.5}, 5 sigma = {band:.5}"
    );
    for (swap, cheat) in &rows {
        println!(
            "  {:<16} PublicInputSwap {}   LocalCheat {}",
            swap.backend, swap.accept, cheat.accept
        );
    }
    for (swap, cheat) in &rows {
        assert_eq!(
            swap.predicted, 0.0,
            "{}: PublicInputSwap must predict deterministic rejection",
            swap.backend
        );
        assert_eq!(
            swap.accept.successes, 0,
            "{}: PublicInputSwap accepted {}/{}",
            swap.backend, swap.accept.successes, swap.accept.trials
        );
        let measured = cheat.accept.successes as f64 / cheat.accept.trials as f64;
        assert!(
            (measured - predicted).abs() < band,
            "{}: LocalCheat measured {measured:.5} vs predicted {predicted:.5} (5 sigma = {band:.5})",
            cheat.backend
        );
    }
}

#[test]
#[ignore = "1,024-view execution cost; run with --ignored"]
fn the_public_input_swap_row_is_rejected_at_1024_views_too() {
    let views = 1024;
    let (c, oracle) = mnist_case(views);
    let nu = c.layout.nu;
    let trials = 40;

    let raw = RawTrace::new;
    let lig = || Ligero::new(Shape::balanced(nu, 4, 4.0));
    let lad = || FoldLadder::new(LadderShape::balanced(nu, 4, 4.0));
    let fx = [fixtures::public_input_swap()];
    let (r_raw, _) = run_case(&c, &oracle, &fx, trials, &raw);
    let (r_lig, _) = run_case(&c, &oracle, &fx, trials, &lig);
    let (r_lad, _) = run_case(&c, &oracle, &fx, trials, &lad);

    let local = c.clone_for_no_instance();
    let lc = closed_form_local_cheat(views, &local, &oracle);
    let inst = MnistInstance::new(views);
    println!(
        "MNIST at {views} views: LocalCheat predicts {:.5} with |Bad| = {}",
        lc.prediction.accept_rate,
        inst.minimal_bad_set_size()
    );
    for rows in [&r_raw, &r_lig, &r_lad] {
        let r = &rows[0];
        println!("  {:<16} PublicInputSwap {}", r.backend, r.accept);
        assert_eq!(
            r.accept.successes, 0,
            "{}: PublicInputSwap accepted at {views} views",
            r.backend
        );
    }
}

#[test]
#[ignore = "measurement"]
fn the_fold_ladder_audit_row_is_at_one_minus_one_over_q_long() {
    const BLOCKS: usize = 5;
    const PER_BLOCK: usize = 40_000;

    let (c, oracle) = mnist_case(16);
    let lad = || FoldLadder::new(LadderShape::balanced(c.layout.nu, 4, 4.0));
    let q = {
        let mut b = lad();
        let mut ch = SeededChallenger::new(1);
        b.commit(&mle::lift(&vec![false; c.layout.n_trace]), &mut ch);
        b.params(1).q_long
    };
    assert_eq!(q, 12, "the 16-view fold ladder reads 12 occurrences");

    let fx = fixtures::oracle_inconsistent(q);
    let predicted = fx.prediction.accept_rate;

    let mut accepted = 0usize;
    let mut total = 0usize;
    println!("MNIST 16 views / fold-ladder / OracleInconsistent, predicted {predicted:.5}");
    for b in 0..BLOCKS {
        let rate = measure(&c, &fx, &oracle, PER_BLOCK, &lad);
        let p = rate.successes as f64 / rate.trials as f64;
        let z = (p - predicted) / (predicted * (1.0 - predicted) / rate.trials as f64).sqrt();
        println!(
            "  block {b}: {}/{} = {p:.5}   z = {z:+.2}",
            rate.successes, rate.trials
        );
        accepted += rate.successes;
        total += rate.trials;
    }
    let pooled = accepted as f64 / total as f64;
    let z = (pooled - predicted) / (predicted * (1.0 - predicted) / total as f64).sqrt();
    println!("  pooled: {accepted}/{total} = {pooled:.5}   z = {z:+.2}");
    assert!(
        z.abs() < 5.0,
        "pooled {pooled:.5} vs predicted {predicted:.5} is z = {z:+.2}"
    );
}
