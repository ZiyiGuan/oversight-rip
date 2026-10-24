use rmip_backend_interleaved::{Ligero, Shape};
use rmip_backend_ladder::tuple::draw_tuple;
use rmip_backend_trace::RawTrace;
use rmip_demo::fixtures;
use rmip_demo::harness::{
    amplification_curve, bounded_error_sampling_queries, measure, mnist_case, run_case,
    run_case_lazy, Case, Registry,
};
use rmip_field::F128;
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_iop::backend::{
    BackendParams, Claim, CommitmentBackend, Coord, LongOracle, QueryPlan, Symbol,
};
use rmip_iop::transcript::{OracleId, RawChallenger, SeededChallenger, TranscriptPrefix};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const SMALL_VIEWS: usize = 16;

struct AuditCountingRawTrace {
    inner: RawTrace,
    checker_queries: Arc<AtomicUsize>,
}

impl CommitmentBackend for AuditCountingRawTrace {
    type Field = F128;
    type Commitment = usize;
    type Verifier = rmip_backend_trace::RawTraceVerifier;

    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
        self.inner.commit(z, ch)
    }

    fn open(&mut self, claims: &[Claim<F128>], ch: &mut dyn RawChallenger) {
        self.inner.open(claims, ch);
    }

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
        self.inner.query_plan(ch)
    }

    fn oracles(&self) -> &[LongOracle<F128>] {
        self.inner.oracles()
    }

    fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
        self.checker_queries.fetch_add(1, Ordering::Relaxed);
        self.inner.reconstruct(id, h, coord)
    }

    fn verifier(&self) -> Self::Verifier {
        self.inner.verifier()
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        self.inner.public_params(trace_len, claims)
    }

    fn params(&self, claims: usize) -> BackendParams {
        self.inner.params(claims)
    }
}

fn q_long_of<B: CommitmentBackend<Field = F128>>(mut b: B, n_trace: usize) -> usize {
    let mut ch = SeededChallenger::new(1);
    b.commit(&rmip_iop::mle::lift(&vec![false; n_trace]), &mut ch);
    b.params(1).q_long
}

fn ladder_coord(
    b: &rmip_backend_ladder::Ladder,
    o: &rmip_mip_prover::OracleMeta,
    ch: &mut dyn RawChallenger,
) -> Coord {
    let s = b.schedule();
    let n_enc = s.shape.n_enc_at(s.table(o.id).level);
    b.coord_of_tuple(o.id, &draw_tuple(ch, n_enc, s.queries))
}

#[test]
fn rejected_iop_executions_still_send_one_checker_query_in_measurements() {
    let (case, oracle) = mnist_case(SMALL_VIEWS);
    let fixture = fixtures::trace_swap();
    let checker_queries = Arc::new(AtomicUsize::new(0));
    let make = || AuditCountingRawTrace {
        inner: RawTrace::new(),
        checker_queries: Arc::clone(&checker_queries),
    };

    let rate = measure(&case, &fixture, &oracle, 7, &make);
    assert_eq!(rate.successes, 0);
    assert_eq!(checker_queries.load(Ordering::Relaxed), 7);

    checker_queries.store(0, Ordering::Relaxed);
    let curve = amplification_curve(&case, &fixture, &oracle, 4, 5, &make);
    assert!(curve.iter().all(|(_, rate, _)| rate.successes == 0));
    assert_eq!(
        checker_queries.load(Ordering::Relaxed),
        5,
        "each rejected chain stops after, not before, its first checker exchange"
    );
}

#[test]
fn the_registry_rows_agree_with_their_definitions() {
    let (c, o) = mnist_case(SMALL_VIEWS);
    let fx = vec![fixtures::honest()];
    let (rows, reg) = run_case(&c, &o, &fx, 40, &RawTrace::new);

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].accept.successes, 40,
        "honest prover rejected in run_case"
    );

    assert_eq!(reg.k_claims, reg.k_rel + reg.t * (reg.b_max + 1));
    assert_eq!(
        reg.k_rel, 3,
        "two PIOP endpoints plus the closed-form Booleanity claim"
    );
    assert_eq!(reg.b_max, 64, "every sparse-judge query is 64 wires");

    assert_eq!((reg.eps_rob.num, reg.eps_rob.den), (7, 16));
    assert_eq!(reg.t, 4);
    assert_eq!(reg.ell, SMALL_VIEWS, "one occurrence per view");

    assert_eq!(reg.q_o, reg.r_for_sigma * reg.t);

    let expected = 1.0 - (1.0 - reg.s_iop) / reg.q_long as f64;
    assert!((reg.s_2p_1 - expected).abs() < 1e-12);
    assert!(
        (reg.s_2p_1 - (1.0 - 1.0 / reg.q_long as f64)).abs() > 1e-9,
        "s_2P,1 must not collapse to the fixture's 1 - 1/q_long"
    );

    assert_eq!(reg.eps_idx, 0.0, "the 16-view case is scan-shaped");
    let miss = rmip_iop::backend::div_error_upward(9.0, 16.0);
    let beta = rmip_iop::backend::pow_error_upward(miss, reg.t);
    let recomposed =
        rmip_iop::backend::sum_errors_upward(&[reg.eps_bind, reg.eps_rel, reg.eps_open, beta]);
    assert_eq!(reg.s_iop, recomposed);
    assert!(
        reg.s_iop > beta,
        "the positive opening term must not be rounded out of the soundness bound"
    );

    assert!(reg.t_mat.as_nanos() > 0, "T_mat was not measured");
    assert!(reg.r_for_sigma > 1);

    assert_eq!(reg.t_check.len(), 2);
    assert_eq!(reg.t_check[0].oracle, rmip_iop::driver::DUMMY_ORACLE);
    assert_eq!(reg.t_check[0].alphabet, rmip_iop::backend::Alphabet::Field);
    assert_eq!(reg.t_check[0].symbols, Coord::from(1u64));
    assert_eq!(reg.t_check[1].symbols, Coord::from(reg.n_trace));

    assert_eq!(
        reg.n_trace,
        8 * SMALL_VIEWS,
        "N = 8r at the power-of-two view counts the demonstration uses"
    );
    assert_eq!(reg.q_long, reg.n_trace);
    assert_eq!(reg.a_long, reg.n_trace as u64);
    assert_eq!(reg.l_long, Coord::from(reg.n_trace + 1));
    let accounted_l_long = reg
        .t_check
        .iter()
        .map(|row| {
            row.symbols
                .checked_mul_u64(row.alphabet.elements() as u64)
                .unwrap()
        })
        .fold(Coord::ZERO, |a, b| a.checked_add(b).unwrap());
    assert_eq!(
        reg.l_long, accounted_l_long,
        "L_long covers every posted oracle"
    );
    assert_eq!(reg.r_open, 0, "raw opening has no prover-message round");
}

#[test]
fn the_index_term_is_the_fifth_summand_of_s_iop_on_an_embedded_case() {
    use rmip_iop::backend::{div_error_upward, pow_error_upward, sum_errors_upward};

    let (c, o) = mnist_case(4096);
    let c = rmip_demo::harness::with_forced_index(c);
    let nu = c.layout.nu;
    let holo = c.holo.clone().expect("the forced index");
    let (rows, reg) = run_case(&c, &o, &[fixtures::honest()], 1, &|| {
        Ligero::new(Shape::balanced(nu, 4, 4.0))
    });
    assert_eq!(
        rows[0].accept.successes, 1,
        "honest prover rejected on the embedded case"
    );

    assert!(
        reg.eps_idx > 0.0,
        "an embedded statement must pay the index debits"
    );
    assert_eq!(
        reg.eps_idx,
        holo.index.error_terms(),
        "eps_idx must be the embedded index's error_terms"
    );

    let miss = div_error_upward(
        (reg.eps_rob.den - reg.eps_rob.num) as f64,
        reg.eps_rob.den as f64,
    );
    let beta = pow_error_upward(miss, reg.t);
    let five = sum_errors_upward(&[reg.eps_bind, reg.eps_rel, reg.eps_open, beta, reg.eps_idx]);
    assert_eq!(
        reg.s_iop, five,
        "s_IOP must be the five-term sum, bit for bit, on an embedded case"
    );

    let four = sum_errors_upward(&[reg.eps_bind, reg.eps_rel, reg.eps_open, beta]);
    assert!(
        four < reg.s_iop,
        "dropping eps_idx must lower s_IOP: {four} is not below {}",
        reg.s_iop
    );

    println!(
        "embedded {} nu={nu} | eps_idx {:.6e} | eps_open {:.6e} | s_IOP {:.6e} | R {}",
        c.family, reg.eps_idx, reg.eps_open, reg.s_iop, reg.r_for_sigma
    );
}

#[test]
fn c_short_counts_the_sampled_query_blocks() {
    let (c, o) = mnist_case(SMALL_VIEWS);
    let (_, reg) = run_case(&c, &o, &[fixtures::honest()], 8, &RawTrace::new);
    let block_bytes = reg.t * (reg.b_max + 1) * 16;
    assert_eq!(
        block_bytes,
        4 * 65 * 16,
        "the MNIST circuit sends 4 blocks of 65 field elements"
    );
    assert!(
        reg.c_short > block_bytes,
        "C_short = {} does not even cover the {block_bytes} bytes of query blocks",
        reg.c_short
    );
    assert_eq!((reg.mu, reg.nu), (7, 7), "the 16-view shape is N = M = 128");
    let (mu, nu) = (reg.mu, reg.nu);
    let rounds_bytes = (mu * 2 + nu * 2 + nu * 2) * 16;
    let endpoint_bytes = (3 + 1) * 16;
    assert_eq!(
        reg.c_short,
        block_bytes + rounds_bytes + endpoint_bytes,
        "C_short drifted from its parts"
    );
}

#[test]
fn r_iop_counts_messages_and_coins_are_reported_separately() {
    let (c, o) = mnist_case(SMALL_VIEWS);
    let (_, reg) = run_case(&c, &o, &[fixtures::honest()], 8, &RawTrace::new);
    assert!(
        reg.coins > reg.r_iop,
        "coins {} must exceed messages {}: tau alone is nu coins in one message",
        reg.coins,
        reg.r_iop
    );
    assert!(reg.r_iop > 0);
}

#[test]
fn the_registry_separates_the_backends() {
    let (c, o) = mnist_case(SMALL_VIEWS);
    let fx = vec![fixtures::honest()];
    let (_, raw) = run_case(&c, &o, &fx, 8, &RawTrace::new);
    let lig = || Ligero::new(Shape::balanced(c.layout.nu, 4, 4.0));
    let (_, code) = run_case(&c, &o, &fx, 8, &lig);

    assert_eq!(raw.backend, "raw-trace");
    assert_eq!(code.backend, "interleaved-rs");
    assert!(code.q_long < raw.q_long);
    assert!(code.r_for_sigma < raw.r_for_sigma);
    assert!(code.q_o < raw.q_o);
    assert_eq!(raw.k_claims, code.k_claims);
    assert_eq!(raw.t, code.t);
    assert_eq!(raw.ell, code.ell);
    assert_ne!(Coord::from(code.a_long), code.l_long);
    assert_ne!(code.a_long, code.q_long as u64);
    assert_eq!(raw.r_open, 0);
    assert_eq!(
        code.r_open, 1,
        "the flat code opening has one prover-message round"
    );
}

#[test]
fn the_amplification_curves_first_point_matches_the_fixture_prediction() {
    let (c, o) = mnist_case(SMALL_VIEWS);
    let nu = c.layout.nu;
    let lig = move || Ligero::new(Shape::balanced(nu, 4, 4.0));
    let q_long = q_long_of(lig(), c.layout.n_trace);
    assert_eq!(
        q_long, 6,
        "the interleaved-RS backend reads six occurrences"
    );
    let fx = fixtures::oracle_inconsistent(q_long);
    let curve = amplification_curve(&c, &fx, &o, 1, 8000, &lig);
    let (r, rate, predicted) = &curve[0];
    assert_eq!(*r, 1);
    let sigma = (predicted * (1.0 - predicted) / 8000.0).sqrt();
    let measured = rate.successes as f64 / rate.trials as f64;
    assert!(
        (measured - predicted).abs() < 4.0 * sigma,
        "amplification R=1 gave {measured:.5}, fixture prediction is {predicted:.5} \
         (4 sigma = {:.5}, {}/{})",
        4.0 * sigma,
        rate.successes,
        rate.trials
    );
}

#[test]
fn the_bounded_error_sampler_is_the_baseline_on_the_mnist_circuit() {
    let sigma = 2f64.powi(-20);
    let hoeffding = |inst: &MnistInstance| -> usize {
        let eps = inst.eps_rob.num as f64 / inst.eps_rob.den as f64;
        ((1.0 / sigma).ln() / (2.0 * eps * eps)).ceil() as usize
    };

    let mut seen = Vec::new();
    for views in [64usize, 256, 1024] {
        let inst = MnistInstance::new(views);
        let (c, o) = mnist_case(views);
        let nu = c.layout.nu;
        let shape = Shape::balanced(nu, 4, 4.0);
        let honest = [fixtures::honest()];
        let (rows_l, flat) = run_case(&c, &o, &honest, 1, &|| Ligero::new(shape));
        let ladder_shape = rmip_backend_ladder::LadderShape::for_trace(nu, 4.0);
        let (rows_b, bun) = run_case_lazy(
            &c,
            &o,
            &honest,
            1,
            &|| rmip_backend_ladder::Ladder::lazy(ladder_shape),
            &ladder_coord,
        );
        assert_eq!(
            rows_l[0].accept.successes, 1,
            "honest prover rejected (flat)"
        );
        assert_eq!(
            rows_b[0].accept.successes, 1,
            "honest prover rejected (ladder)"
        );

        let sampled = bounded_error_sampling_queries(c.inst.ell, c.eps_rob, sigma);
        assert_eq!(
            sampled,
            hoeffding(&inst),
            "{views}: the sampler disagrees with the plain Hoeffding count"
        );
        assert!(
            sampled < c.inst.ell,
            "{views}: the sampler must be a genuine sample, not full evaluation"
        );

        assert!(
            sampled < bun.q_o && bun.q_o < flat.q_o,
            "{views}: expected sampler {sampled} < ladder Q_O {} < flat Q_O {}",
            bun.q_o,
            flat.q_o
        );
        for r in [&flat, &bun] {
            assert_eq!(r.q_o, r.r_for_sigma * r.t, "Q_O must be R*t");
            assert_eq!(r.q_o_distinct_support, views, "the query support is ell");
            assert_eq!(
                r.q_o_distinct,
                r.q_o.min(views),
                "Q_O^dist is capped by Q_O"
            );
            assert!(r.q_o_distinct <= r.q_o);
        }
        println!(
            "{views:>5} views: eps_rob {}/{}  sampler {sampled}  ladder (q_long {}, R {}, Q_O {})  flat (q_long {}, R {}, Q_O {})",
            c.eps_rob.num,
            c.eps_rob.den,
            bun.q_long,
            bun.r_for_sigma,
            bun.q_o,
            flat.q_long,
            flat.r_for_sigma,
            flat.q_o
        );
        seen.push((sampled, flat.r_for_sigma, bun.r_for_sigma, flat.t));
    }

    for w in seen.windows(2) {
        assert!(
            w[0].0.abs_diff(w[1].0) <= 1,
            "the sampler must not track ell"
        );
        assert!(w[0].1.abs_diff(w[1].1) <= 1, "flat R must not track ell");
        assert!(w[0].2.abs_diff(w[1].2) <= 1, "ladder R must not track ell");
        assert_eq!(w[0].3, w[1].3, "t is set by eps_rob, not by ell");
    }

    let small = MnistInstance::new(SMALL_VIEWS);
    assert!(hoeffding(&small) > small.ell());
    assert_eq!(
        bounded_error_sampling_queries(small.ell(), small.eps_rob, sigma),
        small.ell(),
        "a sample larger than the circuit falls back to full evaluation"
    );
}

#[test]
fn bounded_error_sampler_closes_a_hoeffding_rounding_boundary() {
    use rmip_frontend::Ratio;

    let below_exp_minus_one = f64::from_bits(0x3fd7_8b56_362c_ef37);
    assert_eq!(
        bounded_error_sampling_queries(10, Ratio::new(1, 2), below_exp_minus_one),
        3
    );
    assert_eq!(bounded_error_sampling_queries(10, Ratio::new(1, 1), 0.1), 2);

    assert!(std::panic::catch_unwind(|| {
        bounded_error_sampling_queries(10, Ratio::new(1, 2), f64::NAN)
    })
    .is_err());
}

fn synthetic_case(views: usize) -> (Case, SparseTableOracle) {
    let mut predictions = vec![7u8; views - views / 4];
    predictions.extend(vec![1u8; views / 4]);
    let certified = rmip_demo::mnist_survey::certify(0, 7, predictions).expect("a supported R");
    certified
        .build_case("synthetic")
        .expect("an accepting table with a positive radius")
}

#[test]
#[ignore = "runs real N = 2^15 and 2^17 executions to anchor the synthetic harness"]
fn synthetic_check_agrees_with_a_real_execution() {
    use rmip_backend_interleaved::measure_check_at_shape;

    for (views, (c, o)) in [
        (4096usize, mnist_case(4096)),
        (16384, synthetic_case(16384)),
    ] {
        let nu = c.layout.nu;
        let (_, r) = run_case(&c, &o, &[fixtures::honest()], 1, &|| {
            Ligero::new(Shape::balanced(nu, 4, 4.0))
        });
        let real = r.t_open.as_secs_f64();
        let synthetic = measure_check_at_shape(Shape::balanced(nu, 4, 4.0));
        let ratio = synthetic / real;
        println!(
            "R={views:<7} N=2^{nu:<3} real check {:>8.2}us   synthetic {:>8.2}us   {:.2}x",
            real * 1e6,
            synthetic * 1e6,
            ratio
        );
        assert!(
            (0.5..2.0).contains(&ratio),
            "R={views}: synthetic check is {ratio:.2}x the real one. The scaling sweep in \
             `rmip-backend-interleaved` is built on these agreeing; if they do not, its \
             numbers describe fabricated data and nothing else. Band is deliberately \
             wide -- these are ~30-60us medians on a noisy laptop -- so a failure here \
             means a structural difference, not drift."
        );
    }
}

#[test]
#[ignore = "builds synthetic cases to N = 2^21; minutes and some memory"]
fn historical_partial_verifier_scaling() {
    println!(
        "{:<10} {:>10} {:>12} {:>13} {:>9} {:>11} {:>12} {:>9}",
        "views", "N", "compute", "verifier x R", "ratio", "T_exec", "P_check", "proxy x R"
    );

    let mut seen: Vec<(usize, f64)> = Vec::new();
    for views in [16384usize, 65536, 262144] {
        let (c, o) = synthetic_case(views);
        let nu = c.layout.nu;

        let budget = std::time::Duration::from_millis(250);
        let t0 = std::time::Instant::now();
        let mut k: u64 = 0;
        while t0.elapsed() < budget {
            std::hint::black_box(c.circuit.eval(c.inst.x.as_slice(), &o));
            k += 1;
        }
        let compute = t0.elapsed().as_secs_f64() / k as f64;

        let (_, r) = run_case(&c, &o, &[fixtures::honest()], 1, &|| {
            Ligero::new(Shape::balanced(nu, 4, 4.0))
        });
        let verify = (r.t_verify + r.t_mat).as_secs_f64() * r.r_for_sigma as f64;
        let ratio = verify / compute;
        let t_check = r
            .t_check
            .iter()
            .map(|c| c.mean)
            .max()
            .unwrap_or_default()
            .as_secs_f64();
        let combined_with_checker = (r.t_exec.as_secs_f64() + t_check) * r.r_for_sigma as f64;
        println!(
            "{:<10} {:>10} {:>10.1}us {:>11.2}ms {:>8.2}x {:>9.1}ms {:>8.1}us {:>9.1}s",
            views,
            r.n_trace,
            compute * 1e6,
            verify * 1e3,
            ratio,
            r.t_exec.as_secs_f64() * 1e3,
            t_check * 1e6,
            combined_with_checker
        );
        seen.push((r.n_trace, ratio));
    }

    for (n, ratio) in seen {
        assert!(
            ratio.is_finite() && ratio > 0.0,
            "invalid partial ratio at N={n}"
        );
    }
}

#[test]
#[ignore = "builds cases up to N = 2^17; run with --ignored"]
fn how_much_of_the_prover_is_coin_independent() {
    use rmip_backend_interleaved::code;

    println!(
        "{:<20} {:>9} {:>7} {:>11} {:>11} {:>7} {:>13} {:>10}",
        "workload", "N", "R", "T_exec", "encode", "share", "R*T_exec", "amortised"
    );

    fn probe(label: &str, c: &Case, r: &Registry) {
        let shape = Shape::balanced(c.layout.nu, 4, 4.0);
        let (m, n, n_enc) = (shape.m(), shape.n(), shape.n_enc);
        let rows: Vec<Vec<F128>> = (0..m)
            .map(|i| (0..n).map(|j| F128::new((i * n + j) as u128)).collect())
            .collect();
        let dom = code::domain(n_enc);
        let budget = std::time::Duration::from_millis(200);
        let t0 = std::time::Instant::now();
        let mut k: u64 = 0;
        while t0.elapsed() < budget {
            std::hint::black_box(rmip_opt::encode::encode_columns(&rows, &dom));
            k += 1;
        }
        let enc = t0.elapsed().as_secs_f64() / k as f64;
        let tm = r.t_exec.as_secs_f64();
        let total = tm * r.r_for_sigma as f64;
        let amort = enc + (tm - enc).max(0.0) * r.r_for_sigma as f64;
        println!("{:<20} {:>9} {:>7} {:>10.2}ms {:>10.2}ms {:>6.1}% {:>11.2}s {:>8.2}s  (m={} n={} n_enc={})",
            label, r.n_trace, r.r_for_sigma, tm * 1e3, enc * 1e3,
            enc / tm * 100.0, total, amort, m, n, n_enc);
    }

    for (views, (c, o)) in [
        (1024usize, mnist_case(1024)),
        (4096, mnist_case(4096)),
        (16384, synthetic_case(16384)),
    ] {
        let nu = c.layout.nu;
        let (_, r) = run_case(&c, &o, &[fixtures::honest()], 1, &|| {
            Ligero::new(Shape::balanced(nu, 4, 4.0))
        });
        probe(&format!("MNIST R={views}"), &c, &r);
    }
}

#[test]
#[ignore = "runs the prover at N up to 2^17; run with --ignored"]
fn proving_verifying_and_computing() {
    #[allow(clippy::type_complexity)]
    type ExecRow = (
        String,
        usize,
        f64,
        f64,
        usize,
        usize,
        Option<Option<f64>>,
        Option<Option<f64>>,
        f64,
        f64,
    );

    println!("PER EXECUTION -- computing C^O, partial verifier, combined proxy");
    println!(
        "{:<20} {:>9} {:>10} {:>10} {:>10}   {:>8} {:>9} {:>9}",
        "workload", "N", "compute", "verifier", "T_exec", "compute", "verifier", "T_exec"
    );
    println!(
        "{:<20} {:>9} {:>10} {:>10} {:>10}   {:>8} {:>9} {:>9}",
        "", "", "C^O", "1 exec", "1 exec", "= 1x", "vs comp", "vs comp"
    );

    println!();
    println!("VERIFIER  (measured components; current implementation charges T_mat)");
    println!(
        "{:<20} {:>9} {:>10} {:>10} {:>8} {:>10} {:>8} {:>10} {:>5}",
        "workload",
        "N",
        "compute",
        "1 exec",
        "t_verify",
        "+ T_mat",
        "vs comp",
        "current xR",
        "r_IOP"
    );

    fn measure_row(
        label: &str,
        c: &Case,
        o: &SparseTableOracle,
        baseline_calls: usize,
    ) -> (f64, ExecRow) {
        let budget = std::time::Duration::from_millis(250);
        let t0 = std::time::Instant::now();
        let mut n: u64 = 0;
        while t0.elapsed() < budget {
            std::hint::black_box(c.circuit.eval(c.inst.x.as_slice(), o));
            n += 1;
        }
        let compute = t0.elapsed().as_secs_f64() / n as f64;
        let nu = c.layout.nu;
        let (_, r) = run_case(c, o, &[fixtures::honest()], 1, &|| {
            Ligero::new(Shape::balanced(nu, 4, 4.0))
        });
        let amplified_exec = r.t_exec.as_secs_f64() * r.r_for_sigma as f64;
        let verify_one = (r.t_verify + r.t_mat).as_secs_f64();
        let verify = verify_one * r.r_for_sigma as f64;

        let tau = |work: f64| {
            Some(if baseline_calls <= r.q_o {
                None
            } else {
                Some((work - compute) / (baseline_calls - r.q_o) as f64)
            })
        };
        println!(
            "{:<20} {:>9} {:>8.1}us {:>8.2}us {:>8.2}us {:>7.2}us {:>7.2}x {:>8.2}ms {:>5}",
            label,
            r.n_trace,
            compute * 1e6,
            verify_one * 1e6,
            r.t_verify.as_secs_f64() * 1e6,
            r.t_mat.as_secs_f64() * 1e6,
            verify_one / compute,
            verify * 1e3,
            r.r_iop,
        );
        (
            amplified_exec / compute,
            (
                label.to_string(),
                r.n_trace,
                r.t_exec.as_secs_f64(),
                amplified_exec,
                r.r_for_sigma,
                c.layout.nu,
                None,
                tau(verify),
                compute,
                verify_one,
            ),
        )
    }

    let mut ratios: Vec<(String, f64)> = Vec::new();
    let mut exec_rows: Vec<ExecRow> = Vec::new();

    for (views, (c, o)) in [
        (256usize, mnist_case(256)),
        (1024, mnist_case(1024)),
        (4096, mnist_case(4096)),
        (16384, synthetic_case(16384)),
    ] {
        let label = format!("MNIST R={views}");
        let sampled = bounded_error_sampling_queries(c.inst.ell, c.eps_rob, 2f64.powi(-20));
        let (v, row) = measure_row(&label, &c, &o, sampled);
        ratios.push((label, v));
        exec_rows.push(row);
    }

    println!("\nEXECUTION PROXY  (codeword = coin-independent part of commit)");
    println!(
        "{:<20} {:>9} {:>10} {:>10} {:>7} {:>10} {:>11} {:>7}",
        "workload", "N", "T_exec", "codeword", "share", "R*T_exec", "amortised", "saving"
    );
    for (label, n_trace, tm, total, rr, nu, _, _, _, _) in &exec_rows {
        let shape = Shape::balanced(*nu, 4, 4.0);
        let (m, n, n_enc) = (shape.m(), shape.n(), shape.n_enc);
        let rows: Vec<Vec<F128>> = (0..m)
            .map(|i| (0..n).map(|j| F128::new((i * n + j) as u128)).collect())
            .collect();
        let dom = rmip_backend_interleaved::code::domain(n_enc);
        let budget = std::time::Duration::from_millis(200);
        let t0 = std::time::Instant::now();
        let mut k: u64 = 0;
        while t0.elapsed() < budget {
            std::hint::black_box(rmip_opt::encode::encode_columns(&rows, &dom));
            k += 1;
        }
        let enc = t0.elapsed().as_secs_f64() / k as f64;
        let amort = enc + (tm - enc).max(0.0) * *rr as f64;
        println!(
            "{:<20} {:>9} {:>8.1}ms {:>8.1}ms {:>6.1}% {:>9.1}s {:>10.1}s {:>6.1}x",
            label,
            n_trace,
            tm * 1e3,
            enc * 1e3,
            enc / tm * 100.0,
            total,
            amort,
            total / amort
        );
    }

    println!("\nPER EXECUTION -- computing C^O, partial verifier, combined proxy");
    println!(
        "{:<20} {:>9} {:>10} {:>10} {:>10}   {:>8} {:>10} {:>10}",
        "workload", "N", "compute", "verifier", "T_exec", "compute", "verifier", "T_exec"
    );
    for (label, n_trace, tm, _, _, _, _, _, compute, tv) in &exec_rows {
        println!(
            "{:<20} {:>9} {:>8.1}us {:>8.2}us {:>8.2}ms   {:>8} {:>9.2}x {:>9.0}x",
            label,
            n_trace,
            compute * 1e6,
            tv * 1e6,
            tm * 1e3,
            "1x",
            tv / compute,
            tm / compute
        );
    }

    println!("\nBREAK-EVEN per oracle call -- the VERIFIER's decision;");
    println!("the prover's time does not enter it, the prover being untrusted.");
    println!(
        "{:<20} {:>14} {:>18}",
        "workload", "indexed (absent)", "current partial"
    );
    for (label, _, _, _, _, _, indexed, current, _, _) in &exec_rows {
        let f = |b: &Option<Option<f64>>| match b {
            None => "n/a".to_string(),
            Some(None) => "never".to_string(),
            Some(Some(t)) => format!("{:.1} us", t * 1e6),
        };
        println!("{:<20} {:>14} {:>18}", label, f(indexed), f(current));
    }

    for (label, ratio) in &ratios {
        assert!(
            *ratio >= 100.0,
            "{label}: combined execution measured only {ratio:.0}x the cost of computing C^O. \
             Either the execution got dramatically faster or the evaluation baseline \
             is not evaluating the circuit -- check the latter first. The documents \
             state this cost openly and must be corrected if it changed."
        );
    }
}

#[test]
fn an_l_long_above_2_64_prints_exactly_and_json_value_refuses_it() {
    let big: u128 = (1u128 << 66) * 384 + 1;
    assert!(big > u64::MAX as u128);
    assert_eq!(format!("{big}"), "28334198897217871282177");
    assert_eq!(
        serde_json::to_string(&big).expect("serde_json writes a u128 directly"),
        "28334198897217871282177"
    );
    assert!(
        serde_json::to_value(big).is_err(),
        "a serde_json::Value cannot hold a u128 above u64::MAX without arbitrary_precision"
    );
}
