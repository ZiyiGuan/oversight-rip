use rmip_backend_interleaved::{Ligero, Shape};
use rmip_demo::harness::{mnist_case, Case};
use rmip_frontend::oracle::Oracle;
use rmip_iop::backend::{CommitmentBackend, Coord};
use rmip_iop::driver::{Cheat, Config, Statement};
use rmip_iop::transcript::{OsChallenger, RawChallenger, SeededChallenger};
use rmip_mip_prover::{execute_recording as mip_execute, Eager};
use std::time::{Duration, Instant};

fn st(c: &Case) -> Statement<'_> {
    Statement {
        inst: &c.inst,
        sys: &c.sys,
        layout: &c.layout,
        circuit: &c.circuit,
        dc: &c.dc,
        holo: None,
    }
}

fn cfg(c: &Case) -> Config {
    Config {
        t: c.t,
        ..Config::default()
    }
}

const CLOCK_SLACK: f64 = 1.05;

const ARTEFACT_CEILING: f64 = 1.30;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

struct Measured {
    eager: Duration,
    lazy: Duration,
    commit_eager: Duration,
    commit_lazy: Duration,
    r: usize,
    q_long: usize,
    n_enc: usize,
    rate_inv: usize,
}

fn measure<O: Oracle>(c: &Case, oracle: &O, reps: usize) -> Measured {
    let shape = Shape::balanced(c.layout.nu, 4, 4.0);
    let k = 3 + c.t * (c.inst.b_max() + 1);

    let params = {
        let mut b = Ligero::new(shape);
        b.commit(&c.honest_trace, &mut SeededChallenger::new(1));
        b.params(k)
    };
    let eps_rel = rmip_iop::backend::div_error_upward(
        4.0 * c.sys.mu as f64 + 2.0 * c.layout.nu as f64 + 1.0,
        2f64.powi(128),
    );
    let miss = rmip_iop::backend::div_error_upward(
        (c.eps_rob.den - c.eps_rob.num) as f64,
        c.eps_rob.den as f64,
    );
    let beta = rmip_iop::backend::pow_error_upward(miss, c.t);
    let s_iop = rmip_iop::backend::sum_errors_upward(&[
        params.eps_bind,
        params.eps_open,
        eps_rel.max(beta),
    ]);
    let config = cfg(c);
    let q_long = config.audit_query_count(params.q_long);
    let r = rmip_mip::repetitions_for(2f64.powi(-20), s_iop, q_long);

    let mut eager = Vec::new();
    let mut lazy = Vec::new();
    let mut ce = Vec::new();
    let mut cl = Vec::new();
    for i in 0..=reps {
        let mut b = Ligero::new(shape);
        let mut ch = SeededChallenger::new(1);
        let t0 = Instant::now();
        b.commit(&c.honest_trace, &mut ch);
        let a = t0.elapsed();
        let mut b = Ligero::lazy(shape);
        let mut ch = SeededChallenger::new(1);
        let t0 = Instant::now();
        b.commit(&c.honest_trace, &mut ch);
        let d = t0.elapsed();
        if i > 0 {
            ce.push(a);
            cl.push(d);
        }

        let mut b = Eager(Ligero::new(shape));
        let mut ch = OsChallenger::default();
        let t0 = Instant::now();
        let _ = mip_execute(&st(c), &mut b, oracle, &cfg(c), &Cheat::default(), &mut ch);
        let e = t0.elapsed();

        let mut b = Ligero::lazy(shape);
        let mut ch = OsChallenger::default();
        let t0 = Instant::now();
        let _ = mip_execute(&st(c), &mut b, oracle, &cfg(c), &Cheat::default(), &mut ch);
        let l = t0.elapsed();

        if i > 0 {
            eager.push(e);
            lazy.push(l);
        }
    }

    Measured {
        eager: median(eager),
        lazy: median(lazy),
        commit_eager: median(ce),
        commit_lazy: median(cl),
        r,
        q_long,
        n_enc: shape.n_enc,
        rate_inv: shape.n_enc / shape.n(),
    }
}

fn report(name: &str, m: &Measured) {
    let ratio = m.eager.as_secs_f64() / m.lazy.as_secs_f64();
    let predicted = m.rate_inv as f64 * (m.n_enc as f64).log2() / m.q_long as f64;
    println!(
        "{name:<12} commit {:>10.3?} -> {:>10.3?} {:>6.1}x    T_exec {:>10.3?} -> {:>10.3?} \
         {ratio:>5.2}x    R*T_exec {:>10.3?} -> {:>10.3?}",
        m.commit_eager,
        m.commit_lazy,
        m.commit_eager.as_secs_f64() / m.commit_lazy.as_secs_f64(),
        m.eager,
        m.lazy,
        m.eager * m.r as u32,
        m.lazy * m.r as u32,
    );
    println!(
        "{:<12} encode/answer bound {predicted:.1}x, q_long {}, n_enc {}, R {}",
        "", m.q_long, m.n_enc, m.r
    );
}

#[test]
#[ignore = "measurement"]
fn the_mip_prover_against_the_iop_prover() {
    println!(
        "\nLazy MIP execution against materialising IOP execution, \
         backend interleaved-rs\n"
    );
    let (small, small_oracle) = mnist_case(256);
    let ms = measure(&small, &small_oracle, 15);
    report(small.family, &ms);
    let (large, large_oracle) = mnist_case(1024);
    let ml = measure(&large, &large_oracle, 15);
    report(large.family, &ml);

    for (name, m) in [(small.family, &ms), (large.family, &ml)] {
        let (eager, lazy) = (m.eager.as_secs_f64(), m.lazy.as_secs_f64());
        let removed = m.commit_eager.as_secs_f64() - m.commit_lazy.as_secs_f64();
        assert!(
            removed > 0.0,
            "{name}: the lazy commit is not cheaper than the eager one \
             ({removed:.9} s), so there is no removed component to bound with"
        );
        let ratio = eager / lazy;
        let bound = eager / (eager - removed);
        assert!(
            ratio < bound * CLOCK_SLACK,
            "{name}: {ratio:.3}x exceeds {:.3}x -- the eager execution is \
             {eager:.9} s, the component laziness removes is {removed:.9} s of \
             it, and {CLOCK_SLACK} is the clock's margin. A saving this size is \
             not explained by the one component that was removed",
            bound * CLOCK_SLACK
        );
        assert!(
            ratio < ARTEFACT_CEILING,
            "{name}: {ratio:.3}x is at or above the {ARTEFACT_CEILING}x \
             artefact ceiling -- re-measure on a quiet machine, with \
             --test-threads=1, before believing it"
        );
    }
    println!();
}

#[test]
#[ignore = "measurement"]
fn the_saving_grows_with_the_circuit() {
    println!("\nmaterialise-all against answer-q, by trace size\n");
    println!(
        "{:>4} {:>9} {:>7} {:>13} {:>13} {:>7} {:>9}",
        "nu", "n_enc", "q", "eager", "lazy", "ratio", "predicted"
    );
    for nu in [12usize, 14, 16, 18, 20] {
        let shape = Shape::balanced(nu, 4, 4.0);
        let z = vec![rmip_field::F128::ZERO; 1usize << nu];
        let q = shape.queries();

        let reps = if nu >= 18 { 4 } else { 10 };
        let mut eager = Vec::new();
        let mut lazy = Vec::new();
        for rep in 0..=reps {
            let mut b = Ligero::new(shape);
            let t0 = Instant::now();
            b.commit(&z, &mut SeededChallenger::new(1));
            let e = t0.elapsed();

            let mut b = Ligero::lazy(shape);
            let mut ch = SeededChallenger::new(1);
            b.commit(&z, &mut ch);
            let h = ch
                .transcript()
                .prefix_at(rmip_backend_interleaved::CODEWORD_ORACLE);
            let t0 = Instant::now();
            for k in 0..q {
                std::hint::black_box(b.reconstruct(
                    rmip_backend_interleaved::CODEWORD_ORACLE,
                    &h,
                    Coord::from(k * (shape.n_enc / q)),
                ));
            }
            let l = t0.elapsed();

            if rep > 0 {
                eager.push(e);
                lazy.push(l);
            }
        }
        let (e, l) = (median(eager), median(lazy));
        println!(
            "{nu:>4} {:>9} {q:>7} {e:>13.3?} {l:>13.3?} {:>6.2}x {:>8.1}x",
            shape.n_enc,
            e.as_secs_f64() / l.as_secs_f64(),
            (shape.n_enc / shape.n()) as f64 * (shape.n_enc as f64).log2() / q as f64,
        );
    }
    println!();
}

#[test]
#[ignore = "measurement"]
fn where_the_provers_time_goes() {
    use rmip_field::{BinaryField, F128};
    use rmip_iop::lincheck::{Coefficients, Lincheck};
    use rmip_iop::mle::eq_table;
    use rmip_iop::zerocheck::Zerocheck;
    use rmip_iop::{matmle, sumcheck};

    let (c, _oracle) = mnist_case(1024);
    let zf = &c.honest_trace;
    let shape = Shape::balanced(c.layout.nu, 4, 4.0);
    println!(
        "\n{}: N = {} (nu {}), M = {} (mu {}), Z = {}, shape {}x{} -> n_enc {}\n",
        c.family,
        zf.len(),
        c.layout.nu,
        1usize << c.sys.mu,
        c.sys.mu,
        c.sys.a.nnz() + c.sys.b.nnz() + c.sys.c.nnz(),
        shape.m(),
        shape.n(),
        shape.n_enc
    );

    let reps = 7;
    let mut rows: Vec<(&str, Duration)> = Vec::new();
    macro_rules! time {
        ($name:expr, $body:expr) => {{
            let mut v = Vec::new();
            let mut last = None;
            for _ in 0..reps {
                let t0 = Instant::now();
                let r = $body;
                v.push(t0.elapsed());
                last = Some(r);
            }
            rows.push(($name, median(v)));
            last.unwrap()
        }};
    }

    let dom = rmip_backend_interleaved::code::domain(shape.n_enc);
    let rowsv: Vec<Vec<F128>> = zf.chunks(shape.n()).map(|r| r.to_vec()).collect();
    time!("commit: encode_columns (ELIDED when lazy)", {
        std::hint::black_box(rmip_opt::encode::encode_columns(&rowsv, &dom))
    });
    let coeffs_m: Vec<F128> = (0..shape.m()).map(|i| F128::new(i as u128 + 1)).collect();
    time!("commit: proximity combine (m x n)", {
        let mut out = vec![F128::ZERO; shape.n()];
        for (i, r) in rowsv.iter().enumerate() {
            for (j, v) in r.iter().enumerate() {
                out[j] = out[j].add(v.mul(coeffs_m[i]));
            }
        }
        std::hint::black_box(out)
    });

    let (a, b, cm) = (
        time!("zerocheck: mul_vec A", matmle::mul_vec(&c.sys.a, zf)),
        time!("zerocheck: mul_vec B", matmle::mul_vec(&c.sys.b, zf)),
        time!("zerocheck: mul_vec C", matmle::mul_vec(&c.sys.c, zf)),
    );
    let tau: Vec<F128> = (0..c.sys.mu).map(|i| F128::new(i as u128 + 7)).collect();
    time!("zerocheck: Zerocheck::new (3 tables of M, eq factored)", {
        std::hint::black_box(Zerocheck::new(&tau, a.clone(), b.clone(), cm.clone()))
    });
    time!("zerocheck: current_sum (DISCARDED by the driver)", {
        let z = Zerocheck::new(&tau, a.clone(), b.clone(), cm.clone());
        std::hint::black_box(z.current_sum())
    });
    time!("zerocheck: run_mlecheck_recording (mu rounds, degree 2)", {
        let mut z = Zerocheck::new(&tau, a.clone(), b.clone(), cm.clone());
        let mut ch = SeededChallenger::new(3);
        std::hint::black_box(sumcheck::run_mlecheck_recording(&mut z, &tau, &mut ch))
    });

    let r_row: Vec<F128> = (0..c.sys.mu).map(|i| F128::new(i as u128 + 11)).collect();
    let coeffs = Coefficients::sample(&mut SeededChallenger::new(5));
    time!("lincheck: Lincheck::new (incl. zf.clone(), N)", {
        std::hint::black_box(Lincheck::new(
            &c.sys.a,
            &c.sys.b,
            &c.sys.c,
            &r_row,
            coeffs,
            zf.clone(),
        ))
    });
    let lclaim = time!("lincheck: sumcheck::run_recording (nu rounds)", {
        let mut l = Lincheck::new(&c.sys.a, &c.sys.b, &c.sys.c, &r_row, coeffs, zf.clone());
        let mut ch = SeededChallenger::new(5);
        let (_, cl, _) = sumcheck::run_recording(&mut l, F128::ZERO, &mut ch);
        cl
    });
    time!("lincheck: matmle::eval_all  [T_mat, verifier]", {
        std::hint::black_box(matmle::eval_all(
            &c.sys.a,
            &c.sys.b,
            &c.sys.c,
            &r_row,
            &lclaim.point,
        ))
    });

    let k_total = 3 + c.t * (c.inst.b_max() + 1);
    let claims: Vec<rmip_iop::backend::Claim<F128>> = {
        let mut ch = SeededChallenger::new(1);
        let nu = c.layout.nu;
        let true_at = |point: &[F128]| rmip_iop::mle::eval_mle(zf, point);
        let mk = |ch: &mut SeededChallenger| {
            let point: Vec<F128> = (0..nu).map(|_| ch.sample()).collect();
            let value = true_at(&point);
            rmip_iop::backend::Claim { point, value }
        };
        let public = mk(&mut ch);
        let mut v = vec![mk(&mut ch), public.clone(), mk(&mut ch)];
        for _ in 0..c.t {
            let j = rmip_iop::sampling::uniform_index(&mut ch, c.inst.ell);
            for coord in c.inst.loc_row(j) {
                v.push(rmip_iop::backend::Claim {
                    point: rmip_iop::driver::boolean_point(coord, nu),
                    value: zf[coord],
                });
            }
        }
        while v.len() < k_total {
            v.push(public.clone());
        }
        v
    };
    let distinct = {
        let mut d: Vec<&[F128]> = Vec::new();
        for cl in &claims {
            if !d.contains(&cl.point.as_slice()) {
                d.push(cl.point.as_slice());
            }
        }
        d.len()
    };
    time!("batch::reduce (K claims to one)", {
        let mut ch = SeededChallenger::new(9);
        std::hint::black_box(
            rmip_iop::batch::reduce(&claims, zf, &mut ch)
                .expect("the driver's own claim list must batch"),
        )
    });
    println!("  (K = {k_total} claims on {distinct} distinct points)\n");

    time!("open: eq_table(nu_row) + combine (m x n)", {
        let e = eq_table(
            &(0..shape.nu_row)
                .map(|i| F128::new(i as u128 + 3))
                .collect::<Vec<_>>(),
        );
        let mut out = vec![F128::ZERO; shape.n()];
        for (i, r) in rowsv.iter().enumerate() {
            for (j, v) in r.iter().enumerate() {
                out[j] = out[j].add(v.mul(e[i]));
            }
        }
        std::hint::black_box(out)
    });

    let mut lazy = Ligero::lazy(shape);
    let mut lch = SeededChallenger::new(1);
    lazy.commit(zf, &mut lch);
    let h = lch
        .transcript()
        .prefix_at(rmip_backend_interleaved::CODEWORD_ORACLE);
    let q = shape.queries();
    time!("query phase: q reconstructs (the LAZY cost)", {
        for i in 0..q {
            std::hint::black_box(lazy.reconstruct(
                rmip_backend_interleaved::CODEWORD_ORACLE,
                &h,
                Coord::from(i * (shape.n_enc / q)),
            ));
        }
    });

    let total: Duration = rows.iter().map(|(_, d)| *d).sum();
    println!("{:<48} {:>12} {:>8}", "component", "median", "share");
    for (name, d) in &rows {
        println!(
            "{name:<48} {:>12.3?} {:>7.1}%",
            d,
            100.0 * d.as_secs_f64() / total.as_secs_f64()
        );
    }
    println!("{:<48} {:>12.3?}\n", "sum of the above", total);
}
