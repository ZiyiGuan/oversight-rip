use rmip_backend_interleaved::{Ligero, Shape};
use rmip_demo::fixtures;
use rmip_demo::harness::{bounded_error_sampling_queries, mnist_case, run_case, Case};
use rmip_frontend::oracle::Oracle;
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::driver::{Cheat, Config, Statement};
use rmip_iop::transcript::SeededChallenger;
use rmip_mip::{PCheck, SharedAdvice};
use rmip_mip_prover::execute_recording as mip_execute;
use std::time::{Duration, Instant};

fn st(c: &Case) -> Statement<'_> {
    Statement {
        inst: &c.inst,
        sys: &c.sys,
        layout: &c.layout,
        circuit: &c.circuit,
        dc: &c.dc,
        holo: c
            .holo
            .as_deref()
            .map(|e| e as &dyn rmip_iop::mateval::HoloMatEval),
    }
}

use rmip_demo::harness::best_of as best;

fn spread(v: &[Duration]) -> (Duration, Duration, Duration) {
    let mut w = v.to_vec();
    w.sort();
    (w[0], w[w.len() / 2], w[w.len() - 1])
}

fn best_secs(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn spread_secs(v: &[f64]) -> (f64, f64, f64) {
    let mut w = v.to_vec();
    w.sort_by(|a, b| a.partial_cmp(b).expect("timings are never NaN"));
    (w[0], w[w.len() / 2], w[w.len() - 1])
}

fn prover_proxy(
    label: &str,
    arm: &str,
    elapsed: Duration,
    t_verify: Duration,
    t_mat: Duration,
    t_open: Duration,
) -> f64 {
    let (elapsed, t_verify, t_mat, t_open) = (
        elapsed.as_secs_f64(),
        t_verify.as_secs_f64(),
        t_mat.as_secs_f64(),
        t_open.as_secs_f64(),
    );
    let sumchecks = t_verify - t_open;
    assert!(
        sumchecks >= 0.0,
        "{label}/{arm}: t_open {t_open:.9} s exceeds t_verify {t_verify:.9} s, \
         so the in-protocol sumcheck estimate is negative -- the driver's own \
         two timers disagree and no proxy built on them means anything"
    );
    let proxy = elapsed - t_verify - t_mat - sumchecks;
    assert!(
        proxy >= 0.0,
        "{label}/{arm}: prover proxy is NEGATIVE ({proxy:.9} s) -- elapsed \
         {elapsed:.9}, t_verify {t_verify:.9}, t_mat {t_mat:.9}, in-protocol \
         sumchecks {sumchecks:.9}. More is being attributed to the verifier \
         than the execution took, which is an attribution defect and not a \
         measurement to be reported"
    );
    proxy
}

struct SixWay {
    label: String,
    n: usize,
    ell: usize,
    compute: f64,
    compute_memo: f64,
    iop_prover: f64,
    iop_verify: f64,
    t_mat: f64,
    preprocess_verifier: Option<f64>,
    preprocess_prover: Option<f64>,
    p_main: f64,
    p_check: f64,
    p_check_setup: f64,
    mip_verify: f64,
    port_overhead: f64,
    one_call: f64,
    v_batch: f64,
    v_batch_sumcheck: f64,
    v_bool: f64,
    v_oracle: f64,
    q_o_distinct: usize,
    q_o_support: usize,
    pv: (f64, f64, f64),
    pc: (f64, f64, f64),
    s_iop: f64,
    q_long: usize,
    c_pref: usize,
    r_iop: usize,
    r_mip: usize,
    q_o: usize,
    t: usize,
    certificate: Option<usize>,
    sampled: Option<usize>,
}

fn exact_certificate_size(ell: usize, t: usize) -> usize {
    t.min(ell - t + 1)
}

fn iop_repetitions(sigma: f64, s_iop: f64) -> usize {
    assert!(s_iop > 0.0 && s_iop < 1.0);
    let mut r = (sigma.ln() / s_iop.ln()).ceil() as usize;
    while rmip_iop::backend::pow_error_upward(s_iop, r) > sigma {
        r += 1;
    }
    r
}

fn mip_repetitions_tight(sigma: f64, s_iop: f64, q_long: usize) -> usize {
    let s1 = rmip_mip::one_shot_bound(s_iop, q_long);
    let mut r = (sigma.ln() / s1.ln()).ceil() as usize;
    while rmip_mip::repeated_bound(s_iop, q_long, r) > sigma {
        r += 1;
    }
    r
}

fn measure<O: Oracle>(label: &str, c: &Case, o: &O, certificate: Option<usize>) -> SixWay {
    let nu = c.layout.nu;
    let shape = Shape::balanced(nu, 4, 4.0);
    let cfg = Config {
        t: c.t,
        ..Config::default()
    };

    let embedded = c.holo.as_deref();
    let (preprocess_verifier, preprocess_prover) = (c.t_preprocess_verifier, c.t_preprocess_prover);
    let stmt = || st(c);

    let probes: usize = if embedded.is_some() {
        match nu {
            n if n >= 18 => 1,
            n if n >= 16 => 3,
            _ => 9,
        }
    } else {
        9
    };
    #[allow(non_snake_case)]
    let PROBES: usize = probes;

    let (_, reg) = run_case(c, o, &[fixtures::honest()], 1, &|| Ligero::new(shape));

    let mut batch = 1usize;
    loop {
        let t0 = Instant::now();
        for _ in 0..batch {
            std::hint::black_box(c.circuit.eval(c.inst.x.as_slice(), o));
        }
        if t0.elapsed() >= Duration::from_millis(2) || batch >= 1 << 20 {
            break;
        }
        batch *= 4;
    }
    let mut computes = Vec::with_capacity(9);
    for _ in 0..9 {
        let t0 = Instant::now();
        for _ in 0..batch {
            std::hint::black_box(c.circuit.eval(c.inst.x.as_slice(), o));
        }
        computes.push(t0.elapsed() / batch as u32);
    }
    let compute = best(&computes).as_secs_f64();

    let mut a_iop = Vec::with_capacity(PROBES);
    let mut a_port = Vec::with_capacity(PROBES);
    let mut a_lazy = Vec::with_capacity(PROBES);
    let mut mats = Vec::with_capacity(PROBES);
    let mut vers = Vec::with_capacity(PROBES);
    let mut p_iop = Vec::with_capacity(PROBES);
    let mut p_lazy = Vec::with_capacity(PROBES);
    let mut kept = None;
    for i in 0..=PROBES {
        let mut b = Ligero::new(shape);
        let mut ch = SeededChallenger::new(1);
        let t0 = Instant::now();
        let (e_iop, v) = rmip_iop::driver::execute_recording(
            &stmt(),
            &mut b,
            o,
            &cfg,
            &Cheat::default(),
            &mut ch,
        );
        let d_iop = t0.elapsed();
        assert!(
            v.is_ok(),
            "{label}: the IOP driver rejected an honest prover: {v:?}"
        );

        let mut b = Ligero::new(shape);
        let mut ch = SeededChallenger::new(1);
        let t0 = Instant::now();
        let (_, v) = mip_execute(&stmt(), &mut b, o, &cfg, &Cheat::default(), &mut ch);
        let d_port = t0.elapsed();
        assert!(
            v.is_ok(),
            "{label}: the ported path rejected an honest prover: {v:?}"
        );

        let mut b = Ligero::lazy(shape);
        let mut ch = SeededChallenger::new(1);
        let t0 = Instant::now();
        let (e_lazy, v) = mip_execute(&stmt(), &mut b, o, &cfg, &Cheat::default(), &mut ch);
        let d_lazy = t0.elapsed();
        assert!(
            v.is_ok(),
            "{label}: the lazy MIP prover was rejected: {v:?}"
        );

        if i > 0 {
            a_iop.push(d_iop);
            a_port.push(d_port);
            a_lazy.push(d_lazy);
            if embedded.is_none() {
                mats.push(e_iop.t_mat);
            }
            vers.push(e_iop.t_verify);
            let (mat_iop, mat_lazy) = if embedded.is_some() {
                (Duration::ZERO, Duration::ZERO)
            } else {
                (e_iop.t_mat, e_lazy.t_mat)
            };
            p_iop.push(prover_proxy(
                label,
                "iop",
                d_iop,
                e_iop.t_verify,
                mat_iop,
                e_iop.t_open,
            ));
            p_lazy.push(prover_proxy(
                label,
                "lazy",
                d_lazy,
                e_lazy.t_verify,
                mat_lazy,
                e_lazy.t_open,
            ));
        }
        if i == PROBES {
            kept = Some(e_lazy);
        }
    }
    if let Some(h) = embedded.as_ref() {
        use rmip_iop::mateval::HoloMatEval;
        let r_row: Vec<rmip_field::F128> = (0..c.sys.mu)
            .map(|i| rmip_field::F128::new(i as u128 * 7919 + 5))
            .collect();
        let r_col: Vec<rmip_field::F128> = (0..nu)
            .map(|i| rmip_field::F128::new(i as u128 * 104_729 + 11))
            .collect();
        let mut pch = SeededChallenger::new(77);
        let (_t, frames) = h.prove_frames(&c.sys.a, &c.sys.b, &c.sys.c, &r_row, &r_col, &mut pch);
        for _ in 0..9 {
            let mut vch = SeededChallenger::new(77);
            let mut it = frames.iter();
            let mut next = || it.next().cloned().unwrap_or_default();
            let t0 = Instant::now();
            let out = h.verify_frames(&mut next, &r_row, &r_col, &mut vch);
            let d = t0.elapsed();
            assert!(out.is_ok(), "{label}: synthetic holo verify rejected");
            mats.push(d);
        }
    }
    let t_iop = best(&a_iop).as_secs_f64();
    let t_port = best(&a_port).as_secs_f64();
    let iop_verify = best(&vers).as_secs_f64();
    let t_mat = best(&mats).as_secs_f64();
    let iop_prover = best_secs(&p_iop);
    let p_main = best_secs(&p_lazy);
    let port_overhead = t_port - t_iop;
    let (pv_lo, pv_med, pv_hi) = spread_secs(&p_lazy);
    let exec = kept.expect("PROBES >= 1");

    let u = {
        let j = 0usize;
        let loc_j = c.inst.loc_row(j);
        let (qc, _) = loc_j.split_at(loc_j.len() - 1);
        let mut u = rmip_frontend::bits::BitVec::zeros(qc.len());
        for (k, &coord) in qc.iter().enumerate() {
            u.set(k, c.honest_trace[coord] == rmip_field::F128::ONE);
        }
        u
    };
    let reps = 256usize;
    let t0 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(o.query(u.as_slice()));
    }
    let one_call = t0.elapsed().as_secs_f64() / reps as f64;

    let k_total = 3 + c.t * (c.inst.b_max() + 1);
    let synth: Vec<rmip_iop::backend::Claim<rmip_field::F128>> = {
        let mut v: Vec<rmip_iop::backend::Claim<rmip_field::F128>> = Vec::with_capacity(k_total);
        for i in 0..3usize {
            v.push(rmip_iop::backend::Claim {
                point: (0..nu)
                    .map(|j| rmip_field::F128::new((i * nu + j + 1) as u128))
                    .collect(),
                value: rmip_field::F128::new(i as u128 + 1),
            });
        }
        'fill: for j in 0..c.t {
            for coord in c.inst.loc_row(j % c.inst.loc.len()) {
                if v.len() == k_total {
                    break 'fill;
                }
                v.push(rmip_iop::backend::Claim {
                    point: rmip_iop::driver::boolean_point(coord, nu),
                    value: rmip_field::F128::ONE,
                });
            }
        }
        while v.len() < k_total {
            v.push(v[0].clone());
        }
        v
    };
    let gammas: Vec<rmip_field::F128> = (0..rmip_iop::batch::coins_for(k_total))
        .map(|j| rmip_field::F128::new(0x9E3779B9 + j as u128))
        .collect();
    let pt: Vec<rmip_field::F128> = (0..nu)
        .map(|j| rmip_field::F128::new(j as u128 + 3))
        .collect();
    let mut bp = Vec::with_capacity(9);
    for _ in 0..9 {
        let t0 = Instant::now();
        std::hint::black_box(rmip_iop::batch::eval_batching_poly(&synth, &gammas, &pt));
        bp.push(t0.elapsed());
    }
    let v_batch = best(&bp).as_secs_f64();

    let r_col: Vec<rmip_field::F128> = (0..nu)
        .map(|j| rmip_field::F128::new(j as u128 + 11))
        .collect();
    let v_z = rmip_field::F128::new(0xABCDEF);
    let mut bl = Vec::with_capacity(9);
    for _ in 0..9 {
        let t0 = Instant::now();
        std::hint::black_box(rmip_iop::booleanity::claim_from_lincheck_endpoint(
            &r_col, v_z,
        ));
        bl.push(t0.elapsed());
    }
    let v_bool = best(&bl).as_secs_f64();

    let batch_proof = rmip_iop::sumcheck::SumcheckProof {
        round_polys: vec![vec![rmip_field::F128::ZERO; 2]; nu],
    };
    let mut bsv = Vec::with_capacity(9);
    for seed in 1..=9 {
        let mut verifier_ch = SeededChallenger::new(seed);
        let t0 = Instant::now();
        let checked = rmip_iop::sumcheck::verify(
            &batch_proof,
            rmip_field::F128::ZERO,
            nu,
            2,
            &mut verifier_ch,
        );
        bsv.push(t0.elapsed());
        assert!(checked.is_ok(), "synthetic batching sumcheck must verify");
    }
    let v_batch_sumcheck = best(&bsv).as_secs_f64();

    let advice = SharedAdvice { seed: 0xA11CE };
    let mut setup_probes = Vec::with_capacity(PROBES);
    for _ in 0..=PROBES {
        let mut b = Ligero::lazy(shape);
        let mut ch = SeededChallenger::new(advice.seed);
        let t0 = Instant::now();
        b.commit(&c.honest_trace, &mut ch);
        setup_probes.push(t0.elapsed());
    }
    setup_probes.remove(0);
    let p_check_setup = best(&setup_probes).as_secs_f64();

    let mut b = Ligero::lazy(shape);
    let mut own = SeededChallenger::new(advice.seed);
    b.commit(&c.honest_trace, &mut own);
    let checker = PCheck::new(b);

    for occ in &exec.occurrences {
        std::hint::black_box(checker.answer(occ.oracle, &occ.prefix, occ.coord));
    }
    let shots = 200usize;
    let mut checks = Vec::with_capacity(shots);
    for k in 0..shots {
        let occ = &exec.occurrences[k % exec.occurrences.len()];
        let t0 = Instant::now();
        std::hint::black_box(checker.answer(occ.oracle, &occ.prefix, occ.coord));
        checks.push(t0.elapsed());
    }
    let p_check = best(&checks).as_secs_f64();
    let (pc_lo, pc_med, pc_hi) = spread(&checks);

    let q = exec.occurrences.len();
    let answers: Vec<_> = exec
        .occurrences
        .iter()
        .map(|occ| checker.answer(occ.oracle, &occ.prefix, occ.coord))
        .collect();
    for (k, a) in answers.iter().enumerate() {
        assert_eq!(
            a, &exec.occurrences[k].answer,
            "honest provers must agree at occurrence {k}"
        );
    }
    let mut ch = SeededChallenger::new(77);
    let batch = 4096usize;
    let t0 = Instant::now();
    for _ in 0..batch {
        let j = rmip_iop::sampling::uniform_index(&mut ch, q);
        std::hint::black_box(answers[j] == exec.occurrences[j].answer);
    }
    let audit_v = t0.elapsed().as_secs_f64() / batch as f64;
    let v_oracle = one_call * c.t as f64;
    let jointly_measured = v_batch + v_batch_sumcheck + v_bool + v_oracle;
    let iop_verify = iop_verify + t_mat + jointly_measured;
    let (before_iop, before_lazy) = (iop_prover, p_main);
    let iop_prover = iop_prover - jointly_measured;
    let p_main = p_main - jointly_measured;
    for (arm, before, after) in [
        ("iop_prover", before_iop, iop_prover),
        ("p_main", before_lazy, p_main),
    ] {
        assert!(
            after >= 0.0,
            "{label}/{arm}: NEGATIVE ({after:.9} s) -- {before:.9} s of proxy \
             minus jointly measured {jointly_measured:.9} s (batching poly \
             {v_batch:.9}, batching sumcheck {v_batch_sumcheck:.9}, booleanity \
             {v_bool:.9}, oracle calls {v_oracle:.9}). One of those four is \
             being charged to the verifier twice"
        );
    }
    let mip_verify = iop_verify + audit_v;

    let compute_memo =
        (compute - (reg.ell.saturating_sub(reg.q_o_distinct_support)) as f64 * one_call).max(0.0);

    let sigma = 2f64.powi(-20);
    let sampled = certificate.map(|_| bounded_error_sampling_queries(reg.ell, reg.eps_rob, sigma));
    SixWay {
        label: label.to_string(),
        n: reg.n_trace,
        ell: reg.ell,
        compute,
        compute_memo,
        iop_prover,
        iop_verify,
        t_mat,
        preprocess_verifier: preprocess_verifier.map(|d| d.as_secs_f64()),
        preprocess_prover: preprocess_prover.map(|d| d.as_secs_f64()),
        p_main,
        p_check,
        p_check_setup,
        mip_verify,
        port_overhead,
        one_call,
        v_batch,
        v_batch_sumcheck,
        v_bool,
        v_oracle,
        q_o_distinct: reg.q_o_distinct,
        q_o_support: reg.q_o_distinct_support,
        pv: (pv_lo, pv_med, pv_hi),
        pc: (
            pc_lo.as_secs_f64(),
            pc_med.as_secs_f64(),
            pc_hi.as_secs_f64(),
        ),
        s_iop: reg.s_iop,
        q_long: reg.q_long,
        c_pref: reg.c_pref_max,
        r_iop: iop_repetitions(sigma, reg.s_iop),
        r_mip: reg.r_for_sigma,
        q_o: reg.q_o,
        t: reg.t,
        certificate,
        sampled,
    }
}

fn us(x: f64) -> String {
    if x >= 1.0 {
        format!("{:.3} s", x)
    } else if x >= 1e-3 {
        format!("{:.2} ms", x * 1e3)
    } else if x >= 1e-6 {
        format!("{:.2} us", x * 1e6)
    } else {
        format!("{:.1} ns", x * 1e9)
    }
}

#[test]
#[ignore = "measurement"]
fn the_price_of_step_three_tracks_q_long() {
    use rmip_backend_recursive::{FoldLadder, Shape as LadderShape};
    use rmip_backend_trace::RawTrace;

    let (c, o) = mnist_case(1024);
    let nu = c.layout.nu;
    let sigma = 2f64.powi(-20);

    let mut rows: Vec<(&str, usize, f64, usize, f64, usize)> = Vec::new();
    {
        let (_, r) = run_case(&c, &o, &[fixtures::honest()], 1, &|| RawTrace::new());
        rows.push((
            "raw-trace",
            r.q_long,
            r.s_iop,
            r.r_for_sigma,
            r.t_exec.as_secs_f64(),
            r.t,
        ));
    }
    {
        let (_, r) = run_case(&c, &o, &[fixtures::honest()], 1, &|| {
            Ligero::new(Shape::balanced(nu, 4, 4.0))
        });
        rows.push((
            "interleaved-rs",
            r.q_long,
            r.s_iop,
            r.r_for_sigma,
            r.t_exec.as_secs_f64(),
            r.t,
        ));
    }
    {
        let (_, r) = run_case(&c, &o, &[fixtures::honest()], 1, &|| {
            FoldLadder::new(LadderShape::balanced(nu, 4, 4.0))
        });
        rows.push((
            "fold-ladder",
            r.q_long,
            r.s_iop,
            r.r_for_sigma,
            r.t_exec.as_secs_f64(),
            r.t,
        ));
    }

    println!(
        "\n======== THE PRICE OF STEP 3, {}, by backend ========\n",
        c.family
    );
    println!(
        "{:<16} {:>8} {:>8} {:>8} {:>9} {:>8} {:>12} {:>12} {:>9} {:>9}",
        "backend",
        "q_long",
        "s_IOP",
        "R_IOP",
        "R_MIP",
        "ratio",
        "IOP R*T_exec",
        "MIP R*T_exec",
        "Q_O IOP",
        "Q_O MIP"
    );
    for (name, q_long, s_iop, r_mip, t_exec, t) in &rows {
        let r_iop = iop_repetitions(sigma, *s_iop);
        println!(
            "{:<16} {:>8} {:>8.4} {:>8} {:>9} {:>7.1}x {:>12} {:>12} {:>9} {:>9}",
            name,
            q_long,
            s_iop,
            r_iop,
            r_mip,
            *r_mip as f64 / r_iop as f64,
            us(t_exec * r_iop as f64),
            us(t_exec * *r_mip as f64),
            r_iop * t,
            r_mip * t,
        );
    }
    println!(
        "\n  ★ The two time columns are R x T_exec, the COMBINED in-process execution\n  \
         (both parties, one process), not prover time -- the same quantity the\n  \
         T_main -> T_exec rename renamed everywhere else. They are here to show how\n  \
         R moves with q_long, which is what this table is about; for a\n  \
         party-separated prover figure see `six_running_times`, which subtracts the\n  \
         verifier terms per execution.\n\n  \
         Same statement, same trace, same t. Only the commitment scheme moves,\n  \
         and with it q_long -- which multiplies BOTH the repetition count and\n  \
         the oracle calls. raw-trace has q_long = N, which is why a MIP over it\n  \
         is unrunnable while the IOP over it is not: at 1,024 views, over a\n  \
         hundred thousand repetitions against single digits.\n\n  \
         This is the whole case for a backend with q_long independent of N. It is\n  \
         not an efficiency nicety; it is what makes Step 3 affordable at all.\n"
    );
}

#[test]
#[ignore = "measurement"]
fn six_running_times() {
    let mut rows: Vec<SixWay> = Vec::new();

    let (c_largest, o_largest) = mnist_case(4096);
    let _ = measure("warm-up (discarded)", &c_largest, &o_largest, None);

    for r in [256usize, 1024] {
        let (c, o) = mnist_case(r);
        rows.push(measure(
            c.family,
            &c,
            &o,
            Some(exact_certificate_size(r, r / 2 + 1)),
        ));
    }
    rows.push(measure(
        c_largest.family,
        &c_largest,
        &o_largest,
        Some(exact_certificate_size(4096, 4096 / 2 + 1)),
    ));

    println!("\n================ PER BASE EXECUTION ================\n");
    println!(
        "{:<20} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10}",
        "workload",
        "N",
        "compute",
        "IOP prove",
        "IOP verif",
        "P_main",
        "P_check",
        "MIP verif",
        "audit",
        "V's O-calls"
    );
    for r in &rows {
        println!(
            "{:<20} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10}",
            r.label,
            r.n,
            us(r.compute),
            us(r.iop_prover),
            us(r.iop_verify),
            us(r.p_main),
            us(r.p_check),
            us(r.mip_verify),
            us(r.mip_verify - r.iop_verify),
            us(r.one_call * r.t as f64),
        );
    }
    println!("\n  audit prefix communication (reported, not timed as local CPU):");
    for r in &rows {
        println!("    {:<20} C_pref {} bytes", r.label, r.c_pref);
    }
    println!("\n  CURRENT VERIFIER decomposition (all measured components):");
    println!(
        "    {:<20} {:>10} {:>10} {:>11} {:>10} {:>9} {:>9} {:>10}",
        "workload",
        "driver",
        "mat eval",
        "batch poly",
        "batch SC",
        "Boolean",
        "O calls",
        "= current"
    );
    for r in &rows {
        let driver =
            r.iop_verify - r.t_mat - r.v_batch - r.v_batch_sumcheck - r.v_bool - r.v_oracle;
        println!(
            "    {:<20} {:>10} {:>10} {:>11} {:>10} {:>9} {:>9} {:>10}",
            r.label,
            us(driver),
            us(r.t_mat),
            us(r.v_batch),
            us(r.v_batch_sumcheck),
            us(r.v_bool),
            us(r.v_oracle),
            us(r.iop_verify),
        );
    }
    let embedded_rows: Vec<&SixWay> = rows
        .iter()
        .filter(|r| r.preprocess_verifier.is_some())
        .collect();
    if !embedded_rows.is_empty() {
        println!("\n  offline preprocessing, once per circuit (def:preproc's (4.9f)):");
        for r in &embedded_rows {
            println!(
                "    {:<20} {:>10}   verifier (index build + verifier-owned encoding)",
                r.label,
                us(r.preprocess_verifier.unwrap_or(0.0)),
            );
            println!(
                "    {:<20} {:>10}   prover index tables",
                "",
                us(r.preprocess_prover.unwrap_or(0.0)),
            );
        }
    }

    println!(
        "\n  `Execution::t_verify` is only the `driver` column. The current code also\n  \
         performs every other displayed term. `mat eval` is the scan below\n  \
         nu = 15 and the HOLOGRAPHIC VERIFY (synthetic replay of the embedded\n  \
         verify_frames) at and above it, where the shipped driver carries the\n  \
         verifier-owned index; the holo PROVE share of the fused driver block\n  \
         stays in the prover columns. Small\n  \
         transcript/book-keeping operations still run inside the joint driver\n  \
         and cannot be isolated without the prover/verifier split, so `current`\n  \
         is a component sum, not a claim of cycle-exact exhaustive attribution."
    );

    println!(
        "\n  MEASUREMENT FLOOR. Within one process these figures repeat to ~4%;\n  \
         ACROSS processes the envelope is ~28% (three invocations of identical\n  \
         code gave 3.2 / 2.5 / 2.6 ms for one T_exec). Read no cell to better than\n  \
         that, and read the port-control percentages -- which are 0.0-1.3%, well\n  \
         under the floor -- as 'no detectable difference', not as measurements."
    );

    println!("\n  dispersion, min .. median .. max over the probes:");
    for r in &rows {
        println!(
            "    {:<20} P_main {:>10} {:>10} {:>10}    P_check {:>10} {:>10} {:>10}",
            r.label,
            us(r.pv.0),
            us(r.pv.1),
            us(r.pv.2),
            us(r.pc.0),
            us(r.pc.1),
            us(r.pc.2),
        );
    }
    println!("\n  port control (ported-eager minus IOP-driver, should be ~0):");
    for r in &rows {
        println!(
            "    {:<20} {:>10}   ({:+.1}% of the IOP prover)",
            r.label,
            us(r.port_overhead.abs()),
            100.0 * r.port_overhead / r.iop_prover
        );
    }
    println!(
        "\n  `V's O-calls` is the VERIFIER's t calls to O per execution -- step 6 of\n  \
         prot:step2 has the verifier check O(u_h) = a_h. They happen inside\n  \
         `execute`, so they are timed into the two prover columns and appear in\n  \
         neither verifier column. Add them to the verifier and subtract from the\n  \
         prover if you want the parties separated: this is the resource the whole\n  \
         construction exists to economise, and a wall clock is the least\n  \
         interesting thing about it.\n\n  \
         SOUNDNESS IS NOT MATCHED IN THIS TABLE. One base execution errs with\n  \
         probability s_IOP ~ 0.19; the columns are per-execution costs, not\n  \
         costs of a protocol anyone would run. The matched-sigma table is below.\n\n  \
         `compute` is CONTAINED IN the two prover columns: the honest prover builds\n  \
         z^O, whose answer coordinates are O(u_j) at all ell occurrences, so it\n  \
         evaluates C^O in full before proving anything. Rows cannot fall below it.\n\n  \
         `IOP prover` and `P_main` are diagnostic separated proxies: they start\n  \
         from T_exec and subtract measured/replayed verifier components, including\n  \
         synthetic shape-matched terms. They are not exact theoretical T_main.\n\n  \
         `audit` is `MIP verif - IOP verif`. It is the in-process per-execution\n  \
         CPU difference: one exact-uniform index (variable raw coins) and one\n  \
         symbol comparison. Prefix transport is reported above, not timed. The\n  \
         two verifiers otherwise run the same algorithm; their dominant gap is R."
    );

    println!("\n================ AMPLIFIED TO sigma = 2^-20 ================\n");
    println!(
        "{:<20} {:>8} {:>7} {:>7} {:>7} {:>10} {:>11} {:>11} {:>11} {:>11}",
        "workload",
        "s_IOP",
        "q_long",
        "R_IOP",
        "R_MIP",
        "R_MIP/R_IOP",
        "IOP prove",
        "IOP verif",
        "P_main",
        "MIP verif"
    );
    for r in &rows {
        println!(
            "{:<20} {:>8.4} {:>7} {:>7} {:>7} {:>9.1}x {:>11} {:>11} {:>11} {:>11}",
            r.label,
            r.s_iop,
            r.q_long,
            r.r_iop,
            r.r_mip,
            r.r_mip as f64 / r.r_iop as f64,
            us(r.iop_prover * r.r_iop as f64),
            us(r.iop_verify * r.r_iop as f64),
            us(r.p_main * r.r_mip as f64),
            us(r.mip_verify * r.r_mip as f64),
        );
    }
    println!("\n  repetition counts under one estimator (exact log-ratio, both sides):");
    println!(
        "    {:<20} {:>8} {:>9} {:>12} {:>9} {:>9}",
        "workload", "R_IOP", "R_MIP", "R_MIP tight", "tight/IOP", "MIP/tight"
    );
    for r in &rows {
        let tight = mip_repetitions_tight(2f64.powi(-20), r.s_iop, r.q_long);
        println!(
            "    {:<20} {:>8} {:>9} {:>12} {:>8.1}x {:>8.2}x",
            r.label,
            r.r_iop,
            r.r_mip,
            tight,
            tight as f64 / r.r_iop as f64,
            r.r_mip as f64 / tight as f64,
        );
    }
    println!(
        "\n  R_MIP is the exact minimum of (D) (`repetitions_for`), so MIP/tight is\n  \
         1.00 except within one repetition at a rounding boundary: a cross-check\n  \
         of two implementations. The price-of-Step-3 ratio\n  \
         is the `tight/IOP` column."
    );

    println!(
        "\n  Both sides now reach sigma = 2^-20. R_MIP is the exact minimum of (D). R_IOP is the\n  \
         standard independent-repetition formula ceil(ln sigma / ln s_IOP), SUPPLIED\n  \
         HERE -- no IOP repetition count appears anywhere in the documents, whose\n  \
         only amplification theorem is about the MIP.\n\n  \
         R_MIP/R_IOP ~ q_long*ln(1/s_IOP)/(1-s_IOP) is THE PRICE OF STEP 3: what it\n  \
         costs to turn a relativized IOP into a classical two-prover protocol.\n\n  \
         ★ BOTH TOTALS ARE UPPER BOUNDS, and for the same reason. R*T assumes\n  \
         every repetition redoes everything, but `Ligero::commit` and the three\n  \
         matrix-vector products are COIN-INDEPENDENT -- identical in all R runs.\n  \
         Whether the repetitions may share them is an open Step-3 question\n  \
         not an implementation detail, so nothing here assumes\n  \
         an answer. It bounds the MIP side harder, R_MIP being ~10x R_IOP.\n\n  \
         Every row derives t from sigma_loc = 1/8 (`Config::for_instance`), so\n  \
         beta = (1-eps_rob)^t inside s_IOP is at one local target throughout and\n  \
         every row reaches the same sigma = 2^-20."
    );

    println!("\n================ THE SECOND PROVER ================\n");
    println!(
        "{:<20} {:>11} {:>11} {:>7} {:>13} {:>13} {:>9}",
        "workload", "setup (1x)", "per query", "R_MIP", "total P_check", "total P_main", "share"
    );
    for r in &rows {
        let total_check = r.p_check_setup + r.p_check * r.r_mip as f64;
        let total_main = r.p_main * r.r_mip as f64;
        println!(
            "{:<20} {:>11} {:>11} {:>7} {:>13} {:>13} {:>8.4}%",
            r.label,
            us(r.p_check_setup),
            us(r.p_check),
            r.r_mip,
            us(total_check),
            us(total_main),
            100.0 * total_check / total_main,
        );
    }
    println!(
        "\n  P_check is built ONCE, before the protocol, and answers one query per\n  \
         execution: `PCheck::answer` takes `&self` and the audit asks exactly one.\n  \
         So its total is `setup + R * T_check`, not `R * (setup + T_check)`.\n\n  \
         ★ THE TWO PROVERS ARE AMORTISED BY DIFFERENT RULES, and the asymmetry is\n  \
         not a choice made here. P_check's setup is x1 because the CODE does that\n  \
         and the model requires it: `PCheck::new` runs before any execution and\n  \
         `answer` takes `&self`, which is requirement 1 of non-communication.\n  \
         P_main's commit is charged xR because whether R repetitions may share one\n  \
         commitment is an OPEN Step-3 question, and assuming they\n  \
         may would be assuming the answer. So P_check's total is a measurement and\n  \
         P_main's is an upper bound -- they are not the same kind of number."
    );

    println!("\n================ AGAINST COMPUTING IT YOURSELF ================\n");
    println!(
        "{:<20} {:>10} {:>10} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>10} {:>10} {:>8}",
        "workload",
        "compute",
        "memoised",
        "ell",
        "support",
        "cert",
        "sampler",
        "Q_O IOP",
        "Q_O MIP",
        "vs sampler",
        "MIP verif",
        "vs memo"
    );
    for r in &rows {
        let v = r.mip_verify * r.r_mip as f64;
        let q_o_iop = r.r_iop * r.t;
        let calls = match r.sampled {
            None => "-".to_string(),
            Some(sampled) if sampled > r.q_o => {
                format!("{:.1}x saved", sampled as f64 / r.q_o as f64)
            }
            Some(_) => "none".to_string(),
        };
        println!(
            "{:<20} {:>10} {:>10} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>10} {:>10} {:>7.1}x",
            r.label,
            us(r.compute),
            us(r.compute_memo),
            r.ell,
            r.q_o_support,
            r.certificate
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into()),
            r.sampled
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into()),
            q_o_iop,
            r.q_o,
            calls,
            us(v),
            v / r.compute_memo.max(1e-12),
        );
    }
    let per_cell: Vec<f64> = rows
        .iter()
        .filter(|r| r.label.starts_with("MNIST"))
        .map(|r| r.p_check / r.n as f64)
        .collect();
    assert_eq!(per_cell.len(), 3, "three MNIST rows");
    let (lo, hi) = per_cell
        .iter()
        .fold((f64::MAX, 0f64), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(
        hi / lo < 1.5,
        "P_check must be Theta(N): per-cell cost spans {:.2}x across the MNIST rows \
         ({:?} ns/cell) -- re-measure before believing the P_check column",
        hi / lo,
        per_cell.iter().map(|x| x * 1e9).collect::<Vec<_>>()
    );

    for r in &rows {
        assert!(
            r.q_o_distinct <= r.q_o,
            "{}: verifier distinct queries {} exceed Q_O {}",
            r.label,
            r.q_o_distinct,
            r.q_o
        );
        let audit = r.mip_verify - r.iop_verify;
        assert!(
            audit >= 0.0 && audit < 0.05 * r.iop_verify,
            "{}: the audit is {:.1}% of the verifier; it should be exact-index \
             sampling and one symbol compare",
            r.label,
            100.0 * audit / r.iop_verify
        );
    }

    println!(
        "\n  Q_O IOP = R_IOP * t; Q_O MIP = R_MIP * t. Oracularization costs the\n  \
         SAME ~10x on oracle calls that it costs on repetitions -- Step 3 buys a\n  \
         classical two-prover protocol and pays for it in the resource the whole\n  \
         construction exists to economise.\n"
    );
    println!(
        "  Two different questions, and they have opposite answers here.\n\n  \
         ORACLE CALLS: Q_O is flat across the MNIST rows (R_MIP * t is set by\n  \
         eps_rob and q_long, not by N), but a direct bounded-error sampler at\n  \
         the same gap and sigma=2^-20 needs only a few dozen calls by Hoeffding.\n  \
         The protocol therefore saves no oracle calls on these instances. The\n  \
         larger `cert` column is a one-sided exact certificate size, not the\n  \
         complexity of the matching evaluator.\n\n  \
         `support` is the number of distinct strings among all circuit\n  \
         occurrences. A memoising evaluator pays `support`, not `ell`; the MNIST\n  \
         circuit's queries are all distinct, so the two coincide here and the\n  \
         `compute` column, which re-queries at every gate, is the honest\n  \
         baseline. Registry Q_O^dist is the verifier-only upper bound\n  \
         min(Q_O, support), not witness-generation calls.\n\n  \
         LOCAL WORK: the verifier's own seconds are far MORE than just\n  \
         running the circuit. The construction trades local computation for\n  \
         oracle calls; it is worth it only when a call costs more than the\n  \
         local work the protocol adds, which is what the tau*\n  \
         break-even quantifies.\n"
    );
}
