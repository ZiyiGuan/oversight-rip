use crate::fixtures::Fixture;
use crate::stats::Rate;
use rmip_field::{BinaryField, F128};
use rmip_frontend::bits::BitVec;
use rmip_frontend::circuit::OracleCircuit;
use rmip_frontend::deoracle::{OracleSite, OrdinaryCircuit};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::r1cs::{Booleanity, R1cs};
use rmip_frontend::{layout::TraceLayout, Instance, Ratio};
use rmip_iop::backend::AuditMode;
use rmip_iop::backend::{Alphabet, BackendParams, CommitmentBackend, Coord, Symbol};
use rmip_iop::driver::{execute_recording, Cheat, Config, Execution, Reject, Statement};
use rmip_iop::transcript::{OsChallenger, RawChallenger, SeededChallenger};
use rmip_mip::{audit_all, audit_one, PCheck, SharedAdvice};
use rmip_mip_prover::{LazyBackend, OracleMeta};
use std::collections::HashSet;

pub fn best_of(samples: &[std::time::Duration]) -> std::time::Duration {
    *samples.iter().min().expect("at least one sample")
}

pub struct Case {
    pub family: &'static str,
    pub inst: Instance,
    pub sys: R1cs,
    pub layout: TraceLayout,
    pub circuit: OracleCircuit,
    pub dc: OrdinaryCircuit,
    pub sites: std::sync::Arc<[OracleSite]>,
    pub no_instance_x: BitVec,
    pub honest_trace: Vec<rmip_field::F128>,
    pub t: usize,
    pub eps_rob: Ratio,
    pub majority_threshold: usize,
    pub holo: Option<std::sync::Arc<rmip_holo::Embedded>>,
    pub closed_form: Option<std::sync::Arc<rmip_iop::closedform::ChainMatEval>>,
    pub t_preprocess_verifier: Option<std::time::Duration>,
    pub t_preprocess_prover: Option<std::time::Duration>,
    pub t_preprocess: Option<std::time::Duration>,
}

pub fn with_forced_index(mut c: Case) -> Case {
    c.closed_form = None;
    c.holo = None;
    let t0 = std::time::Instant::now();
    let index = rmip_holo::preprocess_verifier(&c.sys.a, &c.sys.b, &c.sys.c, c.sys.mu, c.layout.nu);
    let tv = t0.elapsed();
    let t1 = std::time::Instant::now();
    let wit = rmip_holo::preprocess_prover(&c.sys.a, &c.sys.b, &c.sys.c, c.sys.mu, c.layout.nu);
    let tp = t1.elapsed();
    c.t_preprocess_verifier = Some(tv);
    c.t_preprocess_prover = Some(tp);
    c.t_preprocess = Some(tv + tp);
    c.holo = Some(std::sync::Arc::new(rmip_holo::Embedded { index, wit }));
    c
}

pub fn with_shipped_embedding(mut c: Case) -> Case {
    if c.closed_form.is_none()
        && c.holo.is_none()
        && rmip_iop::mateval::holography_pays(c.layout.nu)
    {
        let t0 = std::time::Instant::now();
        let index =
            rmip_holo::preprocess_verifier(&c.sys.a, &c.sys.b, &c.sys.c, c.sys.mu, c.layout.nu);
        let tv = t0.elapsed();
        let t1 = std::time::Instant::now();
        let wit = rmip_holo::preprocess_prover(&c.sys.a, &c.sys.b, &c.sys.c, c.sys.mu, c.layout.nu);
        let tp = t1.elapsed();
        c.t_preprocess_verifier = Some(tv);
        c.t_preprocess_prover = Some(tp);
        c.t_preprocess = Some(tv + tp);
        c.holo = Some(std::sync::Arc::new(rmip_holo::Embedded { index, wit }));
    }
    c
}

#[derive(Clone, Debug)]
pub struct Row {
    pub backend: &'static str,
    pub family: &'static str,
    pub fixture: &'static str,
    pub accept: Rate,
    pub predicted: f64,
    pub derivation: &'static str,
    pub consistent: bool,
    pub resolves: bool,
}

#[derive(Clone, Debug)]
pub struct TCheck {
    pub oracle: rmip_iop::transcript::OracleId,
    pub alphabet: Alphabet,
    pub symbols: Coord,
    pub worst: std::time::Duration,
    pub mean: std::time::Duration,
}

#[derive(Clone, Debug)]
pub struct Registry {
    pub backend: &'static str,
    pub family: &'static str,
    pub n_trace: usize,
    pub nu: usize,
    pub m: usize,
    pub mu: usize,
    pub z_nnz: usize,
    pub ell: usize,
    pub b_max: usize,
    pub k: usize,
    pub b_tr: usize,
    pub eps_rob: Ratio,
    pub booleanity: Booleanity,
    pub t: usize,
    pub k_claims: usize,
    pub q_long: usize,
    pub a_long: u64,
    pub l_long: Coord,
    pub r_open: usize,
    pub c_pref_max: usize,
    pub eps_bind: f64,
    pub eps_open: f64,
    pub s_iop: f64,
    pub s_2p_1: f64,
    pub r_for_sigma: usize,
    pub q_o: usize,
    pub q_o_distinct: usize,
    pub q_o_distinct_support: usize,
    pub t_check: Vec<TCheck>,
    pub r_iop: usize,
    pub coins: usize,
    pub c_short: usize,
    pub t_mat: std::time::Duration,
    pub t_verify: std::time::Duration,
    pub t_open: std::time::Duration,
    pub t_exec: std::time::Duration,
    pub k_rel: usize,
    pub eps_rel: f64,
    pub eps_idx: f64,
}

impl Case {
    pub fn clone_for_no_instance(&self) -> Case {
        let mut inst = self.inst.clone();
        inst.x = self.no_instance_x.clone();
        Case {
            family: self.family,
            inst,
            sys: self.sys.clone(),
            layout: self.layout.clone(),
            circuit: self.circuit.clone(),
            dc: self.dc.clone(),
            sites: self.sites.clone(),
            no_instance_x: self.no_instance_x.clone(),
            honest_trace: self.honest_trace.clone(),
            t: self.t,
            eps_rob: self.eps_rob,
            majority_threshold: self.majority_threshold,
            holo: self.holo.clone(),
            closed_form: self.closed_form.clone(),
            t_preprocess_verifier: self.t_preprocess_verifier,
            t_preprocess_prover: self.t_preprocess_prover,
            t_preprocess: self.t_preprocess,
        }
    }
}

pub fn statement(c: &Case) -> Statement<'_> {
    Statement {
        inst: &c.inst,
        sys: &c.sys,
        layout: &c.layout,
        circuit: &c.circuit,
        dc: &c.dc,
        holo: matrix_evaluation(c),
    }
}

pub fn matrix_evaluation(c: &Case) -> Option<&dyn rmip_iop::mateval::HoloMatEval> {
    if let Some(cf) = c.closed_form.as_deref() {
        return Some(cf as &dyn rmip_iop::mateval::HoloMatEval);
    }
    c.holo
        .as_deref()
        .map(|e| e as &dyn rmip_iop::mateval::HoloMatEval)
}

pub fn bounded_error_sampling_queries(ell: usize, eps_rob: Ratio, sigma: f64) -> usize {
    fn next_up_nonnegative(value: f64) -> f64 {
        assert!(value.is_finite() && value >= 0.0);
        f64::from_bits(value.to_bits() + 1)
    }

    fn next_down_positive(value: f64) -> f64 {
        assert!(value.is_finite() && value > 0.0);
        f64::from_bits(value.to_bits() - 1)
    }

    fn usize_downward(value: usize) -> f64 {
        let rounded = value as f64;
        if rounded as u128 <= value as u128 {
            rounded
        } else {
            next_down_positive(rounded)
        }
    }

    fn integer_downward(value: u64) -> f64 {
        let rounded = value as f64;
        if rounded as u128 <= value as u128 {
            rounded
        } else {
            next_down_positive(rounded)
        }
    }

    fn integer_upward_u64(value: u64) -> f64 {
        let rounded = value as f64;
        if rounded as u128 >= value as u128 {
            rounded
        } else {
            next_up_nonnegative(rounded)
        }
    }

    fn div_downward(numerator: f64, denominator: f64) -> f64 {
        let rounded = numerator / denominator;
        if rounded == 0.0 || rounded.is_subnormal() {
            return if rounded == 0.0 {
                0.0
            } else {
                next_down_positive(rounded)
            };
        }
        let residual = (-rounded).mul_add(denominator, numerator);
        if residual < 0.0 {
            next_down_positive(rounded)
        } else {
            rounded
        }
    }

    fn div_upward(numerator: f64, denominator: f64) -> f64 {
        let rounded = numerator / denominator;
        assert!(rounded.is_finite() && rounded > 0.0);
        if rounded.is_subnormal() {
            return next_up_nonnegative(rounded);
        }
        let residual = (-rounded).mul_add(denominator, numerator);
        if residual > 0.0 {
            next_up_nonnegative(rounded)
        } else {
            rounded
        }
    }

    fn mul_downward(lhs: f64, rhs: f64) -> f64 {
        if lhs == 0.0 || rhs == 0.0 {
            return 0.0;
        }
        let rounded = lhs * rhs;
        assert!(rounded.is_finite(), "Hoeffding exponent overflowed");
        if rounded == 0.0 || rounded.is_subnormal() {
            return if rounded == 0.0 {
                0.0
            } else {
                next_down_positive(rounded)
            };
        }
        let residual = lhs.mul_add(rhs, -rounded);
        if residual < 0.0 {
            next_down_positive(rounded)
        } else {
            rounded
        }
    }

    assert!(ell > 0, "an evaluator needs a nonempty occurrence list");
    assert!(
        sigma.is_finite() && sigma > 0.0 && sigma < 1.0,
        "target error must be finite and lie in (0,1)"
    );
    assert!(eps_rob.den > 0, "gap denominator must be positive");
    assert!(
        eps_rob.num > 0 && eps_rob.num <= eps_rob.den,
        "the promised relative gap must lie in (0,1]"
    );

    let gap_lower = if eps_rob.num == eps_rob.den {
        1.0
    } else {
        div_downward(
            integer_downward(eps_rob.num),
            integer_upward_u64(eps_rob.den),
        )
    };
    let rate_lower = mul_downward(2.0, mul_downward(gap_lower, gap_lower));
    if rate_lower == 0.0 {
        return ell;
    }

    let log_recip_upper = next_up_nonnegative(-sigma.ln());
    let estimate = div_upward(log_recip_upper, rate_lower);
    let mut sampled = (estimate.ceil() as usize).max(1).min(ell);
    while sampled < ell && mul_downward(rate_lower, usize_downward(sampled)) < log_recip_upper {
        sampled += 1;
    }
    sampled
}

pub fn verifier_query_support(rows: impl IntoIterator<Item = Vec<usize>>, trace: &[F128]) -> usize {
    let mut distinct = HashSet::new();
    for loc_j in rows {
        let (query_coords, answer_coord) = loc_j.split_at(loc_j.len() - 1);
        assert_eq!(
            answer_coord.len(),
            1,
            "loc(j) must end in one answer coordinate"
        );
        let query: Vec<bool> = query_coords
            .iter()
            .map(|&coord| {
                let value = trace[coord];
                assert!(
                    value == F128::ZERO || value == F128::ONE,
                    "the honest trace must be Boolean at every oracle-query coordinate"
                );
                value == F128::ONE
            })
            .collect();
        distinct.insert(query);
    }
    distinct.len()
}

fn mixed_index(i: u64, q: usize) -> usize {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) % q as u64) as usize
}

type Driver<B, O> = fn(
    &Statement<'_>,
    &mut B,
    &O,
    &Config,
    &Cheat,
    &mut dyn RawChallenger,
) -> (Execution, Result<(), Reject>);

pub fn measure<O: Oracle, B: CommitmentBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    fixture: &Fixture,
    oracle: &O,
    trials: usize,
    make: &F,
) -> Rate {
    measure_with(c, fixture, oracle, trials, make, execute_recording::<B, O>)
}

pub fn measure_lazy<O: Oracle, B: LazyBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    fixture: &Fixture,
    oracle: &O,
    trials: usize,
    make: &F,
) -> Rate {
    measure_with(
        c,
        fixture,
        oracle,
        trials,
        make,
        rmip_mip_prover::execute_recording::<B, O>,
    )
}

fn measure_with<O: Oracle, B: CommitmentBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    fixture: &Fixture,
    oracle: &O,
    trials: usize,
    make: &F,
    drive: Driver<B, O>,
) -> Rate {
    let cfg = Config {
        t: c.t,
        ..Default::default()
    };
    let advice = SharedAdvice { seed: 0xC4EC };
    let checker = {
        let committed = rmip_iop::driver::committed_trace(&statement(c), oracle, &fixture.cheat);
        let mut b = make();
        let mut own = SeededChallenger::new(advice.seed);
        b.commit(&committed, &mut own);
        PCheck::new(b)
    };

    let mut accepted = 0;
    for i in 0..trials {
        let mut backend = make();
        let mut ch = OsChallenger::new();
        let (mut e, _iop_verdict) = drive(
            &statement(c),
            &mut backend,
            oracle,
            &cfg,
            &fixture.cheat,
            &mut ch,
        );
        if fixture.corrupt_one_query_answer {
            let victim = mixed_index(i as u64, e.occurrences.len());
            let mut bad = e.occurrences[victim].answer.0.clone();
            bad[0] = bad[0].add(F128::ONE);
            e.occurrences[victim].answer = Symbol(bad);
        }
        if audit_for(&backend, &e, &checker).accepted() {
            accepted += 1;
        }
    }
    Rate::new(accepted, trials)
}

fn audit_for<B: CommitmentBackend<Field = F128>>(
    backend: &B,
    e: &Execution,
    checker: &PCheck<B>,
) -> rmip_mip::Audit {
    match backend.params(1).audit {
        AuditMode::OneOf => {
            let mut audit_ch = OsChallenger::new();
            audit_one(e, checker, &mut audit_ch)
        }
        AuditMode::Sequential => audit_all(e, checker),
        AuditMode::Rewind => {
            if e.iop_accepted() {
                rmip_mip::Audit::Accepted { audited: 0 }
            } else {
                rmip_mip::Audit::UnderlyingRejected { audited: 0 }
            }
        }
    }
}

pub fn measure_all<O: Oracle, B: CommitmentBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    oracle: &O,
    fixtures: &[Fixture],
    trials: usize,
    backend_name: &'static str,
    make: &F,
) -> Vec<Row> {
    fixtures
        .iter()
        .map(|f| {
            let rate = measure(c, f, oracle, trials, make);
            Row {
                backend: backend_name,
                family: c.family,
                fixture: f.name,
                accept: rate,
                predicted: f.prediction.accept_rate,
                derivation: f.prediction.derivation,
                consistent: rate.consistent_with(f.prediction.accept_rate),
                resolves: rate.resolves(f.prediction.accept_rate),
            }
        })
        .collect()
}

pub fn run_case<O: Oracle, B: CommitmentBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    oracle: &O,
    fixtures: &[Fixture],
    trials: usize,
    make: &F,
) -> (Vec<Row>, Registry) {
    let strided = |_: &B, o: &OracleMeta| -> Vec<Coord> {
        let len = usize::try_from(o.len).expect("a materialised oracle's length fits usize");
        let probe_count = len.min(256);
        let step = (len / probe_count.max(1)).max(1);
        (0..len).step_by(step).map(Coord::from).collect()
    };
    run_case_with(
        c,
        oracle,
        fixtures,
        trials,
        make,
        execute_recording::<B, O>,
        &|b: &B| {
            b.oracles()
                .iter()
                .map(|o| OracleMeta {
                    id: o.id,
                    alphabet: o.alphabet,
                    len: Coord::from(o.symbols.len()),
                })
                .collect()
        },
        &strided,
    )
}

pub fn run_case_lazy<O: Oracle, B: LazyBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    oracle: &O,
    fixtures: &[Fixture],
    trials: usize,
    make: &F,
    draw: &dyn Fn(&B, &OracleMeta, &mut dyn RawChallenger) -> Coord,
) -> (Vec<Row>, Registry) {
    let drawn = |b: &B, o: &OracleMeta| -> Vec<Coord> {
        let mut ch = SeededChallenger::new(0x7C8E);
        (0..256)
            .map(|_| {
                let coord = draw(b, o, &mut ch);
                assert!(coord < o.len, "a drawn T_check coordinate is in the domain");
                coord
            })
            .collect()
    };
    run_case_with(
        c,
        oracle,
        fixtures,
        trials,
        make,
        rmip_mip_prover::execute_recording::<B, O>,
        &|b: &B| b.oracle_meta(),
        &drawn,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_case_with<O: Oracle, B: CommitmentBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    oracle: &O,
    fixtures: &[Fixture],
    trials: usize,
    make: &F,
    drive: Driver<B, O>,
    posted: &dyn Fn(&B) -> Vec<OracleMeta>,
    coords: &dyn Fn(&B, &OracleMeta) -> Vec<Coord>,
) -> (Vec<Row>, Registry) {
    let cfg = Config {
        t: c.t,
        ..Default::default()
    };
    let k_claims = cfg.k_rel + c.t * (c.inst.b_max() + 1);

    const PROBES: usize = 5;
    let mut backend = make();
    let mut ch = SeededChallenger::new(1);
    let mut executions: Vec<std::time::Duration> = Vec::with_capacity(PROBES);
    let mut mats: Vec<std::time::Duration> = Vec::with_capacity(PROBES);
    let mut vers: Vec<std::time::Duration> = Vec::with_capacity(PROBES);
    let mut opens: Vec<std::time::Duration> = Vec::with_capacity(PROBES);
    let mut probe = None;
    for i in 0..=PROBES {
        let mut b = make();
        let mut c2 = SeededChallenger::new(1);
        let t_exec_start = std::time::Instant::now();
        let (p, verdict) = drive(
            &statement(c),
            &mut b,
            oracle,
            &cfg,
            &Default::default(),
            &mut c2,
        );
        verdict.expect("run_case needs a yes-instance; use measure_all for a no-instance");
        let elapsed = t_exec_start.elapsed();
        if i > 0 {
            executions.push(elapsed);
            mats.push(p.t_mat);
            vers.push(p.t_verify);
            opens.push(p.t_open);
        }
        if i == PROBES {
            backend = b;
            ch = c2;
            probe = Some(p);
        }
    }
    executions.sort();
    mats.sort();
    let probe = probe.expect("PROBES >= 1");
    let t_exec = executions[executions.len() / 2];
    let params: BackendParams = backend.params(k_claims);
    let posted_oracles = posted(&backend);
    let c_pref_max = posted_oracles
        .iter()
        .map(|oracle| ch.transcript().prefix_at(oracle.id).len())
        .chain(std::iter::once(
            ch.transcript()
                .prefix_at(rmip_iop::driver::DUMMY_ORACLE)
                .len(),
        ))
        .max()
        .unwrap_or(0);

    let mut t_check = vec![TCheck {
        oracle: rmip_iop::driver::DUMMY_ORACLE,
        alphabet: Alphabet::Field,
        symbols: Coord::from(1u64),
        worst: std::time::Duration::ZERO,
        mean: std::time::Duration::ZERO,
    }];
    t_check.extend(posted_oracles.iter().map(|o| {
        let h = ch.transcript().prefix_at(o.id);
        let mut worst = std::time::Duration::ZERO;
        let mut total = std::time::Duration::ZERO;
        let mut n = 0u32;
        for coord in coords(&backend, o) {
            let t0 = std::time::Instant::now();
            let s = backend.reconstruct(o.id, &h, coord);
            let dt = t0.elapsed();
            std::hint::black_box(&s);
            worst = worst.max(dt);
            total += dt;
            n += 1;
        }
        TCheck {
            oracle: o.id,
            alphabet: o.alphabet,
            symbols: o.len,
            worst,
            mean: total / n.max(1),
        }
    }));

    let mut rows = Vec::new();
    for f in fixtures {
        let rate = measure_with(c, f, oracle, trials, make, drive);
        rows.push(Row {
            backend: params.name,
            family: c.family,
            fixture: f.name,
            accept: rate,
            predicted: f.prediction.accept_rate,
            derivation: f.prediction.derivation,
            consistent: rate.consistent_with(f.prediction.accept_rate),
            resolves: rate.resolves(f.prediction.accept_rate),
        });
    }

    let miss = rmip_iop::backend::div_error_upward(
        (c.eps_rob.den - c.eps_rob.num) as f64,
        c.eps_rob.den as f64,
    );
    let beta = rmip_iop::backend::pow_error_upward(miss, c.t);
    let eps_rel = rmip_iop::backend::relation_error(c.sys.mu, c.layout.nu);
    let eps_idx = c.holo.as_ref().map_or(0.0, |e| e.index.error_terms());
    let s_iop = rmip_iop::backend::sum_errors_upward(&[
        params.eps_bind,
        eps_rel,
        params.eps_open,
        beta,
        eps_idx,
    ]);
    let q_long = probe.q_long();
    let a_long = probe.a_long();
    assert_eq!(q_long, cfg.audit_query_count(params.q_long));
    assert_eq!(
        a_long,
        cfg.audit_answer_volume(params.q_long, params.a_long)
    );
    let denominator = rmip_mip::audit_denominator(params.audit, q_long);
    let s_2p_1 = rmip_mip::one_shot_bound(s_iop, denominator);
    let sigma = 2f64.powi(-20);
    let r = rmip_mip::repetitions_for(sigma, s_iop, denominator);

    assert_eq!(
        probe.oracle_calls, c.t,
        "one base execution must make exactly t verifier oracle calls"
    );
    let q_o = r
        .checked_mul(probe.oracle_calls)
        .expect("Q_O = R*t must fit in usize");

    let q_o_distinct_support = verifier_query_support(c.inst.loc_rows(), &c.honest_trace);
    let q_o_distinct = q_o_distinct_support.min(q_o);
    assert!(
        q_o_distinct <= q_o,
        "distinct verifier queries cannot exceed occurrences"
    );
    assert!(
        q_o_distinct <= q_o_distinct_support,
        "the capped distinct count cannot exceed its occurrence support"
    );

    let reg = Registry {
        backend: params.name,
        family: c.family,
        n_trace: c.layout.n_trace,
        nu: c.layout.nu,
        m: c.sys.m,
        mu: c.sys.mu,
        z_nnz: c.sys.z,
        ell: c.inst.ell,
        b_max: c.inst.b_max(),
        k: c.layout.k,
        b_tr: c.inst.b_tr(),
        eps_rob: c.eps_rob,
        booleanity: c.sys.booleanity,
        t: c.t,
        k_claims,
        q_long,
        a_long,
        l_long: params
            .l_long
            .checked_add_u64(rmip_iop::driver::dummy_symbol().0.len() as u64)
            .expect("composite L_long must fit in 256 bits"),
        r_open: params.r_open,
        c_pref_max,
        eps_bind: params.eps_bind,
        eps_open: params.eps_open,
        s_iop,
        s_2p_1,
        r_for_sigma: r,
        q_o,
        q_o_distinct,
        q_o_distinct_support,
        t_check,
        r_iop: probe.rounds,
        coins: probe.coins,
        c_short: probe.short_bytes,
        t_mat: mats[mats.len() / 2],
        t_verify: {
            let mut v = vers.clone();
            v.sort();
            v[v.len() / 2]
        },
        t_open: {
            let mut v = opens.clone();
            v.sort();
            v[v.len() / 2]
        },
        t_exec,
        k_rel: cfg.k_rel,
        eps_rel,
        eps_idx,
    };
    (rows, reg)
}

pub fn amplification_curve<O: Oracle, B: CommitmentBackend<Field = F128>, F: Fn() -> B>(
    c: &Case,
    fixture: &Fixture,
    oracle: &O,
    max_r: usize,
    trials: usize,
    make: &F,
) -> Vec<(usize, Rate, f64)> {
    let cfg = Config {
        t: c.t,
        ..Default::default()
    };
    let advice = SharedAdvice { seed: 0xC4EC };
    let checker = {
        let committed = rmip_iop::driver::committed_trace(&statement(c), oracle, &fixture.cheat);
        let mut b = make();
        let mut own = SeededChallenger::new(advice.seed);
        b.commit(&committed, &mut own);
        PCheck::new(b)
    };

    let mut survived = vec![0usize; max_r];
    for i in 0..trials {
        #[allow(clippy::needless_range_loop)]
        for rep in 0..max_r {
            let mut backend = make();
            let mut ch = OsChallenger::new();
            let (mut e, _iop_verdict) = execute_recording(
                &statement(c),
                &mut backend,
                oracle,
                &cfg,
                &fixture.cheat,
                &mut ch,
            );
            if fixture.corrupt_one_query_answer {
                let victim = mixed_index((i as u64) << 20 ^ rep as u64, e.occurrences.len());
                let mut bad = e.occurrences[victim].answer.0.clone();
                bad[0] = bad[0].add(F128::ONE);
                e.occurrences[victim].answer = Symbol(bad);
            }
            let ok = audit_for(&backend, &e, &checker).accepted();
            if !ok {
                break;
            }
            survived[rep] += 1;
        }
    }

    let p = fixture.prediction.accept_rate;
    (1..=max_r)
        .map(|r| (r, Rate::new(survived[r - 1], trials), p.powi(r as i32)))
        .collect()
}

pub const MNIST_1024_REPETITIONS: usize = 9;

pub fn mnist_case(r: usize) -> (Case, rmip_frontend::mnist::SparseTableOracle) {
    let label: &'static str = Box::leak(format!("MNIST R={r}").into_boxed_str());
    crate::mnist_bench::sparse_mnist_case(r, label)
}
