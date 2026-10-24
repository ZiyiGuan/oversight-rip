use crate::harness::Case;
use crate::party_metrics::MeasuredChallenger;
use rmip_backend_interleaved::{Ligero, Ligero64, Shape};
use rmip_backend_ladder::SingleShotTarget;
use rmip_field::OpeningField;
use rmip_frontend::bits::{BitSlice, BitVec};
use rmip_frontend::oracle::Oracle;
use rmip_iop::backend::{
    div_error_upward, pow_error_upward, sum_errors_upward, AuditMode, CommitmentBackend,
};
use rmip_iop::driver::{Cheat, Config, Statement};
use rmip_iop::driver::{Execution, Occurrence};
use rmip_iop::metrics::{self, Role, Session, Stats};
use rmip_iop::transcript::{OracleId, OsChallenger, RawChallenger, Transcript};
use rmip_mip::{Audit, PCheck};
use rmip_mip_prover::{LazyBackend, OracleMeta};
use rmip_packed::driver::{packed_witness, PackedCheat};
use rmip_packed::statement::PackedStatement;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::io;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Lookup {
    pub query: Vec<u8>,
    pub elapsed: Duration,
}

pub struct ObservedOracle<'a, O> {
    oracle: &'a O,
    calls: RefCell<Vec<Lookup>>,
    record: bool,
    count: std::cell::Cell<u64>,
    elapsed: std::cell::Cell<Duration>,
}

impl<'a, O> ObservedOracle<'a, O> {
    pub fn new(oracle: &'a O) -> Self {
        Self {
            oracle,
            calls: RefCell::new(Vec::new()),
            record: true,
            count: std::cell::Cell::new(0),
            elapsed: std::cell::Cell::new(Duration::ZERO),
        }
    }

    pub fn counting(oracle: &'a O) -> Self {
        Self {
            record: false,
            ..Self::new(oracle)
        }
    }

    pub fn into_calls(self) -> Vec<Lookup> {
        self.calls.into_inner()
    }

    pub fn totals(&self) -> (u64, Duration) {
        (self.count.get(), self.elapsed.get())
    }
}

impl<O: Oracle> Oracle for ObservedOracle<'_, O> {
    fn query(&self, query: &BitSlice) -> bool {
        if !self.record {
            self.count.set(self.count.get() + 1);
            return self.oracle.query(query);
        }
        let start = Instant::now();
        let answer = self.oracle.query(query);
        let elapsed = start.elapsed();
        let _diagnostics = metrics::enter(Role::Diagnostic);
        self.count.set(self.count.get() + 1);
        self.elapsed.set(self.elapsed.get() + elapsed);
        self.calls.borrow_mut().push(Lookup {
            query: BitVec::from_bools(query.to_vec()).to_bytes_be(),
            elapsed,
        });
        answer
    }

    fn query_word(&self, u: u64, width: usize) -> bool {
        if !self.record {
            self.count.set(self.count.get() + 1);
            return self.oracle.query_word(u, width);
        }
        let start = Instant::now();
        let answer = self.oracle.query_word(u, width);
        let elapsed = start.elapsed();
        let _diagnostics = metrics::enter(Role::Diagnostic);
        self.count.set(self.count.get() + 1);
        self.elapsed.set(self.elapsed.get() + elapsed);
        let bits: Vec<bool> = (0..width).map(|k| (u >> k) & 1 == 1).collect();
        self.calls.borrow_mut().push(Lookup {
            query: BitVec::from_bools(bits).to_bytes_be(),
            elapsed,
        });
        answer
    }
}

pub fn statement(case: &Case) -> Statement<'_> {
    Statement {
        inst: &case.inst,
        sys: &case.sys,
        layout: &case.layout,
        circuit: &case.circuit,
        dc: &case.dc,
        holo: crate::harness::matrix_evaluation(case),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MipBackend {
    Flat,
    Ladder,
}

impl MipBackend {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "flat" | "interleaved-rs" => Some(Self::Flat),
            "ladder" | "bundled-ladder" => Some(Self::Ladder),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Flat => "interleaved-rs",
            Self::Ladder => "bundled-ladder",
        }
    }
}

pub const AMPLIFIED_TARGET: f64 = 1.0 / 1_048_576.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Soundness {
    #[default]
    Amplified,
    SingleShot,
    SingleShotAt(u32),
}

impl Soundness {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "amplified" | "repeated" => Some(Self::Amplified),
            "single-shot" | "single_shot" | "single" => Some(Self::SingleShot),
            _ => {
                let bits = s
                    .strip_prefix("single-shot-")
                    .or_else(|| s.strip_prefix("single_shot_"))?;
                let bits: u32 = bits.parse().ok().filter(|&bits| bits > 0)?;
                Some(Self::at(bits))
            }
        }
    }
    pub fn at(bits: u32) -> Self {
        if bits == SingleShotTarget::DEFAULT.bits {
            Self::SingleShot
        } else {
            Self::SingleShotAt(bits)
        }
    }
    pub fn single_shot_target(self) -> Option<SingleShotTarget> {
        match self {
            Self::Amplified => None,
            Self::SingleShot => Some(SingleShotTarget::DEFAULT),
            Self::SingleShotAt(bits) => Some(SingleShotTarget { bits }),
        }
    }
    pub fn target(self) -> f64 {
        match self.single_shot_target() {
            Some(t) if t.bits != 20 => 2f64.powi(-(t.bits as i32)),
            _ => AMPLIFIED_TARGET,
        }
    }
    pub fn is_single_shot(self) -> bool {
        self.single_shot_target().is_some()
    }
    pub fn name(self) -> String {
        match self.single_shot_target() {
            None => "amplified".to_string(),
            Some(target) if target == SingleShotTarget::DEFAULT => "single-shot".to_string(),
            Some(target) => format!("single-shot-{}", target.bits),
        }
    }
    pub fn local_budget(self) -> f64 {
        match self.single_shot_target() {
            None => 0.125,
            Some(target) => target.local_budget(),
        }
    }
    pub fn lambda_bits(self) -> f64 {
        match self.single_shot_target() {
            None => 4.0,
            Some(target) => target.lambda_bits(),
        }
    }
    pub fn collision_budget(self) -> Option<f64> {
        self.single_shot_target()
            .map(SingleShotTarget::collision_budget)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IopBody {
    #[default]
    Flat,
    Packed64,
    Packed128,
}

impl IopBody {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "flat" => Some(Self::Flat),
            "packed" | "packed64" | "packed-64" => Some(Self::Packed64),
            "packed128" | "packed-128" => Some(Self::Packed128),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Packed64 => "packed-64",
            Self::Packed128 => "packed-128",
        }
    }
    pub fn bits(self) -> Option<usize> {
        match self {
            Self::Flat => None,
            Self::Packed64 => Some(64),
            Self::Packed128 => Some(128),
        }
    }
}

fn label_is_digit(x: &BitSlice) -> bool {
    let y: usize = (0..rmip_frontend::mnist::LABEL_BITS)
        .map(|k| usize::from(x[k]) << k)
        .sum();
    y < 10
}

pub fn packed_statement(case: &Case) -> PackedStatement<'_> {
    PackedStatement::new(
        &case.inst,
        &case.circuit,
        case.majority_threshold as u64,
        label_is_digit(case.inst.x.as_slice()),
    )
}

pub fn body_trace_log(case: &Case, body: IopBody) -> usize {
    match body.bits() {
        None => case.layout.nu,
        Some(bits) => {
            packed_statement(case)
                .tree
                .layout(bits.trailing_zeros() as usize)
                .1
        }
    }
}

pub fn body_trace_len(case: &Case, body: IopBody) -> usize {
    match body {
        IopBody::Flat => case.layout.n_trace,
        _ => 1 << body_trace_log(case, body),
    }
}

fn config_for(case: &Case, soundness: Soundness) -> Config {
    Config::for_instance(&case.inst, soundness.local_budget())
}

fn flat_shape(case: &Case, soundness: Soundness, body: IopBody) -> Shape {
    Shape::balanced(body_trace_log(case, body), 4, soundness.lambda_bits())
}

pub fn opening_kappa(body: IopBody) -> usize {
    match body {
        IopBody::Packed64 => 64,
        _ => 128,
    }
}

fn ladder_collision_total(ls: rmip_backend_ladder::LadderShape, kappa: usize) -> f64 {
    rmip_backend_ladder::collision_total(ls, kappa)
}

fn ladder_shape(
    case: &Case,
    soundness: Soundness,
    body: IopBody,
    depth: Option<usize>,
) -> rmip_backend_ladder::LadderShape {
    let nu = body_trace_log(case, body);
    match soundness.single_shot_target() {
        Some(t) => t
            .ladder_shape(nu, depth, opening_kappa(body))
            .unwrap_or_else(|| {
                panic!(
                    "the single-shot target 2^-{} is not reachable at this shape: the per-table field terms exceed it",
                    t.bits
                )
            }),
        None => {
            let lambda = soundness.lambda_bits();
            match depth {
                None => rmip_backend_ladder::LadderShape::for_trace(nu, lambda),
                Some(d) => rmip_backend_ladder::LadderShape::for_trace_at_depth(nu, lambda, d),
            }
        }
    }
}

pub fn parameters(case: &Case) -> Value {
    parameters_for(case, MipBackend::Ladder)
}

pub fn parameters_for(case: &Case, which: MipBackend) -> Value {
    parameters_opts(case, which, Soundness::Amplified)
}

pub fn parameters_opts(case: &Case, which: MipBackend, soundness: Soundness) -> Value {
    parameters_body(case, which, soundness, IopBody::Flat)
}

pub fn parameters_body(
    case: &Case,
    which: MipBackend,
    soundness: Soundness,
    body: IopBody,
) -> Value {
    parameters_body_at_depth(case, which, soundness, body, None)
}

pub fn parameters_body_at_depth(
    case: &Case,
    which: MipBackend,
    soundness: Soundness,
    body: IopBody,
    depth: Option<usize>,
) -> Value {
    assert!(
        !(which == MipBackend::Flat && soundness.is_single_shot()),
        "the flat backend's one-of audit cannot reach 2^-20 in one execution; single-shot needs the ladder"
    );
    let cfg = config_for(case, soundness);
    let shape = flat_shape(case, soundness, body);
    let trace_len = body_trace_len(case, body);
    let claims = match body {
        IopBody::Flat => cfg.k_rel + cfg.t * (case.inst.b_max() + 1),
        _ => 1,
    };
    let (params, eps_col_value, eps_col_total, pins, ladder, lambda_report, queries_report) =
        match which {
            MipBackend::Flat => (
                match body {
                    IopBody::Packed64 => Ligero64::lazy(shape).public_params(trace_len, claims),
                    _ => Ligero::lazy(shape).public_params(trace_len, claims),
                },
                Value::Null,
                Value::Null,
                Value::Null,
                Value::Null,
                soundness.lambda_bits(),
                shape.queries(),
            ),
            MipBackend::Ladder => {
                rmip_backend_ladder::require_unpadded(&cfg);
                let kappa = opening_kappa(body);
                let ls = ladder_shape(case, soundness, body, depth);
                let params = match body {
                    IopBody::Packed64 => {
                        rmip_backend_ladder::Ladder64::lazy(ls).public_params(trace_len, claims)
                    }
                    _ => rmip_backend_ladder::Ladder::lazy(ls).public_params(trace_len, claims),
                };
                (
                    params,
                    json!(rmip_backend_ladder::eps_col_at(ls, 0, kappa)),
                    json!(ladder_collision_total(ls, kappa)),
                    json!(ls.pins),
                    json!({ "depth": ls.depth(), "log_m0": ls.log_m0, "tables": ls.schedule().tables.len() }),
                    ls.lambda_bits,
                    ls.queries(),
                )
            }
        };
    let eps_open_flat = match body {
        IopBody::Packed64 => Ligero64::lazy(shape).public_params(trace_len, claims),
        _ => Ligero::lazy(shape).public_params(trace_len, claims),
    }
    .eps_open;
    let miss = div_error_upward(
        (case.eps_rob.den - case.eps_rob.num) as f64,
        case.eps_rob.den as f64,
    );
    let beta = pow_error_upward(miss, cfg.t);
    let eps_rel = match body.bits() {
        None => rmip_iop::backend::relation_error(case.sys.mu, case.layout.nu),
        Some(bits) => rmip_packed::statement::relation_error(&packed_statement(case).tree, bits),
    };
    let eps_idx = case.holo.as_ref().map_or(0.0, |h| h.index.error_terms());
    let s_iop = sum_errors_upward(&[params.eps_bind, eps_rel, params.eps_open, beta, eps_idx]);
    let q_long = cfg.audit_query_count(params.q_long);
    let denominator = rmip_mip::audit_denominator(params.audit, q_long);
    let sigma = soundness.target();
    let repetitions = rmip_mip::repetitions_for(sigma, s_iop, denominator);
    if soundness.is_single_shot() {
        assert_eq!(
            repetitions, 1,
            "single-shot budgets must meet the target {sigma:e} in one execution; s_iop = {s_iop:e}"
        );
    }
    let embedding = if case.closed_form.is_some() {
        "closed_form"
    } else if case.holo.is_some() {
        "verifier_owned_index"
    } else {
        "matrix_scan"
    };
    let proof_status = match ladder.get("depth").and_then(Value::as_u64) {
        None => "the flat opening: the one-of audit of the committed trace",
        Some(0) | Some(1) => "proved: the ladder at depth 0 and 1 under the sequential audit",
        Some(_) => "CONDITIONAL: the ladder's opening lemma at depth >= 2 is an open obligation; every error figure here assumes it",
    };
    json!({
        "backend": params.name,
        "main_prover": match body {
            IopBody::Flat => "rmip_mip_prover::execute_recording",
            _ => "rmip_mip_prover::execute_recording_packed",
        },
        "body": body.name(), "trace_len": trace_len, "trace_log": body_trace_log(case, body),
        "eps_col": eps_col_value, "eps_col_all_levels": eps_col_total, "eps_open_flat": eps_open_flat,
        "rate_inverse": 4, "lambda_bits": lambda_report,
        "soundness": soundness.name(), "opening_columns": queries_report, "pins": pins,
        "ladder": ladder,
        "proof_status": proof_status,
        "embedding": embedding,
        "audit": match params.audit { AuditMode::OneOf => "one_of", AuditMode::Sequential => "sequential" },
        "local_error_budget": soundness.local_budget(), "amplified_error_budget": sigma,
        "eps_bind": params.eps_bind, "eps_relation": eps_rel,
        "eps_open": params.eps_open, "eps_index": eps_idx, "local_miss_bound": beta,
        "s_iop": s_iop, "s_two_prover_one": rmip_mip::one_shot_bound(s_iop, denominator),
        "base_t": cfg.t, "q_long": q_long, "claim_count": claims,
        "amplification_repetitions": repetitions,
        "amplified_bound": rmip_mip::repeated_bound(s_iop, denominator, repetitions),
        "configured_total_Q_O": repetitions.checked_mul(cfg.t).expect("Q_O overflow"),
        "randomness": "OsChallenger; separate verifier-private audit challenger",
        "ideal_coin_caveat": "a ChaCha20 stream keyed from the OS CSPRNG is a computational implementation; theorem assumes ideal independent coins"
    })
}

pub fn measure_communication(
    transcript: &Transcript,
    execution: &Execution,
    audited: &Occurrence,
    checker_answer: &rmip_iop::backend::Symbol<rmip_field::F128>,
    posted: &[OracleMeta],
) -> io::Result<crate::wire::CommunicationBytes> {
    let domain_of = |oracle: OracleId| posted.iter().find(|m| m.id == oracle).map(|m| m.len);
    crate::wire::measure_execution_with_domains(
        transcript,
        execution,
        audited,
        checker_answer,
        domain_of,
    )
}

const COMMUNICATION_CONVENTION_V1: &str = "actual messages serialized with rmip-wire-v1 framing in both directions; includes all public coin draws, posting notices, main questions/answers and the one checker question/answer; private audit index and coins stay private; in-process protocol bytes exclude OS/network transport headers";

fn communication_convention(communication: crate::wire::CommunicationBytes) -> String {
    let version = communication.widest_version();
    if version == crate::wire::VERSION {
        return COMMUNICATION_CONVENTION_V1.to_string();
    }
    let (threshold, width) = match version {
        crate::wire::VERSION_WIDE => ("2^64", "sixteen-byte"),
        crate::wire::VERSION_WIDE256 => ("2^128", "thirty-two-byte"),
        _ => ("2^256", "128-byte"),
    };
    format!("actual messages serialized with rmip-wire-v1 framing in both directions, except that main and checker questions to an oracle whose posted domain exceeds {threshold} carry a {width} coordinate under rmip-wire-v{version} frames (narrower domains keep the narrower frames); includes all public coin draws, posting notices, main questions/answers and the one checker question/answer; private audit index and coins stay private; in-process protocol bytes exclude OS/network transport headers")
}

pub struct ProtocolRun {
    pub measurements: Value,
    pub verifier_calls: Vec<Lookup>,
    pub main_prover_calls: Vec<Lookup>,
}

fn party_stats(stats: Stats, heap_tracking: bool) -> Value {
    json!({
        "runtime_ms": stats.elapsed.as_secs_f64() * 1000.0,
        "peak_live_heap_bytes": heap_tracking.then_some(stats.peak_heap_bytes),
        "live_heap_bytes_at_end": heap_tracking.then_some(stats.live_heap_bytes),
        "allocated_heap_bytes": heap_tracking.then_some(stats.allocated_heap_bytes),
    })
}

pub fn run_protocol<O: Oracle>(
    case: &Case,
    oracle: &O,
    amplified: bool,
) -> Result<ProtocolRun, String> {
    run_protocol_with(case, oracle, amplified, MipBackend::Ladder)
}

pub fn run_protocol_with<O: Oracle>(
    case: &Case,
    oracle: &O,
    amplified: bool,
    which: MipBackend,
) -> Result<ProtocolRun, String> {
    run_protocol_opts(
        case,
        oracle,
        RunOptions {
            amplified,
            record_lookups: true,
            soundness: Soundness::Amplified,
            body: IopBody::Flat,
            ladder_depth: None,
        },
        which,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunOptions {
    pub amplified: bool,
    pub record_lookups: bool,
    pub soundness: Soundness,
    pub body: IopBody,
    pub ladder_depth: Option<usize>,
}

pub fn run_protocol_opts<O: Oracle>(
    case: &Case,
    oracle: &O,
    opts: RunOptions,
    which: MipBackend,
) -> Result<ProtocolRun, String> {
    let params =
        parameters_body_at_depth(case, which, opts.soundness, opts.body, opts.ladder_depth);
    let shape = flat_shape(case, opts.soundness, opts.body);
    match which {
        MipBackend::Flat => match opts.body {
            IopBody::Packed64 => run_generic(case, oracle, opts, params, &|| Ligero64::lazy(shape)),
            _ => run_generic(case, oracle, opts, params, &|| Ligero::lazy(shape)),
        },
        MipBackend::Ladder => {
            let ls = ladder_shape(case, opts.soundness, opts.body, opts.ladder_depth);
            match opts.body {
                IopBody::Packed64 => run_generic(case, oracle, opts, params, &|| {
                    rmip_backend_ladder::Ladder64::lazy(ls)
                }),
                _ => run_generic(case, oracle, opts, params, &|| {
                    rmip_backend_ladder::Ladder::lazy(ls)
                }),
            }
        }
    }
}

fn run_generic<O: Oracle, B: LazyBackend<Field = rmip_field::F128>>(
    case: &Case,
    oracle: &O,
    opts: RunOptions,
    params: Value,
    make: &dyn Fn() -> B,
) -> Result<ProtocolRun, String> {
    let RunOptions {
        amplified,
        record_lookups,
        soundness,
        body,
        ladder_depth: _,
    } = opts;
    let required = params["amplification_repetitions"].as_u64().unwrap() as usize;
    let count = if amplified { required } else { 1 };
    let word_bytes = crate::wire::ELEMENT_BYTES;
    let native_word_bytes = opening_kappa(body) / 8;
    let cfg = config_for(case, soundness);
    let st = statement(case);
    let packed = match body {
        IopBody::Flat => packed_statement(case),
        _ => packed_statement(case).with_query_plan(),
    };
    match body {
        IopBody::Flat => {}
        IopBody::Packed64 => {
            let _ = <rmip_field::F64 as rmip_packed::PackedField>::skip_tables();
        }
        IopBody::Packed128 => {
            let _ = <rmip_field::F128 as rmip_packed::PackedField>::skip_tables();
        }
    }
    let session = Session::start();
    let (checker, checker_setup, checker_preparation_calls) = {
        let _role = metrics::enter(Role::CheckerProver);
        let checker_start = Instant::now();
        let mut checker_backend = make();
        let (advice, calls): (Vec<rmip_field::F128>, usize) = match body {
            IopBody::Flat => (case.honest_trace.clone(), 0),
            IopBody::Packed64 => {
                let (_, _, w) =
                    packed_witness::<rmip_field::F64, _>(&packed, oracle, &PackedCheat::default());
                (rmip_field::F64::to_words(&w), case.inst.ell)
            }
            IopBody::Packed128 => {
                let (_, _, w) =
                    packed_witness::<rmip_field::F128, _>(&packed, oracle, &PackedCheat::default());
                (w, case.inst.ell)
            }
        };
        checker_backend.commit(&advice, &mut OsChallenger::new());
        (PCheck::new(checker_backend), checker_start.elapsed(), calls)
    };
    let audit_mode = checker.backend().params(1).audit;
    let mut raw = Vec::with_capacity(count);
    let mut verifier_calls = Vec::new();
    let mut main_prover_calls = Vec::new();
    let mut q_o_total = 0usize;
    let mut total_short_bytes = 0usize;
    let mut total_answer_bytes = 0usize;
    let mut total_answer_native_bytes = 0usize;
    let mut checker_question_bytes = 0usize;
    let mut checker_answer_bytes = 0usize;
    let mut checker_answer_native_bytes = 0usize;
    let mut questions_expected = 0usize;
    let mut communication = crate::wire::CommunicationBytes::default();
    let mut all_accepted = true;
    let start = Instant::now();
    for repetition in 0..count {
        let before = session.report();
        let observed = if record_lookups {
            ObservedOracle::new(oracle)
        } else {
            ObservedOracle::counting(oracle)
        };
        let mut main = {
            let _role = metrics::enter(Role::MainProver);
            make()
        };
        let mut public_coins = {
            let _role = metrics::enter(Role::Verifier);
            MeasuredChallenger::new(OsChallenger::new())
        };
        let t0 = Instant::now();
        let (execution, verdict) = match body {
            IopBody::Flat => rmip_mip_prover::execute_recording(
                &st,
                &mut main,
                &observed,
                &cfg,
                &Cheat::default(),
                &mut public_coins,
            ),
            IopBody::Packed64 => {
                rmip_mip_prover::execute_recording_packed::<rmip_field::F64, _, _>(
                    &packed,
                    &mut main,
                    &observed,
                    &cfg,
                    &PackedCheat::default(),
                    &mut public_coins,
                )
            }
            IopBody::Packed128 => {
                rmip_mip_prover::execute_recording_packed::<rmip_field::F128, _, _>(
                    &packed,
                    &mut main,
                    &observed,
                    &cfg,
                    &PackedCheat::default(),
                    &mut public_coins,
                )
            }
        };
        let joint_time = t0.elapsed();
        let posted = {
            let _diagnostics = metrics::enter(Role::Diagnostic);
            main.oracle_meta()
        };
        let ta = Instant::now();
        let (audit, checker_answers): (Audit, Vec<rmip_iop::backend::Symbol<rmip_field::F128>>) = {
            let _role = metrics::enter(Role::Verifier);
            match audit_mode {
                AuditMode::OneOf => {
                    let mut private_audit_coins = MeasuredChallenger::new(OsChallenger::new());
                    let (a, answer) = rmip_mip::checker::audit_with_answer(
                        &execution,
                        &checker,
                        &mut private_audit_coins,
                    );
                    (a, vec![answer])
                }
                AuditMode::Sequential => rmip_mip::audit_all_with_answers(&execution, &checker),
            }
        };
        let audit_time = ta.elapsed();
        {
            let _role = metrics::enter(Role::MainProver);
            drop(main);
        }
        let after = session.report();
        let _diagnostics = metrics::enter(Role::Diagnostic);
        let accepted = verdict.is_ok() && audit.accepted();
        all_accepted &= accepted;
        let (made, _) = observed.totals();
        let mut calls = observed.into_calls();
        if made as usize != case.inst.ell + execution.oracle_calls
            || execution.oracle_calls != cfg.t
        {
            return Err(
                "oracle accounting: expected ell prover calls followed by exactly t verifier calls"
                    .into(),
            );
        }
        if execution.q_long() != params["q_long"].as_u64().unwrap() as usize {
            return Err("protocol: q_long changed from public configuration".into());
        }
        q_o_total += execution.oracle_calls;
        if record_lookups {
            let verifier = calls.split_off(case.inst.ell);
            main_prover_calls.extend(calls);
            verifier_calls.extend(verifier);
        }
        total_short_bytes += execution.short_bytes;
        total_answer_bytes += execution.a_long() as usize * word_bytes;
        total_answer_native_bytes += execution.a_long() as usize * native_word_bytes;
        let domain_of = |oracle: OracleId| posted.iter().find(|m| m.id == oracle).map(|m| m.len);
        let (wire, question_bytes, answer_bytes, audited_json) = match audit_mode {
            AuditMode::OneOf => {
                let audited = match audit {
                    Audit::Accepted { audited }
                    | Audit::UnderlyingRejected { audited }
                    | Audit::Inconsistent { audited } => audited,
                };
                let occurrence = &execution.occurrences[audited];
                let wire = measure_communication(
                    public_coins.transcript(),
                    &execution,
                    occurrence,
                    &checker_answers[0],
                    &posted,
                )
                .map_err(|e| format!("communication serialization: {e}"))?;
                let coordinate_bytes = usize::from(wire.widest_coordinate_bytes());
                let question_bytes = 8 + coordinate_bytes + occurrence.prefix.len();
                let answer_bytes = checker_answers[0].0.len() * word_bytes;
                questions_expected += 1;
                (wire, question_bytes, answer_bytes, json!(audited))
            }
            AuditMode::Sequential => {
                let wire = crate::wire::measure_execution_sequential_with_domains(
                    public_coins.transcript(),
                    &execution,
                    &checker_answers,
                    domain_of,
                )
                .map_err(|e| format!("communication serialization: {e}"))?;
                let coordinate_bytes = usize::from(wire.widest_coordinate_bytes());
                let question_bytes: usize = execution
                    .occurrences
                    .iter()
                    .take(checker_answers.len())
                    .map(|o| 8 + coordinate_bytes + o.prefix.len())
                    .sum();
                let answer_bytes: usize =
                    checker_answers.iter().map(|a| a.0.len() * word_bytes).sum();
                questions_expected += checker_answers.len();
                (wire, question_bytes, answer_bytes, json!("all"))
            }
        };
        communication += wire;
        checker_question_bytes += question_bytes;
        checker_answer_bytes += answer_bytes;
        checker_answer_native_bytes += answer_bytes / word_bytes * native_word_bytes;
        raw.push(json!({
            "base_repetition": repetition, "accepted": accepted,
            "joint_main_verifier_ms": joint_time.as_secs_f64()*1000.0,
            "checker_and_private_audit_ms": audit_time.as_secs_f64()*1000.0,
            "main_prover_runtime_ms": (after.main_prover.elapsed-before.main_prover.elapsed).as_secs_f64()*1000.0,
            "verifier_runtime_ms": (after.verifier.elapsed-before.verifier.elapsed).as_secs_f64()*1000.0,
            "checker_prover_runtime_ms": (after.checker_prover.elapsed-before.checker_prover.elapsed).as_secs_f64()*1000.0,
            "communication": wire.to_json(),
            "verifier_partial_sumcheck_opening_ms": execution.t_verify.as_secs_f64()*1000.0,
            "verifier_opening_check_ms": execution.t_open.as_secs_f64()*1000.0,
            "verifier_phase_cumulative_ms": execution.verifier_phases.iter().map(|(n, d)| json!([n, d.as_secs_f64()*1000.0])).collect::<Vec<_>>(),
            "prover_phase_cumulative_ms": execution.prover_phases.iter().map(|(n, d)| json!([n, d.as_secs_f64()*1000.0])).collect::<Vec<_>>(),
            "prover_piop_phases_ms": execution.prover_piop_phases.iter().map(|(n, level, d)| json!([n, level, d.as_secs_f64()*1000.0])).collect::<Vec<_>>(),
            "matrix_component_ms": execution.t_mat.as_secs_f64()*1000.0,
            "matrix_component_is_joint_prove_verify": matches!(body, IopBody::Flat) && (case.holo.is_some() || case.closed_form.is_some()),
            "holographic_matrix_evaluation_verified": matches!(body, IopBody::Flat) && case.holo.is_some(),
            "matrix_component_convention": if !matches!(body, IopBody::Flat) {
                "packed statement: no separate matrix block, so matrix_component_ms is 0; the \
                 packed lincheck checks the sparse block weights inside the verifier's own time"
            } else if case.closed_form.is_some() {
                "closed-form statement: the block brackets the main prover's and the \
                 verifier's closed-form evaluations of the three matrix MLEs, one each, \
                 so the verifier's share is half of this total"
            } else if case.holo.is_some() {
                "embedded statement: the block brackets prove_frames (main prover) \
                 and verify_frames (verifier); the two shares are attributed to \
                 their roles in party_measurements, so this total is not a \
                 verifier figure"
            } else {
                "scan statement: the block is the verifier's own O(M + N + Z) \
                 evaluation of the three matrix MLEs"
            },
            "actual_verifier_oracle_calls": execution.oracle_calls,
            "q_long": execution.q_long(), "short_bytes": execution.short_bytes,
            "long_answer_bytes": execution.a_long() as usize * word_bytes,
            "verifier_to_checker_question_bytes": question_bytes,
            "checker_to_verifier_answer_bytes": answer_bytes,
            "checker_questions": checker_answers.len(),
            "audited": audited_json,
            "public_rounds": execution.rounds, "public_field_coins": execution.coins,
            "audit": format!("{audit:?}"), "driver_rejection": verdict.err().map(|e|format!("{e:?}"))
        }));
        if (repetition + 1) % 100 == 0 {
            eprintln!(
                "measured: completed {}/{} actual two-prover executions",
                repetition + 1,
                count
            );
        }
    }
    if checker.questions_asked() != questions_expected {
        return Err("checker boundary: the checker was asked a different number of questions than the audit accounts for".into());
    }
    let elapsed = start.elapsed();
    let report = session.finish();
    Ok(ProtocolRun {
        measurements: json!({
            "status": if all_accepted { "accepted" } else { "rejected" },
            "execution_kind": if amplified { "actual_amplified_two_prover" } else { "actual_base_two_prover_smoke" },
            "backend": params["backend"].clone(), "audit_mode": params["audit"].clone(),
            "body": body.name(),
            "configured_repetitions": required, "executed_repetitions": count,
            "actual_total_Q_O": q_o_total,
            "oracle_lookups_recorded": record_lookups,
            "joint_execution_wall_ms": elapsed.as_secs_f64()*1000.0,
            "checker_preparation_ms": checker_setup.as_secs_f64()*1000.0,
            "checker_preparation_oracle_calls": checker_preparation_calls,
            "checker_advice": match body {
                IopBody::Flat => "immutable honest trace retained from separately reported case preparation",
                _ => "the packed words derived from the table in the checker's preparation, as the opening's transcript words",
            },
            "prover_runtime_ms": report.main_prover.elapsed.as_secs_f64()*1000.0,
            "verifier_runtime_ms": report.verifier.elapsed.as_secs_f64()*1000.0,
            "prover_peak_memory_bytes": report.heap_tracking.then_some(report.main_prover.peak_heap_bytes),
            "party_measurements": {
                "main_prover": party_stats(report.main_prover, report.heap_tracking),
                "verifier": party_stats(report.verifier, report.heap_tracking),
                "checker_prover_including_preparation": party_stats(report.checker_prover, report.heap_tracking),
                "diagnostic": party_stats(report.diagnostic, report.heap_tracking),
                "unattributed_runner": party_stats(report.unattributed, report.heap_tracking),
            },
            "measurement_convention": "disjoint actual operation scopes; diagnostic replay and ledger construction excluded from parties; raw instrumented elapsed time includes timer/allocator overhead; prover aliases mean main prover; checker includes one separately reported preparation; joint_execution_wall_ms covers the repetitions only and excludes that preparation, so the party runtimes sum to the joint wall plus checker_preparation_ms",
            "heap_tracking": report.heap_tracking,
            "memory_convention": "measured requested live Rust heap payload by allocation origin across the full amplification; outgoing buffers retained until freed; excludes stack, RSS, allocator bookkeeping, and public case/shared advice allocated before measurement; per-party peaks are not additive",
            "main_short_message_bytes": total_short_bytes, "main_long_answer_bytes": total_answer_bytes,
            "main_long_answer_native_bytes": total_answer_native_bytes,
            "verifier_to_checker_question_bytes": checker_question_bytes,
            "checker_to_verifier_answer_bytes": checker_answer_bytes,
            "checker_to_verifier_answer_native_bytes": checker_answer_native_bytes,
            "payload_convention": "answer bytes count every answered element at its serialized width, the 16-byte transport word of the wire frames (wire::ELEMENT_BYTES), so they agree with long_answer_bytes and the serialized totals; the *_native_bytes figures count the same elements at the opening field's width (kappa/8), what an 8-byte frame for the packed-64 body would carry",
            "accounted_prover_payload_bytes": total_short_bytes+total_answer_bytes+checker_answer_bytes,
            "accounted_partial_two_way_payload_bytes": total_short_bytes+total_answer_bytes+checker_answer_bytes+checker_question_bytes,
            "communication": communication.to_json(),
            "total_serialized_communication_bytes": communication.total_bytes(),
            "communication_convention": communication_convention(communication),
            "checker_questions": checker.questions_asked(), "raw_base_executions": raw,
        }),
        verifier_calls,
        main_prover_calls,
    })
}
