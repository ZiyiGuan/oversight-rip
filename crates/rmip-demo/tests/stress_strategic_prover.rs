use rmip_backend_interleaved::backend::{Ligero, LigeroVerifier, Shape, CODEWORD_ORACLE};
use rmip_backend_interleaved::code;
use rmip_backend_trace::TRACE_ORACLE;
use rmip_demo::fixtures;
use rmip_demo::harness::{mnist_case, Case};
use rmip_demo::weakened::WeakenedTrace;
use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::Instance;
use rmip_iop::backend::{Claim, CommitmentBackend, Coord, Symbol};
use rmip_iop::batch::{BatchMessageProver, HonestBatchProver};
use rmip_iop::common_protocol::verify_interactive_with_opening;
use rmip_iop::driver::{committed_trace, Config, Execution, Occurrence, Statement, DUMMY_ORACLE};
use rmip_iop::lincheck::Coefficients;
use rmip_iop::local_protocol::LocalMessageProver;
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::relation_protocol::{HonestRelationProver, RelationMessageProver};
use rmip_iop::sumcheck::RoundMessageProver;
use rmip_iop::transcript::{Entry, OracleId, OsChallenger, RawChallenger, SeededChallenger};
use rmip_mip::{audit_one, one_shot_bound, PCheck, SharedAdvice};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Deviation {
    None,
    TrueAnswer,
    RewriteLabel,
}

const QUERY_BITS: usize = 64;
const MASK_ID_BIT: usize = 24;
const LABEL_BIT: usize = 56;

fn byte_at(bits: &[bool], i: usize) -> u8 {
    (0..8).fold(0u8, |acc, k| (acc << 1) | u8::from(bits[i * 8 + k]))
}

fn rewrite_label(block: &mut [F128], table: &[u8], want: bool) {
    assert_eq!(
        block.len(),
        QUERY_BITS + 1,
        "one answer after the 64 query wires"
    );
    let bits: Vec<bool> = block[..QUERY_BITS]
        .iter()
        .map(|v| *v == F128::ONE)
        .collect();
    let mask = u32::from_le_bytes([
        byte_at(&bits, MASK_ID_BIT / 8),
        byte_at(&bits, MASK_ID_BIT / 8 + 1),
        byte_at(&bits, MASK_ID_BIT / 8 + 2),
        byte_at(&bits, MASK_ID_BIT / 8 + 3),
    ]) as usize;
    let predicted = table[mask];
    let label = if want { predicted } else { predicted ^ 1 };
    for k in 0..8 {
        block[LABEL_BIT + k] = F128::from_bool((label >> (7 - k)) & 1 == 1);
    }
}

struct Strategic<'a> {
    relation: HonestRelationProver<'a>,
    batching: HonestBatchProver,
    inst: &'a Instance,
    table: &'a [F128],
    oracle: &'a SparseTableOracle,
    deviation: Deviation,
    lies: usize,
}

impl<'a> Strategic<'a> {
    fn new(
        st: &'a Statement<'_>,
        inst: &'a Instance,
        table: &'a [F128],
        oracle: &'a SparseTableOracle,
        deviation: Deviation,
    ) -> Self {
        Self {
            relation: HonestRelationProver::new(st, table),
            batching: HonestBatchProver::for_table(table),
            inst,
            table,
            oracle,
            deviation,
            lies: 0,
        }
    }
}

impl LocalMessageProver for Strategic<'_> {
    fn answer_blocks(&mut self, _sampling_coins: &[F128], sampled: &[usize]) -> Vec<Vec<F128>> {
        sampled
            .iter()
            .map(|&index| {
                let coords = self.inst.loc_row(index);
                let mut block: Vec<F128> = coords.iter().map(|&c| self.table[c]).collect();
                let last = block.len() - 1;
                let bits: Vec<bool> = block[..last].iter().map(|v| *v == F128::ONE).collect();
                let truth = self.oracle.query(&bits);
                let carried = block[last] == F128::ONE;
                if truth == carried {
                    return block;
                }
                match self.deviation {
                    Deviation::None => {}
                    Deviation::TrueAnswer => {
                        block[last] = F128::from_bool(truth);
                        self.lies += 1;
                    }
                    Deviation::RewriteLabel => {
                        rewrite_label(&mut block, &self.oracle.predicted, carried);
                        debug_assert_eq!(
                            self.oracle.query(
                                &block[..last]
                                    .iter()
                                    .map(|v| *v == F128::ONE)
                                    .collect::<Vec<_>>()
                            ),
                            carried
                        );
                        self.lies += 1;
                    }
                }
                block
            })
            .collect()
    }
}

impl RelationMessageProver for Strategic<'_> {
    fn holo_frame(&mut self) -> Vec<F128> {
        self.relation.holo_frame()
    }
    fn observe_public_input_point(&mut self, r_pub: &[F128]) {
        self.relation.observe_public_input_point(r_pub);
    }
    fn begin_zerocheck(&mut self, tau: &[F128]) {
        self.relation.begin_zerocheck(tau);
    }
    fn zerocheck_round_message(&mut self, round: usize) -> Vec<F128> {
        self.relation.zerocheck_round_message(round)
    }
    fn receive_zerocheck_challenge(&mut self, round: usize, challenge: F128) {
        self.relation.receive_zerocheck_challenge(round, challenge);
    }
    fn zerocheck_endpoint(&mut self) -> Vec<F128> {
        self.relation.zerocheck_endpoint()
    }
    fn begin_lincheck(&mut self, row_point: &[F128], coefficients: Coefficients) {
        self.relation.begin_lincheck(row_point, coefficients);
    }
    fn lincheck_round_message(&mut self, round: usize) -> Vec<F128> {
        self.relation.lincheck_round_message(round)
    }
    fn receive_lincheck_challenge(&mut self, round: usize, challenge: F128) {
        self.relation.receive_lincheck_challenge(round, challenge);
    }
    fn lincheck_endpoint(&mut self) -> Vec<F128> {
        self.relation.lincheck_endpoint()
    }
}

impl BatchMessageProver for Strategic<'_> {
    fn begin_batch(&mut self, claims: &[Claim<F128>], gammas: &[F128]) {
        self.batching.begin_batch(claims, gammas);
    }
}

impl RoundMessageProver for Strategic<'_> {
    fn round_message(&mut self, round: usize) -> Vec<F128> {
        self.batching.round_message(round)
    }
    fn receive_challenge(&mut self, round: usize, challenge: F128) {
        self.batching.receive_challenge(round, challenge);
    }
}

struct Setup {
    case: Case,
    oracle: SparseTableOracle,
    table: Vec<F128>,
    local_cheat_rate: f64,
}

fn statement(c: &Case) -> Statement<'_> {
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

const VIEWS: usize = 16;

fn setup() -> Setup {
    let (case, oracle) = mnist_case(VIEWS);
    let case = case.clone_for_no_instance();
    assert!(
        case.holo.is_none(),
        "16 views must be scan-shaped: the live honest relation prover has no \
         holographic arm"
    );
    let inst = MnistInstance::new(VIEWS);
    let lc =
        fixtures::local_cheat_majority(&inst, &case.dc, &case.sites, &case.inst.x, case.t, &oracle);
    let table = committed_trace(&statement(&case), &oracle, &lc.cheat);
    Setup {
        case,
        oracle,
        table,
        local_cheat_rate: lc.prediction.accept_rate,
    }
}

fn config(c: &Case) -> Config {
    Config {
        t: c.t,
        ..Default::default()
    }
}

fn s_iop(c: &Case, params: &rmip_iop::backend::BackendParams) -> f64 {
    let miss = rmip_iop::backend::div_error_upward(
        (c.eps_rob.den - c.eps_rob.num) as f64,
        c.eps_rob.den as f64,
    );
    let beta = rmip_iop::backend::pow_error_upward(miss, c.t);
    let eps_rel = rmip_iop::backend::relation_error(c.sys.mu, c.layout.nu);
    rmip_iop::backend::sum_errors_upward(&[params.eps_bind, eps_rel, params.eps_open, beta, 0.0])
}

fn k_claims(c: &Case) -> usize {
    config(c).k_rel + c.t * (c.inst.b_max() + 1)
}

#[derive(Clone, Copy, Debug)]
struct Outcome {
    iop_accepted: bool,
    mip_accepted: bool,
    lies: usize,
    fabricated: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Tally {
    trials: usize,
    iop: usize,
    mip: usize,
    lied: usize,
    fabricated: usize,
}

impl Tally {
    fn push(&mut self, o: Outcome) {
        self.trials += 1;
        self.iop += usize::from(o.iop_accepted);
        self.mip += usize::from(o.mip_accepted);
        self.lied += usize::from(o.lies > 0);
        self.fabricated += o.fabricated;
    }
    fn mip_rate(&self) -> f64 {
        self.mip as f64 / self.trials as f64
    }
    fn iop_rate(&self) -> f64 {
        self.iop as f64 / self.trials as f64
    }
    fn sigma(&self) -> f64 {
        let p = self.mip_rate();
        (p * (1.0 - p) / self.trials as f64).sqrt()
    }
}

fn execution(iop_accepted: bool, occurrences: Vec<Occurrence>, c: &Case) -> Execution {
    Execution {
        iop_accepted,
        occurrences,
        oracle_calls: c.t,
        claims_submitted: k_claims(c),
        pad_claims: 0,
        rounds: 0,
        coins: 0,
        short_bytes: 0,
        transcript_before_sampling: 0,
        transcript_at_query_start: 0,
        transcript_at_query_end: 0,
        t_mat: std::time::Duration::ZERO,
        t_verify: std::time::Duration::ZERO,
        reduced_claim: None,
        t_open: std::time::Duration::ZERO,
        verifier_phases: Vec::new(),
        prover_phases: Vec::new(),
        prover_piop_phases: Vec::new(),
    }
}

fn fabricate_raw(table: &[F128], claim: &Claim<F128>) -> Option<(usize, F128)> {
    let truth = eval_mle(table, &claim.point);
    if truth == claim.value {
        return None;
    }
    let weights = eq_table(&claim.point);
    let (coord, inverse) = weights
        .iter()
        .enumerate()
        .find_map(|(i, w)| w.inv().map(|iw| (i, iw)))
        .expect("eq(point, .) cannot vanish everywhere");
    Some((coord, claim.value.add(truth).mul(inverse)))
}

fn run_raw(set: &Setup, deviation: Deviation, checker: &PCheck<WeakenedTrace>) -> Outcome {
    let c = &set.case;
    let st = statement(c);
    let cfg = config(c);
    let mut ch = OsChallenger::new();
    ch.post_oracle(DUMMY_ORACLE);
    let mut backend = WeakenedTrace::new();
    backend.commit(&set.table, &mut ch);
    let prefix = ch.transcript().prefix_at(TRACE_ORACLE);

    let mut prover = Strategic::new(&st, &c.inst, &set.table, &set.oracle, deviation);
    let mut returned: Vec<(OracleId, Coord, Symbol<F128>)> = Vec::new();
    let mut fabricated = 0usize;
    let interaction = verify_interactive_with_opening(
        &st,
        &set.oracle,
        &cfg,
        &mut prover,
        |claim, _p, ch| {
            backend.open(std::slice::from_ref(claim), ch);
            let repair = fabricate_raw(&set.table, claim);
            let mut plan = backend.query_plan(ch);
            let mut previous: Option<Symbol<F128>> = None;
            let mut answers = Vec::new();
            while let Some((id, coord)) = plan.next(previous.as_ref()) {
                let mut value = set.table[usize::try_from(coord).expect("coord fits usize")];
                if let Some((victim, delta)) = repair {
                    if Coord::from(victim) == coord {
                        value = value.add(delta);
                        fabricated += 1;
                    }
                }
                let symbol = Symbol(vec![value]);
                previous = Some(symbol.clone());
                answers.push((id, coord, symbol));
            }
            returned = answers.clone();
            (backend.verifier(), answers)
        },
        &mut ch,
    );

    let iop_accepted = interaction.verification.verdict.is_ok();
    let occurrences = returned
        .into_iter()
        .map(|(oracle, coord, answer)| Occurrence {
            oracle,
            prefix: prefix.clone(),
            coord,
            answer,
        })
        .collect();
    let e = execution(iop_accepted, occurrences, c);
    Outcome {
        iop_accepted,
        mip_accepted: audit_one(&e, checker, &mut ch).accepted(),
        lies: prover.lies,
        fabricated,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColumnRule {
    Naive,
    MinimalColumns,
}

fn proximity_coefficients(ch: &dyn RawChallenger, m: usize) -> Vec<F128> {
    let entries = ch.transcript().entries();
    let post = entries
        .iter()
        .position(|e| matches!(e, Entry::OraclePost(id) if *id == CODEWORD_ORACLE))
        .expect("the codeword oracle must have been posted");
    entries[post + 1..]
        .iter()
        .filter_map(|e| match e {
            Entry::Coin(value) => Some(*value),
            _ => None,
        })
        .take(m)
        .collect()
}

fn kernel_vector(mut rows: Vec<Vec<F128>>, n: usize) -> Option<Vec<F128>> {
    let height = rows.len();
    let mut pivot_of_column = vec![usize::MAX; n];
    let mut rank = 0usize;
    for column in 0..n {
        let Some(found) = (rank..height).find(|&i| rows[i][column] != F128::ZERO) else {
            continue;
        };
        rows.swap(rank, found);
        let inverse = rows[rank][column].inv()?;
        for value in rows[rank].iter_mut() {
            *value = value.mul(inverse);
        }
        let pivot = rows[rank].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i == rank || row[column] == F128::ZERO {
                continue;
            }
            let factor = row[column];
            for (value, p) in row.iter_mut().zip(&pivot) {
                *value = value.add(factor.mul(*p));
            }
        }
        pivot_of_column[column] = rank;
        rank += 1;
        if rank == height {
            break;
        }
    }
    let free = (0..n).find(|&c| pivot_of_column[c] == usize::MAX)?;
    let mut d = vec![F128::ZERO; n];
    d[free] = F128::ONE;
    for (column, &pivot) in pivot_of_column.iter().enumerate() {
        if pivot != usize::MAX {
            d[column] = rows[pivot][free];
        }
    }
    Some(d)
}

fn opening_offset(
    shape: &Shape,
    p_col: &[F128],
    delta: F128,
    rule: ColumnRule,
) -> Option<Vec<F128>> {
    if delta == F128::ZERO {
        return None;
    }
    let n = shape.n();
    let direction = match rule {
        ColumnRule::Naive => {
            let mut d = vec![F128::ZERO; n];
            d[n - 1] = F128::ONE;
            d
        }
        ColumnRule::MinimalColumns => {
            let rows: Vec<Vec<F128>> = (0..n - 1)
                .map(|column| {
                    let x = code::domain_point(shape.n_enc, column);
                    (0..n)
                        .map(|j| {
                            let mut unit = vec![F128::ZERO; n];
                            unit[j] = F128::ONE;
                            code::eval_at(&unit, x)
                        })
                        .collect()
                })
                .collect();
            kernel_vector(rows, n)?
        }
    };
    let weight = eval_mle(&direction, p_col);
    let scale = delta.mul(weight.inv()?);
    Some(direction.into_iter().map(|v| v.mul(scale)).collect())
}

fn run_interleaved(
    set: &Setup,
    deviation: Deviation,
    rule: ColumnRule,
    shape: Shape,
    checker: &PCheck<Ligero>,
) -> Outcome {
    let c = &set.case;
    let st = statement(c);
    let cfg = config(c);
    let (m, n) = (shape.m(), shape.n());
    let mut ch = OsChallenger::new();
    ch.post_oracle(DUMMY_ORACLE);
    let mut backend = Ligero::new(shape);
    backend.commit(&set.table, &mut ch);
    let prefix = ch.transcript().prefix_at(CODEWORD_ORACLE);
    let proximity_coeffs = proximity_coefficients(&ch, m);
    assert_eq!(proximity_coeffs.len(), m, "commit draws m proximity coins");
    let proximity = backend.verifier().proximity;

    let mut prover = Strategic::new(&st, &c.inst, &set.table, &set.oracle, deviation);
    let mut returned: Vec<(OracleId, Coord, Symbol<F128>)> = Vec::new();
    let mut fabricated = 0usize;
    let interaction = verify_interactive_with_opening(
        &st,
        &set.oracle,
        &cfg,
        &mut prover,
        |claim, _p, ch| {
            let point = &claim.point;
            let opening_coeffs = eq_table(&point[..shape.nu_row]);
            let honest: Vec<F128> = (0..n)
                .map(|j| {
                    set.table
                        .chunks(n)
                        .zip(&opening_coeffs)
                        .fold(F128::ZERO, |acc, (row, coefficient)| {
                            acc.add(coefficient.mul(row[j]))
                        })
                })
                .collect();
            let p_col = &point[shape.nu_row..];
            let delta = claim.value.add(eval_mle(&honest, p_col));
            let offset = opening_offset(&shape, p_col, delta, rule);
            let opening: Vec<F128> = match &offset {
                None => honest.clone(),
                Some(d) => honest.iter().zip(d).map(|(u, d)| u.add(*d)).collect(),
            };
            let mut message = Vec::with_capacity(opening.len() * 16);
            for value in &opening {
                message.extend_from_slice(&value.to_be_bytes());
            }
            ch.record_ordinary(&message);

            let mut plan = backend.query_plan(ch);
            let mut previous: Option<Symbol<F128>> = None;
            let mut answers = Vec::new();
            while let Some((id, column)) = plan.next(previous.as_ref()) {
                let x = code::domain_point(
                    shape.n_enc,
                    usize::try_from(column).expect("column fits usize"),
                );
                let mut symbol = backend.reconstruct(id, &prefix, column).0;
                let discrepancy = match &offset {
                    None => F128::ZERO,
                    Some(d) => code::eval_at(d, x),
                };
                if discrepancy != F128::ZERO {
                    let (i, j) = independent_pair(&proximity_coeffs, &opening_coeffs)
                        .expect("the two coefficient vectors must be independent");
                    let determinant = proximity_coeffs[i]
                        .mul(opening_coeffs[j])
                        .add(proximity_coeffs[j].mul(opening_coeffs[i]));
                    let inverse = determinant.inv().expect("nonzero by construction");
                    let scaled = discrepancy.mul(inverse);
                    symbol[i] = symbol[i].add(proximity_coeffs[j].mul(scaled));
                    symbol[j] = symbol[j].add(proximity_coeffs[i].mul(scaled));
                    fabricated += 1;
                }
                let symbol = Symbol(symbol);
                previous = Some(symbol.clone());
                answers.push((id, column, symbol));
            }
            returned = answers.clone();
            let verifier = LigeroVerifier::from_public_transcript(
                shape,
                proximity.clone(),
                proximity_coeffs.clone(),
                opening,
                opening_coeffs,
                point.clone(),
            );
            (verifier, answers)
        },
        &mut ch,
    );

    let iop_accepted = interaction.verification.verdict.is_ok();
    let occurrences = returned
        .into_iter()
        .map(|(oracle, coord, answer)| Occurrence {
            oracle,
            prefix: prefix.clone(),
            coord,
            answer,
        })
        .collect();
    let e = execution(iop_accepted, occurrences, c);
    Outcome {
        iop_accepted,
        mip_accepted: audit_one(&e, checker, &mut ch).accepted(),
        lies: prover.lies,
        fabricated,
    }
}

fn independent_pair(a: &[F128], b: &[F128]) -> Option<(usize, usize)> {
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            if a[i].mul(b[j]).add(a[j].mul(b[i])) != F128::ZERO {
                return Some((i, j));
            }
        }
    }
    None
}

fn raw_checker(set: &Setup) -> PCheck<WeakenedTrace> {
    let advice = SharedAdvice { seed: 0xC4EC };
    let mut b = WeakenedTrace::new();
    let mut own = SeededChallenger::new(advice.seed);
    b.commit(&set.table, &mut own);
    PCheck::new(b)
}

fn interleaved_checker(set: &Setup, shape: Shape) -> PCheck<Ligero> {
    let advice = SharedAdvice { seed: 0xC4EC };
    let mut b = Ligero::new(shape);
    let mut own = SeededChallenger::new(advice.seed);
    b.commit(&set.table, &mut own);
    PCheck::new(b)
}

fn measure_raw(set: &Setup, deviation: Deviation, trials: usize) -> Tally {
    let checker = raw_checker(set);
    let mut tally = Tally::default();
    for _ in 0..trials {
        tally.push(run_raw(set, deviation, &checker));
    }
    tally
}

fn measure_interleaved(
    set: &Setup,
    deviation: Deviation,
    rule: ColumnRule,
    shape: Shape,
    trials: usize,
) -> Tally {
    let checker = interleaved_checker(set, shape);
    let mut tally = Tally::default();
    for _ in 0..trials {
        tally.push(run_interleaved(set, deviation, rule, shape, &checker));
    }
    tally
}

fn shape_for(set: &Setup) -> Shape {
    Shape::balanced(set.case.layout.nu, 4, 4.0)
}

fn raw_bound(set: &Setup) -> (f64, usize) {
    let params = WeakenedTrace::new().public_params(set.case.layout.n_trace, k_claims(&set.case));
    let q = params.q_long;
    (one_shot_bound(s_iop(&set.case, &params), q), q)
}

fn interleaved_bound(set: &Setup, shape: Shape) -> (f64, usize) {
    let params = Ligero::new(shape).params(k_claims(&set.case));
    let q = params.q_long;
    (one_shot_bound(s_iop(&set.case, &params), q), q)
}

fn report(label: &str, tally: &Tally, bound: f64, q: usize) {
    println!(
        "  {label:<34} accept {:.5} +- {:.5} ({}/{})  IOP-only {:.5}  \
         deviated {:.3}  fabricated/exec {:.3}  bound(q={q}) {bound:.6}  z {:+.2}",
        tally.mip_rate(),
        tally.sigma(),
        tally.mip,
        tally.trials,
        tally.iop_rate(),
        tally.lied as f64 / tally.trials as f64,
        tally.fabricated as f64 / tally.trials as f64,
        (tally.mip_rate() - bound) / tally.sigma(),
    );
}

fn assert_under_bound(label: &str, tally: &Tally, bound: f64) {
    let ceiling = bound + 5.0 * tally.sigma();
    assert!(
        tally.mip_rate() <= ceiling,
        "{label}: measured {:.6} exceeds one_shot_bound {bound:.6} + 5 sigma = {ceiling:.6}",
        tally.mip_rate()
    );
}

fn sweep(trials_raw: usize, trials_interleaved: usize) {
    let set = setup();
    let shape = shape_for(&set);
    let (bound_raw, q_raw) = raw_bound(&set);
    let (bound_int, q_int) = interleaved_bound(&set, shape);
    let attainable_raw = one_shot_bound(set.local_cheat_rate, q_raw);

    println!(
        "{}, NO-instance: ell = {}, t = {}, LocalCheat (1-|Bad|/ell)^t = {:.6}",
        set.case.family, set.case.inst.ell, set.case.t, set.local_cheat_rate
    );
    println!("--- raw-trace backend, q_long = {q_raw} ---");
    println!("  declared bound (q-1+s_IOP)/q = {bound_raw:.6}, exact-s form = {attainable_raw:.6}");
    let control = measure_raw(&set, Deviation::None, trials_raw / 8);
    report("LocalCheat (no deviation)", &control, bound_raw, q_raw);
    let answer = measure_raw(&set, Deviation::TrueAnswer, trials_raw);
    report("strategic, true answer", &answer, bound_raw, q_raw);
    let wire = measure_raw(&set, Deviation::RewriteLabel, trials_raw);
    report("strategic, rewritten query label", &wire, bound_raw, q_raw);

    println!(
        "--- interleaved-rs backend, q_long = {q_int}, n_enc = {} ---",
        shape.n_enc
    );
    let int_control = measure_interleaved(
        &set,
        Deviation::None,
        ColumnRule::MinimalColumns,
        shape,
        trials_interleaved / 8,
    );
    report("LocalCheat (no deviation)", &int_control, bound_int, q_int);
    let int_min = measure_interleaved(
        &set,
        Deviation::TrueAnswer,
        ColumnRule::MinimalColumns,
        shape,
        trials_interleaved,
    );
    report("strategic, fewest columns", &int_min, bound_int, q_int);
    let int_naive = measure_interleaved(
        &set,
        Deviation::TrueAnswer,
        ColumnRule::Naive,
        shape,
        trials_interleaved,
    );
    report("strategic, naive column rule", &int_naive, bound_int, q_int);

    for (label, tally) in [
        ("raw/true answer", &answer),
        ("raw/rewritten label", &wire),
        ("interleaved/fewest columns", &int_min),
        ("interleaved/naive", &int_naive),
    ] {
        assert!(
            tally.iop_rate() > 0.99,
            "{label}: the IOP alone should accept almost always, got {:.5}",
            tally.iop_rate()
        );
    }

    for (label, tally, bound) in [
        ("raw/control", &control, bound_raw),
        ("raw/true answer", &answer, bound_raw),
        ("raw/rewritten label", &wire, bound_raw),
        ("interleaved/control", &int_control, bound_int),
        ("interleaved/fewest columns", &int_min, bound_int),
        ("interleaved/naive", &int_naive, bound_int),
    ] {
        assert_under_bound(label, tally, bound);
    }

    let sigma =
        (set.local_cheat_rate * (1.0 - set.local_cheat_rate) / control.trials as f64).sqrt();
    assert!(
        (control.mip_rate() - set.local_cheat_rate).abs() < 5.0 * sigma,
        "the no-deviation control should be the LocalCheat rate {:.5}, got {:.5}",
        set.local_cheat_rate,
        control.mip_rate()
    );
    assert!(
        answer.mip_rate() > 10.0 * set.local_cheat_rate,
        "the strategic prover should dwarf the LocalCheat escape"
    );

    let attainable_sigma = (attainable_raw * (1.0 - attainable_raw) / answer.trials as f64).sqrt();
    assert!(
        (answer.mip_rate() - attainable_raw).abs() < 5.0 * attainable_sigma,
        "the raw-trace strategic prover should attain (q - 1 + s)/q = {attainable_raw:.6}, \
         got {:.6}",
        answer.mip_rate()
    );
    assert_eq!(
        answer.fabricated, answer.lied,
        "exactly one occurrence is fabricated per deviating execution"
    );
}

#[test]
fn the_strategic_prover_is_caught_only_by_the_audit_and_stays_under_the_one_shot_bound() {
    sweep(20_000, 5_000);
}

#[test]
#[ignore = "measurement"]
fn strategic_prover_acceptance_at_measurement_trial_counts() {
    sweep(1_000_000, 150_000);
}
