use rmip_field::F128;
use rmip_frontend::oracle::Oracle;
use rmip_iop::backend::{
    Claim, CommitmentBackend, Coord, OpeningVerifier, OracleRecordingChallenger,
};
use rmip_iop::driver::{dummy_symbol, Config, Execution, Occurrence, Reject, DUMMY_ORACLE};
use rmip_iop::transcript::RawChallenger;

use crate::majority::{combined_words, MajorityError};
use crate::PackedField;
use crate::statement::PackedStatement;
use crate::tree::LevelWitness;

#[derive(Clone, Debug, Default)]
pub struct PackedCheat {
    pub public_input_bits: Vec<usize>,
    pub flip_answer: Option<usize>,
    pub corrupt_word: Option<(usize, u128)>,
}

pub fn packed_witness<F: PackedField, O: Oracle>(
    st: &PackedStatement<'_>,
    oracle: &O,
    cheat: &PackedCheat,
) -> (Vec<u64>, Vec<LevelWitness<F>>, Vec<F>) {
    let mut x = st.inst.x.clone();
    for &b in &cheat.public_input_bits {
        x.set(b, !x.get(b));
    }
    let mut answers = crate::statement::packed_answer_words(st, x.as_slice(), oracle);
    if let Some(v) = cheat.flip_answer {
        answers[v / 64] ^= 1 << (v % 64);
    }
    let witness = st.tree.witness_from_words::<F>(&answers);
    let mut words = combined_words(&st.tree, &witness);
    if let Some((i, v)) = cheat.corrupt_word {
        words[i] = F::from_bits(v);
    }
    (answers, witness, words)
}

pub fn execute_recording_packed<F, B, O>(
    st: &PackedStatement<'_>,
    backend: &mut B,
    oracle: &O,
    cfg: &Config,
    cheat: &PackedCheat,
    ch: &mut dyn RawChallenger,
) -> (Execution, Result<(), Reject>)
where
    F: PackedField,
    B: CommitmentBackend<Field = F128>,
    O: Oracle,
{
    let _execution_role = rmip_iop::metrics::enter(rmip_iop::metrics::Role::Verifier);
    let mut verifier_phases: Vec<(&'static str, std::time::Duration)> = Vec::new();
    let mut prover_phases: Vec<(&'static str, std::time::Duration)> = Vec::new();
    let mut mark = |name: &'static str| {
        verifier_phases.push((
            name,
            rmip_iop::metrics::elapsed_now(rmip_iop::metrics::Role::Verifier).unwrap_or_default(),
        ));
        prover_phases.push((
            name,
            rmip_iop::metrics::elapsed_now(rmip_iop::metrics::Role::MainProver).unwrap_or_default(),
        ));
    };
    macro_rules! measured {
        ($role:ident, $expression:expr) => {{
            let _role = rmip_iop::metrics::enter(rmip_iop::metrics::Role::$role);
            $expression
        }};
    }
    let inst = st.inst;
    let mut oracle_challenger = OracleRecordingChallenger::new(ch);
    let ch = &mut oracle_challenger;

    ch.post_public_oracle(DUMMY_ORACLE);
    let dummy_prefix = ch.transcript().prefix_at(DUMMY_ORACLE);

    mark("prelude");
    let (answers, witness, words) =
        measured!(MainProver, packed_witness::<F, O>(st, oracle, cheat));
    let zf = measured!(MainProver, F::to_words(&words));
    mark("witness");

    let has_rows = false;
    let k_total = 1usize;
    let params = backend.public_params(zf.len(), k_total);
    mark("public_params");
    assert!(
        !(params.q_long == 1 && cfg.pad_queries_to.is_some()),
        "Config::pad_queries_to = {:?} is refused for a backend declaring q_long = 1 \
         (e.g. rmip-backend-bundled): padding restores the 1/q audit dilution",
        cfg.pad_queries_to
    );
    assert!(
        !(params.audit.refuses_padding() && cfg.pad_queries_to.is_some()),
        "Config::pad_queries_to = {:?} is refused for a sequentially audited or rewound backend",
        cfg.pad_queries_to
    );
    mark("asserts");
    let precommit_admissible = params.check_admissible(has_rows);
    mark("admissible");
    let precommit_q_vetted = precommit_admissible.is_ok();

    mark("params");
    measured!(MainProver, backend.commit(&zf, ch));

    let mut deferred: Option<Reject> = None;
    macro_rules! defer {
        ($r:expr) => {
            if deferred.is_none() {
                deferred = Some($r);
            }
        };
    }

    if let Err(error) = precommit_admissible {
        defer!(Reject::InadmissibleBackend(error));
    }

    if let Some(error) = ch.error().cloned() {
        defer!(Reject::OracleMutation(error));
    }
    if let Err(error) = ch.frozen().validate_current(backend.oracles()) {
        defer!(Reject::OracleMutation(error));
    }

    let params_after_commit = backend.params(k_total);
    if !params.same_fixed_schedule(&params_after_commit) {
        defer!(Reject::BackendParamsChanged);
    }
    let postcommit_q_vetted = match params_after_commit.check_admissible(has_rows) {
        Ok(()) => true,
        Err(error) => {
            defer!(Reject::InadmissibleBackend(error));
            false
        }
    };
    let audit_length_is_vetted = precommit_q_vetted && postcommit_q_vetted;

    mark("commit");

    if !st.valid {
        defer!(Reject::InvalidPublicInput);
    }
    let tables = F::skip_tables();

    mark("admissibility");
    let views = st.views();
    let mut sampled = Vec::with_capacity(cfg.t);
    for _ in 0..cfg.t {
        sampled.push(rmip_iop::sampling::uniform_index(ch, views));
    }

    mark("sampling");
    let piop_start = ch.transcript().entries().len();
    let mut piop_phases = crate::phases::Phases::new();
    let proof = measured!(
        MainProver,
        crate::majority::prove_piop_with_phases(
            tables,
            &st.tree,
            &witness,
            &words,
            &sampled,
            ch,
            Some(&mut piop_phases),
        )
    );

    mark("piop_prover");
    let mut oracle_calls = 0;
    for (h, &view) in sampled.iter().enumerate() {
        let bit = crate::majority::sampled_bit(&st.tree, view, proof.sampled_words[h]);
        let u = crate::statement::query_bits(st, inst.x.as_slice(), view);
        oracle_calls += 1;
        if oracle.query(u.as_slice()) != bit {
            defer!(Reject::OracleMismatch { occurrence: h });
        }
    }

    mark("claimed_blocks");
    let coins = rmip_iop::replay::coins_since(ch.transcript(), piop_start);
    let t_verify_start = std::time::Instant::now();
    let mut replay = rmip_iop::replay::CoinReplay::new(&coins);
    let verified = crate::majority::verify_piop(
        tables,
        &st.tree,
        st.threshold,
        &sampled,
        &proof,
        &mut replay,
    );
    let t_piop_verify = t_verify_start.elapsed();
    let opened: Vec<Claim<F128>> = match verified {
        Ok(claim) => {
            if !replay.replayed_exactly() {
                defer!(Reject::PackedRelation("coin schedule".into()));
            }
            if claim != proof.batched {
                defer!(Reject::PackedRelation("batched claim".into()));
            }
            vec![claim]
        }
        Err(MajorityError::BelowThreshold { count, threshold }) => {
            defer!(Reject::BelowThreshold { count, threshold });
            vec![proof.batched.clone()]
        }
        Err(e) => {
            defer!(Reject::PackedRelation(format!("{e:?}")));
            vec![proof.batched.clone()]
        }
    };
    let pad_claims = 0usize;

    mark("batch");
    measured!(MainProver, backend.open(&opened, ch));
    if let Some(error) = ch.error().cloned() {
        defer!(Reject::OracleMutation(error));
    }
    if let Err(error) = ch.frozen().validate_current(backend.oracles()) {
        defer!(Reject::OracleMutation(error));
    }
    let frozen_oracles = ch.frozen().clone();

    let params_after_open = backend.params(k_total);
    if !params.same_fixed_schedule(&params_after_open) {
        defer!(Reject::BackendParamsChanged);
    }
    if let Err(error) = params_after_open.check_admissible(has_rows) {
        defer!(Reject::InadmissibleBackend(error));
    }

    mark("open");
    let expected_q = cfg.audit_query_count(params.q_long);

    let verifier = backend.verifier();
    let transcript_before_sampling = ch.transcript().entries().len();
    let mut plan = backend.query_plan(ch);
    let transcript_at_query_start = ch.transcript().entries().len();
    for e in &ch.transcript().entries()[transcript_before_sampling..transcript_at_query_start] {
        match e {
            rmip_iop::transcript::Entry::Coin(_) => {}
            rmip_iop::transcript::Entry::OraclePost(id) => {
                defer!(Reject::LateOraclePost { oracle: *id });
            }
            rmip_iop::transcript::Entry::Ordinary(_) => {
                defer!(Reject::LateBindingMessage);
            }
        }
    }
    if let Some(error) = ch.error().cloned() {
        defer!(Reject::OracleMutation(error));
    }
    let mut occurrences = Vec::new();
    let mut answered = Vec::new();
    let mut prefixes: Vec<(
        rmip_iop::transcript::OracleId,
        rmip_iop::transcript::TranscriptPrefix,
    )> = Vec::new();
    let mut plan_stopped = false;
    while occurrences.len() < params.q_long {
        let Some((id, coord)) = plan.next(answered.last().map(|(_, _, a)| a)) else {
            plan_stopped = true;
            break;
        };
        let Some(o) = frozen_oracles.get(id) else {
            defer!(Reject::OracleInconsistent {
                oracle: id,
                coord: Box::new(coord)
            });
            plan_stopped = true;
            break;
        };
        let Some(ix) = usize::try_from(coord).ok().filter(|&i| i < o.symbols.len()) else {
            defer!(Reject::OracleInconsistent {
                oracle: id,
                coord: Box::new(coord)
            });
            plan_stopped = true;
            break;
        };
        let answer = measured!(MainProver, o.symbols[ix].clone());
        let cached = prefixes
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, p)| p.clone());
        let prefix_opt = match cached {
            Some(p) => Some(p),
            None => {
                let fresh = ch.transcript().try_prefix_at(id);
                if let Some(p) = &fresh {
                    prefixes.push((id, p.clone()));
                }
                fresh
            }
        };
        let Some(prefix) = prefix_opt else {
            defer!(Reject::OracleInconsistent {
                oracle: id,
                coord: Box::new(coord)
            });
            plan_stopped = true;
            break;
        };
        occurrences.push(Occurrence {
            oracle: id,
            prefix,
            coord,
            answer: answer.clone(),
        });
        answered.push((id, coord, answer));
    }
    let mut real_occurrences = occurrences.len();
    let last_answer = answered.last().map(|(_, _, a)| a);
    if !plan_stopped && occurrences.len() == params.q_long && plan.next(last_answer).is_some() {
        real_occurrences += 1;
    }
    if real_occurrences != params.q_long {
        defer!(Reject::QueryCount {
            expected: params.q_long,
            got: real_occurrences
        });
    }
    let pad_to = if audit_length_is_vetted {
        expected_q
    } else {
        occurrences.len().max(1)
    };
    while occurrences.len() < pad_to {
        occurrences.push(Occurrence {
            oracle: DUMMY_ORACLE,
            prefix: dummy_prefix.clone(),
            coord: Coord::ZERO,
            answer: dummy_symbol(),
        });
    }

    mark("query");
    let t_open_start = std::time::Instant::now();
    let opening_ok = verifier.check(&opened, &answered);
    let t_open = t_open_start.elapsed();
    mark("decide");
    if !opening_ok {
        defer!(Reject::OpeningMismatch { claim: usize::MAX });
    }
    let params_after_queries = backend.params(k_total);
    if !params.same_fixed_schedule(&params_after_queries) {
        defer!(Reject::BackendParamsChanged);
    }
    if let Err(error) = params_after_queries.check_admissible(has_rows) {
        defer!(Reject::InadmissibleBackend(error));
    }
    let verdict = match deferred {
        Some(r) => Err(r),
        None => Ok(()),
    };

    let _diagnostics_role = rmip_iop::metrics::enter(rmip_iop::metrics::Role::Diagnostic);
    let mut rounds = 0usize;
    let mut coins = 0usize;
    let mut short_bytes = 0usize;
    let mut in_coin_run = false;
    for e in ch.transcript().entries() {
        match e {
            rmip_iop::transcript::Entry::Coin(_) => {
                coins += 1;
                if !in_coin_run {
                    rounds += 1;
                    in_coin_run = true;
                }
            }
            rmip_iop::transcript::Entry::Ordinary(b) => {
                short_bytes += b.len();
                in_coin_run = false;
            }
            rmip_iop::transcript::Entry::OraclePost(_) => in_coin_run = false,
        }
    }

    let execution = Execution {
        iop_accepted: verdict.is_ok(),
        reduced_claim: if cfg.batch_claims {
            opened.first().cloned()
        } else {
            None
        },
        transcript_before_sampling,
        transcript_at_query_start,
        transcript_at_query_end: ch.transcript().entries().len(),
        occurrences,
        oracle_calls,
        claims_submitted: k_total,
        pad_claims,
        rounds,
        coins,
        short_bytes,
        t_mat: std::time::Duration::ZERO,
        t_verify: {
            t_piop_verify + t_open
        },
        t_open,
        verifier_phases,
        prover_phases,
        prover_piop_phases: piop_phases.entries().to_vec(),
    };
    drop(_diagnostics_role);
    measured!(MainProver, {
        drop(witness);
        drop(words);
        drop(zf);
        drop(answers);
    });
    (execution, verdict)
}
