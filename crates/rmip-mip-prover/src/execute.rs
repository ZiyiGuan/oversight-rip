use crate::lazy::LazyBackend;
use rmip_iop::backend::{
    Claim, Coord, FreezeError, FrozenOracleSet, OpeningVerifier, OracleRecordingChallenger,
};
use rmip_iop::driver::{
    boolean_point, committed_trace, dummy_symbol, Cheat, Config, Execution, Occurrence, Reject,
    Statement, DUMMY_ORACLE,
};
use rmip_iop::lincheck::{Coefficients, Lincheck};
use rmip_iop::matmle;
use rmip_iop::mle::{eval_mle, lift};
use rmip_iop::sumcheck;
use rmip_iop::transcript::RawChallenger;
use rmip_iop::zerocheck::RowClaims;
use rmip_iop::zerocheck::Zerocheck;
use rmip_field::{BinaryField, F128};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::trace;

pub(crate) fn validate_registered_oracles<B>(
    backend: &B,
    frozen: &FrozenOracleSet<F128>,
) -> Result<(), FreezeError>
where
    B: LazyBackend<Field = F128>,
{
    let registered = frozen.as_slice();
    let current = backend.oracle_meta();
    for (posted, now) in registered.iter().zip(&current) {
        if posted.id != now.id {
            return Err(FreezeError::RemovedOrReordered(posted.id));
        }
        if posted.alphabet != now.alphabet || posted.len() != now.len {
            return Err(FreezeError::Replaced(posted.id));
        }
    }
    if let Some(extra) = current.get(registered.len()) {
        return Err(FreezeError::Unregistered(extra.id));
    }
    if let Some(missing) = registered.get(current.len()) {
        return Err(FreezeError::Missing(missing.id));
    }
    Ok(())
}

pub fn execute<B, O>(
    st: &Statement<'_>,
    backend: &mut B,
    oracle: &O,
    cfg: &Config,
    cheat: &Cheat,
    ch: &mut dyn RawChallenger,
) -> Result<Execution, Reject>
where
    B: LazyBackend<Field = F128>,
    O: Oracle,
{
    let (e, verdict) = execute_recording(st, backend, oracle, cfg, cheat, ch);
    verdict.map(|()| e)
}

pub fn execute_recording<B, O>(
    st: &Statement<'_>,
    backend: &mut B,
    oracle: &O,
    cfg: &Config,
    cheat: &Cheat,
    ch: &mut dyn RawChallenger,
) -> (Execution, Result<(), Reject>)
where
    B: LazyBackend<Field = F128>,
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
    let (inst, sys, layout) = (st.inst, st.sys, st.layout);
    let mu = sys.mu;
    let nu = layout.nu;
    let mut oracle_challenger = OracleRecordingChallenger::new(ch);
    let ch = &mut oracle_challenger;

    ch.post_public_oracle(DUMMY_ORACLE);
    let dummy_prefix = ch.transcript().prefix_at(DUMMY_ORACLE);

    mark("prelude");
    let zf = measured!(MainProver, committed_trace(st, oracle, cheat));
    mark("witness");

    let has_rows = matches!(
        sys.booleanity,
        rmip_frontend::r1cs::Booleanity::ExplicitRows
            | rmip_frontend::r1cs::Booleanity::AnswerBlockRows
    );
    let k_total = cfg.k_rel + cfg.t * (inst.b_max() + 1);
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
    if let Err(error) = validate_registered_oracles(backend, ch.frozen()) {
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

    mark("admissibility");
    let r_pub: Vec<F128> = (0..layout.k).map(|_| ch.sample()).collect();
    let r_prime: Vec<F128> = std::iter::repeat_n(F128::ZERO, nu - layout.k)
        .chain(r_pub.iter().copied())
        .collect();
    let p_block = trace::public_block(layout, inst.x.as_slice());
    let v_pub = eval_mle(&lift(p_block.as_slice()), &r_pub);
    let public_claim = Claim {
        point: r_prime,
        value: v_pub,
    };

    mark("public_input");
    let tau: Vec<F128> = (0..mu).map(|_| ch.sample()).collect();
    let mut zc = measured!(
        MainProver,
        Zerocheck::new(
            &tau,
            matmle::mul_vec(&sys.a, &zf),
            matmle::mul_vec(&sys.b, &zf),
            matmle::mul_vec(&sys.c, &zf),
        )
    );
    let zerocheck_from = ch.transcript().entries().len();
    let (zproof, zclaim) = sumcheck::run_mlecheck_recording(&mut zc, &tau, ch);
    let zerocheck_coins = rmip_iop::replay::coins_since(ch.transcript(), zerocheck_from);
    let r_row = zclaim.point.clone();
    let rc: RowClaims = measured!(MainProver, zc.row_claims());
    let rc = if cheat.wrong_zerocheck_endpoint {
        RowClaims {
            v_a: rc.v_a.add(F128::ONE),
            ..rc
        }
    } else {
        rc
    };
    measured!(MainProver, {
        let mut row_claim_message = Vec::with_capacity(3 * 16);
        for v in [rc.v_a, rc.v_b, rc.v_c] {
            row_claim_message.extend_from_slice(&v.to_be_bytes());
        }
        ch.record_ordinary(&row_claim_message);
    });
    if rc.v_a.mul(rc.v_b).add(rc.v_c) != zclaim.expected {
        defer!(Reject::ZerocheckEndpoint);
    }

    mark("zerocheck");
    let coeffs = Coefficients::sample(ch);
    let batched = coeffs.batch(rc);
    let mut lc = measured!(
        MainProver,
        Lincheck::new(&sys.a, &sys.b, &sys.c, &r_row, coeffs, zf.clone())
    );
    let batched = if cheat.corrupt_lincheck_round {
        batched.add(F128::ONE)
    } else {
        batched
    };
    let lincheck_from = ch.transcript().entries().len();
    let (lproof, lclaim, lbad) = sumcheck::run_recording(&mut lc, batched, ch);
    let lincheck_coins = rmip_iop::replay::coins_since(ch.transcript(), lincheck_from);
    if let Some(e) = lbad {
        defer!(Reject::Lincheck(e));
    }
    let t_mat_start = std::time::Instant::now();
    let (ma, mb, mc) = match st.holo {
        Some(h) => {
            let entries_before = ch.transcript().entries().len();
            let ([pa, pb, pc], frames) = measured!(
                MainProver,
                h.prove_frames(&sys.a, &sys.b, &sys.c, &r_row, &lclaim.point, ch)
            );
            let coins = rmip_iop::replay::coins_since(ch.transcript(), entries_before);
            let (checked, exact, leftover) = measured!(Verifier, {
                let mut replay = rmip_iop::replay::CoinReplay::new(&coins);
                let mut supplied = frames.iter();
                let out = {
                    let mut next = || supplied.next().cloned().unwrap_or_default();
                    h.verify_frames(&mut next, &r_row, &lclaim.point, &mut replay)
                };
                (out, replay.replayed_exactly(), supplied.next().is_some())
            });
            match checked {
                Ok(v) if exact && !leftover && v == [pa, pb, pc] => (v[0], v[1], v[2]),
                Ok(v) => {
                    defer!(Reject::HolographicFrames);
                    (v[0], v[1], v[2])
                }
                Err(_) => {
                    defer!(Reject::HolographicFrames);
                    (F128::ZERO, F128::ZERO, F128::ZERO)
                }
            }
        }
        None => matmle::eval_all(&sys.a, &sys.b, &sys.c, &r_row, &lclaim.point),
    };
    let t_mat = t_mat_start.elapsed();
    let l_at = coeffs
        .lambda_a()
        .mul(ma)
        .add(coeffs.lambda_b().mul(mb))
        .add(coeffs.lambda_c().mul(mc));
    let v_z = measured!(MainProver, {
        if cheat.wrong_lincheck_endpoint {
            lc.z_claim().add(F128::ONE)
        } else {
            lc.z_claim()
        }
    });
    measured!(MainProver, ch.record_ordinary(&v_z.to_be_bytes()));
    if l_at.mul(v_z) != lclaim.expected {
        defer!(Reject::LincheckEndpoint);
    }
    let lincheck_claim = Claim {
        point: lclaim.point.clone(),
        value: v_z,
    };

    mark("lincheck");
    let (sigma, v_sigma) = rmip_iop::booleanity::claim_from_lincheck_endpoint(&lclaim.point, v_z);
    let booleanity_claim = Claim {
        point: sigma,
        value: v_sigma,
    };

    mark("booleanity");
    let mut sampled = Vec::with_capacity(cfg.t);
    for _ in 0..cfg.t {
        sampled.push(rmip_iop::sampling::uniform_index(ch, inst.ell));
    }

    mark("sampling");
    let mut oracle_calls = 0;
    let mut coord_claims = Vec::new();
    for (h, &j) in sampled.iter().enumerate() {
        let loc_j = inst.loc_row(j);
        let (query_coords, answer_coord) = loc_j.split_at(loc_j.len() - 1);

        let mut u = rmip_frontend::bits::BitVec::zeros(query_coords.len());
        for (k, &coord) in query_coords.iter().enumerate() {
            let v = measured!(
                MainProver,
                match cheat.supply_non_boolean_block {
                    Some(bad) if h == 0 && k == 0 => F128::new(bad),
                    _ => zf[coord],
                }
            );
            if v != F128::ZERO && v != F128::ONE {
                defer!(Reject::NonBooleanBlock { occurrence: h });
            }
            u.set(k, v == F128::ONE);
        }
        let av = measured!(
            MainProver,
            match cheat.non_boolean_answer {
                Some(bad) if bad == h => F128::new(0x9999),
                _ => zf[answer_coord[0]],
            }
        );
        if av != F128::ZERO && av != F128::ONE {
            defer!(Reject::NonBooleanBlock { occurrence: h });
        }
        let mut a = av == F128::ONE;

        if cheat.bad_oracle_answer == Some(h) {
            a = !a;
        }
        if cheat.trace_swap && h == 0 && u.len() >= 2 {
            u.set(0, !u.get(0));
            u.set(1, !u.get(1));
        }

        measured!(MainProver, {
            let mut block = Vec::with_capacity((u.len() + 1) * 16);
            for k in 0..u.len() {
                block.extend_from_slice(&F128::from_bool(u.get(k)).to_be_bytes());
            }
            block.extend_from_slice(&F128::from_bool(a).to_be_bytes());
            ch.record_ordinary(&block);
        });

        oracle_calls += 1;
        if oracle.query(u.as_slice()) != a {
            defer!(Reject::OracleMismatch { occurrence: h });
        }
        for (k, &coord) in loc_j.iter().enumerate() {
            let bit = if k + 1 == loc_j.len() { a } else { u.get(k) };
            coord_claims.push(Claim {
                point: boolean_point(coord, nu),
                value: F128::from_bool(bit),
            });
        }
    }

    mark("claimed_blocks");
    let mut claims = vec![lincheck_claim, public_claim.clone(), booleanity_claim];
    claims.extend(coord_claims);
    let before_pad = claims.len();
    while claims.len() < k_total {
        claims.push(public_claim.clone());
    }
    let pad_claims = k_total - before_pad;
    assert_eq!(claims.len(), k_total, "K must be a public constant");

    mark("pad");
    let opened: Vec<Claim<F128>> = if cfg.batch_claims {
        match rmip_iop::batch::reduce(&claims, &zf, ch) {
            Ok(c) => vec![c],
            Err(e) => {
                defer!(Reject::Batching(e));
                vec![public_claim.clone()]
            }
        }
    } else {
        claims.clone()
    };

    mark("batch");
    measured!(MainProver, backend.open(&opened, ch));
    if let Some(error) = ch.error().cloned() {
        defer!(Reject::OracleMutation(error));
    }
    if let Err(error) = validate_registered_oracles(backend, ch.frozen()) {
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
        if coord >= o.len() {
            defer!(Reject::OracleInconsistent {
                oracle: id,
                coord: Box::new(coord)
            });
            plan_stopped = true;
            break;
        }
        let cached = prefixes.iter().find(|(i, _)| *i == id).map(|(_, p)| p.clone());
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
        let posted = if Coord::from(o.symbols.len()) == o.len() {
            let ix = usize::try_from(coord).expect("coord < o.len() == o.symbols.len()");
            Some(o.symbols[ix].clone())
        } else {
            None
        };
        let live = measured!(MainProver, backend.symbol_at(id, &prefix, coord));
        let answer = match posted {
            Some(posted) => {
                if live != posted {
                    defer!(Reject::OracleMutation(FreezeError::Replaced(id)));
                }
                posted
            }
            None => live,
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
        t_mat,
        t_verify: {
            let t0 = std::time::Instant::now();
            let (_, zerocheck_fault) = sumcheck::verify_mlecheck_recording(
                &zproof,
                &tau,
                F128::ZERO,
                2,
                &mut rmip_iop::replay::CoinReplay::new(&zerocheck_coins),
            );
            let (_, lincheck_fault) = sumcheck::verify_recording(
                &lproof,
                batched,
                lproof.round_polys.len(),
                2,
                &mut rmip_iop::replay::CoinReplay::new(&lincheck_coins),
            );
            let replayed = t0.elapsed();
            assert!(
                verdict.is_err() || (zerocheck_fault.is_none() && lincheck_fault.is_none()),
                "the timing replay must accept what the live verifier accepted: {zerocheck_fault:?} {lincheck_fault:?}"
            );
            replayed + t_open
        },
        t_open,
        verifier_phases,
        prover_phases,
        prover_piop_phases: Vec::new(),
    };
    drop(_diagnostics_role);
    measured!(MainProver, {
        drop(lc);
        drop(zc);
        drop(zf);
    });
    (execution, verdict)
}
