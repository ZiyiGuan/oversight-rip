//! Grind-free statistical parameters for the extracted BLAKE3 IOP.
//!
//! This is a sum of the native query and algebraic error terms, never the
//! minimum per-round work factor used by a Fiat--Shamir profile. The native
//! statistical composition is an explicit assumption; this is not a proof of
//! the two-prover compiler or its additional collision/audit terms.

use super::{
    F128, F256, LigeritoProfile, LigeritoSecurityConfig, ProverConfig, SoundnessRegime,
    VerifierConfig, embedded_security_config, johnson_interleaved_list_log2, paper_johnson_log_a,
    paper_per_query_bits,
};
use crate::pcs::LOG_PACKING;

pub const TARGET_BITS: usize = 40;
/// A quarter of the target error for queries, a quarter for native algebra;
/// the remaining half is reserved for compiler error and unused slack.
pub const FAMILY_BUDGET_BITS: usize = TARGET_BITS + 2;

#[derive(Clone, Debug)]
pub struct StatisticalErrorTerm {
    pub name: String,
    pub error: f64,
}

#[derive(Clone, Debug)]
pub struct InteractiveSecurityReport {
    pub target_bits: usize,
    pub base_field_bits: usize,
    pub fold_field_bits: usize,
    pub queries: Vec<usize>,
    pub query_error: f64,
    pub native_nonquery_error: f64,
    pub native_error: f64,
    pub terms: Vec<StatisticalErrorTerm>,
}

impl InteractiveSecurityReport {
    pub fn native_error_bits(&self) -> f64 {
        -self.native_error.log2()
    }
}

/// The exact runtime field sizes are used in every algebraic denominator.
/// The Fast ladder is reused at the same packed dimension, so 64-bit packing
/// selects the 128-bit reference geometry at `m + 1`, rather than silently
/// dropping half the witness. No native grinding contribution is retained.
pub fn configs_for(
    log_n: usize,
    log_batch_size: usize,
) -> Result<(ProverConfig, VerifierConfig, InteractiveSecurityReport), String> {
    let m = log_n + LOG_PACKING;
    let encoded = embedded_security_config(m, LigeritoProfile::Interactive40)
        .ok_or_else(|| format!("no interactive ladder registered at m={m}"))?;
    // These reference TOMLs describe F128/F256 geometry. Do not invoke their
    // legacy per-round security validator under a different runtime field.
    let source: LigeritoSecurityConfig =
        toml::from_str(encoded).map_err(|e| format!("interactive ladder: {e}"))?;
    if source.log_n != log_n || source.initial_k != log_batch_size {
        return Err("interactive ladder does not match the packed dimension/batch size".into());
    }
    if source.levels.is_empty()
        || source.initial_k
            + source
                .levels
                .iter()
                .skip(1)
                .map(|lv| lv.k_recursive - 1)
                .sum::<usize>()
            + source.final_block.yr_log_n
            != log_n
    {
        return Err("invalid interactive recursive dimensions".into());
    }
    let mut dimension = log_n;
    for (j, lv) in source.levels.iter().enumerate() {
        if j > 0 {
            dimension += 1;
        }
        if lv.log_msg_cols + lv.log_num_interleaved != dimension
            || lv.log_num_interleaved != lv.k_recursive
            || lv.regime != SoundnessRegime::JohnsonOod
            || lv.ood_samples != if j == 0 { 1 } else { 2 }
        {
            return Err(format!(
                "unsupported interactive ladder shape/regime at L{j}"
            ));
        }
        dimension = lv.log_msg_cols;
    }
    let per_level_budget_bits = FAMILY_BUDGET_BITS as f64 + (source.levels.len() as f64).log2();
    let mut queries = Vec::with_capacity(source.levels.len());
    for (j, lv) in source.levels.iter().enumerate() {
        let eta = lv
            .eta
            .ok_or_else(|| format!("missing Johnson slack at L{j}"))?;
        let per_query = paper_per_query_bits(lv.log_inv_rate, eta);
        if !per_query.is_finite() || per_query <= 0.0 {
            return Err(format!("invalid Johnson query bound at L{j}"));
        }
        let q = (per_level_budget_bits / per_query).floor() as usize + 1;
        if q > 1usize << (lv.log_msg_cols + lv.log_inv_rate) {
            return Err(format!(
                "40-bit target needs more queries than positions at L{j}"
            ));
        }
        queries.push(q);
    }
    let report = blake3_error_report(m, &source, queries.clone())?;
    let rates: Vec<_> = source.levels.iter().map(|lv| lv.log_inv_rate).collect();
    let columns: Vec<_> = source
        .levels
        .iter()
        .skip(1)
        .map(|lv| lv.log_msg_cols)
        .collect();
    let folds: Vec<_> = source
        .levels
        .iter()
        .skip(1)
        .map(|lv| lv.k_recursive)
        .collect();
    let ood: Vec<_> = source.levels.iter().map(|lv| lv.ood_samples).collect();
    let zero = vec![0; queries.len()];
    let prover = ProverConfig {
        log_inv_rates: rates.clone(),
        recursive_steps: folds.len(),
        initial_log_msg_cols: source.levels[0].log_msg_cols,
        initial_log_num_interleaved: log_batch_size,
        initial_k: log_batch_size,
        recursive_log_msg_cols: columns.clone(),
        recursive_ks: folds.clone(),
        queries: queries.clone(),
        grinding_bits: zero.clone(),
        fold_grinding_bits: zero.clone(),
        claim_batch_grinding_bits: zero.clone(),
        consistency_batch_grinding_bits: zero.clone(),
        ood_samples: ood.clone(),
        merkle_hash: source.merkle_hash()?,
        stratified: vec![],
    }
    .with_default_stratified();
    let verifier = VerifierConfig {
        log_inv_rates: rates,
        recursive_steps: folds.len(),
        initial_log_msg_cols: source.levels[0].log_msg_cols,
        initial_log_num_interleaved: log_batch_size,
        initial_k: log_batch_size,
        recursive_log_msg_cols: columns,
        recursive_ks: folds,
        queries,
        grinding_bits: zero.clone(),
        fold_grinding_bits: zero.clone(),
        claim_batch_grinding_bits: zero.clone(),
        consistency_batch_grinding_bits: zero,
        ood_samples: ood,
        merkle_hash: source.merkle_hash()?,
        stratified: vec![],
    }
    .with_default_stratified();
    prover.validate_stratified()?;
    verifier.validate_stratified()?;
    Ok((prover, verifier, report))
}

fn blake3_error_report(
    m: usize,
    source: &LigeritoSecurityConfig,
    queries: Vec<usize>,
) -> Result<InteractiveSecurityReport, String> {
    // A partial BLAKE3 batch can have padded circuit dimension one above
    // the dense commitment's m (11707 live bits out of 16384 per block).
    // Charge that larger dimension so the report also covers partial batches.
    let m = m + 1;
    let base_inv = (-(F128::BITS as f64)).exp2();
    let fold_inv = (-(F256::BITS as f64)).exp2();
    let mut terms = Vec::new();
    let mut add = |name: String, error: f64| {
        terms.push(StatisticalErrorTerm { name, error });
    };
    let l0 = &source.levels[0];
    let l0_list = johnson_interleaved_list_log2(l0.log_inv_rate, l0.eta.unwrap()).exp2();
    // The genus-95 cover's Hasse--Weil loss plus removed exceptional points
    // is bounded conservatively by 256/sqrt(|F|). This is 2^-56 at 128
    // bits and 2^-24 at 64 bits; retaining 2^-56 in a small field is wrong.
    let cover_fraction = 1.0 - 256.0 * (-(F128::BITS as f64) / 2.0).exp2();
    if cover_fraction <= 0.0 {
        return Err("field too small for the AG sampler point-count bound".into());
    }
    for (name, numerator) in [
        ("zerocheck_initial", m as f64),
        ("zerocheck_exceptional_denominators", m as f64),
        (
            "ag_round1_and_lincheck_skip",
            // The base code has dimension 64 on the genus-95 curve:
            // deg(D) = dim + genus - 1. The round-one combined zero
            // bound is pinned by the AG kernel's own constant test.
            (crate::zerocheck::ag_skip::R1_ZERO_BOUND
                + crate::genus95_curve_code::BASE_MESSAGE_BITS
                + 95
                - 1) as f64
                / cover_fraction,
        ),
        ("zerocheck_quadratic_rounds", (2 * (m - 6)) as f64),
        ("lincheck_alpha_beta_and_rounds", 18.0),
        ("two_ring_switches", (2 * LOG_PACKING) as f64),
        ("outer_and_inner_claim_batches", 2.0 * l0_list),
        ("merged_quadratic_rounds", (2 * (m - LOG_PACKING)) as f64),
        ("multipoint_gamma", (2 * F128::BITS - 1) as f64),
        (
            "multipoint_quadratic_rounds",
            (2 * (m - LOG_PACKING)) as f64,
        ),
        (
            "anchor_quadratic_rounds",
            (4 * (m - LOG_PACKING + 1)) as f64,
        ),
    ] {
        add(name.into(), numerator * base_inv);
    }
    let mut query_error = 0.0;
    for (j, (lv, &q)) in source.levels.iter().zip(&queries).enumerate() {
        let eta = lv.eta.unwrap();
        let list = johnson_interleaved_list_log2(lv.log_inv_rate, eta).exp2();
        let mu = (lv.log_msg_cols + lv.log_num_interleaved) as f64;
        let query = (-(q as f64) * paper_per_query_bits(lv.log_inv_rate, eta)).exp2();
        query_error += query;
        add(format!("L{j}_query"), query);
        let mca = paper_johnson_log_a(
            lv.log_inv_rate,
            eta,
            lv.log_msg_cols,
            lv.log_num_interleaved,
        )
        .exp2()
            * fold_inv;
        let mut ood = (mu * base_inv).powi(lv.ood_samples as i32) * list * list / 2.0;
        if j == 0 {
            ood *= (mu + LOG_PACKING as f64) * base_inv;
        }
        let batches = lv.ood_samples
            + usize::from(j > 0)
            + usize::from(j > 0 && j == source.levels.len() - 1);
        for (kind, error) in [
            ("mca_all_folds", lv.k_recursive as f64 * mca),
            (
                "sumcheck_all_folds",
                lv.k_recursive as f64 * 2.0 * list * fold_inv,
            ),
            ("ood", ood),
            ("claim_batches", batches as f64 * list * base_inv),
            (
                "consistency_batch",
                list * (q as f64).log2().ceil() * base_inv,
            ),
        ] {
            add(format!("L{j}_{kind}"), error);
        }
    }
    let native_nonquery_error: f64 = terms
        .iter()
        .filter(|t| !t.name.ends_with("_query"))
        .map(|t| t.error)
        .sum();
    let native_error = query_error + native_nonquery_error;
    let budget = (-(FAMILY_BUDGET_BITS as f64)).exp2();
    if !native_error.is_finite() || query_error > budget || native_nonquery_error > budget {
        return Err(format!(
            "40-bit statistical target unavailable: query bits {:.6}, native nonquery bits {:.6}; each family needs {} bits",
            -query_error.log2(),
            -native_nonquery_error.log2(),
            FAMILY_BUDGET_BITS
        ));
    }
    Ok(InteractiveSecurityReport {
        target_bits: TARGET_BITS,
        base_field_bits: F128::BITS,
        fold_field_bits: F256::BITS,
        queries,
        query_error,
        native_nonquery_error,
        native_error,
        terms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::ligerito::embedded_initial_k;

    #[test]
    fn interactive40_uses_summed_bound_and_no_grinding() {
        for m in [22, 26, 28, 30, 32] {
            let batch = embedded_initial_k(m, LigeritoProfile::Interactive40).unwrap();
            let (p, v, report) = configs_for(m - LOG_PACKING, batch).unwrap();
            assert_eq!(p.queries, v.queries);
            assert_eq!(p.queries, report.queries);
            assert_eq!(report.base_field_bits, F128::BITS);
            assert_eq!(report.fold_field_bits, F256::BITS);
            assert!(report.native_error < 2.0f64.powi(-41));
            let summed: f64 = report.terms.iter().map(|t| t.error).sum();
            assert!((summed - report.native_error).abs() < report.native_error * 1e-12);
            for schedule in [
                &p.grinding_bits,
                &p.fold_grinding_bits,
                &p.claim_batch_grinding_bits,
                &p.consistency_batch_grinding_bits,
            ] {
                assert!(schedule.iter().all(|&bits| bits == 0));
            }
        }
    }

    #[test]
    fn understated_query_count_cannot_pass_the_summed_target() {
        let source: LigeritoSecurityConfig =
            toml::from_str(embedded_security_config(30, LigeritoProfile::Interactive40).unwrap())
                .unwrap();
        assert!(blake3_error_report(30, &source, vec![1; source.levels.len()]).is_err());
    }

    #[test]
    fn wrong_batch_and_unregistered_shape_are_rejected() {
        assert!(configs_for(30 - LOG_PACKING, 0).is_err());
        assert!(configs_for(60 - LOG_PACKING, 6).is_err());
    }

    #[test]
    #[cfg(not(feature = "field64"))]
    fn default_field_queries_match_the_independent_40bit_ledger() {
        for (m, expected) in [
            (26, vec![96, 31, 19, 14]),
            (28, vec![97, 32, 19, 14, 12]),
            (30, vec![97, 32, 19, 14, 12]),
        ] {
            let batch = embedded_initial_k(m, LigeritoProfile::Interactive40).unwrap();
            let (p, _, _) = configs_for(m - LOG_PACKING, batch).unwrap();
            assert_eq!(p.queries, expected);
        }
    }
}
