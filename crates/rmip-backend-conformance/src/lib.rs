use rmip_field::BinaryField;
use rmip_iop::backend::{Alphabet, Claim, CommitmentBackend, Coord, Regime, Symbol};
use rmip_iop::transcript::{Entry, OracleId, RawChallenger, SeededChallenger, TranscriptPrefix};

#[derive(Debug)]
pub struct Violation {
    pub item: u8,
    pub what: String,
}

fn fail(item: u8, what: impl Into<String>) -> Violation {
    Violation {
        item,
        what: what.into(),
    }
}

fn prefix_is_strictly_before_the_posting(
    entries: &[Entry],
    id: OracleId,
    h: &TranscriptPrefix,
) -> bool {
    let Some(pos) = entries
        .iter()
        .position(|e| matches!(e, Entry::OraclePost(x) if *x == id))
    else {
        return false;
    };
    let Ok(decoded) = h.decode() else {
        return false;
    };
    h.entry_count() == pos && decoded == entries[..pos]
}

pub fn check<B: CommitmentBackend>(
    backend: &mut B,
    z: &[B::Field],
    claims: &[Claim<B::Field>],
    frontend_has_booleanity_rows: bool,
) -> Vec<Violation> {
    let mut v = Vec::new();
    let mut ch = SeededChallenger::new(0xC0F1);

    let params = backend.public_params(z.len(), claims.len());
    let p0 = backend.public_params(z.len(), 0);
    let p1 = backend.public_params(z.len(), claims.len() + 8);

    backend.commit(z, &mut ch);
    let post_commit_entries = ch.transcript().entries().len();
    let params_after_commit = backend.params(claims.len());
    backend.open(claims, &mut ch);
    let params_after_open = backend.params(claims.len());

    let oracles = backend.oracles();
    let pre_sampling_entries = ch.transcript().entries().len();
    let mut plan = backend.query_plan(&mut ch);
    let post_sampling_entries = ch.transcript().entries().len();
    let mut occurrences = Vec::new();
    let mut prev: Option<Symbol<B::Field>> = None;
    while let Some((id, coord)) = plan.next(prev.as_ref()) {
        match oracles.iter().find(|o| o.id == id) {
            None => {
                v.push(fail(1, format!("query names unposted oracle {id}")));
                break;
            }
            Some(o) => {
                let Some(ix) = usize::try_from(coord).ok().filter(|&i| i < o.symbols.len()) else {
                    v.push(fail(
                        1,
                        format!("coord {coord} out of range for oracle {id}"),
                    ));
                    break;
                };
                prev = Some(o.symbols[ix].clone());
                occurrences.push((id, ix));
            }
        }
        if occurrences.len() > params.q_long {
            break;
        }
    }
    if occurrences.len() != params.q_long {
        v.push(fail(
            1,
            format!(
                "plan yielded {} occurrences, params says q_long = {}",
                occurrences.len(),
                params.q_long
            ),
        ));
    }
    let raw_symbol_count: usize = oracles.iter().map(|o| o.symbols.len()).sum();
    if params.q_long == raw_symbol_count {
        let expected: std::collections::HashSet<_> = oracles
            .iter()
            .flat_map(|o| (0..o.symbols.len()).map(move |coord| (o.id, coord)))
            .collect();
        let actual: std::collections::HashSet<_> = occurrences.iter().copied().collect();
        if actual != expected || actual.len() != occurrences.len() {
            v.push(fail(
                1,
                format!(
                    "plan claims a full read of {raw_symbol_count} symbols but does not cover each tagged coordinate exactly once"
                ),
            ));
        }
    }

    for &(id, coord) in &occurrences {
        let o = oracles.iter().find(|o| o.id == id).unwrap();
        let Some(h) = ch.transcript().try_prefix_at(id) else {
            continue;
        };
        if backend.reconstruct(id, &h, Coord::from(coord)) != o.symbols[coord] {
            v.push(fail(
                2,
                format!("reconstruct disagrees with oracles() at ({id}, {coord})"),
            ));
            break;
        }
    }

    for entry in ch.transcript().entries() {
        let Entry::OraclePost(id) = entry else {
            continue;
        };
        if !oracles.iter().any(|oracle| oracle.id == *id) {
            v.push(fail(
                3,
                format!("oracle {id} was announced without registered content"),
            ));
            break;
        }
    }
    for o in oracles {
        let announced = ch
            .transcript()
            .entries()
            .iter()
            .any(|e| matches!(e, Entry::OraclePost(x) if *x == o.id));
        if !announced {
            v.push(fail(
                3,
                format!("oracle {} was never announced via post_oracle", o.id),
            ));
            continue;
        }
        let h = ch.transcript().prefix_at(o.id);
        if !prefix_is_strictly_before_the_posting(ch.transcript().entries(), o.id, &h) {
            v.push(fail(
                3,
                format!(
                    "oracle {}'s prefix is not exactly the public events strictly before its posting",
                    o.id
                ),
            ));
        }
    }

    if p1.eps_open < p0.eps_open {
        v.push(fail(4, "eps_open decreased as the claim count grew"));
    }
    let params_after_queries = backend.params(claims.len());
    for (boundary, current) in [
        ("commit", &params_after_commit),
        ("open", &params_after_open),
        ("queries", &params_after_queries),
    ] {
        if !params.same_fixed_schedule(current) {
            v.push(fail(
                4,
                format!("protocol-fixed backend parameters changed after {boundary}"),
            ));
        }
    }
    if let Err(e) = params.check_admissible(frontend_has_booleanity_rows) {
        v.push(fail(4, e));
    }
    if let Regime::JohnsonOod { ood_samples } = params.regime {
        let window = &ch.transcript().entries()[..post_commit_entries];
        let coins = window
            .iter()
            .filter(|e| matches!(e, Entry::Coin(_)))
            .count();
        let last_coin = window.iter().rposition(|e| matches!(e, Entry::Coin(_)));
        let responded = match last_coin {
            Some(i) => window[i + 1..]
                .iter()
                .any(|e| matches!(e, Entry::Ordinary(_))),
            None => false,
        };
        if post_commit_entries == 0 {
            v.push(fail(
                4,
                "JohnsonOod backend produced no commitment-phase transcript",
            ));
        } else if coins < ood_samples {
            v.push(fail(
                4,
                format!(
                    "{} declares ood_samples = {ood_samples} but drew only {coins} coins \
                     inside `commit`; the out-of-domain points are not sampled before the cut",
                    params.name
                ),
            ));
        } else if !responded {
            v.push(fail(
                4,
                format!(
                    "{}: no prover response follows the last commitment-phase coin, so the \
                     out-of-domain exchange does not COMPLETE before the cut",
                    params.name
                ),
            ));
        }
    }

    for o in oracles {
        let Some(h) = ch.transcript().try_prefix_at(o.id) else {
            continue;
        };
        for coord in [
            0usize,
            o.symbols.len() / 2,
            o.symbols.len().saturating_sub(1),
        ] {
            if coord >= o.symbols.len() {
                continue;
            }
            if backend.reconstruct(o.id, &h, Coord::from(coord)) != o.symbols[coord] {
                v.push(fail(
                    6,
                    format!("oracle {} not reconstructible at {coord}", o.id),
                ));
                break;
            }
        }
    }

    if params.alphabet.len() != oracles.len() {
        v.push(fail(
            7,
            format!(
                "declared {} alphabets for {} oracles",
                params.alphabet.len(),
                oracles.len()
            ),
        ));
    }
    for (o, a) in oracles.iter().zip(&params.alphabet) {
        if let Some(bad) = o.symbols.iter().find(|s| s.0.len() != a.elements()) {
            v.push(fail(
                7,
                format!(
                    "oracle {} has a {}-element symbol, alphabet declares {}",
                    o.id,
                    bad.0.len(),
                    a.elements()
                ),
            ));
            break;
        }
    }
    let expected_a_long: u64 = occurrences
        .iter()
        .map(|(id, _)| {
            let idx = oracles.iter().position(|o| o.id == *id).unwrap();
            params
                .alphabet
                .get(idx)
                .copied()
                .unwrap_or(Alphabet::Field)
                .elements() as u64
        })
        .sum();
    if params.a_long != expected_a_long {
        v.push(fail(
            7,
            format!(
                "A_long = {} but the plan returns {expected_a_long} elements",
                params.a_long
            ),
        ));
    }
    let expected_l_long = oracles.iter().try_fold(Coord::ZERO, |total, oracle| {
        total.checked_add(oracle.raw_len()?)
    });
    if let Some(expected_l_long) = expected_l_long {
        if params.l_long != expected_l_long {
            v.push(fail(
                7,
                format!(
                    "L_long = {} but oracles hold {expected_l_long}",
                    params.l_long
                ),
            ));
        }
    } else {
        v.push(fail(7, "posted-oracle length overflows u128 accounting"));
    }

    if oracles.iter().any(|o| o.symbols.is_empty()) {
        v.push(fail(8, "an oracle was posted with no symbols"));
    }
    for e in &ch.transcript().entries()[pre_sampling_entries..post_sampling_entries] {
        match e {
            Entry::Coin(_) => {}
            Entry::OraclePost(id) => {
                v.push(fail(
                    8,
                    format!("oracle {id} posted during local-validity sampling"),
                ));
                break;
            }
            Entry::Ordinary(_) => {
                v.push(fail(
                    8,
                    "ordinary message sent during local-validity sampling",
                ));
                break;
            }
        }
    }

    let coordinate_count: u64 = oracles.iter().map(|o| o.symbols.len() as u64).sum();
    let is_proper_subset = (params.q_long as u64) < coordinate_count;
    if is_proper_subset {
        if post_sampling_entries == pre_sampling_entries {
            v.push(fail(
                9,
                format!(
                    "{} queries {} of {coordinate_count} coordinates and declares eps_open = {:.6}, but drew no verifier coin to choose them",
                    params.name, params.q_long, params.eps_open
                ),
            ));
        } else {
            let coords = |seed: u64| -> Vec<(rmip_iop::transcript::OracleId, Coord)> {
                let mut c = SeededChallenger::new(seed);
                let mut p = backend.query_plan(&mut c);
                let mut out = Vec::new();
                let mut prev: Option<Symbol<B::Field>> = None;
                while out.len() <= params.q_long {
                    let Some(x) = p.next(prev.as_ref()) else {
                        break;
                    };
                    prev = oracles
                        .iter()
                        .find(|o| o.id == x.0)
                        .and_then(|o| usize::try_from(x.1).ok().and_then(|i| o.symbols.get(i)))
                        .cloned();
                    out.push(x);
                }
                out
            };
            let samples: Vec<_> = (0..64).map(|k| coords(0xA1 + k)).collect();
            let base = &samples[0];
            if samples[1..].iter().all(|x| x == base) {
                v.push(fail(
                    9,
                    format!(
                        "{} spends coins but its query set does not depend on them: 64 different challengers gave the same {} coordinates",
                        params.name,
                        base.len()
                    ),
                ));
            } else if samples.iter().all(|x| {
                x.len() == params.q_long
                    && x.iter().all(|(id, coord)| {
                        oracles
                            .iter()
                            .find(|o| o.id == *id)
                            .is_some_and(|o| *coord < Coord::from(o.symbols.len()))
                    })
            }) {
                let traces: Vec<Vec<_>> = (0..params.q_long)
                    .map(|pos| samples.iter().map(|run| run[pos]).collect())
                    .collect();
                let copied = (0..params.q_long).any(|a| {
                    (a + 1..params.q_long)
                        .any(|b| traces[a][0].0 == traces[b][0].0 && traces[a] == traces[b])
                });
                if copied {
                    v.push(fail(
                        9,
                        format!(
                            "{} repeats one sampled coordinate in multiple same-oracle query positions instead of drawing independent samples",
                            params.name
                        ),
                    ));
                } else {
                    let misses_half = traces.iter().any(|trace| {
                        let id = trace[0].0;
                        if trace.iter().any(|x| x.0 != id) {
                            return false;
                        }
                        let Some(len) =
                            oracles.iter().find(|o| o.id == id).map(|o| o.symbols.len())
                        else {
                            return false;
                        };
                        len >= 2
                            && (!trace.iter().any(|x| x.1 < Coord::from(len / 2))
                                || !trace.iter().any(|x| x.1 >= Coord::from(len / 2)))
                    });
                    if misses_half {
                        v.push(fail(
                            9,
                            format!(
                                "{} sampled query coordinates never reach both halves of an oracle across 64 challenger seeds",
                                params.name
                            ),
                        ));
                    }
                }
            }
        }
    }

    v
}

pub fn check_deterministic<B: CommitmentBackend, F: Fn() -> B>(
    make: F,
    z: &[B::Field],
    claims: &[Claim<B::Field>],
) -> Vec<Violation> {
    let run = || {
        let mut b = make();
        let mut ch = SeededChallenger::new(0xDE7);
        b.commit(z, &mut ch);
        b.open(claims, &mut ch);
        let syms: Vec<Vec<Vec<u8>>> = b
            .oracles()
            .iter()
            .map(|o| {
                o.symbols
                    .iter()
                    .map(|s| s.0.iter().flat_map(|f| f.to_be_bytes()).collect())
                    .collect()
            })
            .collect();
        (ch.transcript().canonical(), syms)
    };
    if run() != run() {
        vec![fail(
            5,
            "same coins produced a different transcript or different oracles",
        )]
    } else {
        Vec::new()
    }
}

pub fn check_prefix_only_reconstruction<B: CommitmentBackend, F: Fn() -> B>(
    make: F,
    z: &[B::Field],
    claims: &[Claim<B::Field>],
) -> Vec<Violation> {
    let mut v = Vec::new();

    let mut prover = make();
    let mut ch = SeededChallenger::new(0x9110);
    prover.commit(z, &mut ch);
    prover.open(claims, &mut ch);

    let mut checker = make();
    let mut own = SeededChallenger::new(0xFACE);
    checker.commit(z, &mut own);

    for o in prover.oracles() {
        let Some(h) = ch.transcript().try_prefix_at(o.id) else {
            v.push(fail(10, format!("oracle {} was never announced", o.id)));
            continue;
        };
        for coord in [
            0usize,
            o.symbols.len() / 3,
            o.symbols.len().saturating_sub(1),
        ] {
            if coord >= o.symbols.len() {
                continue;
            }
            if checker.reconstruct(o.id, &h, Coord::from(coord)) != o.symbols[coord] {
                v.push(fail(
                    10,
                    format!(
                        "oracle {} coord {coord}: an instance that ran only `commit` \
                         could not reproduce it from the legal prefix — `reconstruct` \
                         is reading state that the checker does not have",
                        o.id
                    ),
                ));
                break;
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_threes_precedence_check_rejects_a_prefix_carrying_a_later_coin() {
        let mut ch = SeededChallenger::new(0x5E7);
        ch.record_ordinary(b"a prover message before anything is posted");
        ch.sample();
        ch.post_oracle(0);
        ch.sample();
        ch.post_oracle(1);

        let entries = ch.transcript().entries();
        let h0 = ch.transcript().prefix_at(0);
        let h1 = ch.transcript().prefix_at(1);
        let whole = ch.transcript().canonical();

        assert!(prefix_is_strictly_before_the_posting(entries, 0, &h0));
        assert!(prefix_is_strictly_before_the_posting(entries, 1, &h1));
        assert_eq!(h0.entry_count(), 2);
        assert_eq!(h1.entry_count(), 4);

        assert!(
            !prefix_is_strictly_before_the_posting(entries, 0, &h1),
            "oracle 1's prefix carries oracle 0's posting and a later coin"
        );
        for id in [0, 1] {
            assert!(
                !prefix_is_strictly_before_the_posting(entries, id, &whole),
                "the whole transcript is not a legal prefix for oracle {id}"
            );
        }
        assert!(
            !prefix_is_strictly_before_the_posting(entries, 9, &h0),
            "an oracle that was never announced has no legal prefix"
        );
    }
}
