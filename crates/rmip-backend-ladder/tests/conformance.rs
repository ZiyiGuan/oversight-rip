use rmip_backend_ladder::{commit_steps, events_of, open_steps, Kind, Ladder, LadderShape, Step};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{AuditMode, Claim, CommitmentBackend, Coord, OpeningVerifier};
use rmip_iop::mle::eval_mle;
use rmip_iop::transcript::{Entry, RawChallenger, SeededChallenger, Transcript};

fn toy_depth1() -> LadderShape {
    LadderShape {
        nu: 6,
        log_m0: 3,
        rate_inv: 4,
        lambda_bits: 1.0,
        pins: 2,
        log_n_min: 2,
    }
}

fn toy_depth2() -> LadderShape {
    LadderShape {
        nu: 7,
        log_m0: 3,
        rate_inv: 4,
        lambda_bits: 1.0,
        pins: 2,
        log_n_min: 2,
    }
}

fn table(nu: usize) -> Vec<F128> {
    (0..1usize << nu)
        .map(|i| F128::from_bool((i * 7 + 3) % 5 < 2))
        .collect()
}

fn one_claim(z: &[F128], nu: usize) -> Vec<Claim<F128>> {
    let p: Vec<F128> = (0..nu)
        .map(|i| F128::new(7919 * (i as u128 + 3) + 11))
        .collect();
    vec![Claim {
        value: eval_mle(z, &p),
        point: p,
    }]
}

fn conforms(shape: LadderShape) {
    let z = table(shape.nu);
    let c = one_claim(&z, shape.nu);
    let mut b = Ladder::new(shape);
    let v = rmip_backend_conformance::check(&mut b, &z, &c, false);
    assert!(v.is_empty(), "{v:#?}");
    let v = rmip_backend_conformance::check_deterministic(|| Ladder::new(shape), &z, &c);
    assert!(v.is_empty(), "{v:#?}");
    let v =
        rmip_backend_conformance::check_prefix_only_reconstruction(|| Ladder::new(shape), &z, &c);
    assert!(v.is_empty(), "{v:#?}");
}

#[test]
fn the_depth_one_ladder_conforms() {
    conforms(toy_depth1());
}

#[test]
fn the_depth_two_ladder_conforms() {
    conforms(toy_depth2());
}

#[test]
fn a_pinless_depth_zero_ladder_conforms_and_declares_rewinding() {
    let shape = LadderShape {
        nu: 6,
        log_m0: 3,
        rate_inv: 4,
        lambda_bits: 1.0,
        pins: 0,
        log_n_min: 3,
    };
    conforms(shape);
    let z = table(shape.nu);
    let c = one_claim(&z, shape.nu);
    let mut b = Ladder::new(shape);
    let mut ch = SeededChallenger::new(3);
    b.commit(&z, &mut ch);
    b.open(&c, &mut ch);
    let p = b.params(1);
    assert_eq!(p.audit, AuditMode::Rewind);
    assert_eq!(p.q_long, 1);
    assert_eq!(rmip_backend_ladder::eps_col_at(shape, 0, 128), 0.0);
    assert_eq!(rmip_backend_ladder::collision_total(shape, 128), 0.0);
    let s = b.schedule();
    let q = s.queries;
    let escape = 2.0 * (1.0 - shape.delta()).powi(q as i32);
    assert!(p.eps_open >= escape && p.eps_open < escape * (1.0 + 1e-6) + 1e-9);
    let mut coins = 0;
    let mut messages = 0;
    let mut posts = 0;
    for e in ch.transcript().entries() {
        match e {
            Entry::Coin(_) => coins += 1,
            Entry::Ordinary(_) => messages += 1,
            Entry::OraclePost(_) => posts += 1,
        }
    }
    assert_eq!(posts, 1, "one table");
    assert_eq!(
        coins,
        shape.m0() + 1,
        "the proximity coins and the batching coin, no pin coins"
    );
    assert_eq!(
        messages, 3,
        "the echoed point, the proximity reply, the opening reply"
    );
    for st in commit_steps(s).into_iter().chain(open_steps(s)) {
        if let Step::PinCoins(_) | Step::ClearPin(_) = st {
            assert_eq!(
                events_of(s, st),
                (0, 0, 0),
                "a pin step of a pin-free ladder is empty"
            );
        }
    }
}

#[test]
#[should_panic(expected = "needs at least one pinning functional")]
fn a_pinless_shape_below_depth_zero_is_refused() {
    let mut shape = toy_depth1();
    shape.pins = 0;
    let _ = shape.schedule();
}

#[test]
fn a_depth_zero_ladder_is_the_bundled_backend_and_conforms() {
    conforms(LadderShape {
        nu: 6,
        log_m0: 3,
        rate_inv: 4,
        lambda_bits: 1.0,
        pins: 1,
        log_n_min: 3,
    });
}

#[test]
fn every_table_is_pinned_before_it_is_tested() {
    let shape = toy_depth2();
    let z = table(shape.nu);
    let c = one_claim(&z, shape.nu);
    let mut b = Ladder::new(shape);
    let mut ch = SeededChallenger::new(5);
    b.commit(&z, &mut ch);
    b.open(&c, &mut ch);
    let s = b.schedule().clone();
    let entries = ch.transcript().entries().to_vec();
    let mut predicted: Vec<char> = Vec::new();
    for st in commit_steps(&s).into_iter().chain(open_steps(&s)) {
        let (coins, msgs, posts) = events_of(&s, st);
        predicted.extend(std::iter::repeat_n('c', coins));
        predicted.extend(std::iter::repeat_n('m', msgs));
        predicted.extend(std::iter::repeat_n('p', posts));
    }
    let actual: Vec<char> = entries
        .iter()
        .map(|e| match e {
            Entry::Coin(_) => 'c',
            Entry::Ordinary(_) => 'm',
            Entry::OraclePost(_) => 'p',
        })
        .collect();
    assert_eq!(
        actual, predicted,
        "the transcript is the step lists' events"
    );
    let steps: Vec<Step> = commit_steps(&s).into_iter().chain(open_steps(&s)).collect();
    let pos = |pred: &dyn Fn(&Step) -> bool| steps.iter().position(pred).expect("step present");
    for t in &s.tables {
        let posted = pos(&|st| *st == Step::BuildAndPost(t.id));
        let pinned = pos(&|st| *st == Step::PinCoins(t.id));
        let tested = pos(&|st| *st == Step::ProxCoins(t.id));
        assert!(posted < pinned && pinned < tested, "table {}", t.id);
        let fixed = match s.child_rows(t.id, rmip_backend_ladder::Reply::Pin(0)) {
            Some((child, _, _)) => pos(&|st| *st == Step::BuildAndPost(child)),
            None => pos(&|st| *st == Step::ClearPin(t.id)),
        };
        assert!(
            pinned < fixed && fixed < tested,
            "table {}: pin reply fixed before test",
            t.id
        );
    }
}

#[test]
fn the_plan_reads_one_path() {
    let shape = toy_depth2();
    let b = Ladder::new(shape);
    let mut ch = SeededChallenger::new(9);
    let mut plan = b.query_plan(&mut ch);
    let mut got = Vec::new();
    while let Some(x) = plan.next(None) {
        got.push(x);
    }
    let s = b.schedule();
    assert_eq!(got.len(), s.tables.len());
    let base = b.tuple_at(0, got[0].1).expect("level-0 tuple");
    for (k, (id, theta)) in got.iter().enumerate() {
        assert_eq!(*id, k, "posting order");
        let j = b.tuple_at(*id, *theta).expect("in domain");
        let want: Vec<usize> = base.iter().map(|&c| c >> s.table(*id).level).collect();
        assert_eq!(j, want, "table {id} reads the path");
    }
}

#[test]
fn parameters_are_declared_honestly() {
    let shape = toy_depth2();
    let mut b = Ladder::new(shape);
    let z = table(shape.nu);
    let mut ch = SeededChallenger::new(1);
    b.commit(&z, &mut ch);
    let p = b.params(1);
    let s = b.schedule();
    assert_eq!(p.name, "bundled-ladder");
    assert_eq!(p.audit, AuditMode::Sequential);
    assert_eq!(p.q_long, s.tables.len());
    assert_eq!(p.q_long, 6);
    assert_eq!(p.eps_bind, 0.0);
    let q = s.queries;
    let delta = shape.delta();
    let escape = 2.0 * (1.0 - delta).powi(q as i32);
    assert!(p.eps_open >= escape && p.eps_open < escape * (1.0 + 1e-6) + 1e-9);
    let a: u64 = s.tables.iter().map(|t| (q * t.rows.len()) as u64).sum();
    assert_eq!(p.a_long, a);
    assert_eq!(p.alphabet.len(), s.tables.len());
    assert_eq!(p.schedule.len(), s.depth + 1);
    assert!(p.check_admissible(false).is_ok());
}

#[test]
fn each_identity_has_an_isolating_adversary() {
    let shape = toy_depth2();
    let z = table(shape.nu);
    let c = one_claim(&z, shape.nu);
    let mut b = Ladder::new(shape);
    let mut ch = SeededChallenger::new(3);
    b.commit(&z, &mut ch);
    b.open(&c, &mut ch);
    let mut plan = b.query_plan(&mut ch);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        let o = b.oracles().iter().find(|o| o.id == id).unwrap();
        answers.push((
            id,
            coord,
            o.symbols[usize::try_from(coord).unwrap()].clone(),
        ));
    }
    assert!(b.verifier().check(&c, &answers), "honest accepts");
    let s = b.schedule().clone();
    let bump = |x: F128| x.add(F128::ONE);
    let mut cases = 0usize;
    for t in &s.tables {
        for i in 0..shape.pins {
            let mut v = b.verifier();
            v.states[t.id].pin_coins[i][0] = bump(v.states[t.id].pin_coins[i][0]);
            assert!(!v.check(&c, &answers), "table {} pin coin {i}", t.id);
            cases += 1;
        }
        let mut v = b.verifier();
        v.states[t.id].prox_coins[0] = bump(v.states[t.id].prox_coins[0]);
        assert!(!v.check(&c, &answers), "table {} prox coin", t.id);
        cases += 1;
        if t.level == s.depth {
            let mut v = b.verifier();
            v.states[t.id].prox_reply[0] = bump(v.states[t.id].prox_reply[0]);
            assert!(!v.check(&c, &answers), "table {} clear prox reply", t.id);
            let mut v = b.verifier();
            v.states[t.id].pin_replies[0][1] = bump(v.states[t.id].pin_replies[0][1]);
            assert!(!v.check(&c, &answers), "table {} clear pin reply", t.id);
            cases += 2;
            if t.has_open() {
                let mut v = b.verifier();
                v.states[t.id].open_reply[0] = bump(v.states[t.id].open_reply[0]);
                assert!(!v.check(&c, &answers), "table {} clear open reply", t.id);
                cases += 1;
            }
        }
    }
    let mut wrong = c.clone();
    wrong[0].value = bump(wrong[0].value);
    assert!(!b.verifier().check(&wrong, &answers));
    let mut v = b.verifier();
    v.point[0] = bump(v.point[0]);
    assert!(!v.check(&c, &answers));
    for k in 0..answers.len() {
        let mut tampered = answers.clone();
        tampered[k].2 .0[0] = bump(tampered[k].2 .0[0]);
        assert!(!b.verifier().check(&c, &tampered), "answer {k} tampered");
        let mut dropped = answers.clone();
        dropped.remove(k);
        assert!(!b.verifier().check(&c, &dropped), "answer {k} dropped");
    }
    assert!(
        cases >= 6 * 3,
        "{cases} single-field corruptions were tried"
    );
    let kinds: Vec<Kind> = s.tables.iter().map(|t| t.kind).collect();
    assert!(kinds.contains(&Kind::OpenPin));
}

#[test]
fn the_batching_coin_is_the_last_opening_event_and_honest_answers_pass_under_any_coin() {
    let shape = toy_depth2();
    let z = table(shape.nu);
    let c = one_claim(&z, shape.nu);
    let mut b = Ladder::new(shape);
    let mut ch = SeededChallenger::new(5);
    b.commit(&z, &mut ch);
    b.open(&c, &mut ch);
    let entries = ch.transcript().entries();
    assert!(
        matches!(entries.last(), Some(Entry::Coin(_))),
        "the opening phase ends with the batching coin"
    );
    let last_post = entries
        .iter()
        .rposition(|e| matches!(e, Entry::OraclePost(_)))
        .expect("tables were posted");
    let last_message = entries
        .iter()
        .rposition(|e| matches!(e, Entry::Ordinary(_)))
        .expect("clear replies were sent");
    assert!(last_post < entries.len() - 1 && last_message < entries.len() - 1);
    let mut plan = b.query_plan(&mut ch);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        let o = b.oracles().iter().find(|o| o.id == id).unwrap();
        answers.push((
            id,
            coord,
            o.symbols[usize::try_from(coord).unwrap()].clone(),
        ));
    }
    let mut tampered = answers.clone();
    tampered[2].2 .0[1] = tampered[2].2 .0[1].add(F128::ONE);
    let drawn = b.verifier();
    assert!(
        drawn.check(&c, &answers),
        "honest accepts under the drawn coin"
    );
    assert!(
        !drawn.check(&c, &tampered),
        "tampered rejects under the drawn coin"
    );
    for seed in [0u128, 1, 2, 0xdead_beef, u128::MAX] {
        let mut v = b.verifier();
        v.batch_coin = F128::new(seed);
        assert!(v.check(&c, &answers), "honest accepts under coin {seed:#x}");
        assert!(
            !v.check(&c, &tampered),
            "tampered rejects under coin {seed:#x}"
        );
    }
}

#[test]
fn the_main_prover_answers_from_its_own_state_as_the_checker_replays() {
    let shape = toy_depth2();
    let z = table(shape.nu);
    let c = one_claim(&z, shape.nu);
    let mut main = Ladder::lazy(shape);
    let mut ch = SeededChallenger::new(21);
    main.commit(&z, &mut ch);
    main.open(&c, &mut ch);
    let mut checker = Ladder::lazy(shape);
    checker.commit(&z, &mut SeededChallenger::new(22));
    let s = main.schedule().clone();
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut draws = SeededChallenger::new(23);
    for t in &s.tables {
        let h = ch.transcript().prefix_at(t.id);
        for _ in 0..3 {
            let n_enc = s.shape.n_enc_at(t.level);
            let j = rmip_backend_ladder::tuple::draw_tuple(&mut draws, n_enc, s.queries);
            let coord = main.coord_of_tuple(t.id, &j);
            assert_eq!(
                main.reconstruct(t.id, &h, coord),
                checker.reconstruct(t.id, &h, coord),
                "table {}: the main prover's own states and the checker's replay",
                t.id
            );
        }
        if t.id > 0 {
            let mut entries = h.decode().unwrap();
            let last_coin = entries
                .iter()
                .rposition(|e| matches!(e, Entry::Coin(_)))
                .expect("a coin precedes the posting");
            if let Entry::Coin(x) = &mut entries[last_coin] {
                *x = x.add(F128::ONE);
            }
            let mut other = Transcript::new();
            for e in &entries {
                match e {
                    Entry::Coin(x) => other.record_coin(*x),
                    Entry::Ordinary(m) => other.record_ordinary(m.clone()),
                    Entry::OraclePost(i) => other.post_oracle(*i),
                }
            }
            let h2 = other.canonical();
            assert_eq!(h2.len(), h.len());
            seed ^= seed << 7;
            let j: Vec<usize> = (0..s.queries)
                .map(|k| (seed as usize).wrapping_mul(k + 1) % s.shape.n_enc_at(t.level))
                .collect();
            let coord = main.coord_of_tuple(t.id, &j);
            assert_eq!(
                main.reconstruct(t.id, &h2, coord),
                checker.reconstruct(t.id, &h2, coord),
                "table {}: both replay a foreign prefix",
                t.id
            );
        }
    }
}

#[test]
fn the_cached_replay_equals_a_cold_one() {
    let shape = toy_depth2();
    let z = table(shape.nu);
    let c = one_claim(&z, shape.nu);
    let mut prover = Ladder::new(shape);
    let mut ch = SeededChallenger::new(21);
    prover.commit(&z, &mut ch);
    prover.open(&c, &mut ch);
    let mut warm = Ladder::new(shape);
    warm.commit(&z, &mut SeededChallenger::new(99));
    let s = prover.schedule().clone();
    for t in &s.tables {
        let h = ch.transcript().prefix_at(t.id);
        let mut cold = Ladder::new(shape);
        cold.commit(&z, &mut SeededChallenger::new(77));
        let coord = Coord::from(5u64 % u64::try_from(prover.domain_of(t.id)).unwrap());
        let a = warm.reconstruct(t.id, &h, coord);
        let b2 = cold.reconstruct(t.id, &h, coord);
        let posted = prover.oracles()[t.id].symbols[usize::try_from(coord).unwrap()].clone();
        assert_eq!(a, posted, "warm replay of table {}", t.id);
        assert_eq!(b2, posted, "cold replay of table {}", t.id);
    }
}
