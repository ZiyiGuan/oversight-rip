use rmip_backend_ladder::{LadderOf, LadderShape, SingleShotTarget};
use rmip_field::{F128, F64};
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use rmip_packed::execute::check_opening;
use rmip_packed::majority::{
    combined_words, count_from_top, prove, prove_piop_with_phases, sampled_bit, verify_piop,
    MajorityError, MajorityProof,
};
use rmip_packed::subfield::SkipTables;
use rmip_packed::tree::{LevelWitness, Tree};
use rmip_packed::PackedField;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn answers(views: usize, seed: u64) -> Vec<bool> {
    let mut s = seed;
    (0..views).map(|_| lcg(&mut s) % 100 < 94).collect()
}

fn sampled(views: usize, t: usize, seed: u64) -> Vec<usize> {
    let mut s = seed;
    (0..t).map(|_| lcg(&mut s) as usize % views).collect()
}

struct Setup<F: PackedField> {
    tree: Tree,
    tables: SkipTables<F>,
    shape: LadderShape,
    sampled: Vec<usize>,
    honest: Vec<bool>,
    threshold: u64,
}

fn ladder_shape(total_log: usize, kappa: usize) -> LadderShape {
    let target = SingleShotTarget::from_env();
    let depth = std::env::var("RMIP_LADDER_DEPTH")
        .ok()
        .and_then(|d| d.parse().ok());
    target
        .ladder_shape(total_log, depth, kappa)
        .unwrap_or_else(|| {
            panic!(
                "no ladder shape within the 2^-{} target at nu = {total_log}, depth {depth:?}, kappa {kappa}",
                target.bits
            )
        })
}

fn setup<F: PackedField>(views: usize, seed: u64) -> Setup<F> {
    let tree = Tree::new(views, 6, 6);
    let (_, total_log) = tree.layout(F::BITS.trailing_zeros() as usize);
    let shape = ladder_shape(total_log, F::BITS);
    Setup {
        tree,
        tables: SkipTables::<F>::new(),
        shape,
        sampled: sampled(views, 26, seed),
        honest: answers(views, seed),
        threshold: (views / 2) as u64,
    }
}

fn run<F: PackedField>(
    s: &Setup<F>,
    witness: &[LevelWitness<F>],
    seed: u64,
) -> (MajorityProof<F>, Result<(), MajorityError>, bool) {
    let mut backend = LadderOf::<F>::lazy(s.shape);
    let mut pch = SeededChallenger::new(seed);
    let proof = prove(
        &mut backend,
        &s.tables,
        &s.tree,
        witness,
        &s.sampled,
        &mut pch,
    );

    let mut vch = SeededChallenger::new(seed);
    let mut replay = LadderOf::<F>::lazy(s.shape);
    replay.commit(&F::to_words(&combined_words(&s.tree, witness)), &mut vch);
    let piop = verify_piop(
        &s.tables,
        &s.tree,
        s.threshold,
        &s.sampled,
        &proof.piop,
        &mut vch,
    );
    let batched = match &piop {
        Ok(b) => b.clone(),
        Err(e) => return (proof, Err(e.clone()), false),
    };
    assert_eq!(
        batched, proof.piop.batched,
        "the two sides derive the same claim"
    );
    replay.open(std::slice::from_ref(&batched), &mut vch);
    let mut plan = replay.query_plan(&mut vch);
    let checked = check_opening(&backend.verifier(), plan.as_mut(), &batched, &proof.answers);
    if let Err(e) = &checked {
        eprintln!(
            "check_opening failed: {e:?}; prover batched {:?}, verifier batched {:?}",
            proof.piop.batched.value, batched.value
        );
    }
    let opening = checked.is_ok();
    assert_eq!(
        pch.transcript().canonical(),
        vch.transcript().canonical(),
        "one transcript"
    );
    (proof, Ok(()), opening)
}

fn honest_accepted<F: PackedField>(views: usize, seed: u64) {
    {
        let s = setup::<F>(views, seed);
        let witness = s.tree.witness::<F>(&s.honest);
        let count = s.tree.count(witness.last().unwrap());
        assert!(count >= s.threshold);
        let (proof, piop, opening) = run(&s, &witness, seed);
        assert_eq!(piop, Ok(()), "views {views}");
        assert!(opening, "views {views}: the opening checks");
        assert_eq!(
            count_from_top(&s.tree, &proof.piop.levels.last().unwrap().lincheck.z_vec),
            Some(count)
        );
        println!(
            "views {views} at {} bits: {} levels, {} answers",
            F::BITS,
            s.tree.levels.len(),
            proof.answers.len()
        );
    }
}

#[test]
fn the_majority_opens_on_the_ladder() {
    for (views, seed) in [(4096usize, 1u64), (1 << 14, 2), (1 << 16, 3)] {
        honest_accepted::<F128>(views, seed);
        honest_accepted::<F64>(views, seed);
    }
}

#[test]
fn a_flipped_sampled_answer_shows_in_the_sampled_word() {
    let s = setup::<F64>(4096, 4);
    let mut lying = s.honest.clone();
    let view = s.sampled[3];
    lying[view] ^= true;
    let witness = s.tree.witness::<F64>(&lying);
    let (proof, piop, _) = run(&s, &witness, 4);
    assert_eq!(piop, Ok(()));
    for (i, &v) in s.sampled.iter().enumerate() {
        let bit = sampled_bit(&s.tree, v, proof.piop.sampled_words[i]);
        assert_eq!(bit, lying[v]);
    }
    assert_ne!(
        sampled_bit(&s.tree, view, proof.piop.sampled_words[3]),
        s.honest[view]
    );
}

fn copy_violation_rejected<F: PackedField>(seed: u64) {
    let s = setup::<F>(1 << 14, seed);
    let mut witness = s.tree.witness::<F>(&s.honest);
    let mut inputs = s.tree.outputs_as_next_inputs(0, &witness[0]);
    let per_block = s.tree.levels[1].block.in_count * s.tree.levels[1].block.in_bits;
    let bits = s.tree.levels[1].block.in_bits;
    inputs[per_block + 5 * bits + 2] ^= true;
    let sliced = s.tree.slice_inputs(1, &inputs);
    witness[1] = s.tree.level_witness::<F>(1, &sliced);
    let mut next = s.tree.outputs_as_next_inputs(1, &witness[1]);
    let depth = s.tree.levels.len();
    for (level, slot) in witness.iter_mut().enumerate().skip(2) {
        let sliced = s.tree.slice_inputs(level, &next);
        let w = s.tree.level_witness::<F>(level, &sliced);
        if level + 1 < depth {
            next = s.tree.outputs_as_next_inputs(level, &w);
        }
        *slot = w;
    }
    for (l, lv) in s.tree.levels.iter().enumerate() {
        assert!(
            lv.r1cs().satisfies(&witness[l].z),
            "level {l} still satisfies its R1CS"
        );
    }
    assert_ne!(
        s.tree.count(witness.last().unwrap()),
        s.honest.iter().filter(|&&b| b).count() as u64
    );
    let (_, piop, _) = run(&s, &witness, seed);
    assert!(
        matches!(piop, Err(MajorityError::RingSwitch { level: 1, .. })),
        "the copy claim on level 1's inputs fails: {piop:?}"
    );
}

#[test]
fn a_copy_violation_is_rejected() {
    copy_violation_rejected::<F128>(5);
    copy_violation_rejected::<F64>(6);
}

#[test]
fn an_inflated_count_is_rejected() {
    let s = setup::<F128>(4096, 6);
    let witness = s.tree.witness::<F128>(&s.honest);
    let mut backend = LadderOf::<F128>::lazy(s.shape);
    let mut pch = SeededChallenger::new(6);
    let mut proof = prove(
        &mut backend,
        &s.tables,
        &s.tree,
        &witness,
        &s.sampled,
        &mut pch,
    );
    let honest_count = s.tree.count(witness.last().unwrap());
    let top = proof.piop.levels.last_mut().unwrap();
    let out_base = s.tree.levels.last().unwrap().block.out_base;
    top.lincheck.z_vec[out_base + 1] += F128::ONE;
    let inflated = count_from_top(&s.tree, &top.lincheck.z_vec);
    assert_ne!(inflated, Some(honest_count));
    let mut vch = SeededChallenger::new(6);
    let mut replay = LadderOf::<F128>::lazy(s.shape);
    replay.commit(&combined_words(&s.tree, &witness), &mut vch);
    let piop = verify_piop(
        &s.tables,
        &s.tree,
        s.threshold,
        &s.sampled,
        &proof.piop,
        &mut vch,
    );
    assert!(
        matches!(
            piop,
            Err(MajorityError::Lincheck { .. }) | Err(MajorityError::Batch { .. })
        ),
        "rejected by the lincheck or the batcher: {piop:?}"
    );
}

#[test]
fn a_count_below_the_threshold_is_rejected() {
    let s = setup::<F128>(4096, 7);
    let witness = s.tree.witness::<F128>(&s.honest);
    let count = s.tree.count(witness.last().unwrap());
    let mut backend = LadderOf::<F128>::lazy(s.shape);
    let mut pch = SeededChallenger::new(7);
    let proof = prove(
        &mut backend,
        &s.tables,
        &s.tree,
        &witness,
        &s.sampled,
        &mut pch,
    );
    let mut vch = SeededChallenger::new(7);
    let mut replay = LadderOf::<F128>::lazy(s.shape);
    replay.commit(&combined_words(&s.tree, &witness), &mut vch);
    let piop = verify_piop(
        &s.tables,
        &s.tree,
        count + 1,
        &s.sampled,
        &proof.piop,
        &mut vch,
    );
    assert_eq!(
        piop,
        Err(MajorityError::BelowThreshold {
            count,
            threshold: count + 1
        })
    );
}

fn throughput<F: PackedField>(views_log: usize) {
    use std::time::Instant;
    let views = 1usize << views_log;
    let s = setup::<F>(views, 11);
    let t0 = Instant::now();
    let witness = s.tree.witness::<F>(&s.honest);
    let witness_time = t0.elapsed();
    let (_, total_log) = s.tree.layout(F::BITS.trailing_zeros() as usize);
    let mut backend = LadderOf::<F>::lazy(s.shape);
    let mut pch = SeededChallenger::new(11);
    let t1 = Instant::now();
    let proof = prove(
        &mut backend,
        &s.tables,
        &s.tree,
        &witness,
        &s.sampled,
        &mut pch,
    );
    let prove_time = t1.elapsed();
    let mut vch = SeededChallenger::new(11);
    let mut replay = LadderOf::<F>::lazy(s.shape);
    replay.commit(&F::to_words(&combined_words(&s.tree, &witness)), &mut vch);
    let t2 = Instant::now();
    let batched = verify_piop(
        &s.tables,
        &s.tree,
        s.threshold,
        &s.sampled,
        &proof.piop,
        &mut vch,
    )
    .expect("verifies");
    let verify_time = t2.elapsed();
    assert_eq!(batched, proof.piop.batched);
    println!(
        "2^{views_log} views at {} bits: {} levels, vector 2^{total_log} words; witness {:.3} s ({:.1} ns/view), prover {:.3} s ({:.1} ns/view), verifier PIOP {:.1} ms",
        F::BITS,
        s.tree.levels.len(),
        witness_time.as_secs_f64(),
        witness_time.as_nanos() as f64 / views as f64,
        prove_time.as_secs_f64(),
        prove_time.as_nanos() as f64 / views as f64,
        verify_time.as_secs_f64() * 1e3
    );
}

#[test]
#[ignore = "measurement"]
fn majority_prover_throughput() {
    let sizes: Vec<usize> = std::env::var("RMIP_VIEWS_LOG")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![20, 22, 24]);
    for views_log in sizes {
        throughput::<F128>(views_log);
        throughput::<F64>(views_log);
    }
}

fn prover_phases_on<F: PackedField, B: CommitmentBackend<Field = F128>>(
    name: &str,
    views_log: usize,
    make: &dyn Fn() -> B,
    s: &Setup<F>,
    witness: &[LevelWitness<F>],
    words: &[F],
    zf: &[F128],
) -> (std::time::Duration, String, String) {
    use rmip_packed::majority::verify_piop_with_phases;
    use rmip_packed::phases::Phases;
    use std::time::Instant;
    let mut backend = make();
    let backend = &mut backend;
    let mut ch = SeededChallenger::new(11);
    let t0 = Instant::now();
    backend.commit(zf, &mut ch);
    let t_commit = t0.elapsed();
    let mut phases = Phases::new();
    let t1 = Instant::now();
    let piop = prove_piop_with_phases(
        &s.tables,
        &s.tree,
        witness,
        words,
        &s.sampled,
        &mut ch,
        Some(&mut phases),
    );
    let t_piop = t1.elapsed();
    let t2 = Instant::now();
    backend.open(std::slice::from_ref(&piop.batched), &mut ch);
    let t_open = t2.elapsed();
    let t3 = Instant::now();
    let mut plan = backend.query_plan(&mut ch);
    let mut prev = None;
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(prev.as_ref()) {
        let prefix = ch.transcript().prefix_at(id);
        let symbol = backend.reconstruct(id, &prefix, coord);
        answers.push((id, coord, symbol.clone()));
        prev = Some(symbol);
    }
    let t_answers = t3.elapsed();
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let prover_line = format!(
        "2^{views_log} views at {} bits, {name} opening: commit {:.1} ms; PIOP {:.1} ms [{}]; open {:.1} ms; {} answers {:.1} ms; prover total {:.1} ms",
        F::BITS,
        ms(t_commit),
        ms(t_piop),
        phases.report(),
        ms(t_open),
        answers.len(),
        ms(t_answers),
        ms(t_commit + t_piop + t_open + t_answers)
    );
    let mut vb = make();
    let mut vch = SeededChallenger::new(11);
    vb.commit(zf, &mut vch);
    let mut vphases = Phases::new();
    let t5 = Instant::now();
    let batched = verify_piop_with_phases(
        &s.tables,
        &s.tree,
        s.threshold,
        &s.sampled,
        &piop,
        &mut vch,
        Some(&mut vphases),
    )
    .expect("verifies");
    let t_vpiop = t5.elapsed();
    assert_eq!(batched, piop.batched);
    vb.open(std::slice::from_ref(&batched), &mut vch);
    let mut vplan = vb.query_plan(&mut vch);
    let verifier = vb.verifier();
    let t6 = Instant::now();
    let ok = check_opening(&verifier, vplan.as_mut(), &batched, &answers).is_ok();
    let t_check = t6.elapsed();
    assert!(ok, "the opening verifies");
    let verifier_line = format!(
        "2^{views_log} views at {} bits, {name} opening, verifier: PIOP {:.2} ms [{}]; opening check {:.2} ms",
        F::BITS,
        ms(t_vpiop),
        vphases.report(),
        ms(t_check)
    );
    (
        t_commit + t_piop + t_open + t_answers,
        prover_line,
        verifier_line,
    )
}

fn prover_phases<F: PackedField>(views_log: usize) {
    use std::time::Instant;
    let views = 1usize << views_log;
    let s = setup::<F>(views, 11);
    let witness = s.tree.witness::<F>(&s.honest);
    let (_, total_log) = s.tree.layout(F::BITS.trailing_zeros() as usize);
    let t = Instant::now();
    let words = combined_words(&s.tree, &witness);
    let zf = F::to_words(&words);
    println!(
        "2^{views_log} views at {} bits: words and their conversion {:.1} ms",
        F::BITS,
        t.elapsed().as_secs_f64() * 1e3
    );
    {
        let mut ch = SeededChallenger::new(3);
        let _ = rmip_packed::majority::prove_piop(
            &s.tables, &s.tree, &witness, &words, &s.sampled, &mut ch,
        );
    }
    let shape = ladder_shape(total_log, F::BITS);
    let mut runs: Vec<(std::time::Duration, String, String)> = (0..3)
        .map(|_| {
            prover_phases_on(
                "ladder",
                views_log,
                &|| LadderOf::<F>::lazy(shape),
                &s,
                &witness,
                &words,
                &zf,
            )
        })
        .collect();
    runs.sort_by_key(|r| r.0);
    let (_, prover_line, verifier_line) = &runs[1];
    println!(
        "{prover_line} (median of three by prover total; {} columns, {} pins at a 2^-{} target, opening error 2^{:.2}; the opening term only, not the whole-protocol bound)",
        shape.queries(),
        shape.pins,
        SingleShotTarget::from_env().bits,
        rmip_backend_ladder::opening_error(shape, F::BITS).log2()
    );
    println!("{verifier_line}");
}

#[test]
#[ignore = "measurement"]
fn majority_prover_phases() {
    let sizes: Vec<usize> = std::env::var("RMIP_VIEWS_LOG")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![20, 22]);
    for views_log in sizes {
        prover_phases::<F128>(views_log);
        prover_phases::<F64>(views_log);
    }
}

fn level_zero_own_work<F: PackedField>(views_log: usize) {
    use rmip_packed::phases::Phases;
    use std::time::Instant;
    let views = 1usize << views_log;
    let full = Tree::new(views, 6, 6);
    let tree = Tree {
        levels: vec![full.levels[0].clone()],
        ..full
    };
    let witness = tree.witness::<F>(&answers(views, 11));
    let words = combined_words(&tree, &witness);
    let zf = F::to_words(&words);
    let (_, total_log) = tree.layout(F::BITS.trailing_zeros() as usize);
    let shape = ladder_shape(total_log, F::BITS);
    let tables = SkipTables::<F>::new();
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let mut samples: Vec<(std::time::Duration, String)> = Vec::new();
    for run in 0..4 {
        let mut backend = LadderOf::<F>::lazy(shape);
        let mut ch = SeededChallenger::new(11);
        let t0 = Instant::now();
        backend.commit(&zf, &mut ch);
        let t_commit = t0.elapsed();
        let mut phases = Phases::new();
        let t1 = Instant::now();
        let piop = prove_piop_with_phases(
            &tables,
            &tree,
            &witness,
            &words,
            &[],
            &mut ch,
            Some(&mut phases),
        );
        let t_piop = t1.elapsed() - phases.total("cross-level batch");
        let t2 = Instant::now();
        backend.open(std::slice::from_ref(&piop.batched), &mut ch);
        let t_open = t2.elapsed();
        let t3 = Instant::now();
        let mut plan = backend.query_plan(&mut ch);
        let mut prev = None;
        let mut count = 0;
        while let Some((id, coord)) = plan.next(prev.as_ref()) {
            let prefix = ch.transcript().prefix_at(id);
            let symbol = backend.reconstruct(id, &prefix, coord);
            count += 1;
            prev = Some(symbol);
        }
        let t_answers = t3.elapsed();
        if run == 0 {
            continue;
        }
        let total = t_commit + t_piop + t_open + t_answers;
        samples.push((
            total,
            format!(
                "2^{views_log} views at {} bits, level 0 alone ({} pins): commit {:.1} ms; PIOP {:.1} ms [zerocheck {:.1}, lincheck {:.1}, ring switch {:.1}, batch {:.1}]; open {:.1} ms; {count} answers {:.1} ms; own work {:.1} ms ({:.1} ns/view)",
                F::BITS,
                shape.pins,
                ms(t_commit),
                ms(t_piop),
                ms(phases.total("zerocheck")),
                ms(phases.total("lincheck")),
                ms(phases.total("ring switch")),
                ms(phases.total("batch")),
                ms(t_open),
                ms(t_answers),
                ms(total),
                total.as_nanos() as f64 / views as f64
            ),
        ));
    }
    samples.sort_by_key(|s| s.0);
    println!("{}", samples[1].1);
}

#[test]
#[ignore = "measurement"]
fn level_zero_prover_throughput() {
    let sizes: Vec<usize> = std::env::var("RMIP_VIEWS_LOG")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![20, 22, 24]);
    for views_log in sizes {
        level_zero_own_work::<F128>(views_log);
        level_zero_own_work::<F64>(views_log);
    }
}
