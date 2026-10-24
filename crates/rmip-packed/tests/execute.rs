use rmip_backend_ladder::{Ladder, LadderShape};
use rmip_field::F128;
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use rmip_packed::execute::{check_opening, prove, verify_piop, PackedError, PackedProof};
use rmip_packed::lincheck::{BlockR1cs, SparseMatrix};
use rmip_packed::subfield::SkipTables;
use rmip_packed::zerocheck::PackedBits;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn instance(m: usize, k_log: usize, seed: u64) -> (BlockR1cs, PackedBits<F128>) {
    let k = 1 << k_log;
    let r1cs = BlockR1cs::new(
        m,
        k_log,
        SparseMatrix::identity(k),
        SparseMatrix::identity(k),
    );
    let mut s = seed;
    let bits: Vec<bool> = (0..1usize << m).map(|_| lcg(&mut s) & 1 == 1).collect();
    (r1cs, PackedBits::from_bits(m, &bits))
}

#[test]
fn the_packed_iop_opens_on_the_ladder() {
    for (m, k_log, seed) in [(13usize, 8usize, 1u64), (16, 9, 2)] {
        let tables = SkipTables::<F128>::new();
        let (r1cs, z) = instance(m, k_log, seed);
        assert!(r1cs.satisfies(&z));
        let mut shape = LadderShape::for_trace(m - 7, 23.0);
        shape.pins = 3;
        let mut backend = Ladder::lazy(shape);
        let mut pch = SeededChallenger::new(seed);
        let proof = prove(&mut backend, &tables, &r1cs, &z, &mut pch);
        assert_eq!(
            proof.answers.len(),
            shape.schedule().tables.len(),
            "one occurrence per table"
        );

        let mut vch = SeededChallenger::new(seed);
        let mut replay = Ladder::lazy(shape);
        replay.commit(&z.words, &mut vch);
        let batched = verify_piop(&tables, &r1cs, &proof, &mut vch).expect("the PIOP verifies");
        assert_eq!(
            batched, proof.batched,
            "the two sides derive the same batched claim"
        );
        replay.open(std::slice::from_ref(&batched), &mut vch);
        let mut plan = replay.query_plan(&mut vch);
        check_opening(&backend.verifier(), plan.as_mut(), &batched, &proof.answers)
            .expect("the opening checks");
        assert_eq!(
            pch.transcript().canonical(),
            vch.transcript().canonical(),
            "one transcript"
        );
    }
}

#[test]
#[ignore = "measurement"]
fn packed_prover_throughput() {
    use std::time::Instant;
    for m in [22usize, 24, 26] {
        let tables = SkipTables::<F128>::new();
        let (r1cs, z) = instance(m, 10, 9);
        let mut shape = LadderShape::for_trace(m - 7, 23.0);
        shape.pins = 5;
        let pins = shape.pins;
        let mut best = f64::MAX;
        let mut proof = None;
        for _ in 0..2 {
            let mut backend = Ladder::lazy(shape);
            let mut pch = SeededChallenger::new(3);
            let t = Instant::now();
            let p = prove(&mut backend, &tables, &r1cs, &z, &mut pch);
            best = best.min(t.elapsed().as_secs_f64());
            proof = Some((p, backend));
        }
        let (proof, backend) = proof.unwrap();
        let mut vch = SeededChallenger::new(3);
        let mut replay = Ladder::lazy(shape);
        replay.commit(&z.words, &mut vch);
        let t = Instant::now();
        let batched = verify_piop(&tables, &r1cs, &proof, &mut vch).unwrap();
        replay.open(std::slice::from_ref(&batched), &mut vch);
        let mut plan = replay.query_plan(&mut vch);
        check_opening(&backend.verifier(), plan.as_mut(), &batched, &proof.answers).unwrap();
        let verifier_s = t.elapsed().as_secs_f64();
        println!(
            "m = {m}: prover {:.3} s ({:.2} ns per bit), verifier {:.1} ms (including the replayed commit's share of coins), pins {pins}",
            best,
            best * 1e9 / (1u64 << m) as f64,
            verifier_s * 1e3
        );
    }
}

#[test]
#[ignore = "measurement"]
fn packed_prover_phases() {
    use rmip_iop::backend::Claim;
    use rmip_packed::ringswitch;
    use rmip_packed::{lincheck, zerocheck};
    use std::time::Instant;
    let m = 24usize;
    let tables = SkipTables::<F128>::new();
    let (r1cs, z) = instance(m, 10, 9);
    let mut shape = LadderShape::for_trace(m - 7, 23.0);
    shape.pins = 5;
    let mut backend = Ladder::lazy(shape);
    let mut ch = SeededChallenger::new(3);
    let bits = (1u64 << m) as f64;
    let mut lap = Instant::now();
    let phase = |name: &str, lap: &mut Instant| {
        let s = lap.elapsed().as_secs_f64();
        println!("  {name:14} {:.3} s  {:.2} ns per bit", s, s * 1e9 / bits);
        *lap = Instant::now();
    };
    backend.commit(&z.words, &mut ch);
    phase("commit", &mut lap);
    let a = r1cs.apply(&r1cs.a0, &z);
    let b = r1cs.apply(&r1cs.b0, &z);
    phase("apply A, B", &mut lap);
    let (_, zc) = zerocheck::prove(&tables, &a, &b, &z, &mut ch);
    phase("zerocheck", &mut lap);
    let (_, lc) = lincheck::prove(&tables, &r1cs, &z, &zc.rho, &mut ch);
    phase("lincheck", &mut lap);
    let claims = vec![lc.bit_claim(&tables)];
    let (_, ws) = ringswitch::prove_many(&z.words, &claims, &mut ch);
    phase("ring switch", &mut lap);
    let families: Vec<rmip_packed::batch::ConjugateFamily<F128>> = ws
        .iter()
        .map(|c| rmip_packed::batch::ConjugateFamily::from_word_claims(c))
        .collect();
    let (_, batched_word) = rmip_packed::batch::prove(&z.words, &families, &[], &mut ch);
    let batched = Claim {
        point: batched_word.point.iter().rev().copied().collect(),
        value: batched_word.value,
    };
    phase("batch", &mut lap);
    backend.open(std::slice::from_ref(&batched), &mut ch);
    phase("open", &mut lap);
    let mut plan = backend.query_plan(&mut ch);
    let (id, coord) = plan.next(None).unwrap();
    let _ = backend.reconstruct(id, &ch.transcript().prefix_at(id), coord);
    phase("answer columns", &mut lap);
}

#[test]
fn a_forged_c_claim_is_rejected() {
    use rmip_iop::backend::{Claim, Symbol};
    use rmip_packed::batch::ConjugateFamily;
    use rmip_packed::{batch, lincheck, ringswitch, zerocheck};
    let m = 13usize;
    let k_log = 8usize;
    let k = 1 << k_log;
    let r1cs = BlockR1cs::new(
        m,
        k_log,
        SparseMatrix {
            k,
            rows: vec![vec![]; k],
        },
        SparseMatrix::identity(k),
    );
    let z = PackedBits::<F128>::from_bits(m, &vec![true; 1 << m]);
    let fake_c = PackedBits::<F128>::from_bits(m, &vec![false; 1 << m]);
    assert!(!r1cs.satisfies(&z));
    let a = r1cs.apply(&r1cs.a0, &z);
    let b = r1cs.apply(&r1cs.b0, &z);
    let tables = SkipTables::<F128>::new();
    let mut shape = LadderShape::for_trace(m - 7, 23.0);
    shape.pins = 3;
    for (seed, forger_appends_c) in [(1u64, true), (2, true), (3, false), (4, false)] {
        let mut backend = Ladder::lazy(shape);
        let mut pch = SeededChallenger::new(seed);
        backend.commit(&z.words, &mut pch);
        let (zc_proof, zc) = zerocheck::prove(&tables, &a, &b, &fake_c, &mut pch);
        let (lc_proof, lc) = lincheck::prove(&tables, &r1cs, &z, &zc.rho, &mut pch);
        let claims = vec![lc.bit_claim(&tables)];
        let (switches, mut word_claims) = ringswitch::prove_many(&z.words, &claims, &mut pch);
        if forger_appends_c {
            word_claims.push(ringswitch::word_claims(&zc.c_claim, &zc.c_s_hat));
        }
        let families: Vec<ConjugateFamily<F128>> = word_claims
            .iter()
            .map(|c| ConjugateFamily::from_word_claims(c))
            .collect();
        let (batch_proof, word) = batch::prove(&z.words, &families, &[], &mut pch);
        let batched = Claim {
            point: word.point.iter().rev().copied().collect(),
            value: word.value,
        };
        backend.open(std::slice::from_ref(&batched), &mut pch);
        let mut plan = backend.query_plan(&mut pch);
        let mut answers = Vec::new();
        let mut previous: Option<Symbol<F128>> = None;
        while let Some((id, coord)) = plan.next(previous.as_ref()) {
            let answer = backend.reconstruct(id, &pch.transcript().prefix_at(id), coord);
            previous = Some(answer.clone());
            answers.push((id, coord, answer));
        }
        let proof = PackedProof {
            zerocheck: zc_proof,
            lincheck: lc_proof,
            switches,
            batch: batch_proof,
            batched,
            answers,
        };
        let mut replay = Ladder::lazy(shape);
        let mut vch = SeededChallenger::new(seed);
        replay.commit(&z.words, &mut vch);
        let rejected = match verify_piop(&tables, &r1cs, &proof, &mut vch) {
            Err(_) => true,
            Ok(verified) => {
                replay.open(std::slice::from_ref(&verified), &mut vch);
                let mut vplan = replay.query_plan(&mut vch);
                check_opening(
                    &replay.verifier(),
                    vplan.as_mut(),
                    &verified,
                    &proof.answers,
                )
                .is_err()
            }
        };
        assert!(
            rejected,
            "seed {seed}, forger appends c: {forger_appends_c}"
        );
    }
    let mut backend = Ladder::lazy(shape);
    let mut pch = SeededChallenger::new(17);
    let normal = prove(&mut backend, &tables, &r1cs, &z, &mut pch);
    let mut vch = SeededChallenger::new(17);
    Ladder::lazy(shape).commit(&z.words, &mut vch);
    assert!(matches!(
        verify_piop(&tables, &r1cs, &normal, &mut vch),
        Err(PackedError::Zerocheck(_))
    ));
}
