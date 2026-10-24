use binius_compute::GlobalAllocator;
use binius_field::Field as _;
use binius_math::FieldBuffer;
use rmip_field::{BinaryField, F128};
use rmip_iop::binius_channel::{down, up, ProverTranscript, RecordedSumcheck, B128};
use rmip_iop::sumcheck;
use rmip_iop::transcript::SeededChallenger;
use rmip_iop::zerocheck::Zerocheck;

fn tbl(nv: usize, seed: u64) -> Vec<F128> {
    let mut s = seed | 1;
    (0..(1usize << nv))
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            F128::new(s as u128 * 104_729 + 3)
        })
        .collect()
}

fn satisfying(nv: usize, seed: u64) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
    let a = tbl(nv, seed);
    let b = tbl(nv, seed + 1);
    let c = a.iter().zip(&b).map(|(&x, &y)| x.mul(y)).collect();
    (a, b, c)
}

fn buf(v: &[F128]) -> FieldBuffer<B128, Vec<B128>> {
    FieldBuffer::from_values(&v.iter().copied().map(up).collect::<Vec<_>>())
}

#[test]
fn the_endpoints_are_the_multilinears_at_the_challenge_point() {
    for nv in [3usize, 6, 9] {
        let (a, b, c) = satisfying(nv, 500 + nv as u64);
        let tau: Vec<F128> = (0..nv).map(|i| F128::new(i as u128 * 7919 + 5)).collect();

        let mut zc = Zerocheck::new(&tau, a.clone(), b.clone(), c.clone());
        assert_eq!(zc.current_sum(), F128::ZERO, "the trace must satisfy");
        let mut ch = SeededChallenger::new(11);
        let (_proof, claim) = sumcheck::run_mlecheck_recording(&mut zc, &tau, &mut ch);
        let rows = zc.row_claims();

        let r = &claim.point;
        assert_eq!(rows.v_a, rmip_iop::mle::eval_mle(&a, r), "nv={nv}: v_a");
        assert_eq!(rows.v_b, rmip_iop::mle::eval_mle(&b, r), "nv={nv}: v_b");
        assert_eq!(rows.v_c, rmip_iop::mle::eval_mle(&c, r), "nv={nv}: v_c");

        assert_eq!(
            rows.v_a.mul(rows.v_b).add(rows.v_c),
            claim.expected,
            "nv={nv}: the prover's terminal value is the verifier's"
        );
    }
}

#[test]
fn binius_mlecheck_verifier_accepts_over_our_transcript() {
    let nv = 6usize;
    let (a, b, c) = satisfying(nv, 900);
    let tau: Vec<F128> = (0..nv)
        .map(|i| F128::new(i as u128 * 104_729 + 7))
        .collect();
    let tau_rev: Vec<B128> = tau.iter().rev().copied().map(up).collect();

    let alloc = GlobalAllocator;
    let prover = binius_ip_prover::sumcheck::quadratic_mlecheck_prover(
        &alloc,
        [buf(&a), buf(&b), buf(&c)],
        |[x, y, z]: [B128; 3]| x * y - z,
        |[x, y, _z]: [B128; 3]| x * y,
        tau_rev.clone(),
        B128::ZERO,
    );
    let mut pch = SeededChallenger::new(23);
    let mut pc = ProverTranscript::new(&mut pch);
    let out = binius_ip_prover::sumcheck::prove_single_mlecheck(prover, &mut pc);
    let sent = pc.sent.clone();

    let rounds: Vec<Vec<F128>> = sent.iter().take(nv).cloned().collect();
    let mut vch = SeededChallenger::new(23);
    let mut chan = RecordedSumcheck::new(&rounds, nv, 2, &mut vch);
    let v = binius_ip::mlecheck::verify::<B128, _>(&tau_rev, 2, B128::ZERO, &mut chan)
        .expect("their verifier accepts their prover");
    assert!(chan.finish().is_none(), "no fault on an honest transcript");
    assert_eq!(
        v.challenges.iter().copied().map(down).collect::<Vec<_>>(),
        out.challenges.iter().copied().map(down).collect::<Vec<_>>(),
        "verifier and prover agree on the challenge point"
    );
}

#[test]
fn their_prover_and_ours_send_identical_bytes() {
    for nv in [3usize, 6, 9, 13, 16] {
        let (a, b, c) = satisfying(nv, 700 + nv as u64);
        let tau: Vec<F128> = (0..nv).map(|i| F128::new(i as u128 * 31 + 7)).collect();

        let tau_rev: Vec<B128> = tau.iter().rev().copied().map(up).collect();
        let alloc = GlobalAllocator;
        let prover = binius_ip_prover::sumcheck::quadratic_mlecheck_prover(
            &alloc,
            [buf(&a), buf(&b), buf(&c)],
            |[x, y, z]: [B128; 3]| x * y - z,
            |[x, y, _z]: [B128; 3]| x * y,
            tau_rev,
            B128::ZERO,
        );
        let mut pch = SeededChallenger::new(19);
        let mut pc = ProverTranscript::new(&mut pch);
        let out = binius_ip_prover::sumcheck::prove_single_mlecheck(prover, &mut pc);
        let theirs = pc.sent.clone();

        let mut zc = Zerocheck::new(&tau, a.clone(), b.clone(), c.clone());
        let mut och = SeededChallenger::new(19);
        let (proof, claim) = sumcheck::run_mlecheck_recording(&mut zc, &tau, &mut och);

        assert_eq!(
            proof.round_polys, theirs,
            "nv={nv}: the kernels must send identical bytes"
        );
        let rows = zc.row_claims();
        let their_evals: Vec<F128> = out.multilinear_evals.iter().copied().map(down).collect();
        assert_eq!(
            their_evals,
            vec![rows.v_a, rows.v_b, rows.v_c],
            "nv={nv}: identical endpoints"
        );
        let _ = claim;
    }
}
