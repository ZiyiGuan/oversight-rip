use rmip_field::{BinaryField, F128};
use rmip_iop::binius_channel::{down, up, RecordedSumcheck};
use rmip_iop::sumcheck::{self, SumcheckError, SumcheckProof, Summand};
use rmip_iop::transcript::{RawChallenger, SeededChallenger, Transcript};

struct Counting<C> {
    inner: C,
    coins: usize,
}
impl<C: RawChallenger> RawChallenger for Counting<C> {
    fn sample(&mut self) -> F128 {
        self.coins += 1;
        self.inner.sample()
    }
    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.inner.record_ordinary(bytes)
    }
    fn post_oracle(&mut self, id: rmip_iop::transcript::OracleId) {
        self.inner.post_oracle(id)
    }
    fn transcript(&self) -> &Transcript {
        self.inner.transcript()
    }
}
fn counting(seed: u64) -> Counting<SeededChallenger> {
    Counting {
        inner: SeededChallenger::new(seed),
        coins: 0,
    }
}

struct Product {
    tables: Vec<Vec<F128>>,
}
impl Summand for Product {
    fn num_vars(&self) -> usize {
        self.tables[0].len().trailing_zeros() as usize
    }
    fn degree(&self) -> usize {
        self.tables.len()
    }
    fn round_evals(&self) -> Vec<F128> {
        sumcheck::round_evals_over(&self.tables, self.tables.len(), |vals| {
            vals.iter().fold(F128::ONE, |acc, &v| acc.mul(v))
        })
    }
    fn fold(&mut self, challenge: F128) {
        for t in &mut self.tables {
            rmip_iop::mle::fold_in_place(t, challenge);
        }
    }
    fn final_value(&self) -> F128 {
        self.tables.iter().fold(F128::ONE, |acc, t| acc.mul(t[0]))
    }
}

fn table(nv: usize, seed: u64) -> Vec<F128> {
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

fn honest(nv: usize, degree: usize, seed: u64) -> (SumcheckProof, F128) {
    let tables: Vec<Vec<F128>> = (0..degree).map(|d| table(nv, seed + d as u64)).collect();
    let mut s = Product { tables };
    let claim = {
        let n = 1usize << nv;
        (0..n)
            .map(|i| s.tables.iter().fold(F128::ONE, |a, t| a.mul(t[i])))
            .fold(F128::ZERO, |a, b| a.add(b))
    };
    let mut ch = SeededChallenger::new(seed);
    let (proof, _) = sumcheck::run(&mut s, claim, &mut ch).expect("honest run");
    (proof, claim)
}

fn agree(proof: &SumcheckProof, claim: F128, nv: usize, degree: usize, what: &str) {
    let mut ours = counting(9);
    let (our_claim, our_fault) = sumcheck::verify_recording(proof, claim, nv, degree, &mut ours);

    let mut theirs = counting(9);
    let (their_claim, their_fault) = {
        let mut chan = RecordedSumcheck::new(&proof.round_polys, nv, degree, &mut theirs);
        let out = binius_ip::sumcheck::verify::<rmip_iop::binius_channel::B128, _>(
            nv,
            degree,
            up(claim),
            &mut chan,
        )
        .expect("this channel never fails");
        let fault = chan.finish();
        (
            sumcheck::SumcheckClaim {
                point: out.challenges.into_iter().map(down).collect(),
                expected: down(out.eval),
            },
            fault,
        )
    };

    assert_eq!(our_fault, their_fault, "{what}: same first fault");
    assert_eq!(
        our_claim.point, their_claim.point,
        "{what}: same challenge point"
    );
    assert_eq!(
        our_claim.expected, their_claim.expected,
        "{what}: same terminal claim"
    );
    assert_eq!(ours.coins, theirs.coins, "{what}: SAME COIN COUNT");
    assert_eq!(
        ours.transcript().entries().len(),
        theirs.transcript().entries().len(),
        "{what}: same transcript length"
    );
}

#[test]
fn the_bridge_is_a_field_isomorphism() {
    let mut s = 0x9E3779B97F4A7C15u64;
    for _ in 0..1000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let a = F128::new(s as u128 * 0x9E37 + 7);
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let b = F128::new(s as u128 * 0x1EB1 + 11);
        assert_eq!(down(up(a)), a, "round trip");
        assert_eq!(down(up(a) * up(b)), a.mul(b), "multiplication is preserved");
        assert_eq!(down(up(a) + up(b)), a.add(b), "addition is preserved");
    }
}

#[test]
fn the_engines_agree_on_honest_transcripts() {
    for &(nv, degree) in &[(4usize, 2usize), (5, 3), (3, 6), (4, 12)] {
        let (proof, claim) = honest(nv, degree, 100 + degree as u64);
        agree(
            &proof,
            claim,
            nv,
            degree,
            &format!("honest nv={nv} degree={degree}"),
        );
    }
}

#[test]
fn the_engines_agree_on_malformed_transcripts() {
    let (base, claim) = honest(5, 3, 77);

    let mut tampered = base.clone();
    tampered.round_polys[2][1] = tampered.round_polys[2][1].add(F128::ONE);
    agree(&tampered, claim, 5, 3, "tampered coefficient");

    let mut short = base.clone();
    short.round_polys.truncate(3);
    agree(&short, claim, 5, 3, "short proof");

    let mut long = base.clone();
    long.round_polys.push(vec![F128::ONE; 3]);
    agree(&long, claim, 5, 3, "extra rounds");

    let mut narrow = base.clone();
    narrow.round_polys[1].truncate(1);
    agree(&narrow, claim, 5, 3, "narrow round");
    let mut wide = base.clone();
    wide.round_polys[1].push(F128::ONE);
    agree(&wide, claim, 5, 3, "wide round");

    let mut empty_round = base.clone();
    empty_round.round_polys[4].clear();
    agree(&empty_round, claim, 5, 3, "empty round");

    let empty = SumcheckProof {
        round_polys: Vec::new(),
    };
    agree(&empty, claim, 5, 3, "empty proof");
}

#[test]
fn the_named_faults_are_produced() {
    let (base, claim) = honest(5, 3, 55);
    let mut ch = counting(1);
    let empty = SumcheckProof {
        round_polys: Vec::new(),
    };
    let mut chan = RecordedSumcheck::new(&empty.round_polys, 5, 3, &mut ch);
    let _ = binius_ip::sumcheck::verify::<rmip_iop::binius_channel::B128, _>(
        5,
        3,
        up(claim),
        &mut chan,
    );
    assert!(matches!(
        chan.finish(),
        Some(SumcheckError::WrongRoundCount {
            expected: 5,
            got: 0
        })
    ));

    let mut narrow = base;
    narrow.round_polys[1].truncate(1);
    let mut ch2 = counting(1);
    let mut chan2 = RecordedSumcheck::new(&narrow.round_polys, 5, 3, &mut ch2);
    let _ = binius_ip::sumcheck::verify::<rmip_iop::binius_channel::B128, _>(
        5,
        3,
        up(claim),
        &mut chan2,
    );
    assert!(matches!(
        chan2.finish(),
        Some(SumcheckError::WrongDegree {
            round: 1,
            expected: 3,
            got: 1
        })
    ));
}

fn pt(n: usize, seed: u64) -> Vec<F128> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            F128::new(s as u128 * 104_729 + 3)
        })
        .collect()
}

#[test]
fn eq_table_matches_binius_eq_ind() {
    for nv in 1..=8usize {
        let r = pt(nv, 300 + nv as u64);
        let ours = rmip_iop::mle::eq_table(&r);
        let rb: Vec<rmip_iop::binius_channel::B128> = r.iter().rev().copied().map(up).collect();
        let theirs: Vec<F128> = binius_math::multilinear::eq::eq_ind_partial_eval_scalars(&rb)
            .into_iter()
            .map(down)
            .collect();
        assert_eq!(ours, theirs, "eq convention differs at nv={nv}");
    }
}

#[test]
fn eval_mle_matches_binius_evaluate() {
    for nv in 1..=8usize {
        let table = pt(1 << nv, 400 + nv as u64);
        let r = pt(nv, 500 + nv as u64);
        let ours = rmip_iop::mle::eval_mle(&table, &r);
        let mut tb: Vec<rmip_iop::binius_channel::B128> = table.iter().copied().map(up).collect();
        let rb: Vec<rmip_iop::binius_channel::B128> = r.iter().rev().copied().map(up).collect();
        let theirs =
            down(binius_math::multilinear::evaluate::evaluate_inplace_scalars(&mut tb[..], &rb));
        assert_eq!(ours, theirs, "evaluation convention differs at nv={nv}");
    }
}

#[test]
fn univariate_evaluation_matches_binius_with_no_reversal() {
    for degree in 1..=12usize {
        let coeffs = pt(degree + 1, 600 + degree as u64);
        let x = pt(1, 700 + degree as u64)[0];
        let ours = rmip_iop::mle::horner(&coeffs, x);
        let cb: Vec<rmip_iop::binius_channel::B128> = coeffs.iter().copied().map(up).collect();
        let theirs = down(binius_math::univariate::evaluate_univariate(&cb, &up(x)));
        assert_eq!(ours, theirs, "univariate Horner differs at degree={degree}");
    }
}

fn lagrange_by_hand(points: &[F128], values: &[F128], x: F128) -> F128 {
    let mut acc = F128::ZERO;
    for (j, &vj) in values.iter().enumerate() {
        let mut term = vj;
        for (k, &pk) in points.iter().enumerate() {
            if k == j {
                continue;
            }
            let den = points[j]
                .add(pk)
                .inv()
                .expect("interpolation points are distinct");
            term = term.mul(x.add(pk)).mul(den);
        }
        acc = acc.add(term);
    }
    acc
}

#[test]
fn interpolation_domain_matches_binius() {
    for degree in 1..=8usize {
        let points = rmip_iop::mle::interpolation_points(degree);
        let values = pt(points.len(), 800 + degree as u64);
        let x = pt(1, 900 + degree as u64)[0];
        assert!(
            !points.contains(&x),
            "the off-domain point must be off the domain"
        );

        let ours =
            rmip_iop::mle::InterpolationDomain::from_points(points.clone()).extrapolate(&values, x);

        let pb: Vec<rmip_iop::binius_channel::B128> = points.iter().copied().map(up).collect();
        let vb: Vec<rmip_iop::binius_channel::B128> = values.iter().copied().map(up).collect();
        let theirs = down(
            binius_math::univariate::EvaluationDomain::from_points(pb).extrapolate(&vb, up(x)),
        );
        assert_eq!(ours, theirs, "interpolation differs at degree={degree}");
        assert_eq!(
            ours,
            lagrange_by_hand(&points, &values, x),
            "the engine differs from a textbook Lagrange sum at degree={degree}"
        );

        for (i, &p) in points.iter().enumerate() {
            let ours_at = rmip_iop::mle::InterpolationDomain::from_points(points.clone())
                .extrapolate(&values, p);
            assert_eq!(ours_at, values[i], "ours must be total at point {i}");
            let pb2: Vec<rmip_iop::binius_channel::B128> = points.iter().copied().map(up).collect();
            let vb2: Vec<rmip_iop::binius_channel::B128> = values.iter().copied().map(up).collect();
            let theirs_at = down(
                binius_math::univariate::EvaluationDomain::from_points(pb2)
                    .extrapolate(&vb2, up(p)),
            );
            assert_eq!(theirs_at, values[i], "theirs must be total at point {i}");
        }
    }
}
