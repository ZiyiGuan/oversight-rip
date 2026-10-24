use crate::backend::Claim;
use crate::booleanity;
use crate::driver::Statement;
use crate::lincheck::{Coefficients, Lincheck};
use crate::matmle;
use crate::mle::{eval_mle, lift};
use crate::sumcheck::{self, RoundMessageProver, SumcheckError, SumcheckProof, Summand};
use crate::transcript::RawChallenger;
use crate::zerocheck::{RowClaims, Zerocheck};
use rmip_field::{BinaryField, F128};
use rmip_frontend::trace;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationProof {
    pub zerocheck: SumcheckProof,
    pub zerocheck_endpoint: Vec<F128>,
    pub lincheck: SumcheckProof,
    pub lincheck_endpoint: Vec<F128>,
    pub holo_frames: Vec<Vec<F128>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationError {
    MatrixEvaluation,
    MalformedEndpoint {
        phase: &'static str,
        expected: usize,
        got: usize,
    },
    Zerocheck(SumcheckError),
    ZerocheckEndpoint,
    Lincheck(SumcheckError),
    LincheckEndpoint,
}

pub struct RelationVerification {
    pub claims: Vec<Claim<F128>>,
    pub public_input_claim: Claim<F128>,
    pub t_mat: Duration,
    pub verdict: Result<(), RelationError>,
}

pub struct RelationProverOutput {
    pub proof: RelationProof,
    pub claims: Vec<Claim<F128>>,
    pub public_input_claim: Claim<F128>,
}

pub trait RelationMessageProver {
    fn observe_public_input_point(&mut self, r_pub: &[F128]);
    fn begin_zerocheck(&mut self, tau: &[F128]);
    fn zerocheck_round_message(&mut self, round: usize) -> Vec<F128>;
    fn receive_zerocheck_challenge(&mut self, round: usize, challenge: F128);
    fn zerocheck_endpoint(&mut self) -> Vec<F128>;

    fn begin_lincheck(&mut self, row_point: &[F128], coefficients: Coefficients);
    fn lincheck_round_message(&mut self, round: usize) -> Vec<F128>;
    fn receive_lincheck_challenge(&mut self, round: usize, challenge: F128);
    fn lincheck_endpoint(&mut self) -> Vec<F128>;
    fn holo_frame(&mut self) -> Vec<F128>;
}

pub struct HonestRelationProver<'a> {
    a: &'a rmip_frontend::r1cs::SparseF2,
    b: &'a rmip_frontend::r1cs::SparseF2,
    c: &'a rmip_frontend::r1cs::SparseF2,
    z: Vec<F128>,
    zerocheck: Option<Zerocheck>,
    lincheck: Option<Lincheck>,
}

impl<'a> HonestRelationProver<'a> {
    pub fn new(st: &'a Statement<'_>, z: &[F128]) -> Self {
        assert_eq!(
            z.len(),
            st.layout.n_trace,
            "trace length must match the public layout"
        );
        Self {
            a: &st.sys.a,
            b: &st.sys.b,
            c: &st.sys.c,
            z: z.to_vec(),
            zerocheck: None,
            lincheck: None,
        }
    }
}

impl RelationMessageProver for HonestRelationProver<'_> {
    fn observe_public_input_point(&mut self, _r_pub: &[F128]) {}

    fn begin_zerocheck(&mut self, tau: &[F128]) {
        self.zerocheck = Some(Zerocheck::new(
            tau,
            matmle::mul_vec(self.a, &self.z),
            matmle::mul_vec(self.b, &self.z),
            matmle::mul_vec(self.c, &self.z),
        ));
    }

    fn zerocheck_round_message(&mut self, _round: usize) -> Vec<F128> {
        self.zerocheck
            .as_mut()
            .expect("zerocheck phase was not started")
            .round_message()
    }

    fn receive_zerocheck_challenge(&mut self, _round: usize, challenge: F128) {
        self.zerocheck
            .as_mut()
            .expect("zerocheck phase was not started")
            .receive_challenge(challenge);
    }

    fn holo_frame(&mut self) -> Vec<F128> {
        unreachable!(
            "live holographic proving is not implemented; the honest \
             interactive provers run scan-shaped statements and the SNARG \
             replays a tape"
        )
    }

    fn zerocheck_endpoint(&mut self) -> Vec<F128> {
        let row = self
            .zerocheck
            .as_mut()
            .expect("zerocheck phase was not started")
            .row_claims();
        vec![row.v_a, row.v_b, row.v_c]
    }

    fn begin_lincheck(&mut self, row_point: &[F128], coefficients: Coefficients) {
        self.lincheck = Some(Lincheck::new(
            self.a,
            self.b,
            self.c,
            row_point,
            coefficients,
            self.z.clone(),
        ));
    }

    fn lincheck_round_message(&mut self, _round: usize) -> Vec<F128> {
        {
            let evals = self
                .lincheck
                .as_ref()
                .expect("lincheck phase was not started")
                .round_evals();
            let coeffs = crate::mle::monomial_from_evals(&evals);
            coeffs[..coeffs.len() - 1].to_vec()
        }
    }

    fn receive_lincheck_challenge(&mut self, _round: usize, challenge: F128) {
        self.lincheck
            .as_mut()
            .expect("lincheck phase was not started")
            .fold(challenge);
    }

    fn lincheck_endpoint(&mut self) -> Vec<F128> {
        vec![self
            .lincheck
            .as_ref()
            .expect("lincheck phase was not started")
            .z_claim()]
    }
}

struct ZerocheckChannel<'a, P: ?Sized>(&'a mut P);

impl<P: RelationMessageProver + ?Sized> RoundMessageProver for ZerocheckChannel<'_, P> {
    fn round_message(&mut self, round: usize) -> Vec<F128> {
        self.0.zerocheck_round_message(round)
    }

    fn receive_challenge(&mut self, round: usize, challenge: F128) {
        self.0.receive_zerocheck_challenge(round, challenge);
    }
}

struct LincheckChannel<'a, P: ?Sized>(&'a mut P);

impl<P: RelationMessageProver + ?Sized> RoundMessageProver for LincheckChannel<'_, P> {
    fn round_message(&mut self, round: usize) -> Vec<F128> {
        self.0.lincheck_round_message(round)
    }

    fn receive_challenge(&mut self, round: usize, challenge: F128) {
        self.0.receive_lincheck_challenge(round, challenge);
    }
}

fn set_first(slot: &mut Option<RelationError>, error: RelationError) {
    if slot.is_none() {
        *slot = Some(error);
    }
}

fn record_field_message(ch: &mut dyn RawChallenger, values: &[F128]) {
    let mut encoded = Vec::with_capacity(values.len() * 16);
    for value in values {
        encoded.extend_from_slice(&value.to_be_bytes());
    }
    ch.record_ordinary(&encoded);
}

fn public_claim(st: &Statement<'_>, ch: &mut dyn RawChallenger) -> (Claim<F128>, Vec<F128>) {
    let r_pub: Vec<F128> = (0..st.layout.k).map(|_| ch.sample()).collect();
    let point: Vec<F128> = std::iter::repeat_n(F128::ZERO, st.layout.nu - st.layout.k)
        .chain(r_pub.iter().copied())
        .collect();
    let block = trace::public_block(st.layout, st.inst.x.as_slice());
    (
        Claim {
            point,
            value: eval_mle(&lift(block.as_slice()), &r_pub),
        },
        r_pub,
    )
}

pub fn prove(st: &Statement<'_>, z: &[F128], ch: &mut dyn RawChallenger) -> RelationProof {
    prove_with_claims(st, z, ch).proof
}

pub fn prove_with_claims(
    st: &Statement<'_>,
    z: &[F128],
    ch: &mut dyn RawChallenger,
) -> RelationProverOutput {
    assert_eq!(
        z.len(),
        st.layout.n_trace,
        "trace length must match the public layout"
    );
    let (public, _) = public_claim(st, ch);

    let tau: Vec<F128> = (0..st.sys.mu).map(|_| ch.sample()).collect();
    let mut zc = Zerocheck::new(
        &tau,
        matmle::mul_vec(&st.sys.a, z),
        matmle::mul_vec(&st.sys.b, z),
        matmle::mul_vec(&st.sys.c, z),
    );
    let (zerocheck, zclaim) = sumcheck::run_mlecheck_recording(&mut zc, &tau, ch);
    let zpoint = zclaim.point;
    let rows = zc.row_claims();
    let zerocheck_endpoint = vec![rows.v_a, rows.v_b, rows.v_c];
    record_field_message(ch, &zerocheck_endpoint);

    let coeffs = Coefficients::sample(ch);
    let mut lc = Lincheck::new(&st.sys.a, &st.sys.b, &st.sys.c, &zpoint, coeffs, z.to_vec());
    let (lincheck, lpoint, _) = sumcheck::prove(&mut lc, ch);
    let holo_frames = match st.holo {
        Some(h) => {
            let (_vals, frames) =
                h.prove_frames(&st.sys.a, &st.sys.b, &st.sys.c, &zpoint, &lpoint, ch);
            frames
        }
        None => Vec::new(),
    };
    let lincheck_endpoint = vec![lc.z_claim()];
    record_field_message(ch, &lincheck_endpoint);

    let lincheck_claim = Claim {
        point: lpoint.clone(),
        value: lincheck_endpoint[0],
    };
    let (sigma, v_sigma) = booleanity::claim_from_lincheck_endpoint(&lpoint, lincheck_endpoint[0]);
    let booleanity_claim = Claim {
        point: sigma,
        value: v_sigma,
    };
    RelationProverOutput {
        proof: RelationProof {
            zerocheck,
            zerocheck_endpoint,
            lincheck,
            lincheck_endpoint,
            holo_frames,
        },
        claims: vec![lincheck_claim, public.clone(), booleanity_claim],
        public_input_claim: public,
    }
}

pub fn verify(
    st: &Statement<'_>,
    proof: &RelationProof,
    ch: &mut dyn RawChallenger,
) -> RelationVerification {
    let (public, _) = public_claim(st, ch);
    let mut first = None;

    let tau: Vec<F128> = (0..st.sys.mu).map(|_| ch.sample()).collect();
    let (zclaim, zbad) =
        sumcheck::verify_mlecheck_recording(&proof.zerocheck, &tau, F128::ZERO, 2, ch);
    if let Some(error) = zbad {
        set_first(&mut first, RelationError::Zerocheck(error));
    }

    record_field_message(ch, &proof.zerocheck_endpoint);
    if proof.zerocheck_endpoint.len() != 3 {
        set_first(
            &mut first,
            RelationError::MalformedEndpoint {
                phase: "zerocheck",
                expected: 3,
                got: proof.zerocheck_endpoint.len(),
            },
        );
    }
    let row = RowClaims {
        v_a: proof
            .zerocheck_endpoint
            .first()
            .copied()
            .unwrap_or(F128::ZERO),
        v_b: proof
            .zerocheck_endpoint
            .get(1)
            .copied()
            .unwrap_or(F128::ZERO),
        v_c: proof
            .zerocheck_endpoint
            .get(2)
            .copied()
            .unwrap_or(F128::ZERO),
    };
    if row.v_a.mul(row.v_b).add(row.v_c) != zclaim.expected {
        set_first(&mut first, RelationError::ZerocheckEndpoint);
    }

    let coeffs = Coefficients::sample(ch);
    let batched = coeffs.batch(row);
    let (lclaim, lbad) = sumcheck::verify_recording(&proof.lincheck, batched, st.layout.nu, 2, ch);
    if let Some(error) = lbad {
        set_first(&mut first, RelationError::Lincheck(error));
    }

    let holo_vals = match st.holo {
        Some(h) => {
            let mut it = proof.holo_frames.iter();
            let out = {
                let mut next = || it.next().cloned().unwrap_or_default();
                h.verify_frames(&mut next, &zclaim.point, &lclaim.point, ch)
            };
            if it.next().is_some() {
                set_first(&mut first, RelationError::MatrixEvaluation);
            }
            match out {
                Ok(v) => Some(v),
                Err(_) => {
                    set_first(&mut first, RelationError::MatrixEvaluation);
                    Some([F128::ZERO; 3])
                }
            }
        }
        None => None,
    };

    record_field_message(ch, &proof.lincheck_endpoint);
    if proof.lincheck_endpoint.len() != 1 {
        set_first(
            &mut first,
            RelationError::MalformedEndpoint {
                phase: "lincheck",
                expected: 1,
                got: proof.lincheck_endpoint.len(),
            },
        );
    }
    let v_z = proof
        .lincheck_endpoint
        .first()
        .copied()
        .unwrap_or(F128::ZERO);

    let t0 = Instant::now();
    let (ma, mb, mc) = match holo_vals {
        Some([ma, mb, mc]) => (ma, mb, mc),
        None => matmle::eval_all(
            &st.sys.a,
            &st.sys.b,
            &st.sys.c,
            &zclaim.point,
            &lclaim.point,
        ),
    };
    let t_mat = t0.elapsed();
    let l_at = coeffs
        .lambda_a()
        .mul(ma)
        .add(coeffs.lambda_b().mul(mb))
        .add(coeffs.lambda_c().mul(mc));
    if l_at.mul(v_z) != lclaim.expected {
        set_first(&mut first, RelationError::LincheckEndpoint);
    }

    let lincheck = Claim {
        point: lclaim.point.clone(),
        value: v_z,
    };
    let (sigma, v_sigma) = booleanity::claim_from_lincheck_endpoint(&lclaim.point, v_z);
    let booleanity = Claim {
        point: sigma,
        value: v_sigma,
    };
    RelationVerification {
        claims: vec![lincheck, public.clone(), booleanity],
        public_input_claim: public,
        t_mat,
        verdict: first.map_or(Ok(()), Err),
    }
}

pub fn verify_interactive<P: RelationMessageProver + ?Sized>(
    st: &Statement<'_>,
    prover: &mut P,
    ch: &mut dyn RawChallenger,
) -> (RelationProof, RelationVerification) {
    let (public, r_pub) = public_claim(st, ch);
    prover.observe_public_input_point(&r_pub);
    let mut first = None;

    let tau: Vec<F128> = (0..st.sys.mu).map(|_| ch.sample()).collect();
    prover.begin_zerocheck(&tau);
    let (zerocheck, zclaim, zbad) = {
        let mut phase = ZerocheckChannel(prover);
        sumcheck::verify_mlecheck_interactive(&mut phase, &tau, F128::ZERO, 2, ch)
    };
    if let Some(error) = zbad {
        set_first(&mut first, RelationError::Zerocheck(error));
    }

    let zerocheck_endpoint = prover.zerocheck_endpoint();
    record_field_message(ch, &zerocheck_endpoint);
    if zerocheck_endpoint.len() != 3 {
        set_first(
            &mut first,
            RelationError::MalformedEndpoint {
                phase: "zerocheck",
                expected: 3,
                got: zerocheck_endpoint.len(),
            },
        );
    }
    let row = RowClaims {
        v_a: zerocheck_endpoint.first().copied().unwrap_or(F128::ZERO),
        v_b: zerocheck_endpoint.get(1).copied().unwrap_or(F128::ZERO),
        v_c: zerocheck_endpoint.get(2).copied().unwrap_or(F128::ZERO),
    };
    if row.v_a.mul(row.v_b).add(row.v_c) != zclaim.expected {
        set_first(&mut first, RelationError::ZerocheckEndpoint);
    }

    let coefficients = Coefficients::sample(ch);
    let batched = coefficients.batch(row);
    prover.begin_lincheck(&zclaim.point, coefficients);
    let (lincheck, lclaim, lbad) = {
        let mut phase = LincheckChannel(prover);
        sumcheck::verify_interactive(&mut phase, batched, st.layout.nu, 2, ch)
    };
    if let Some(error) = lbad {
        set_first(&mut first, RelationError::Lincheck(error));
    }

    let (holo_vals, holo_frames) = match st.holo {
        Some(h) => {
            let mut frames: Vec<Vec<F128>> = Vec::new();
            let vals = {
                let mut next = || {
                    let f = prover.holo_frame();
                    frames.push(f.clone());
                    f
                };
                h.verify_frames(&mut next, &zclaim.point, &lclaim.point, ch)
            };
            match vals {
                Ok(v) => (Some(v), frames),
                Err(_) => {
                    set_first(&mut first, RelationError::MatrixEvaluation);
                    (Some([F128::ZERO; 3]), frames)
                }
            }
        }
        None => (None, Vec::new()),
    };

    let lincheck_endpoint = prover.lincheck_endpoint();
    record_field_message(ch, &lincheck_endpoint);
    if lincheck_endpoint.len() != 1 {
        set_first(
            &mut first,
            RelationError::MalformedEndpoint {
                phase: "lincheck",
                expected: 1,
                got: lincheck_endpoint.len(),
            },
        );
    }
    let v_z = lincheck_endpoint.first().copied().unwrap_or(F128::ZERO);

    let t0 = Instant::now();
    let (ma, mb, mc) = match holo_vals {
        Some([ma, mb, mc]) => (ma, mb, mc),
        None => matmle::eval_all(
            &st.sys.a,
            &st.sys.b,
            &st.sys.c,
            &zclaim.point,
            &lclaim.point,
        ),
    };
    let t_mat = t0.elapsed();
    let l_at = coefficients
        .lambda_a()
        .mul(ma)
        .add(coefficients.lambda_b().mul(mb))
        .add(coefficients.lambda_c().mul(mc));
    if l_at.mul(v_z) != lclaim.expected {
        set_first(&mut first, RelationError::LincheckEndpoint);
    }

    let lincheck_claim = Claim {
        point: lclaim.point.clone(),
        value: v_z,
    };
    let (sigma, v_sigma) = booleanity::claim_from_lincheck_endpoint(&lclaim.point, v_z);
    let booleanity_claim = Claim {
        point: sigma,
        value: v_sigma,
    };
    let proof = RelationProof {
        zerocheck,
        zerocheck_endpoint,
        lincheck,
        lincheck_endpoint,
        holo_frames,
    };
    let verification = RelationVerification {
        claims: vec![lincheck_claim, public.clone(), booleanity_claim],
        public_input_claim: public,
        t_mat,
        verdict: first.map_or(Ok(()), Err),
    };
    (proof, verification)
}
