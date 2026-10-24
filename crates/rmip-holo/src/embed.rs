use crate::index::{HoloIndex, ProverIndex};
use rmip_field::{BinaryField, F128};
use rmip_frontend::SparseF2;
use rmip_iop::backend::Claim;
use rmip_iop::batch;
use rmip_iop::mateval::{HoloMatEval, MatEvalFault};
use rmip_iop::metrics::{self, Role};
use rmip_iop::replay::decode_elements;
use rmip_iop::sumcheck::{self, SumcheckProof};
use rmip_iop::transcript::{OracleId, RawChallenger, Transcript};

pub struct Embedded {
    pub index: HoloIndex,
    pub wit: ProverIndex,
}

struct Capturing<'a> {
    inner: &'a mut dyn RawChallenger,
    frames: Vec<Vec<F128>>,
}

impl RawChallenger for Capturing<'_> {
    fn sample(&mut self) -> F128 {
        let _role = metrics::enter(Role::Verifier);
        self.inner.sample()
    }
    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.frames.push(decode_elements(bytes));
        self.inner.record_ordinary(bytes);
    }
    fn post_oracle(&mut self, id: OracleId) {
        self.inner.post_oracle(id);
    }
    fn post_long_oracle(&mut self, oracle: &rmip_iop::backend::LongOracle<F128>) {
        self.inner.post_long_oracle(oracle);
    }
    fn transcript(&self) -> &Transcript {
        self.inner.transcript()
    }
}

fn encode(values: &[F128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 16);
    for v in values {
        out.extend_from_slice(&v.to_be_bytes());
    }
    out
}

fn framed(mut frame: Vec<F128>, want: usize, fault: &mut Option<MatEvalFault>) -> Vec<F128> {
    if frame.len() != want && fault.is_none() {
        *fault = Some(MatEvalFault);
    }
    frame.resize(want, F128::ZERO);
    frame
}

impl HoloMatEval for Embedded {
    fn prove_frames(
        &self,
        a: &SparseF2,
        b: &SparseF2,
        c: &SparseF2,
        r_row: &[F128],
        r_col: &[F128],
        ch: &mut dyn RawChallenger,
    ) -> ([F128; 3], Vec<Vec<F128>>) {
        let _role = metrics::enter(Role::MainProver);
        let mut cap = Capturing {
            inner: ch,
            frames: Vec::new(),
        };
        let (_proof, t) =
            crate::protocol::prove(&self.index, &self.wit, [a, b, c], r_row, r_col, &mut cap);
        (t, cap.frames)
    }

    fn verify_frames(
        &self,
        next: &mut dyn FnMut() -> Vec<F128>,
        r_row: &[F128],
        r_col: &[F128],
        ch: &mut dyn RawChallenger,
    ) -> Result<[F128; 3], MatEvalFault> {
        let _role = metrics::enter(Role::Verifier);
        let index = &self.index;
        if !index.is_well_formed() || r_row.len() != index.mu || r_col.len() != index.nu {
            return Err(MatEvalFault);
        }
        let shape = crate::index::shape_for(index.v_pre);
        let width = index.mu + index.nu;
        let lam = crate::pack::LAMBDA;
        let mut fault: Option<MatEvalFault> = None;

        let mut t = [F128::ZERO; 3];
        let mut claims = Vec::new();
        for (m, tm) in t.iter_mut().enumerate() {
            let tf = framed(next(), 1, &mut fault);
            *tm = tf[0];
            ch.record_ordinary(&encode(&tf));

            let (parity, i0, j0) = index.corrections[m];
            let corr = if parity {
                crate::protocol::eq_at_index(r_row, i0).mul(crate::protocol::eq_at_index(r_col, j0))
            } else {
                F128::ZERO
            };
            let target = tm.add(corr);

            let rounds: Vec<Vec<F128>> = (0..index.zeta).map(|_| next()).collect();
            let proof = SumcheckProof {
                round_polys: rounds,
            };
            let (claim, bad) = sumcheck::verify_recording(&proof, target, index.zeta, width, ch);
            if bad.is_some() && fault.is_none() {
                fault = Some(MatEvalFault);
            }

            let packed = framed(next(), index.groups * lam, &mut fault);
            ch.record_ordinary(&encode(&packed));

            let orbit = crate::pack::orbit(&claim.point, lam);
            let mut values = Vec::with_capacity(width);
            for chunk in packed.chunks(lam) {
                values.extend(
                    crate::pack::unpack(chunk, lam).unwrap_or_else(|| vec![F128::ZERO; lam]),
                );
            }
            values.truncate(width);
            let product = values
                .iter()
                .enumerate()
                .map(|(k, &v)| {
                    let rk = if k < index.mu {
                        r_row[k]
                    } else {
                        r_col[k - index.mu]
                    };
                    F128::ONE.add(rk).add(v)
                })
                .fold(F128::ONE, |acc, f| acc.mul(f));
            if product != claim.expected && fault.is_none() {
                fault = Some(MatEvalFault);
            }
            for (g, chunk) in packed.chunks(lam).enumerate() {
                for (v, &val) in chunk.iter().enumerate() {
                    claims.push(Claim {
                        point: index.claim_point(m, g, &orbit[v]),
                        value: val,
                    });
                }
            }
        }

        let batch_rounds: Vec<Vec<F128>> = (0..index.v_pre).map(|_| next()).collect();
        let bproof = batch::BatchProof {
            sumcheck: SumcheckProof {
                round_polys: batch_rounds,
            },
        };
        let out = batch::verify_proof(&claims, index.v_pre, &bproof, ch);
        if out.verdict.is_err() && fault.is_none() {
            fault = Some(MatEvalFault);
        }
        let reduced = out.claim;

        let opening = framed(next(), shape.n(), &mut fault);
        ch.record_ordinary(&encode(&opening));

        let cols = crate::opening::columns(shape, ch);
        let sound = crate::opening::check(
            &index.codeword,
            shape,
            &reduced.point,
            reduced.value,
            &opening,
            &cols,
        );
        if !sound && fault.is_none() {
            fault = Some(MatEvalFault);
        }

        match fault {
            None => Ok(t),
            Some(f) => Err(f),
        }
    }
}
