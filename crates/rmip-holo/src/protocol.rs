use crate::index::{HoloIndex, ProverIndex};
use rmip_field::{BinaryField, F128};
use rmip_frontend::SparseF2;
use rmip_iop::backend::Claim;
use rmip_iop::batch::{self, BatchProof};
use rmip_iop::metrics::{self, Role};
use rmip_iop::mle::fold_in_place;
use rmip_iop::sumcheck::{self, SumcheckProof, Summand};
use rmip_iop::transcript::RawChallenger;

struct Product {
    tables: Vec<Vec<F128>>,
    r: Vec<F128>,
}

impl Summand for Product {
    fn num_vars(&self) -> usize {
        self.tables[0].len().trailing_zeros() as usize
    }
    fn degree(&self) -> usize {
        self.tables.len()
    }
    fn round_evals(&self) -> Vec<F128> {
        let r = &self.r;
        sumcheck::round_evals_over(&self.tables, self.tables.len(), |vals| {
            vals.iter()
                .zip(r)
                .map(|(&b, &rk)| F128::ONE.add(rk).add(b))
                .fold(F128::ONE, |acc, f| acc.mul(f))
        })
    }
    fn fold(&mut self, challenge: F128) {
        for t in &mut self.tables {
            fold_in_place(t, challenge);
        }
    }
    fn final_value(&self) -> F128 {
        self.r
            .iter()
            .zip(&self.tables)
            .map(|(&rk, t)| F128::ONE.add(rk).add(t[0]))
            .fold(F128::ONE, |acc, f| acc.mul(f))
    }
}

pub(crate) fn eq_at_index(r: &[F128], s: usize) -> F128 {
    let len = r.len();
    r.iter()
        .enumerate()
        .map(|(k, &rk)| {
            if (s >> (len - 1 - k)) & 1 == 1 {
                rk
            } else {
                F128::ONE.add(rk)
            }
        })
        .fold(F128::ONE, |acc, f| acc.mul(f))
}

fn encode(values: &[F128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 16);
    for v in values {
        out.extend_from_slice(&v.to_be_bytes());
    }
    out
}

#[derive(Clone, Debug)]
pub struct HoloProof {
    pub t: [F128; 3],
    pub sumchecks: Vec<SumcheckProof>,
    pub endpoints: Vec<Vec<F128>>,
    pub batching: BatchProof,
    pub opening: Vec<F128>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejected {
    Sumcheck {
        matrix: usize,
        error: sumcheck::SumcheckError,
    },
    Endpoint {
        matrix: usize,
    },
    Batching(batch::BatchProofError),
    OpeningMismatch,
    IndexShape,
    Arity,
    ProofShape,
}

pub fn prove(
    index: &HoloIndex,
    wit: &ProverIndex,
    mats: [&SparseF2; 3],
    r_row: &[F128],
    r_col: &[F128],
    ch: &mut dyn RawChallenger,
) -> (HoloProof, [F128; 3]) {
    let _role = metrics::enter(Role::MainProver);
    let [a, b, c] = mats;
    assert_eq!(
        r_row.len(),
        index.mu,
        "r_row must have the index's row arity"
    );
    assert_eq!(
        r_col.len(),
        index.nu,
        "r_col must have the index's column arity"
    );
    assert!(
        index.is_well_formed(),
        "prove needs an index preprocess could have built"
    );
    let shape = crate::index::shape_for(index.v_pre);

    let (ta, tb, tc) = rmip_iop::matmle::eval_all(a, b, c, r_row, r_col);
    let t = [ta, tb, tc];
    let mut r = r_row.to_vec();
    r.extend_from_slice(r_col);

    let mut sumchecks = Vec::with_capacity(3);
    let mut endpoints = Vec::with_capacity(3);
    let mut claims = Vec::new();
    for (m, &tm) in t.iter().enumerate() {
        ch.record_ordinary(&encode(&[tm]));
        let (parity, i0, j0) = index.corrections[m];
        let corr = if parity {
            eq_at_index(r_row, i0).mul(eq_at_index(r_col, j0))
        } else {
            F128::ZERO
        };
        let target = tm.add(corr);
        let mut summand = Product {
            tables: wit.tables[m].clone(),
            r: r.clone(),
        };
        let (proof, claim) =
            sumcheck::run(&mut summand, target, ch).expect("the honest product sumcheck accepts");
        let orbit = crate::pack::orbit(&claim.point, crate::pack::LAMBDA);
        let mut packed_evals = Vec::with_capacity(index.groups * crate::pack::LAMBDA);
        for g in 0..index.groups {
            let slot = m * index.groups + g;
            let base = slot << index.zeta;
            let col = &wit.flat[base..base + (1 << index.zeta)];
            for pt in &orbit {
                packed_evals.push(rmip_iop::mle::eval_mle(col, pt));
            }
        }
        ch.record_ordinary(&encode(&packed_evals));
        for (g, chunk) in packed_evals.chunks(crate::pack::LAMBDA).enumerate() {
            for (v, &val) in chunk.iter().enumerate() {
                claims.push(Claim {
                    point: index.claim_point(m, g, &orbit[v]),
                    value: val,
                });
            }
        }
        sumchecks.push(proof);
        endpoints.push(packed_evals);
    }

    let out = batch::prove_recording(&claims, &wit.flat, ch);
    out.verdict.expect("honest batching does not vanish");
    let opening = crate::opening::open(&wit.flat, shape, &out.claim.point, ch);

    {
        let _verifier = metrics::enter(Role::Verifier);
        let _ = crate::opening::columns(shape, ch);
    }

    (
        HoloProof {
            t,
            sumchecks,
            endpoints,
            batching: out.proof,
            opening,
        },
        t,
    )
}

pub fn verify(
    index: &HoloIndex,
    proof: &HoloProof,
    r_row: &[F128],
    r_col: &[F128],
    ch: &mut dyn RawChallenger,
) -> Result<[F128; 3], Rejected> {
    let _role = metrics::enter(Role::Verifier);
    if !index.is_well_formed() {
        return Err(Rejected::IndexShape);
    }
    if r_row.len() != index.mu || r_col.len() != index.nu {
        return Err(Rejected::Arity);
    }
    if proof.sumchecks.len() != 3 || proof.endpoints.len() != 3 {
        return Err(Rejected::ProofShape);
    }
    let shape = crate::index::shape_for(index.v_pre);
    let width = index.mu + index.nu;

    let mut claims = Vec::new();
    for m in 0..3 {
        ch.record_ordinary(&encode(&[proof.t[m]]));
        let (parity, i0, j0) = index.corrections[m];
        let corr = if parity {
            eq_at_index(r_row, i0).mul(eq_at_index(r_col, j0))
        } else {
            F128::ZERO
        };
        let target = proof.t[m].add(corr);
        let claim = sumcheck::verify(&proof.sumchecks[m], target, index.zeta, width, ch)
            .map_err(|error| Rejected::Sumcheck { matrix: m, error })?;
        let packed_evals = &proof.endpoints[m];
        ch.record_ordinary(&encode(packed_evals));
        if packed_evals.len() != index.groups * crate::pack::LAMBDA {
            return Err(Rejected::Endpoint { matrix: m });
        }
        let orbit = crate::pack::orbit(&claim.point, crate::pack::LAMBDA);
        let mut values = Vec::with_capacity(width);
        for chunk in packed_evals.chunks(crate::pack::LAMBDA) {
            let recovered = crate::pack::unpack(chunk, crate::pack::LAMBDA)
                .ok_or(Rejected::Endpoint { matrix: m })?;
            values.extend(recovered);
        }
        values.truncate(width);
        if values.len() != width {
            return Err(Rejected::Endpoint { matrix: m });
        }
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
        if product != claim.expected {
            return Err(Rejected::Endpoint { matrix: m });
        }
        for (g, chunk) in packed_evals.chunks(crate::pack::LAMBDA).enumerate() {
            for (v, &val) in chunk.iter().enumerate() {
                claims.push(Claim {
                    point: index.claim_point(m, g, &orbit[v]),
                    value: val,
                });
            }
        }
    }

    let out = batch::verify_proof(&claims, index.v_pre, &proof.batching, ch);
    out.verdict.map_err(Rejected::Batching)?;
    let reduced = out.claim;
    ch.record_ordinary(&encode(&proof.opening));

    let cols = crate::opening::columns(shape, ch);
    if !crate::opening::check(
        &index.codeword,
        shape,
        &reduced.point,
        reduced.value,
        &proof.opening,
        &cols,
    ) {
        return Err(Rejected::OpeningMismatch);
    }

    Ok(proof.t)
}
