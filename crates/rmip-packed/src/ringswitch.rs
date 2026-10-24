use rmip_iop::transcript::RawChallenger;

use crate::subfield::SkipTables;
use crate::zerocheck::{eq_table, record, PackedBits};
use crate::PackedField;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitClaim<F> {
    pub coeffs: Vec<F>,
    pub point: Vec<F>,
    pub value: F,
}

impl<F: PackedField> BitClaim<F> {
    pub fn from_skip_point(tables: &SkipTables<F>, z_skip: F, rho: &[F], value: F) -> Self {
        let lanes = tables.fold_weights(z_skip);
        let in_word = F::BITS.trailing_zeros() as usize - 6;
        let mut coeffs = Vec::with_capacity(F::BITS);
        let eq_in = eq_table(&rho[..in_word]);
        for &e in &eq_in {
            for &l in &lanes {
                coeffs.push(l * e);
            }
        }
        Self {
            coeffs,
            point: rho[in_word..].to_vec(),
            value,
        }
    }

    pub fn evaluate(&self, bits: &PackedBits<F>) -> F {
        let eq = eq_table(&self.point);
        let mut acc = F::ZERO;
        for (i, &w) in bits.words.iter().enumerate() {
            acc += eq[i] * self.apply(w);
        }
        acc
    }

    fn apply(&self, w: F) -> F {
        let b = w.to_bits();
        let mut acc = F::ZERO;
        for (r, c) in self.coeffs.iter().enumerate() {
            if (b >> r) & 1 == 1 {
                acc += *c;
            }
        }
        acc
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordClaim<F> {
    pub point: Vec<F>,
    pub value: F,
}

pub fn eval_words<F: PackedField>(words: &[F], point: &[F]) -> F {
    let eq = eq_table(point);
    assert_eq!(eq.len(), words.len());
    words
        .iter()
        .zip(&eq)
        .fold(F::ZERO, |acc, (w, e)| acc + *w * *e)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingSwitchProof<F> {
    pub s_hat: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RingSwitchError {
    BadLength {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    PartialEvaluationsMismatch,
}

pub struct FrobeniusBasis<F> {
    pub pow: Vec<Vec<F>>,
}

impl<F: PackedField> FrobeniusBasis<F> {
    pub fn new() -> Self {
        let n = F::BITS;
        let mut pow = Vec::with_capacity(n);
        let mut row: Vec<F> = (0..n).map(|r| F::from_bits(1u128 << r)).collect();
        for _ in 0..n {
            pow.push(row.clone());
            for x in row.iter_mut() {
                *x = x.square();
            }
        }
        Self { pow }
    }

    pub fn twist(&self, s: &[F], j: usize) -> F {
        F::dot(&self.pow[j], s)
    }

    pub fn inverse_twists(&self, gammas: &[F]) -> Vec<F> {
        let n = F::BITS;
        (0..n)
            .map(|r| {
                gammas
                    .iter()
                    .enumerate()
                    .fold(F::ZERO, |acc, (j, g)| acc + *g * self.pow[(n - j) % n][r])
            })
            .collect()
    }
}

impl<F: PackedField> Default for FrobeniusBasis<F> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn frobenius_inverse<F: PackedField>(x: F, j: usize) -> F {
    frobenius(x, (F::BITS - j % F::BITS) % F::BITS)
}

pub fn frobenius<F: PackedField>(x: F, j: usize) -> F {
    let mut y = x;
    for _ in 0..j {
        y = y.square();
    }
    y
}

pub fn conjugate_points<F: PackedField>(base: &[F]) -> Vec<Vec<F>> {
    let kappa = F::BITS;
    let mut points = vec![Vec::new(); kappa];
    points[0] = base.to_vec();
    let mut cur: Vec<F> = base.iter().map(|x| x.square()).collect();
    for j in (1..kappa).rev() {
        points[j] = cur.clone();
        for x in cur.iter_mut() {
            *x = x.square();
        }
    }
    points
}

pub fn word_claims<F: PackedField>(claim: &BitClaim<F>, s_hat: &[F]) -> Vec<WordClaim<F>> {
    let basis = F::frobenius_basis();
    conjugate_points(&claim.point)
        .into_iter()
        .enumerate()
        .map(|(j, point)| WordClaim {
            point,
            value: frobenius_inverse(basis.twist(s_hat, j), j),
        })
        .collect()
}

fn partial_evaluations<F: PackedField>(words: &[F], claim: &BitClaim<F>) -> Vec<F> {
    let eq = eq_table(&claim.point);
    assert_eq!(eq.len(), words.len());
    let mut s_hat = vec![F::ZERO; F::BITS];
    for (w, &e) in words.iter().zip(&eq) {
        if e == F::ZERO {
            continue;
        }
        let mut b = w.to_bits();
        while b != 0 {
            let r = b.trailing_zeros() as usize;
            s_hat[r] += e;
            b &= b - 1;
        }
    }
    s_hat
}

pub fn prove_many<F: PackedField>(
    words: &[F],
    claims: &[BitClaim<F>],
    ch: &mut dyn RawChallenger,
) -> (Vec<RingSwitchProof<F>>, Vec<Vec<WordClaim<F>>>) {
    let mut proofs = Vec::with_capacity(claims.len());
    let mut out = Vec::with_capacity(claims.len());
    for claim in claims {
        let s_hat = partial_evaluations(words, claim);
        record(ch, &s_hat);
        out.push(word_claims(claim, &s_hat));
        proofs.push(RingSwitchProof { s_hat });
    }
    (proofs, out)
}

pub fn prove<F: PackedField>(
    words: &[F],
    claim: &BitClaim<F>,
    ch: &mut dyn RawChallenger,
) -> (RingSwitchProof<F>, Vec<WordClaim<F>>) {
    let (mut proofs, mut claims) = prove_many(words, std::slice::from_ref(claim), ch);
    (proofs.remove(0), claims.remove(0))
}

pub fn verify_many<F: PackedField>(
    claims: &[BitClaim<F>],
    proofs: &[RingSwitchProof<F>],
    ch: &mut dyn RawChallenger,
) -> Result<Vec<Vec<WordClaim<F>>>, RingSwitchError> {
    let kappa = F::BITS;
    if proofs.len() != claims.len() {
        return Err(RingSwitchError::BadLength {
            what: "proofs",
            expected: claims.len(),
            got: proofs.len(),
        });
    }
    let check = |what: &'static str, got: usize| {
        if got == kappa {
            Ok(())
        } else {
            Err(RingSwitchError::BadLength {
                what,
                expected: kappa,
                got,
            })
        }
    };
    let mut out = Vec::with_capacity(claims.len());
    for (claim, proof) in claims.iter().zip(proofs) {
        check("s_hat", proof.s_hat.len())?;
        check("coeffs", claim.coeffs.len())?;
        let v = claim
            .coeffs
            .iter()
            .zip(&proof.s_hat)
            .fold(F::ZERO, |acc, (c, s)| acc + *c * *s);
        if v != claim.value {
            return Err(RingSwitchError::PartialEvaluationsMismatch);
        }
        record(ch, &proof.s_hat);
        out.push(word_claims(claim, &proof.s_hat));
    }
    Ok(out)
}

pub fn verify<F: PackedField>(
    claim: &BitClaim<F>,
    proof: &RingSwitchProof<F>,
    ch: &mut dyn RawChallenger,
) -> Result<Vec<WordClaim<F>>, RingSwitchError> {
    let mut out = verify_many(std::slice::from_ref(claim), std::slice::from_ref(proof), ch)?;
    Ok(out.remove(0))
}
