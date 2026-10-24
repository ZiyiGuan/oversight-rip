use rmip_backend_interleaved::{code, Ligero, Shape, CODEWORD_ORACLE};
use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{Claim, CommitmentBackend, Coord, OpeningVerifier, Symbol};
use rmip_iop::batch;
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::transcript::{Entry, RawChallenger, SeededChallenger};

fn shape(nu: usize) -> Shape {
    Shape::balanced(nu, 4, 4.0)
}

fn table(nu: usize) -> Vec<F128> {
    (0..1usize << nu)
        .map(|i| F128::from_bool((i * 7 + 3) % 5 < 2))
        .collect()
}

fn one_claim(z: &[F128], nu: usize) -> Vec<Claim<F128>> {
    let raw: Vec<Claim<F128>> = (0..6)
        .map(|k| {
            let p: Vec<F128> = (0..nu)
                .map(|i| F128::new((k + 3) * 7717 + i as u128))
                .collect();
            Claim {
                value: eval_mle(z, &p),
                point: p,
            }
        })
        .collect();
    let mut ch = SeededChallenger::new(9);
    vec![batch::reduce(&raw, z, &mut ch).expect("batching verifies")]
}

fn recovered_proximity_coeffs(ch: &SeededChallenger, m: usize) -> Vec<F128> {
    let entries = ch.transcript().entries();
    assert_eq!(
        entries[0],
        Entry::OraclePost(CODEWORD_ORACLE),
        "commit posts the codeword first"
    );
    let coins: Vec<F128> = entries[1..=m]
        .iter()
        .map(|e| match e {
            Entry::Coin(c) => *c,
            other => panic!("expected a commitment-phase coin, found {other:?}"),
        })
        .collect();
    assert!(
        matches!(entries[m + 1], Entry::Ordinary(_)),
        "the proximity response follows the m coins"
    );
    coins
}

fn dot(c: &[F128], sym: &[F128]) -> F128 {
    c.iter()
        .zip(sym)
        .map(|(a, b)| a.mul(*b))
        .fold(F128::ZERO, |x, y| x.add(y))
}

fn symbol_at(rows: &[Vec<F128>], x: F128) -> Symbol<F128> {
    Symbol(rows.iter().map(|r| code::eval_at(r, x)).collect())
}

#[test]
fn an_off_domain_column_satisfies_both_identities_and_is_rejected() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let sh = shape(nu);
    let mut b = Ligero::new(sh);
    let mut ch = SeededChallenger::new(0x0FFD_0111);
    b.commit(&z, &mut ch);
    let proximity_coeffs = recovered_proximity_coeffs(&ch, sh.m());
    b.open(&claims, &mut ch);

    let v = b.verifier();
    let mut plan = b.query_plan(&mut ch);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        answers.push((
            id,
            coord,
            b.oracles()[0].symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert!(v.check(&claims, &answers), "honest columns must verify");

    let opening_coeffs = eq_table(&claims[0].point[..sh.nu_row]);
    let (_, probe_col, probe_sym) = &answers[0];
    assert_eq!(
        dot(&proximity_coeffs, &probe_sym.0),
        code::eval_at(&v.proximity, F128::new(u128::try_from(*probe_col).unwrap())),
        "the recovered proximity coefficients are not the verifier's"
    );
    assert_eq!(
        dot(&opening_coeffs, &probe_sym.0),
        code::eval_at(&v.opening, F128::new(u128::try_from(*probe_col).unwrap())),
        "the recomputed opening coefficients are not the verifier's"
    );

    let rows: Vec<Vec<F128>> = z.chunks(sh.n()).map(|r| r.to_vec()).collect();
    let off = sh.n_enc;
    let x_off = F128::new(off as u128);
    let sym_off = symbol_at(&rows, x_off);
    assert_eq!(
        sym_off.0.len(),
        sh.m(),
        "the width clause must not be what fires"
    );

    assert_eq!(
        dot(&proximity_coeffs, &sym_off.0),
        code::eval_at(&v.proximity, x_off),
        "the proximity identity must HOLD off the domain, or this test pins nothing"
    );
    assert_eq!(
        dot(&opening_coeffs, &sym_off.0),
        code::eval_at(&v.opening, x_off),
        "the opening identity must HOLD off the domain, or this test pins nothing"
    );

    let mut extended = answers.clone();
    extended.push((CODEWORD_ORACLE, Coord::from(off), sym_off.clone()));
    assert!(
        !v.check(&claims, &extended),
        "an answer at coordinate n_enc was accepted"
    );
    let alone = [(CODEWORD_ORACLE, Coord::from(off), sym_off.clone())];
    assert!(
        !v.check(&claims, &alone),
        "an answer at coordinate n_enc was accepted on its own"
    );

    let mut wide = sh;
    wide.n_enc = sh.n_enc * 2;
    let wide_verifier = rmip_backend_interleaved::LigeroVerifier::from_public_transcript(
        wide,
        v.proximity.clone(),
        proximity_coeffs.clone(),
        v.opening.clone(),
        opening_coeffs.clone(),
        claims[0].point.clone(),
    );
    assert!(
        wide_verifier.check(&claims, &extended),
        "with the range clause widened, the off-domain answer must be accepted - \
         otherwise some OTHER clause is doing the rejecting"
    );
}

#[test]
fn the_same_construction_inside_the_domain_is_accepted() {
    let nu = 10;
    let z = table(nu);
    let claims = one_claim(&z, nu);
    let sh = shape(nu);
    let mut b = Ligero::new(sh);
    let mut ch = SeededChallenger::new(0x0FFD_0222);
    b.commit(&z, &mut ch);
    b.open(&claims, &mut ch);

    let v = b.verifier();
    let mut plan = b.query_plan(&mut ch);
    let mut answers = Vec::new();
    while let Some((id, coord)) = plan.next(None) {
        answers.push((
            id,
            coord,
            b.oracles()[0].symbols[usize::try_from(coord).expect("coord fits usize")].clone(),
        ));
    }
    assert!(v.check(&claims, &answers), "honest columns must verify");

    let rows: Vec<Vec<F128>> = z.chunks(sh.n()).map(|r| r.to_vec()).collect();
    let inside = sh.n_enc - 1;
    assert!(
        answers.iter().all(|(_, c, _)| *c != Coord::from(inside)),
        "the control column must be one the plan did not draw"
    );
    let sym_in = symbol_at(&rows, F128::new(inside as u128));
    assert_eq!(
        sym_in,
        b.oracles()[0].symbols[inside],
        "the construction must reproduce the posted column on the domain"
    );

    let mut extended = answers.clone();
    extended.push((CODEWORD_ORACLE, Coord::from(inside), sym_in.clone()));
    assert!(
        v.check(&claims, &extended),
        "an in-domain column built the same way was rejected"
    );
    let alone = [(CODEWORD_ORACLE, Coord::from(inside), sym_in)];
    assert!(
        v.check(&claims, &alone),
        "an in-domain column built the same way was rejected on its own"
    );
}
