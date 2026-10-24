use rmip_field::{F128, F64};
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use rmip_packed::lincheck::{self, BlockR1cs, LincheckError, SparseMatrix};
use rmip_packed::ringswitch::BitClaim;
use rmip_packed::subfield::SkipTables;
use rmip_packed::zerocheck::{self, eval_bits, PackedBits, ZerocheckClaim};
use rmip_packed::PackedField;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn random_matrix(k: usize, per_row: usize, s: &mut u64) -> SparseMatrix {
    let rows = (0..k)
        .map(|_| {
            let mut cols: Vec<usize> = (0..per_row).map(|_| lcg(s) as usize % k).collect();
            cols.sort_unstable();
            cols.dedup();
            cols
        })
        .collect();
    SparseMatrix { k, rows }
}

fn random_bits<F: PackedField>(m: usize, s: &mut u64) -> PackedBits<F> {
    let bits: Vec<bool> = (0..1usize << m).map(|_| lcg(s) & 1 == 1).collect();
    PackedBits::from_bits(m, &bits)
}

fn random_point<F: PackedField>(n: usize, s: &mut u64) -> Vec<F> {
    (0..n)
        .map(|_| F::from_bits(((lcg(s) as u128) << 64) | lcg(s) as u128))
        .collect()
}

fn assert_claim_true<F: PackedField>(
    tables: &SkipTables<F>,
    z: &PackedBits<F>,
    c: &lincheck::LincheckClaim<F>,
) {
    let mut point = c.r_rest.clone();
    point.extend_from_slice(&c.outer);
    assert_eq!(eval_bits(tables, z, c.r_skip, &point), c.w);
}

fn honest<F: PackedField>(m: usize, k_log: usize, seed: u64) {
    let tables = SkipTables::<F>::new();
    let mut s = seed;
    let r1cs = BlockR1cs::new(
        m,
        k_log,
        random_matrix(1 << k_log, 3, &mut s),
        random_matrix(1 << k_log, 2, &mut s),
    );
    let z = random_bits::<F>(m, &mut s);
    let a = r1cs.apply(&r1cs.a0, &z);
    let b = r1cs.apply(&r1cs.b0, &z);
    let z_skip = F::from_bits(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128);
    let rho = random_point::<F>(m - 6, &mut s);
    let v_a = eval_bits(&tables, &a, z_skip, &rho);
    let v_b = eval_bits(&tables, &b, z_skip, &rho);
    let zc = ZerocheckClaim {
        r: Vec::new(),
        z: z_skip,
        rho: rho.clone(),
        a_eval: v_a,
        b_eval: v_b,
        c_claim: BitClaim {
            coeffs: Vec::new(),
            point: Vec::new(),
            value: F::ZERO,
        },
        c_s_hat: Vec::new(),
    };

    let mut pch = SeededChallenger::new(seed);
    let (proof, claim) = lincheck::prove(&tables, &r1cs, &z, &rho, &mut pch);
    assert_eq!(proof.z_vec.len(), 1 << k_log);
    let mut vch = SeededChallenger::new(seed);
    let verified = lincheck::verify(&tables, &r1cs, &zc, &proof, &mut vch).expect("accepted");
    assert_eq!(verified, claim);
    assert_eq!(pch.transcript().canonical(), vch.transcript().canonical());
    assert_claim_true(&tables, &z, &claim);
    let outer = &rho[k_log - 6..];
    for j_in in 0..1usize << k_log {
        let eq = zerocheck::eq_table(outer);
        let mut expected = F::ZERO;
        for (j_out, w) in eq.iter().enumerate() {
            if z.bit(j_in + (1 << k_log) * j_out) {
                expected += *w;
            }
        }
        assert_eq!(proof.z_vec[j_in], expected, "Z[{j_in}]");
    }
}

fn wrong_fold_rejected<F: PackedField>(m: usize, k_log: usize, seed: u64) {
    let tables = SkipTables::<F>::new();
    let mut s = seed;
    let r1cs = BlockR1cs::new(
        m,
        k_log,
        random_matrix(1 << k_log, 3, &mut s),
        random_matrix(1 << k_log, 2, &mut s),
    );
    let z = random_bits::<F>(m, &mut s);
    let a = r1cs.apply(&r1cs.a0, &z);
    let b = r1cs.apply(&r1cs.b0, &z);
    let z_skip = F::from_bits(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128);
    let rho = random_point::<F>(m - 6, &mut s);
    let v_a = eval_bits(&tables, &a, z_skip, &rho);
    let v_b = eval_bits(&tables, &b, z_skip, &rho);
    let zc = ZerocheckClaim {
        r: Vec::new(),
        z: z_skip,
        rho: rho.clone(),
        a_eval: v_a,
        b_eval: v_b,
        c_claim: BitClaim {
            coeffs: Vec::new(),
            point: Vec::new(),
            value: F::ZERO,
        },
        c_s_hat: Vec::new(),
    };
    let mut pch = SeededChallenger::new(seed);
    let (mut proof, _) = lincheck::prove(&tables, &r1cs, &z, &rho, &mut pch);
    proof.z_vec[3] += F::ONE;
    let mut vch = SeededChallenger::new(seed);
    assert!(matches!(
        lincheck::verify(&tables, &r1cs, &zc, &proof, &mut vch),
        Err(LincheckError::AMismatch) | Err(LincheckError::BMismatch)
    ));
    let (proof, _) = lincheck::prove(&tables, &r1cs, &z, &rho, &mut SeededChallenger::new(seed));
    let mut vch = SeededChallenger::new(seed);
    assert_eq!(
        lincheck::verify(
            &tables,
            &r1cs,
            &ZerocheckClaim {
                a_eval: v_a + F::ONE,
                ..zc.clone()
            },
            &proof,
            &mut vch
        ),
        Err(LincheckError::AMismatch)
    );
}

fn end_to_end<F: PackedField>(m: usize, k_log: usize, seed: u64) {
    let tables = SkipTables::<F>::new();
    let mut s = seed;
    let k = 1 << k_log;
    let r1cs = BlockR1cs::new(
        m,
        k_log,
        SparseMatrix::identity(k),
        SparseMatrix::identity(k),
    );
    let z = random_bits::<F>(m, &mut s);
    assert!(r1cs.satisfies(&z));
    let a = r1cs.apply(&r1cs.a0, &z);
    let b = r1cs.apply(&r1cs.b0, &z);

    let mut pch = SeededChallenger::new(seed);
    let (zc_proof, zc_claim) = zerocheck::prove(&tables, &a, &b, &z, &mut pch);
    let (lc_proof, lc_claim) = lincheck::prove(&tables, &r1cs, &z, &zc_claim.rho, &mut pch);

    let mut vch = SeededChallenger::new(seed);
    let zc_v = zerocheck::verify(&tables, m, &zc_proof, &mut vch).expect("zerocheck accepts");
    let lc_v =
        lincheck::verify(&tables, &r1cs, &zc_v, &lc_proof, &mut vch).expect("lincheck accepts");
    assert_eq!(lc_v, lc_claim);
    assert_eq!(pch.transcript().canonical(), vch.transcript().canonical());
    assert_eq!(zc_v.c_claim.evaluate(&z), zc_v.c_claim.value);
    assert_claim_true(&tables, &z, &lc_v);
}

#[test]
fn an_honest_lincheck_reduces_to_a_true_claim() {
    for (m, k_log, seed) in [(8usize, 6usize, 1u64), (10, 7, 2), (13, 8, 3)] {
        honest::<F64>(m, k_log, seed);
        honest::<F128>(m, k_log, seed);
    }
}

#[test]
fn a_wrong_fold_is_rejected() {
    wrong_fold_rejected::<F64>(10, 7, 21);
    wrong_fold_rejected::<F128>(10, 7, 22);
}

#[test]
fn zerocheck_then_lincheck_end_to_end() {
    for (m, k_log, seed) in [(9usize, 6usize, 31u64), (13, 8, 32), (16, 9, 33)] {
        end_to_end::<F64>(m, k_log, seed);
        end_to_end::<F128>(m, k_log, seed);
    }
}

#[test]
fn the_row_fold_matches_the_bits() {
    use rmip_packed::lincheck::{fold_rows, fold_rows_by_bits};
    use rmip_packed::zerocheck::{eq_table, PackedBits};
    fn check<F: PackedField>(m: usize, k_log: usize, seed: u64) {
        let mut s = seed;
        let bits: Vec<bool> = (0..1usize << m)
            .map(|i| (lcg(&mut s) % 100 < 28) && (i % 977 != 3))
            .collect();
        let z = PackedBits::<F>::from_bits(m, &bits);
        let rows = 1usize << (m - k_log);
        let outer: Vec<F> = (0..m - k_log)
            .map(|_| F::from_bits(u128::from(lcg(&mut s)) << 7 | 5))
            .collect();
        let weights = eq_table(&outer);
        assert_eq!(weights.len(), rows);
        assert_eq!(
            fold_rows(1 << k_log, &z, &outer),
            fold_rows_by_bits(1 << k_log, &z, &weights)
        );
    }
    check::<F64>(16, 8, 1);
    check::<F128>(16, 8, 2);
    check::<F64>(13, 9, 3);
    check::<F128>(11, 7, 4);
    check::<F64>(20, 8, 5);
    check::<F64>(9, 7, 6);
    check::<F128>(12, 9, 7);
    check::<F64>(7, 7, 8);
}
