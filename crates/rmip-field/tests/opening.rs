use rmip_field::{BinaryField, OpeningField, F128, F64};

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 11
}

fn elements<F: OpeningField>(n: usize, seed: u64) -> Vec<F> {
    let mut s = seed;
    (0..n)
        .map(|_| F::from_bits((u128::from(lcg(&mut s)) << 64) | u128::from(lcg(&mut s))))
        .collect()
}

#[test]
fn words_round_trip_and_junk_is_refused() {
    for x in elements::<F64>(64, 1) {
        let w = x.to_word();
        assert_eq!(
            w.to_u128() >> 64,
            0,
            "a 64-bit element sits in the low bits"
        );
        assert_eq!(F64::from_word(w), Some(x));
        let junk = F128::new(w.to_u128() | (1u128 << 100));
        assert_eq!(F64::from_word(junk), None, "a bit above κ is refused");
    }
    for x in elements::<F128>(64, 2) {
        assert_eq!(x.to_word(), x);
        assert_eq!(F128::from_word(x), Some(x));
    }
    let c = F128::new((7u128 << 64) | 12345);
    assert_eq!(F64::from_coin(c), F64::new(12345));
    assert_eq!(F128::from_coin(c), c);
}

#[test]
fn bytes_round_trip() {
    let xs = elements::<F64>(37, 3);
    let bytes = F64::to_bytes(&xs);
    assert_eq!(bytes.len(), 8 * xs.len());
    assert_eq!(F64::from_bytes(&bytes), xs);
    let ys = elements::<F128>(37, 4);
    let bytes = F128::to_bytes(&ys);
    assert_eq!(bytes.len(), 16 * ys.len());
    assert_eq!(F128::from_bytes(&bytes), ys);
    let x = F64::new(0x0123_4567_89ab_cdef);
    let wide = <F64 as BinaryField>::to_be_bytes(x);
    assert_eq!(&wide[..8], &[0u8; 8]);
    assert_eq!(&wide[8..], &x.to_u64().to_be_bytes());
    assert_eq!(<F64 as BinaryField>::from_be_bytes(wide), x);
}

#[test]
fn kernels_agree_with_the_fold() {
    fn check<F: OpeningField>(seed: u64) {
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 64, 100] {
            let a = elements::<F>(n, seed + n as u64);
            let b = elements::<F>(n, seed + 1000 + n as u64);
            let fold = a
                .iter()
                .zip(&b)
                .fold(F::ZERO, |acc, (x, y)| acc.add(x.mul(*y)));
            assert_eq!(F::dot(&a, &b), fold, "dot at n = {n}");
            let c = elements::<F>(1, seed + 2000 + n as u64)[0];
            let mut acc = a.clone();
            F::axpy(&mut acc, c, &b);
            let expected: Vec<F> = a.iter().zip(&b).map(|(x, y)| x.add(c.mul(*y))).collect();
            assert_eq!(acc, expected, "axpy at n = {n}");
            let rows: Vec<Vec<F>> = (0..3).map(|k| elements::<F>(n, seed + 3000 + k)).collect();
            let coeffs = elements::<F>(3, seed + 4000);
            let mut lazy = vec![F::Unreduced::default(); n];
            let mut eager = vec![F::ZERO; n];
            for (ci, row) in coeffs.iter().zip(&rows) {
                F::axpy_unreduced(&mut lazy, *ci, row);
                F::axpy(&mut eager, *ci, row);
            }
            assert_eq!(
                F::reduce_all(&lazy),
                eager,
                "unreduced combination at n = {n}"
            );
            let mut acc = F::Unreduced::default();
            for (x, y) in a.iter().zip(&b) {
                F::accumulate(&mut acc, *x, *y);
            }
            assert_eq!(F::reduce(acc), fold, "accumulate at n = {n}");
        }
    }
    check::<F64>(5);
    check::<F128>(6);
}

#[test]
#[ignore = "measurement"]
fn kernel_throughput() {
    use std::time::Instant;
    let n = 2048usize;
    let reps = 8192usize;
    let a64 = elements::<F64>(n, 11);
    let b64 = elements::<F64>(n, 12);
    let a128 = elements::<F128>(n, 13);
    let b128 = elements::<F128>(n, 14);
    let per = |t: Instant| t.elapsed().as_secs_f64() * 1e9 / (n * reps) as f64;
    for _ in 0..2 {
        let t = Instant::now();
        let mut acc = F64::ZERO;
        for _ in 0..reps {
            let mut s = F64::ZERO;
            for (x, y) in a64.iter().zip(&b64) {
                s += *x * *y;
            }
            acc += s;
        }
        std::hint::black_box(acc);
        let scalar64 = per(t);
        let t = Instant::now();
        let mut acc = F64::ZERO;
        for _ in 0..reps {
            acc += F64::dot(&a64, &b64);
        }
        std::hint::black_box(acc);
        let vector64 = per(t);
        let t = Instant::now();
        let mut acc = <F128 as BinaryField>::ZERO;
        for _ in 0..reps {
            acc = acc.add(F128::dot(&a128, &b128));
        }
        std::hint::black_box(acc);
        let f128 = per(t);
        println!(
            "GF(2^64) scalar {scalar64:.2} ns/mul, GF(2^64) vector kernel {vector64:.2} ns/mul, GF(2^128) {f128:.2} ns/mul; per bit the 64-bit opening is {:.2}x the 128-bit one",
            2.0 * vector64 / f128
        );
    }
}
