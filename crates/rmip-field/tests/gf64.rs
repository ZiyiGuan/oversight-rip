use rmip_field::gf64::{dot, dot8, mul2, reference_mul, scale_add8, square8, MODULUS_TAIL};
use rmip_field::{BinaryField, F128, F64};

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11 ^ (*state << 7)
}

#[test]
fn gf64_agrees_with_its_reference() {
    let mut s = 0x64_u64;
    for _ in 0..20_000 {
        let a = F64::new(lcg(&mut s));
        let b = F64::new(lcg(&mut s));
        assert_eq!(a * b, reference_mul(a, b), "{a:?} * {b:?}");
        assert_eq!(a.square(), reference_mul(a, a), "square of {a:?}");
    }
    for a in [0u64, 1, 2, u64::MAX, 1 << 63, 0x8000_0000_0000_001B] {
        for b in [0u64, 1, 2, u64::MAX, 1 << 63, 0x1B] {
            let (a, b) = (F64::new(a), F64::new(b));
            assert_eq!(a * b, reference_mul(a, b));
        }
    }
}

#[test]
fn the_paired_kernel_and_the_dot_product_agree_with_the_scalar_path() {
    let mut s = 0x2A_u64;
    for _ in 0..5_000 {
        let a = [F64::new(lcg(&mut s)), F64::new(lcg(&mut s))];
        let b = [F64::new(lcg(&mut s)), F64::new(lcg(&mut s))];
        assert_eq!(mul2(a, b), [a[0] * b[0], a[1] * b[1]]);
    }
    for n in [0usize, 1, 3, 4, 7, 64, 1000] {
        let a: Vec<F64> = (0..n).map(|_| F64::new(lcg(&mut s))).collect();
        let b: Vec<F64> = (0..n).map(|_| F64::new(lcg(&mut s))).collect();
        let expected = a
            .iter()
            .zip(&b)
            .fold(F64::ZERO, |acc, (x, y)| acc + *x * *y);
        assert_eq!(dot(&a, &b), expected, "n = {n}");
    }
}

#[test]
fn gf64_reduces_by_the_modulus() {
    assert_eq!(F64::new(1 << 63) * F64::X, F64::new(MODULUS_TAIL));
    assert_eq!(F64::X.pow(64), F64::new(MODULUS_TAIL));
    assert_eq!(
        F64::new(1 << 63) * F64::new(1 << 63),
        F64::new(1 << 62) * F64::new(MODULUS_TAIL)
    );
}

#[test]
fn gf64_is_a_field() {
    let mut s = 0xF1E1D_u64;
    for _ in 0..2_000 {
        let a = F64::new(lcg(&mut s));
        let b = F64::new(lcg(&mut s));
        let c = F64::new(lcg(&mut s));
        assert_eq!(a * b, b * a);
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!(a * F64::ONE, a);
        assert_eq!(a * F64::ZERO, F64::ZERO);
        assert_eq!(a + a, F64::ZERO);
        if a != F64::ZERO {
            assert_eq!(a * a.inv().unwrap(), F64::ONE, "inverse of {a:?}");
        }
        assert_eq!(a.pow(3), a * a * a);
        assert_eq!(a.pow(0), F64::ONE);
    }
    assert_eq!(F64::ZERO.inv(), None);
    assert_eq!(F64::from_bool(true), F64::ONE);
    assert_eq!(
        F64::from_be_bytes(F64::new(0x0123_4567_89AB_CDEF).to_be_bytes()).to_u64(),
        0x0123_4567_89AB_CDEF
    );
    let (a, b) = (F64::new(lcg(&mut s)), F64::new(lcg(&mut s)));
    assert_eq!((a + b).square(), a.square() + b.square());
    assert_eq!(F64::X.square(), F64::X * F64::X);
    assert_ne!(F64::X.square(), F64::X);
}

fn degree(p: u128) -> Option<u32> {
    (p != 0).then(|| 127 - p.leading_zeros())
}

fn poly_gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        while let (Some(da), Some(db)) = (degree(a), degree(b)) {
            if da < db {
                break;
            }
            a ^= b << (da - db);
        }
        std::mem::swap(&mut a, &mut b);
    }
    a
}

#[test]
fn the_modulus_is_irreducible() {
    let p: u128 = (1u128 << 64) | MODULUS_TAIL as u128;
    let mut x_pow = F64::X;
    for k in 1..=64 {
        x_pow = x_pow.square();
        if k == 32 {
            let diff = (x_pow + F64::X).to_u64() as u128;
            assert_eq!(poly_gcd(p, diff), 1, "gcd(x^(2^32) − x, P) must be 1");
        }
    }
    assert_eq!(x_pow, F64::X, "x^(2^64) ≡ x (mod P)");
    assert_eq!(poly_gcd(0b101, 0b11), 0b11);
}

#[test]
#[ignore = "measurement"]
fn throughput_of_the_two_fields() {
    use std::hint::black_box;
    use std::time::Instant;
    let n = 1usize << 22;
    let mut s = 0xBEEF_u64;
    let a64: Vec<F64> = (0..n).map(|_| F64::new(lcg(&mut s))).collect();
    let b64: Vec<F64> = (0..n).map(|_| F64::new(lcg(&mut s))).collect();
    let a128: Vec<F128> = (0..n)
        .map(|_| F128::new(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128))
        .collect();
    let b128: Vec<F128> = (0..n)
        .map(|_| F128::new(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128))
        .collect();
    let mut best64 = f64::MAX;
    let mut best128 = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        let mut acc = [F64::ZERO; 4];
        for i in 0..n {
            acc[i & 3] += a64[i] * b64[i];
        }
        black_box(acc);
        best64 = best64.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let mut acc = [F128::ZERO; 4];
        for i in 0..n {
            acc[i & 3] = acc[i & 3].add(a128[i].mul(b128[i]));
        }
        black_box(acc);
        best128 = best128.min(t.elapsed().as_secs_f64());
    }
    let mut best64v = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        black_box(dot(&a64, &b64));
        best64v = best64v.min(t.elapsed().as_secs_f64());
    }
    let ns64 = best64 * 1e9 / n as f64;
    let ns64v = best64v * 1e9 / n as f64;
    let ns128 = best128 * 1e9 / n as f64;
    println!(
        "GF(2^64) scalar:   {ns64:.3} ns per multiply, {:.4} ns per packed bit (64 per element)",
        ns64 / 64.0
    );
    println!(
        "GF(2^64) paired:   {ns64v:.3} ns per multiply, {:.4} ns per packed bit (64 per element)",
        ns64v / 64.0
    );
    println!(
        "GF(2^128): {ns128:.3} ns per multiply, {:.4} ns per packed bit (128 per element)",
        ns128 / 128.0
    );
    println!(
        "GF(2^128) over GF(2^64) paired: per multiply {:.2}x, per packed bit {:.2}x",
        ns128 / ns64v,
        (ns128 / 128.0) / (ns64v / 64.0)
    );
}

#[test]
fn the_eight_lane_kernels_agree_with_the_scalar_path() {
    let mut s = 0x77_u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F64::new(s >> 3 ^ s << 29)
    };
    for _ in 0..200 {
        let a: [F64; 8] = std::array::from_fn(|_| next());
        let b: [F64; 8] = std::array::from_fn(|_| next());
        let c = next();
        let expected_dot = a
            .iter()
            .zip(&b)
            .fold(F64::ZERO, |acc, (x, y)| acc + *x * *y);
        assert_eq!(dot8(&a, &b), expected_dot);
        let mut sq = a;
        square8(&mut sq);
        for (x, y) in a.iter().zip(&sq) {
            assert_eq!(*y, x.square());
            assert_eq!(*y, *x * *x);
        }
        let mut acc = b;
        scale_add8(&mut acc, c, &a);
        for ((x, y), z) in a.iter().zip(&b).zip(&acc) {
            assert_eq!(*z, *y + c * *x);
        }
    }
}
