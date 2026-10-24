use rmip_field::{BinaryField, F128};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> F128 {
        let mut hi = self.0;
        for _ in 0..2 {
            hi ^= hi << 13;
            hi ^= hi >> 7;
            hi ^= hi << 17;
        }
        self.0 = hi;
        let lo = hi.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        F128::new(((hi as u128) << 64) | lo as u128)
    }
}

#[test]
fn additive_group_is_xor_and_self_inverse() {
    let mut r = Rng::new(1);
    for _ in 0..1000 {
        let (a, b) = (r.next(), r.next());
        assert_eq!(a.add(b), F128::new(a.to_u128() ^ b.to_u128()));
        assert_eq!(a.add(b), b.add(a));
        assert_eq!(a.add(a), F128::ZERO);
        assert_eq!(a.add(F128::ZERO), a);
    }
}

#[test]
fn multiplication_is_commutative_associative_distributive() {
    let mut r = Rng::new(2);
    for _ in 0..1000 {
        let (a, b, c) = (r.next(), r.next(), r.next());
        assert_eq!(a.mul(b), b.mul(a));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(F128::ONE), a);
        assert_eq!(a.mul(F128::ZERO), F128::ZERO);
        assert_eq!(a.square(), a.mul(a));
    }
}

#[test]
fn inverse_is_correct() {
    let mut r = Rng::new(3);
    assert_eq!(F128::ZERO.inv(), None, "zero has no inverse");
    assert_eq!(F128::ONE.inv(), Some(F128::ONE));
    for _ in 0..500 {
        let a = r.next();
        if a == F128::ZERO {
            continue;
        }
        let ai = a.inv().expect("nonzero is invertible");
        assert_eq!(a.mul(ai), F128::ONE, "{a:?} * {ai:?}");
    }
}

#[test]
fn squaring_is_additive() {
    let mut r = Rng::new(4);
    for _ in 0..500 {
        let (a, b) = (r.next(), r.next());
        assert_eq!(a.add(b).square(), a.square().add(b.square()));
    }
}

#[test]
fn only_zero_and_one_are_idempotent() {
    assert_eq!(F128::ZERO.square(), F128::ZERO);
    assert_eq!(F128::ONE.square(), F128::ONE);
    let mut r = Rng::new(5);
    let mut checked = 0;
    for _ in 0..2000 {
        let a = r.next();
        if a == F128::ZERO || a == F128::ONE {
            continue;
        }
        assert_ne!(
            a.square(),
            a,
            "{a:?} is a third idempotent, which is impossible"
        );
        checked += 1;
    }
    assert!(checked > 1900);
}

#[test]
fn prime_subfield_embedding() {
    assert_eq!(F128::from_bool(false), F128::ZERO);
    assert_eq!(F128::from_bool(true), F128::ONE);
    assert_eq!(F128::ONE.add(F128::ONE), F128::ZERO);
    assert_eq!(F128::ONE.mul(F128::ONE), F128::ONE);
}

#[test]
fn serialization_is_big_endian_and_round_trips() {
    assert_eq!(
        F128::ONE.to_be_bytes(),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    );
    assert_eq!(
        F128::X.to_be_bytes()[15],
        2,
        "x is the coefficient of degree 1"
    );
    let mut r = Rng::new(6);
    for _ in 0..1000 {
        let a = r.next();
        assert_eq!(F128::from_be_bytes(a.to_be_bytes()), a);
    }
}

#[test]
fn byte_reflection_is_an_involution() {
    let reflect = |v: u128| -> u128 {
        let mut out = [0u8; 16];
        for (i, b) in v.to_be_bytes().iter().rev().enumerate() {
            out[i] = b.reverse_bits();
        }
        u128::from_be_bytes(out)
    };
    let mut r = Rng::new(7);
    for _ in 0..500 {
        let a = r.next();
        assert_eq!(reflect(reflect(a.to_u128())), a.to_u128());
    }
}
