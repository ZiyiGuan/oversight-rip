use rmip_field::ghash::{self, software};
use rmip_field::{reference, BinaryField, F128};

fn rng(seed: u64) -> impl FnMut() -> u128 {
    let mut s = seed | 1;
    move || {
        let mut o = [0u64; 2];
        for w in o.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *w = s;
        }
        ((o[0] as u128) << 64) | o[1] as u128
    }
}

const EDGES: [u128; 8] = [
    0,
    1,
    2,
    1 << 63,
    1 << 64,
    1 << 127,
    u128::MAX,
    (1 << 7) | (1 << 2) | (1 << 1) | 1,
];

#[test]
fn the_portable_kernel_agrees_with_the_reference_and_with_the_shipped_multiply() {
    let mut next = rng(0x243f_6a88_85a3_08d3);
    let check = |a: u128, b: u128| {
        let (x, y) = (F128::new(a), F128::new(b));
        let want = reference::mul(a, b);
        assert_eq!(
            software::ghash_mul(x, y).to_u128(),
            want,
            "sw mul {a:#x}*{b:#x}"
        );
        assert_eq!(
            software::ghash_mul_unreduced(x, y).reduce().to_u128(),
            want,
            "sw deferred {a:#x}*{b:#x}"
        );
        assert_eq!(x.mul(y).to_u128(), want, "dispatched mul {a:#x}*{b:#x}");
        assert_eq!(
            software::ghash_square(x).to_u128(),
            reference::square(a),
            "sw square {a:#x}"
        );
        assert_eq!(
            ghash::ghash_square(x).to_u128(),
            reference::square(a),
            "dispatched square {a:#x}"
        );
    };
    for _ in 0..20_000 {
        let (a, b) = (next(), next());
        check(a, b);
    }
    for a in EDGES {
        for b in EDGES {
            check(a, b);
        }
    }
}
