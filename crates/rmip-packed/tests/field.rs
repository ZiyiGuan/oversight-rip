use rmip_field::OpeningField;
use rmip_field::{F128, F64};
use rmip_packed::{pack, skip_domain, unpack, PackedField};
use std::collections::HashSet;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn packing_round_trips<F: PackedField>() {
    let mut s = 0x9AC_u64;
    let bits: Vec<bool> = (0..F::BITS * 37).map(|_| lcg(&mut s) & 1 == 1).collect();
    let elements = pack::<F>(&bits);
    assert_eq!(elements.len(), 37);
    assert_eq!(unpack(&elements), bits);
    for (i, e) in elements.iter().enumerate() {
        for r in 0..F::BITS {
            assert_eq!((e.to_bits() >> r) & 1 == 1, bits[F::BITS * i + r]);
        }
    }
    let other: Vec<bool> = (0..F::BITS * 37).map(|_| lcg(&mut s) & 1 == 1).collect();
    let xor: Vec<bool> = bits.iter().zip(&other).map(|(a, b)| a ^ b).collect();
    let summed: Vec<F> = pack::<F>(&bits)
        .iter()
        .zip(pack::<F>(&other))
        .map(|(a, b)| *a + b)
        .collect();
    assert_eq!(summed, pack::<F>(&xor));
}

fn the_subfield_and_the_skip_domain<F: PackedField>() {
    let g = F::subfield_8_generator();
    assert_ne!(g, F::ONE);
    assert_eq!(g.pow(255), F::ONE, "order divides 255");
    for d in [85u128, 51, 15, 5, 3, 1] {
        assert_ne!(g.pow(d), F::ONE, "order is exactly 255, not {d}");
    }
    assert_eq!(g.pow(256), g, "g lies in GF(2^8)");
    let full = skip_domain::<F>(8);
    let distinct: HashSet<u128> = full.iter().map(|p| p.to_bits()).collect();
    assert_eq!(distinct.len(), 256);
    for p in &full {
        assert_eq!(p.pow(256), *p, "every point is in the subfield");
    }
    let domain = skip_domain::<F>(6);
    assert_eq!(domain.len(), 64);
    assert_eq!(domain[0], F::ZERO);
    let members: HashSet<u128> = domain.iter().map(|p| p.to_bits()).collect();
    assert_eq!(members.len(), 64);
    for a in &domain {
        for b in &domain {
            assert!(
                members.contains(&(*a + *b).to_bits()),
                "closed under addition"
            );
        }
    }
    let xi = F::subfield_8_root();
    assert_eq!(
        xi.pow(8) + xi.pow(4) + xi.pow(3) + xi + F::ONE,
        F::ZERO,
        "ξ is a root"
    );
    assert_eq!(xi.pow(256), xi, "ξ lies in GF(2^8)");
    assert_eq!(domain[1], F::ONE);
    assert_eq!(domain[2], xi);
    assert_eq!(domain[3], F::ONE + xi);
    assert_eq!(domain[4], xi * xi);
}

#[test]
fn packing_round_trips_at_both_fields() {
    packing_round_trips::<F64>();
    packing_round_trips::<F128>();
}

#[test]
fn the_subfield_and_the_skip_domain_at_both_fields() {
    the_subfield_and_the_skip_domain::<F64>();
    the_subfield_and_the_skip_domain::<F128>();
}

#[test]
fn group_orders_and_widths() {
    assert_eq!(<F64 as PackedField>::group_order(), u64::MAX as u128);
    assert_eq!(<F128 as PackedField>::group_order(), u128::MAX);
    assert_eq!(
        <F64 as rmip_field::OpeningField>::from_bits(u128::MAX).to_bits(),
        u64::MAX as u128
    );
    assert_eq!(
        <F128 as rmip_field::OpeningField>::from_bits(u128::MAX).to_bits(),
        u128::MAX
    );
}
