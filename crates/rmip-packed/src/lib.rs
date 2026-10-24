pub mod batch;
pub mod circuit;
#[rustfmt::skip]
pub mod driver;
pub mod execute;
pub mod lincheck;
pub mod majority;
pub mod phases;
pub mod ringswitch;
pub mod statement;
pub mod subfield;
pub mod tree;
pub mod zerocheck;

use std::ops::{Add, AddAssign, Mul, MulAssign};

use rmip_field::{OpeningField, F128, F64};

pub trait PackedField:
    OpeningField
    + rmip_opt::ntt::NttField
    + Add<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + MulAssign
{
    const BITS: usize;
    fn subfield_8_generator() -> Self;
    fn subfield_8_root() -> Self;
    fn frobenius_basis() -> &'static ringswitch::FrobeniusBasis<Self>;
    fn skip_tables() -> &'static subfield::SkipTables<Self>;
    fn group_order() -> u128 {
        if Self::BITS == 128 {
            u128::MAX
        } else {
            (1u128 << Self::BITS) - 1
        }
    }
}

pub fn subfield_8_generator<F: PackedField>() -> F {
    let cofactor = F::group_order() / 255;
    assert_eq!(
        F::group_order() % 255,
        0,
        "255 divides 2^κ − 1 when 8 divides κ"
    );
    for v in 2u128.. {
        let g = F::from_bits(v).pow(cofactor);
        if g != F::ONE && g.pow(85) != F::ONE && g.pow(51) != F::ONE && g.pow(15) != F::ONE {
            return g;
        }
    }
    unreachable!("the field has elements of every order dividing 2^κ − 1")
}

pub fn subfield_8_root<F: PackedField>() -> F {
    let g = F::subfield_8_generator();
    let mut x = F::ONE;
    for _ in 0..255 {
        let v = x.pow(8) + x.pow(4) + x.pow(3) + x + F::ONE;
        if v == F::ZERO {
            return x;
        }
        x *= g;
    }
    unreachable!("the polynomial splits in GF(2^8)")
}

pub fn skip_domain<F: PackedField>(k: usize) -> Vec<F> {
    assert!(k <= 8, "the skip domain lies in GF(2^8)");
    let xi = F::subfield_8_root();
    let mut basis = Vec::with_capacity(k);
    let mut power = F::ONE;
    for _ in 0..k {
        basis.push(power);
        power *= xi;
    }
    (0..1usize << k)
        .map(|j| {
            let mut point = F::ZERO;
            for (i, b) in basis.iter().enumerate() {
                if (j >> i) & 1 == 1 {
                    point += *b;
                }
            }
            point
        })
        .collect()
}

#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn pack<F: PackedField>(bits: &[bool]) -> Vec<F> {
    assert_eq!(bits.len() % F::BITS, 0, "a whole number of elements");
    bits.chunks_exact(F::BITS)
        .map(|chunk| {
            let mut v = 0u128;
            for (r, &b) in chunk.iter().enumerate() {
                if b {
                    v |= 1u128 << r;
                }
            }
            F::from_bits(v)
        })
        .collect()
}

pub fn unpack<F: PackedField>(elements: &[F]) -> Vec<bool> {
    let mut bits = Vec::with_capacity(elements.len() * F::BITS);
    for e in elements {
        let v = e.to_bits();
        for r in 0..F::BITS {
            bits.push((v >> r) & 1 == 1);
        }
    }
    bits
}

impl PackedField for F64 {
    const BITS: usize = 64;
    fn subfield_8_generator() -> Self {
        static G: std::sync::OnceLock<F64> = std::sync::OnceLock::new();
        *G.get_or_init(subfield_8_generator::<F64>)
    }
    fn subfield_8_root() -> Self {
        static X: std::sync::OnceLock<F64> = std::sync::OnceLock::new();
        *X.get_or_init(subfield_8_root::<F64>)
    }
    fn frobenius_basis() -> &'static ringswitch::FrobeniusBasis<Self> {
        static B: std::sync::OnceLock<ringswitch::FrobeniusBasis<F64>> = std::sync::OnceLock::new();
        B.get_or_init(ringswitch::FrobeniusBasis::new)
    }
    fn skip_tables() -> &'static subfield::SkipTables<Self> {
        static T: std::sync::OnceLock<subfield::SkipTables<F64>> = std::sync::OnceLock::new();
        T.get_or_init(subfield::SkipTables::new)
    }
}

impl PackedField for F128 {
    const BITS: usize = 128;
    fn subfield_8_generator() -> Self {
        static G: std::sync::OnceLock<F128> = std::sync::OnceLock::new();
        *G.get_or_init(subfield_8_generator::<F128>)
    }
    fn subfield_8_root() -> Self {
        static X: std::sync::OnceLock<F128> = std::sync::OnceLock::new();
        *X.get_or_init(subfield_8_root::<F128>)
    }
    fn frobenius_basis() -> &'static ringswitch::FrobeniusBasis<Self> {
        static B: std::sync::OnceLock<ringswitch::FrobeniusBasis<F128>> =
            std::sync::OnceLock::new();
        B.get_or_init(ringswitch::FrobeniusBasis::new)
    }
    fn skip_tables() -> &'static subfield::SkipTables<Self> {
        static T: std::sync::OnceLock<subfield::SkipTables<F128>> = std::sync::OnceLock::new();
        T.get_or_init(subfield::SkipTables::new)
    }
}
