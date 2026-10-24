use crate::{BinaryField, F128, F64};

pub trait OpeningField: BinaryField {
    const BYTES: usize;
    fn from_bits(bits: u128) -> Self;
    fn to_bits(self) -> u128;

    #[inline]
    fn to_word(self) -> F128 {
        F128::new(self.to_bits())
    }
    #[inline]
    fn from_word(w: F128) -> Option<Self> {
        let bits = w.to_u128();
        let clear = Self::KAPPA >= 128 || bits >> Self::KAPPA == 0;
        clear.then(|| Self::from_bits(bits))
    }
    #[inline]
    fn from_coin(c: F128) -> Self {
        Self::from_bits(c.to_u128())
    }
    fn write_be(self, out: &mut Vec<u8>);
    fn read_be(bytes: &[u8]) -> Self;
    fn dot(a: &[Self], b: &[Self]) -> Self;
    fn axpy(acc: &mut [Self], c: Self, row: &[Self]);
    type Unreduced: Copy + Default + Send + Sync + std::ops::BitXorAssign;
    fn product(self, rhs: Self) -> Self::Unreduced;
    fn reduce(p: Self::Unreduced) -> Self;
    #[inline]
    fn accumulate(acc: &mut Self::Unreduced, a: Self, b: Self) {
        *acc ^= a.product(b);
    }
    fn axpy_unreduced(acc: &mut [Self::Unreduced], c: Self, row: &[Self]);
    fn reduce_all(acc: &[Self::Unreduced]) -> Vec<Self>;

    fn to_words(elements: &[Self]) -> Vec<F128> {
        elements.iter().map(|e| e.to_word()).collect()
    }
    fn from_words(words: &[F128]) -> Option<Vec<Self>> {
        words.iter().map(|w| Self::from_word(*w)).collect()
    }
    fn to_bytes(elements: &[Self]) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::BYTES * elements.len());
        for e in elements {
            e.write_be(&mut out);
        }
        out
    }
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        assert_eq!(bytes.len() % Self::BYTES, 0, "whole elements");
        bytes.chunks_exact(Self::BYTES).map(Self::read_be).collect()
    }
}

impl OpeningField for F128 {
    const BYTES: usize = 16;
    #[inline]
    fn from_bits(bits: u128) -> Self {
        F128::new(bits)
    }
    #[inline]
    fn to_bits(self) -> u128 {
        self.to_u128()
    }
    #[inline]
    fn to_word(self) -> F128 {
        self
    }
    #[inline]
    fn from_word(w: F128) -> Option<Self> {
        Some(w)
    }
    fn write_be(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&BinaryField::to_be_bytes(self));
    }
    fn read_be(bytes: &[u8]) -> Self {
        <F128 as BinaryField>::from_be_bytes(bytes.try_into().expect("16 bytes"))
    }
    fn dot(a: &[Self], b: &[Self]) -> Self {
        crate::ghash::dot_lazy(a, b)
    }
    fn axpy(acc: &mut [Self], c: Self, row: &[Self]) {
        for (a, r) in acc.iter_mut().zip(row) {
            *a = a.add(c.mul(*r));
        }
    }
    type Unreduced = crate::F256Unreduced;
    #[inline]
    fn product(self, rhs: Self) -> Self::Unreduced {
        self.mul_unreduced(rhs)
    }
    #[inline]
    fn accumulate(acc: &mut Self::Unreduced, a: Self, b: Self) {
        crate::ghash::accumulate(acc, a, b);
    }
    #[inline]
    fn reduce(p: Self::Unreduced) -> Self {
        p.reduce()
    }
    fn axpy_unreduced(acc: &mut [Self::Unreduced], c: Self, row: &[Self]) {
        crate::ghash::axpy_unreduced(acc, c, row);
    }
    fn reduce_all(acc: &[Self::Unreduced]) -> Vec<Self> {
        acc.iter().map(|p| p.reduce()).collect()
    }
}

impl OpeningField for F64 {
    const BYTES: usize = 8;
    #[inline]
    fn from_bits(bits: u128) -> Self {
        F64::new(bits as u64)
    }
    #[inline]
    fn to_bits(self) -> u128 {
        self.to_u64() as u128
    }
    fn write_be(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_u64().to_be_bytes());
    }
    fn read_be(bytes: &[u8]) -> Self {
        F64::new(u64::from_be_bytes(bytes.try_into().expect("8 bytes")))
    }
    fn dot(a: &[Self], b: &[Self]) -> Self {
        crate::gf64::dot_lazy(a, b)
    }
    fn axpy(acc: &mut [Self], c: Self, row: &[Self]) {
        let n = acc.len().min(row.len());
        let (acc, row) = (&mut acc[..n], &row[..n]);
        let (acc8, acc_rest) = acc.as_chunks_mut::<8>();
        let (row8, row_rest) = row.as_chunks::<8>();
        for (a, r) in acc8.iter_mut().zip(row8) {
            crate::gf64::scale_add8(a, c, r);
        }
        for (a, r) in acc_rest.iter_mut().zip(row_rest) {
            *a += c * *r;
        }
    }
    type Unreduced = crate::gf64::Product64;
    #[inline]
    fn product(self, rhs: Self) -> Self::Unreduced {
        self.mul_unreduced(rhs)
    }
    #[inline]
    fn accumulate(acc: &mut Self::Unreduced, a: Self, b: Self) {
        crate::gf64::accumulate(acc, a, b);
    }
    #[inline]
    fn reduce(p: Self::Unreduced) -> Self {
        p.reduce()
    }
    fn axpy_unreduced(acc: &mut [Self::Unreduced], c: Self, row: &[Self]) {
        crate::gf64::axpy_unreduced(acc, c, row);
    }
    fn reduce_all(acc: &[Self::Unreduced]) -> Vec<Self> {
        crate::gf64::reduce_products(acc)
    }
}
