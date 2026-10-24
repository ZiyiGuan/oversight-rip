use crate::bits::{BitSlice, BitVec};
use std::cell::RefCell;
use std::collections::HashSet;

pub fn encode(u: &BitSlice) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + u.len().div_ceil(8));
    out.extend_from_slice(&(u.len() as u64).to_be_bytes());
    out.extend_from_slice(&BitVec::from_bools(u.to_vec()).to_bytes_be());
    out
}

pub trait Oracle {
    fn query(&self, u: &BitSlice) -> bool;

    fn query_word(&self, u: u64, width: usize) -> bool {
        let bits: Vec<bool> = (0..width).map(|k| (u >> k) & 1 == 1).collect();
        self.query(&bits)
    }
}

pub fn pack_query(u: &BitSlice) -> u64 {
    assert!(u.len() <= 64, "a word holds 64 bits");
    u.iter()
        .enumerate()
        .fold(0u64, |acc, (k, &b)| acc | (u64::from(b) << k))
}

impl<O: Oracle + ?Sized> Oracle for &O {
    fn query(&self, u: &BitSlice) -> bool {
        (**self).query(u)
    }
    fn query_word(&self, u: u64, width: usize) -> bool {
        (**self).query_word(u, width)
    }
}

pub struct CountingOracle<O: Oracle> {
    inner: O,
    occurrences: RefCell<usize>,
    distinct: RefCell<HashSet<Vec<bool>>>,
}

impl<O: Oracle> CountingOracle<O> {
    pub fn new(inner: O) -> Self {
        Self {
            inner,
            occurrences: RefCell::new(0),
            distinct: RefCell::new(HashSet::new()),
        }
    }

    pub fn occurrences(&self) -> usize {
        *self.occurrences.borrow()
    }

    pub fn distinct(&self) -> usize {
        self.distinct.borrow().len()
    }
}

impl<O: Oracle> Oracle for CountingOracle<O> {
    fn query(&self, u: &BitSlice) -> bool {
        *self.occurrences.borrow_mut() += 1;
        self.distinct.borrow_mut().insert(u.to_vec());
        self.inner.query(u)
    }
    fn query_word(&self, u: u64, width: usize) -> bool {
        *self.occurrences.borrow_mut() += 1;
        let bits: Vec<bool> = (0..width).map(|k| (u >> k) & 1 == 1).collect();
        self.distinct.borrow_mut().insert(bits);
        self.inner.query_word(u, width)
    }
}
