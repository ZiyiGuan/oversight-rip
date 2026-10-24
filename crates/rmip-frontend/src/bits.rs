#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct BitVec(Vec<bool>);

pub type BitSlice = [bool];

impl BitVec {
    pub fn zeros(n: usize) -> Self {
        Self(vec![false; n])
    }

    pub fn from_bools(v: Vec<bool>) -> Self {
        Self(v)
    }

    pub fn from_index(value: usize, width: usize) -> Self {
        assert!(
            width >= usize::BITS as usize || value < (1usize << width),
            "index {value} needs more than {width} bits"
        );
        let mut v = Vec::with_capacity(width);
        for i in (0..width).rev() {
            v.push(i < usize::BITS as usize && (value >> i) & 1 == 1);
        }
        Self(v)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &BitSlice {
        &self.0
    }

    pub fn get(&self, i: usize) -> bool {
        self.0[i]
    }

    pub fn set(&mut self, i: usize, v: bool) {
        self.0[i] = v;
    }

    pub fn push(&mut self, v: bool) {
        self.0.push(v);
    }

    pub fn popcount(&self) -> usize {
        self.0.iter().filter(|b| **b).count()
    }

    pub fn parity(&self) -> bool {
        self.popcount() % 2 == 1
    }

    pub fn to_bytes_be(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.0.len().div_ceil(8)];
        for (i, b) in self.0.iter().enumerate() {
            if *b {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }
}

impl std::fmt::Debug for BitVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{}", if *b { 1 } else { 0 })?;
        }
        Ok(())
    }
}
