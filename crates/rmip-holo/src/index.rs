use rmip_backend_interleaved::{code, Shape};
use rmip_field::F128;
use rmip_frontend::SparseF2;
use rmip_iop::backend::Symbol;

pub struct HoloIndex {
    pub mu: usize,
    pub nu: usize,
    pub zeta: usize,
    pub corrections: Vec<(bool, usize, usize)>,
    pub v_pre: usize,
    pub groups: usize,
    pub codeword: Vec<Symbol<F128>>,
}

pub struct ProverIndex {
    pub tables: Vec<Vec<Vec<F128>>>,
    pub flat: Vec<F128>,
}

pub fn shape_for(v_pre: usize) -> Shape {
    let base = Shape::balanced(v_pre, 4, 4.0);
    if base.nu_row + base.nu_col != v_pre || base.nu_col == 0 {
        return base;
    }
    let nu_row = v_pre - v_pre / 2;
    let nu_col = v_pre - nu_row;
    Shape {
        nu_row,
        nu_col,
        n_enc: (1 << nu_col) * 4,
        lambda_bits: base.lambda_bits,
        regime: rmip_backend_interleaved::CodeRegime::UniqueDecoding,
    }
}

const MAX_V_PRE: usize = 48;

impl HoloIndex {
    pub fn is_well_formed(&self) -> bool {
        let width = self.mu + self.nu;
        if width == 0 || self.corrections.len() != 3 {
            return false;
        }
        if self.groups != width.div_ceil(crate::pack::LAMBDA) {
            return false;
        }
        if self.zeta > self.v_pre || self.v_pre > MAX_V_PRE {
            return false;
        }
        let Some(slots) = 3usize.checked_mul(self.groups) else {
            return false;
        };
        if slots > 1usize << (self.v_pre - self.zeta) {
            return false;
        }
        self.codeword.len() == shape_for(self.v_pre).n_enc
    }

    pub fn claim_point(&self, m: usize, g: usize, orbit_pt: &[F128]) -> Vec<F128> {
        let slot = m * self.groups + g;
        let prefix_vars = self.v_pre - self.zeta;
        let mut p = Vec::with_capacity(self.v_pre);
        for b in (0..prefix_vars).rev() {
            p.push(if (slot >> b) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            });
        }
        p.extend_from_slice(orbit_pt);
        p
    }

    pub fn error_terms(&self) -> f64 {
        use rmip_iop::backend::{div_error_upward, pow_error_upward, sum_errors_upward};
        let field = 2f64.powi(128);
        let eps_idx = div_error_upward((3 * (self.mu + self.nu) * self.zeta) as f64, field);
        let shape = shape_for(self.v_pre);
        let rho = div_error_upward(shape.n() as f64, shape.n_enc as f64);
        let escape = pow_error_upward(rho, shape.queries());
        let k_idx = 3 * self.groups * crate::pack::LAMBDA;
        let log_k = k_idx.next_power_of_two().trailing_zeros() as usize;
        let linear = div_error_upward((log_k + self.v_pre) as f64, field);
        sum_errors_upward(&[eps_idx, escape, linear])
    }
}

pub fn preprocess(
    a: &SparseF2,
    b: &SparseF2,
    c: &SparseF2,
    mu: usize,
    nu: usize,
) -> (HoloIndex, ProverIndex) {
    let wit = preprocess_prover(a, b, c, mu, nu);
    (encode(meta(a, b, c, mu, nu), &wit.flat), wit)
}

pub fn preprocess_verifier(
    a: &SparseF2,
    b: &SparseF2,
    c: &SparseF2,
    mu: usize,
    nu: usize,
) -> HoloIndex {
    let meta = meta(a, b, c, mu, nu);
    let width = mu + nu;
    let mut flat = vec![F128::ZERO; 1 << meta.v_pre];
    for (m, mat) in [a, b, c].into_iter().enumerate() {
        for g in 0..meta.groups {
            let lo = g * crate::pack::LAMBDA;
            let hi = (lo + crate::pack::LAMBDA).min(width);
            let group = bit_columns(mat, mu, nu, meta.zeta, lo, hi);
            pack_slot(&mut flat, meta.zeta, m * meta.groups + g, group);
        }
    }
    encode(meta, &flat)
}

pub fn preprocess_prover(
    a: &SparseF2,
    b: &SparseF2,
    c: &SparseF2,
    mu: usize,
    nu: usize,
) -> ProverIndex {
    let Meta {
        zeta,
        v_pre,
        groups,
        ..
    } = meta(a, b, c, mu, nu);
    let width = mu + nu;
    let tables: Vec<Vec<Vec<F128>>> = [a, b, c]
        .into_iter()
        .map(|mat| bit_columns(mat, mu, nu, zeta, 0, width))
        .collect();

    let mut flat = vec![F128::ZERO; 1 << v_pre];
    for (m, cols) in tables.iter().enumerate() {
        for g in 0..groups {
            let lo = g * crate::pack::LAMBDA;
            let hi = (lo + crate::pack::LAMBDA).min(width);
            pack_slot(&mut flat, zeta, m * groups + g, cols[lo..hi].to_vec());
        }
    }
    ProverIndex { tables, flat }
}

struct Meta {
    mu: usize,
    nu: usize,
    zeta: usize,
    corrections: Vec<(bool, usize, usize)>,
    v_pre: usize,
    groups: usize,
}

fn meta(a: &SparseF2, b: &SparseF2, c: &SparseF2, mu: usize, nu: usize) -> Meta {
    let zeta = [a, b, c]
        .iter()
        .map(|m| m.nz.len().max(1).next_power_of_two().trailing_zeros() as usize)
        .max()
        .unwrap();
    let width = mu + nu;
    let groups = width.div_ceil(crate::pack::LAMBDA);
    let slots = 3 * groups;
    let prefix_vars = slots.next_power_of_two().trailing_zeros() as usize;
    let v_pre = zeta + prefix_vars;

    let mut corrections = Vec::with_capacity(3);
    for m in [a, b, c] {
        assert!(
            !m.nz.is_empty(),
            "a matrix with no nonzeros has T_mat = 0; index it as such"
        );
        let (i0, j0) = m.nz[0];
        let pad = (1usize << zeta) - m.nz.len();
        corrections.push((pad % 2 == 1, i0, j0));
    }
    Meta {
        mu,
        nu,
        zeta,
        corrections,
        v_pre,
        groups,
    }
}

fn bit_columns(
    m: &SparseF2,
    mu: usize,
    nu: usize,
    zeta: usize,
    lo: usize,
    hi: usize,
) -> Vec<Vec<F128>> {
    let (i0, j0) = m.nz[0];
    let mut cols: Vec<Vec<F128>> = vec![Vec::with_capacity(1 << zeta); hi - lo];
    for z in 0..(1usize << zeta) {
        let (i, j) = if z < m.nz.len() { m.nz[z] } else { (i0, j0) };
        for (off, col) in cols.iter_mut().enumerate() {
            let k = lo + off;
            let bit = if k < mu {
                (i >> (mu - 1 - k)) & 1
            } else {
                (j >> (nu - 1 - (k - mu))) & 1
            };
            col.push(if bit == 1 { F128::ONE } else { F128::ZERO });
        }
    }
    cols
}

fn pack_slot(flat: &mut [F128], zeta: usize, slot: usize, mut group: Vec<Vec<F128>>) {
    while group.len() < crate::pack::LAMBDA {
        group.push(vec![F128::ZERO; 1 << zeta]);
    }
    let packed = crate::pack::pack(&group);
    flat[slot << zeta..(slot << zeta) + (1 << zeta)].copy_from_slice(&packed);
}

fn encode(meta: Meta, flat: &[F128]) -> HoloIndex {
    let Meta {
        mu,
        nu,
        zeta,
        corrections,
        v_pre,
        groups,
    } = meta;
    let shape = shape_for(v_pre);
    assert_eq!(
        flat.len(),
        shape.m() * shape.n(),
        "index length must match its shape"
    );
    let rows: Vec<Vec<F128>> = flat.chunks(shape.n()).map(<[F128]>::to_vec).collect();
    let dom = code::domain(shape.n_enc);
    let codeword: Vec<Symbol<F128>> = rmip_opt::encode::encode_columns(&rows, &dom)
        .into_iter()
        .map(Symbol)
        .collect();

    HoloIndex {
        mu,
        nu,
        zeta,
        corrections,
        v_pre,
        groups,
        codeword,
    }
}
