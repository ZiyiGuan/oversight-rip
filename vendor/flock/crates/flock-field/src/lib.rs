//! Binary field arithmetic for Flock protocols.

pub use gf2_8::F8;
pub use gf2_128::{F128, F256Unreduced, mul_by_x};
pub use gf2_256::{F256, QUADRATIC_NONRESIDUE, mul_by_x_inv};
pub use phi8::{PHI_8_TABLE, phi8};
#[cfg_attr(feature = "field64", path = "f64_slice.rs")]
pub mod f128_slice;
/// RMIP: two-lane kernels for the optional 64-bit field on PMULL targets.
#[cfg(all(feature = "field64", target_arch = "aarch64", target_feature = "aes"))]
pub mod f64_neon;
#[cfg_attr(feature = "field64", path = "gf2_64.rs")]
pub mod gf2_128;
#[cfg_attr(feature = "field64", path = "gf2_64_extension.rs")]
pub mod gf2_256;
pub mod gf2_8;
#[cfg_attr(feature = "field64", path = "phi8_64.rs")]
pub mod phi8;

/// Width of the active base field. Legacy `F128` names are retained at the API seam.
pub const BASE_FIELD_BITS: usize = F128::BITS;
pub const BASE_FIELD_BYTES: usize = F128::BYTES;
pub const EXTENSION_FIELD_BITS: usize = F256::BITS;

pub mod test_rng;
