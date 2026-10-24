//! The genus-95 routines use the crate-wide [`crate::field::F128`] alias.
//! It selects GF(2^128) by default and GF(2^64) with `field64`. The curve,
//! rational functions and binary message code are defined over F2 and stay
//! identical; the evaluation field and its Artin–Schreier solver change.
//!
//! This module previously carried its own standalone `u128`-backed
//! implementation; it now re-exports the shared type and supplies the few
//! helpers the genus-95 code needs on top of flock's API:
//!   - [`F128Ext::inverse`] — `Option`-returning inverse (`None` at zero),
//!     matching the original signature; flock's inherent `inv()` maps 0 → 0.
//!   - [`F128Ext::random`] — uniform sampling from an `RngCore`.
//!   - [`F128Ext::to_bits`]/[`F128Ext::from_bits`] — polynomial-basis access in a `u128` container for the
//!     Artin–Schreier GF(2) linear algebra (coefficient i is bit i).
//!
//! `square` is provided by the shared field implementation, so the genus-95
//! code calls `value.square()` directly.
//!
//! Field addition is XOR; the shared `F128` spells it `+` (via `Add`), so the
//! genus-95 code uses `+`/`+=` like the rest of the repository.

use rand_core::RngCore;

pub use crate::field::F128;

/// Genus-95-specific helpers layered onto the shared [`F128`].
pub(crate) trait F128Ext: Sized {
    fn inverse(self) -> Option<Self>;
    fn random(rng: &mut impl RngCore) -> Self;
    fn to_bits(self) -> u128;
    fn from_bits(bits: u128) -> Self;
}

impl F128Ext for F128 {
    #[inline(always)]
    fn inverse(self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.inv())
        }
    }

    #[inline(always)]
    fn random(rng: &mut impl RngCore) -> Self {
        let lo = rng.next_u64();
        let hi = if F128::BITS == 128 { rng.next_u64() } else { 0 };
        F128::new(lo, hi)
    }

    #[inline(always)]
    fn to_bits(self) -> u128 {
        self.to_u128()
    }

    #[inline(always)]
    fn from_bits(bits: u128) -> Self {
        F128::from_u128(bits)
    }
}
