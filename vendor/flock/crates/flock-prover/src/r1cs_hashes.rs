//! Monolithic per-block R1CS encoders for cryptographic hashes (BLAKE3,
//! SHA-2). Each submodule packages: per-instance witness
//! layout, sparse `(A_0, B_0)` matrix construction (`C_0 = I`), `prove_fast`
//! helpers (the c-aliased fast path), and a `*Setup` convenience type
//! wrapping R1CS + PCS params.
//!
//! Submodules share low-level bit-packing / matrix-row utilities via
//! [`common`].
//!
//! The `field64` feature currently exposes BLAKE3 and its shared utilities
//! only. Other hash/tower encodings retain fixed 128-bit layouts and are
//! deliberately unavailable instead of silently using a mismatched layout.

pub mod blake3;
/// Shared low-level bit-packing / R1CS-row utilities (carry-save adders,
/// fused adders, lin-id slot helpers) used by the per-hash encoders.
pub mod common;
/// The Fiat–Shamir chain: BLAKE3 over a transcript with a finalize forked at
/// every squeeze — the FS chain's witness generator, over [`blake3`]'s rows.
#[cfg(not(feature = "field64"))]
pub mod fs_chain;
/// The recursion tower's Merkle glue gates (`SwapTable`, `BitSpreadTable`,
/// `PowMaskTable`, `FamilyTransposeTileTable`): small R1CS tables that join
/// the tower's in-circuit Merkle openings to the BLAKE3 compressions.
#[cfg(not(feature = "field64"))]
pub mod merkle_glue;
/// Merkle-path layout and node-hash spec (`MerkleTreeLayout`, `HashSpec`,
/// `ChunkPathInput`, `SLOT_WORDS`) that the tower's Merkle gates and
/// [`merkle_glue`] build on.
#[cfg(not(feature = "field64"))]
pub mod merkle_r1cs;
#[cfg(not(feature = "field64"))]
pub mod sha2;
