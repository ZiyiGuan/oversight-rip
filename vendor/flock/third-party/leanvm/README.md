# leanVM arithmetic and encoder provenance

Upstream: https://github.com/leanEthereum/leanVM

Pinned commit: `a0ca7858f2533ce51abdc20bb635c007e19b26b3`.
Adapted on 2026-10-17. Copyright (c) 2026 leanEthereum.

Both upstream root license files are retained byte-for-byte here:
[LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
The upstream GF64 source explicitly credits
[`binius-zk/binius64`](https://github.com/binius-zk/binius64) under Apache-2.0;
that attribution is preserved for the adapted field arithmetic. These
notices supplement Flock's existing notices; they do not replace them.

Source correspondence at the pinned commit:

| Upstream source | Adapted behavior | Local destination |
|---|---|---|
| [`crates/primitives/src/field/gf2_64.rs`, lines 49–74](https://github.com/leanEthereum/leanVM/blob/a0ca7858f2533ce51abdc20bb635c007e19b26b3/crates/primitives/src/field/gf2_64.rs#L49-L74) | Itoh–Tsujii inversion, 63 squarings and eight other products | `crates/flock-field/src/gf2_64.rs` |
| [`gf2_64.rs`, lines 223–277](https://github.com/leanEthereum/leanVM/blob/a0ca7858f2533ce51abdc20bb635c007e19b26b3/crates/primitives/src/field/gf2_64.rs#L223-L277) | Immediate scalar and paired PMULL reductions, including the pair's empty compiler barrier | `crates/flock-field/src/f64_neon.rs` |
| [`crates/primitives/src/field/neon.rs`, lines 15–35](https://github.com/leanEthereum/leanVM/blob/a0ca7858f2533ce51abdc20bb635c007e19b26b3/crates/primitives/src/field/neon.rs#L15-L35) | SHA3-gated three-input XOR and ordinary-XOR fallback | `crates/flock-field/src/f64_neon.rs` |
| [`crates/pcs/src/ntt/additive_ntt_f64.rs`, lines 94–117](https://github.com/leanEthereum/leanVM/blob/a0ca7858f2533ce51abdc20bb635c007e19b26b3/crates/pcs/src/ntt/additive_ntt_f64.rs#L94-L117) | Joint radix-eight twiddle computation using the linear subset-sum identity | `crates/flock-core/src/ntt/additive_ntt_f128.rs` |
| [`additive_ntt_f64.rs`, lines 438–500](https://github.com/leanEthereum/leanVM/blob/a0ca7858f2533ce51abdc20bb635c007e19b26b3/crates/pcs/src/ntt/additive_ntt_f64.rs#L438-L500) | Deep three-layer fusion with two/one-layer tails, adapted to Flock's existing cache driver | `crates/flock-core/src/ntt/additive_ntt_f128.rs` |
| [`additive_ntt_f64.rs`, lines 941–1012](https://github.com/leanEthereum/leanVM/blob/a0ca7858f2533ce51abdc20bb635c007e19b26b3/crates/pcs/src/ntt/additive_ntt_f64.rs#L941-L1012) | Eight independent lane products per row-loop iteration | `crates/flock-core/src/ntt/additive_ntt_f128/kernels/aarch64_field64.rs` |

The local base field has the same polynomial and bit representation as
upstream. Flock's quadratic extension, Ligerito protocol, deferred-product
accumulator layout, and transcript remain local. No upstream cubic extension,
WHIR protocol, gathered encoder driver or lane-order reversal is imported.

The local inversion deliberately uses counted self-products internally,
preserving Flock's distinction between native inversion work and one
inversion constraint; its counter constant is updated from 127 to 71.
The existing standalone-square counter convention is retained.

See [PROVENANCE item 23](../../PROVENANCE.md) for the complete local edit inventory.
