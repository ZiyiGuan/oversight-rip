# Provenance: vendored code in this workspace

Two things are vendored, both from Flock, both under `Apache-2.0 OR MIT`, both
originating in binius64. This file covers the field; the additive NTT is
recorded in the same terms below.

## The `GF(2^128)` arithmetic in `rmip-field`

Vendored under four conditions: a plain copy at a pinned commit, both licence texts
preserved, per-file provenance headers, and this file "recording the commit and
every change made."

## Upstream

| | |
|---|---|
| project | Flock — <https://github.com/succinctlabs/flock> |
| commit | `e636760f8dae78306f804554fb4244993758b011` (`e636760`, 2026-08-17) |
| path | `crates/flock-core/src/field/gf2_128.rs` and `gf2_128/{aarch64,x86_64,portable}.rs` |
| licence | `Apache-2.0 OR MIT` |
| copyright | The Binius Developers; Irreducible, Inc.; Succinct Labs, Benedikt Bunz, William Wang |

Flock's own header records that its default `Mul` is a port of `mul_clmul` from
**binius64** (<https://github.com/binius-zk/binius64>,
`crates/field/src/arch/shared/ghash.rs`). So the arithmetic here is Binius's
algorithm, by way of Flock's port of it. That matters for the choice: the binius64
*crate* was measured on this machine at **0.39×–0.57×** our own speed, because
its `M256`/`M512` packed types have no native aarch64 GHASH implementation and
fall back to emulation over NEON — `arch/mod.rs` lists `packed_ghash_256` and
`packed_ghash_512` for `x86_64` only. Flock's flat `{ lo, hi }` port of the same
algorithm has no such layer and is the fastest of the three. The algorithm was
never the problem; the generic packed machinery was.

Our licence is `MIT OR Apache-2.0`, identical, so this is a copy under the same
terms rather than a relicensing. `LICENSE-APACHE` and `LICENSE-MIT` in this
directory are Flock's, copied unmodified.

## Location

The plan names `vendor/flock/` as the vendoring location. That entry was written
for the code-based *backend* (raw Ligerito), which is not vendored — the shipped
backend is the clean-room `rmip-backend-interleaved`. This is a single self-contained field
module, so it lives with the crate that uses it, at
`crates/rmip-field/src/ghash/`. Nothing else of Flock's is present in this tree.

## Every change made

`ghash.rs`, from `gf2_128.rs`:

1. **`serde` derives dropped.** `#[derive(..., Serialize, Deserialize)]` became
   `#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]`. This crate has no
   dependencies and the swap was not going to add one for a derive we never use.
   `Debug` was dropped from the derive too, and hand-written in `f128.rs` so the
   output stays `F128(0x…)` as it was before the swap.
2. **`new(lo: u64, hi: u64)` renamed `from_parts`.** `F128::new` in this
   workspace takes a `u128` and is called from all over the tree, mostly
   outside this crate; the rename keeps both spellings available. (No call-site
   count here on purpose: the argument does not turn on one, and the figure
   that used to stand here — "96 sites across 29 files, 80 of them outside
   this crate" — had drifted to 255 across 68 files by 2026-09-17. `f128.rs`'s
   module header carried the same defect and the same remedy was applied
   there.)
3. **Their `inv` removed.** It returns `ONE` for zero; `BinaryField::inv`
   returns `Option` and is the crate's only inverse. Keeping both would have
   left two functions of the same name disagreeing about `0`.
4. **`#[path = "gf2_128/portable.rs"]` retargeted** to `ghash/portable.rs`.
5. **Two dispatchers made `pub`** (`ghash_mul_unreduced`, `ghash_square`) and
   **one added** (`ghash_mul_schoolbook`), so `f128.rs` and `rmip-bench` can
   reach them safely without `unsafe` or `#[target_feature]` at the call site.
   The added one is a `#[cfg]` dispatch over routines that already existed.
6. **Module documentation added**, and one doc comment corrected: their
   `square` doc cites "the genus-95 samplers' x-power chains", which is Flock's
   workload, not ours. The replacement records what `bin/compare` measures for
   it here, which is *not* a speedup — see below.

7. **Upstream's `#[cfg(test)] mod tests` was not copied** — 323 lines and 16
   tests, the largest single deletion by volume. They test the same field, but
   against a bit-loop written inside that module, and this crate already has an
   independent one in `reference.rs` that is used by `f128.rs`'s tests. Copying
   theirs would have meant maintaining two portable references that must agree
   with each other as well as with the vendored code. The arch-level tests they
   contain — NEON against portable, the four aarch64 variants against each
   other — are covered here by `every_variant_agrees`, which additionally
   compares against a reference that shares nothing with the vendored file.

8. **Three NEON kernels added to `ghash/aarch64.rs`** (2026-10, for the
   deferred-reduction opening arithmetic): `ghash_dot_lazy_neon`,
   `ghash_accumulate_neon` and `ghash_axpy_unreduced_neon`, reached through
   the safe wrappers `dot_lazy`, `accumulate` and `axpy_unreduced` added to
   `ghash.rs`. They read and write `F256Unreduced` as four consecutive words
   through a raw pointer, so that struct is `repr(C)` and the kernel file
   asserts its offsets at compile time (a review on
   2026-10-18 found the default representation assumed). This item
   was missing until then: the statement below that the architecture
   files are verbatim was true of `x86_64.rs` and `portable.rs` only.

`ghash/{x86_64,portable}.rs`: **verbatim**, header only; `ghash/aarch64.rs`:
verbatim plus the three kernels of item 8. Verify with

```sh
for f in aarch64 x86_64 portable; do
  diff <(sed -n '9,$p' crates/rmip-field/src/ghash/$f.rs) \
       /path/to/flock/crates/flock-core/src/field/gf2_128/$f.rs
done
```

No arithmetic was modified in any file.

## What was kept out, and why

`gf2_8.rs` (GF(2^8)), `phi8.rs`, and `f128_slice.rs` are not vendored. Nothing
in this protocol uses a small binary field or the slice helpers; copying them
would be dead code carrying a licence obligation.

## How a divergence would be caught

`crates/rmip-field/src/reference.rs` is the hand-written `GF(2^128)` this crate
shipped before the swap, reduced to a `#[cfg]`-free bit-loop over `u128`.
`differential_agreement_with_the_portable_reference` and `every_variant_agrees`
in `f128.rs` check the vendored code against it on 20,000 random pairs plus 64
edge cases, per test, on every target — including `mul_by_x`, the dedicated
square, and deferred reduction, each of which is a *different algorithm* rather
than a second spelling of the same one.

`the_wire_format_did_not_move` pins the sixteen bytes `to_be_bytes` produces,
which is what every Merkle root, Fiat–Shamir transcript and figure in `RUN.txt`
depends on. The representation changed from `F128(u128)` to `{ lo, hi }`; `lo`
holds the coefficients of `x^0..x^63` in both, so the split is an identity.

## What the swap actually bought

Measured by `bin/compare` and `bin/endtoend` against `rmip_opt::field::mul` —
the predecessor Karatsuba, over the *same* hardware `clmul64`, so this is our
arithmetic against theirs and not against a software bit-loop:

| regime | ratio | who wins |
|---|---|---|
| dependent chain (latency) | 0.73× | predecessor |
| independent operands (throughput) | 1.35× | adopted |
| dot product, 4 accumulators | 2.01× | adopted |
| dedicated square vs `mul(a, a)` | 1.00× throughput, 0.77× latency | neither |
| deferred reduction vs reduce-each | 0.93× | reduce-each |

The protocol's hot paths are throughput-bound, so the middle rows are the ones that reach a total — measured end to end
across three post-swap runs at **1.62–1.69× less prover work** and 1.63–2.00×
less verifier work at the selected backend. Only the first of those is stable:
`T_mat` is a ~60 µs median and swings 23% run to run, and the pre-swap side of
that ratio is a single sample. The fold-ladder backend is the exception: it encodes serially and is
therefore ~1.05× *slower* after the swap (85.1 ms against 80.7 ms per
execution). It is documented as deliberately worse than the selected
backend, so this costs nothing that is claimed anywhere.

**Two of these findings are negative and are recorded because they were
expected to be positive.** The dedicated square halves the carry-less products
and does not reduce cost at all here, because both forms end in a reduction and
`ghash_reduce` is a scalar shift-XOR chain about as long as the product it
saves. Deferred reduction — XOR-ing 256-bit products and reducing once — is a
real algorithmic saving that this machine declines to pay out, because
`ghash_mul_unreduced_neon` extracts four lanes to scalar registers per term
where `ghash_mul_binius` keeps its work in vector registers and extracts twice.
Both are kept: they are what the adopted field ships, they are correct, and the
PMULL count is what matters on a machine where PMULL is the scarce resource.
Neither is credited with a speedup anywhere in these documents.

An earlier evaluation of this swap reported **3.1×** for the throughput row and
**6.1×** for deferred reduction — against the 1.35× and 0.93× above. Both were
artefacts of comparing across harnesses — our arm timed in a
single-accumulator dependent chain, theirs in a multi-accumulator dot product,
so the ratio was latency against throughput and measured neither field. That is
the third benchmark defect of this kind in this workspace, after a
fixed-operand loop the compiler hoisted and a pair of benchmark arms that both
called `F128::mul`; `bin/endtoend`'s header records all three.


---

# Provenance: the additive NTT in `rmip-opt`

## Upstream

| | |
|---|---|
| project | Flock — <https://github.com/succinctlabs/flock> |
| commit | `e636760f8dae78306f804554fb4244993758b011` (`e636760`, 2026-08-17) |
| path | `crates/flock-core/src/ntt/additive_ntt_f128.rs` |
| licence | `Apache-2.0 OR MIT` |
| copyright | Irreducible, Inc.; Succinct Labs, Benedikt Bunz, William Wang |

Flock's header records that the iterative LCH skeleton with neighbours-last
ordering derives from **binius64**'s `NeighborsLastReference`
(`crates/math/src/ntt/reference.rs`), with the interleaved SoA layout, the fused
two-layer butterfly and the parallelisation original to Flock. The same
provenance chain as the field, and the same reason for preferring Flock's:
binius64's `AdditiveNTT` trait takes `FieldSliceMut<P: PackedField>`, the packed
abstraction measured at 0.39×–0.57× our speed on aarch64. Flock's operates on
plain `&mut [F128]`.

## Why it was taken

`rmip_opt::encode::encode_columns` evaluated `m` rows at `n_enc` points by
Horner: `m · n · n_enc` multiplies, `Θ(N^1.5)` at a balanced shape. Measured
`N^1.495`, and it was the **only** superlinear term in the prover. Flock's
transform is `Θ(N log N)` and measured **765× faster at `N = 2²²`**.

## What was taken

`generate_evals_from_subspace`, `span_get`, `AdditiveNttF128::{new, standard,
log_domain_size, twiddle}`, `forward_transform_interleaved_scalar_from_layer`,
`log2_pow2`. Renamed `AdditiveNtt`; `F128::new(lo, hi)` → `from_parts`;
operators → `BinaryField::{add, mul}`; `.inv()` → `.inv().expect(..)` because
this crate's inverse returns `Option`. The arithmetic is unmodified.

## What was NOT taken, and why

The NEON and AVX kernels, the rayon paths, the cache-blocked batched variants,
the A/B toggles, and the inverse transform. The kernels and rayon are where
Flock's remaining performance is, and taking them means taking `rayon` into a
crate that depends on `rmip-field` and nothing else. The scalar core is the
asymptotic win.

`novel_basis_at` in `rmip-opt/src/ntt.rs` is **ours**, not Flock's.

## The consequence that is not a performance one

An additive NTT evaluates a polynomial in the **LCH novel basis**, not the
monomial basis, so the same message encodes to different symbols. Every codeword
value, every transcript and every figure in `RUN.txt` moved. What did **not**
move, and this is what the protocol rests on:

- the **evaluation domain is identical** — `standard(dim)` spans
  `{0, 1, …, 2^dim − 1}` under the integer encoding, exactly `code::domain`.
  Verified rather than assumed.
- encoding stays `F`-linear, so the proximity and opening combinations and
  `folding_and_encoding_commute` are untouched.
- `X̂_0 = 1`, so a constant message still encodes to a constant codeword.
- same code, same domain, same dimension ⇒ the distance, `ε_bind = 0` and the
  unique-decoding radius are unchanged. **No soundness parameter moves.**

## How a divergence would be caught

`the_basis_evaluates_what_the_transform_computes` (`rmip-opt/src/ntt.rs`) checks
`novel_basis_at` against `forward_interleaved` on **every column** at four
shapes — a tensor expansion of linear-map values against a butterfly network,
two genuinely different computations of the same quantity.
`optimised_encoding_matches_the_reference` pins `encode_columns` against the
naive `O(n · n_enc)` reference in the *same* basis; that reference was moved to
the novel basis rather than weakening the test to compare properties.

Noted honestly: that second test is no longer fully independent, since the
reference now shares `novel_basis_at` with the fast path. What it still covers
uniquely is the **interleaving** — that coefficient `k` of row `i` lands at
`data[k·m + i]` and `chunks(m)` recovers `out[c][i]`. A transposed layout passes
every test in `ntt.rs` and fails there.
