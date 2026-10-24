# Provenance of vendor/flock

Upstream: https://github.com/succinctlabs/flock
Commit: b684b1258e4b1f202bec24afd660ace851b09e5e (2026-09-11T08:09:24+03:00,
"Merge pull request #55 from succinctlabs/codex/expand-rust-function-names"),
the revision this repository had pinned as a git dependency since
2026-10-13.
Vendored: 2026-10-16, by copying cargo's checkout of that revision
(~/.cargo/git/checkouts/flock-4487d88407b3b86b/b684b12) without its
`target/`, `.git/` and `.cargo-ok`. Licence: MIT OR Apache-2.0
(LICENSE-MIT and LICENSE-APACHE kept). The upstream `.cargo/config.toml`
sets `rustflags = ["-C", "target-cpu=native"]`, which its own bench
binaries were built with; this workspace's `.cargo/config.toml` adopts
the same flag for matched builds.

Identity check: `flock_proof_digest` (crates/rmip-flock/tests/adapter.rs)
proves 2^8 BLAKE3 compressions on the fast union path with the Fiat–Shamir
domain `rmip-flock-blake3-v0`, one thread, and asserts the serialized
proof's length and a 64-bit FNV-1a fold. Against cargo's checkout:
276372 bytes, fnv1a e0d65954522a2e38. The default vendored build must retain the
same asserted digest; the opt-in field64 build has a distinct proof instantiation.

SHA-256 over the sorted file list's SHA-256s at vendoring: 48424005a787cb4dc5944409de20fe0dbf4480dd37f5cff318596c0eec402a71

## Changes to upstream files

Every edit to a file under vendor/flock is recorded here with its reason.
The unmodified pinned revision remains the regression reference for every
native “Flock SNARG” number we report. Items 1–18 established the
byte-identical default SNARG path and the extracted interactive realization.
Item 19 adds an opt-in field/configuration port. The default digest assertion
was rerun successfully after the port: 276372 bytes, fnv1a e0d65954522a2e38.
It remains the required compatibility check; validation of the new field and
configuration is separate and does not imply a completed MIP backend.
Earlier boundary edits use `rmip, 2026-10-16` markers. The field/configuration
port is recorded by the complete grouped file inventory in item 19 rather
than by a marker on every changed line.

1. `crates/flock-transcript/src/challenger.rs`: `pub enum Compilation
   { Snarg, Interactive }` and three default methods on the `Challenger`
   trait, `compilation()` (default `Snarg`), `post_oracle(table, row_len)
   -> id` and `read_oracle(id, position) -> Option<Vec<F128>>` (both
   panic on a SNARG challenger). No existing implementation changes.
2. `crates/flock-core/src/challenger.rs`: re-exports `Compilation`.
3. `crates/flock-core/src/pcs/commit.rs`: `commit_with`, `commit_into_with`,
   `commit_lane_major_with` and `commit_merkle_with` take a
   `Compilation`; the existing `commit`, `commit_into`, `commit_lane_major`
   and `commit_merkle` are one-line wrappers passing `Snarg`;
   `finalize_commit` threads the parameter. Under `Interactive`,
   `commit_merkle_with` returns the encoded codeword with an empty tree and
   an empty cap, and the leaf pipeline is not taken.
4. `crates/flock-core/src/pcs/ligerito.rs`: `ligero_commit_with` (the
   existing `ligero_commit` wraps it with `Snarg`; under `Interactive` no
   tree is built); `LigeroWitness::cap` returns an empty slice on an empty
   tree; `merkle_paths_for` returns no paths on an empty tree;
   `verify_level_opens_or_oracle` (new) dispatches on the challenger's
   compilation: `Snarg` to the unchanged `verify_level_opens`,
   `Interactive` to reading the posted oracle's rows at the queried
   positions and requiring the opened rows to equal them, with an empty
   cap and no paths.
5. `crates/flock-core/src/pcs/ligerito/extension.rs` (the F256 split
   opener the fast union path runs): the prover reads
   `challenger.compilation()`; under `Interactive` it skips the L0 tree
   assertion, sends an empty initial cap, encodes each recursive level
   with `ligero_commit_with` and posts its table with
   `challenger.post_oracle` right after each `commit_split`; the verifier's
   three leaf checks call `verify_level_opens_or_oracle` with the oracle
   ids 0 (L0) and the level index.
6. `crates/flock-core/src/pcs.rs` (`open_batch_merged`): the two
   assertions on the commitment cap run under `Snarg` only.
7. `crates/flock-prover/src/prover.rs` (`prove_union_with_binding_zerocheck`):
   the L0 commit calls the `_with` variants with the challenger's
   compilation, and under `Interactive` posts the L0 codeword as oracle 0
   (rows of `num_ntts` words, position-major).

8. `crates/flock-core/src/zerocheck/ag_skip.rs` (`RoundGrindProver`) and
   `crates/flock-transcript/src/transcript_record.rs`
   (`RecordingChallenger`): the two challenger wrappers forward
   `compilation`, `post_oracle` and `read_oracle` to the wrapped challenger.
9. `crates/flock-core/src/pcs.rs` (`open_batch_mixed_ligerito_seeded`, the
   inner open): its commitment-cap assertion runs under `Snarg` only, as in
   item 6.

10. `crates/flock-core/src/genus95_curve_code/rng.rs` and `mod.rs`:
    `ChallengerRng`, an `RngCore` over a challenger's fresh coins (one
    `F128` per `u64`, one per 16-byte block), exported with the DRBGs.
11. `crates/flock-core/src/zerocheck/ag_skip.rs`:
    `sample_round_one_interactive` (the verifier's coins drive
    `sample_random_evaluation_point` through `ChallengerRng`; no seed
    expansion, no nonce, no proof of work); the prover takes it under
    `Interactive` with `r1_nonce = 0`, the verifier takes it under
    `Interactive` and rejects a nonzero `r1_nonce`.
12. `crates/flock-core/src/lincheck.rs` (`SkipPoint::sample_fresh_interactive`
    and the prover's and verifier's fresh-skip-point sites) and
    `crates/flock-core/src/lincheck/union.rs` (the union verifier's site):
    the same for the lincheck's fresh AG skip point, the nonce slot zero
    when the schedule has one.

13. `crates/flock-transcript/src/challenger.rs`: `Challenger::expect_oracle`
    (default `true`), the verifier's side of a posting, forwarded by the two
    wrappers (item 8).
14. `crates/flock-core/src/pcs.rs` (`open_batch_merged`): under `Interactive`
    the assist (on the forked child) and the inner opening (on the parent)
    run sequentially in the verifier's order, assist first, instead of the
    parallel join; a SNARG run keeps the join.
15. `crates/flock-core/src/pcs/commit.rs`: `statement_bytes(&PcsParams)`,
    the parameters as one statement message; `crates/flock-prover/src/prover.rs`:
    under `Interactive` the statement binding and that message precede the
    L0 posting (oracle 0's prefix carries them) and the later binding is
    skipped; `crates/flock-core/src/verifier.rs`: after its binding, under
    `Interactive`, the same message and `expect_oracle(0, num_ntts,
    n_positions)`; `crates/flock-core/src/pcs/ligerito/extension.rs`: the
    verifier's `expect_oracle(level, 2^log_lanes, block_len)` at each
    recursive cap, before the empty cap is observed, where the prover posted.
16. `crates/flock-core/src/zerocheck/ag_skip.rs` and `crates/flock-core/src/lincheck.rs`,
    `lincheck/union.rs`: the interactive samplers return `Option`; the
    verifier's exhaustion is `AgVerifyError::SamplerExhausted` /
    `LincheckError::SamplerExhausted` (new variants) instead of a panic; the
    prover's is a documented abort (`expect`).

17. `crates/flock-prover/src/prover.rs` (the circuit binding): refuses the
    interactive compilation with a message, since its wiring argument runs
    on a parallel fork whose interactive order is not specified; the
    interactive compilation covers the mixed (BLAKE3) binding only.

18. Formatting only (2026-10-16): `cargo fmt` over the six files of items 1
    to 3, 5, 7, 13 and 15, nine hunks, all inside the lines those items
    added (import lists and panic messages they lengthened past rustfmt's
    width); no upstream line changed, no behaviour changed, the SNARG
    digest test unchanged.


19. Optional `field64` arithmetic and explicit `Interactive40` parameters
    (2026-10-16). This changes the compiled arithmetic/packing instantiation
    only when the feature is selected. The default field tower, profile
    discriminants, native statement bytes and SNARG proof digest remain the
    compatibility baseline. Existing enum variants keep their serialized
    indices; `Interactive40` is appended.

    The field crate adds an actual eight-byte GF(2^64) base field modulo
    `x^64+x^4+x^3+x+1`, its sixteen-byte quadratic extension, and the matching
    byte-field embedding. Legacy integration names `F128` and `F256` are
    retained, with `BITS`, `BYTES` and `LOG_BITS` authoritative. Field
    construction rejects truncating a nonzero high limb. Feature forwarding
    is explicit through the four crate manifests. `FIELD64.md` records the
    arithmetic, encoding, coin transport and supported build scope.

    The protocol port makes packing, field-byte serialization, unsafe byte
    views, witness allocations, ring-switch dimensions, inverse Frobenius,
    AG interpolation/sampling and NTT dispatch use that actual field width.
    The AG skip remains six Boolean coordinates. Its inner Boolean cube
    has `LOG_BITS` coordinates, so it cannot pack 128 independent Boolean
    coefficients into a 64-bit field. Native 128-bit SIMD layouts are
    excluded where they are not width-generic; the selected field's
    hardware multiplication remains available in generic kernels. Test
    fixtures with fixed default-field layouts are either width-aware or
    explicitly restricted to the default build, and active-width byte
    views are checked rather than reading sixteen bytes from eight-byte
    elements. The BLAKE3 witness/layout, sparse-product, lincheck and
    poisoned-dead-padding checks run at both widths; only native SNARG
    roundtrips retain a default-field gate. The field64 padding test uses
    deterministic test coins and compares both proof messages and posted
    oracle digests, without claiming those coins provide security.

    `Blake3Setup::with_interactive_target(count, 40)` selects the new profile.
    Its Ligerito configuration disables all grinding and sums every native
    query/algebraic error family with runtime field denominators; each
    family is allocated at most 2^-42, leaving compiler/slack budget within
    the 2^-40 target. The AG cover denominator uses its field-dependent
    Hasse–Weil loss. The report conservatively covers a padded BLAKE3
    circuit one dimension above the dense commitment for partial batches.
    It imports conditional native statistical soundness; it does not prove
    the imported theorems or include MIP pinning/checker error. Interactive
    statement bytes bind the target and both field widths. Prover/verifier
    guards prevent interpreting this statistical profile as a SNARG.

    The smaller packing uses the reference ladder at the same packed
    dimension (`m + 1` in the default-field TOML), rather than dropping
    witness coordinates. Consequently the 64/128-bit m=30 configuration has
    **six native oracles**, domain logs `[19,18,17,16,15,14]`, and queries
    `[97,32,20,14,12,10]`. Its reported conditional native sum is approximately
    2^-42.921122698. The old five-oracle F128 pin-cost ledger does not apply
    unchanged. The new `interactive_security_report` example reproduces
    these runtime configurations without allocating a witness.

    Supported field64 scope is the shared Boolean AG BLAKE3 interactive
    oracle reference with the mixed binding. The recursive tower, mixed
    SHA-256/BLAKE3 registry, SHA-256, Fiat–Shamir chain and Merkle gate/glue
    modules are unavailable under that feature. Mixed/tower bundle flavors
    are rejected by proof-IO header parsing; their types are not exposed.
    Union verification rejects circuit/wiring and element compositions in
    field64 mode. This is not a completed two-prover MIP backend or a
    claim about a smaller-field SNARG. Root-workspace adapter/tests and
    measurements are recorded outside this upstream-file inventory.

    Files changed by this item (relative to vendor/flock; new additions
    are marked). This inventory records edits, not a claim that every test
    or end-to-end measurement has completed:

    ```text
    crates/flock-core/Cargo.toml
    crates/flock-core/src/circuit.rs
    crates/flock-core/src/circuit/builder.rs
    crates/flock-core/src/element_r1cs.rs
    crates/flock-core/src/element_r1cs/union.rs
    crates/flock-core/src/genus95_curve_code/artin_schreier.rs
    crates/flock-core/src/genus95_curve_code/field.rs
    crates/flock-core/src/genus95_curve_code/rng.rs
    crates/flock-core/src/genus95_curve_code/round1.rs
    crates/flock-core/src/genus95_curve_code/sampling.rs
    crates/flock-core/src/genus95_curve_code/tests.rs
    crates/flock-core/src/lincheck.rs
    crates/flock-core/src/lincheck/kernels.rs
    crates/flock-core/src/lincheck/union.rs
    crates/flock-core/src/matrix_fold.rs
    crates/flock-core/src/ntt/additive_ntt_f128.rs
    crates/flock-core/src/ntt/additive_ntt_f128/kernels.rs
    crates/flock-core/src/pcs.rs
    crates/flock-core/src/pcs/commit.rs
    crates/flock-core/src/pcs/jagged.rs
    crates/flock-core/src/pcs/ligerito.rs
    crates/flock-core/src/pcs/ligerito/interactive.rs (new)
    crates/flock-core/src/pcs/pack.rs
    crates/flock-core/src/pcs/ring_switch.rs
    crates/flock-core/src/pcs/tensor_algebra.rs
    crates/flock-core/src/product_gkr.rs
    crates/flock-core/src/r1cs.rs
    crates/flock-core/src/schedule.rs
    crates/flock-core/src/scratch.rs
    crates/flock-core/src/union.rs
    crates/flock-core/src/verifier.rs
    crates/flock-core/src/zerocheck.rs
    crates/flock-core/src/zerocheck/ag_skip.rs
    crates/flock-core/src/zerocheck/multilinear.rs
    crates/flock-core/src/zerocheck/multilinear/kernels.rs
    crates/flock-core/src/zerocheck/univariate_skip.rs
    crates/flock-core/src/zerocheck/univariate_skip_optimized.rs
    crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels.rs
    crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels/aarch64.rs
    crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels/bit_transpose_aarch64.rs (new)
    crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels/portable.rs
    crates/flock-field/Cargo.toml
    crates/flock-field/FIELD64.md (new)
    crates/flock-field/src/f64_slice.rs (new)
    crates/flock-field/src/gf2_128.rs
    crates/flock-field/src/gf2_256.rs
    crates/flock-field/src/gf2_64.rs (new)
    crates/flock-field/src/gf2_64_extension.rs (new)
    crates/flock-field/src/lib.rs
    crates/flock-field/src/phi8_64.rs (new)
    crates/flock-field/src/test_rng.rs
    crates/flock-prover/Cargo.toml
    crates/flock-prover/examples/interactive_security_report.rs (new)
    crates/flock-prover/src/lib.rs
    crates/flock-prover/src/proof_io.rs
    crates/flock-prover/src/prover.rs
    crates/flock-prover/src/r1cs_hashes.rs
    crates/flock-prover/src/r1cs_hashes/blake3.rs
    crates/flock-prover/src/r1cs_hashes/common.rs
    crates/flock-transcript/Cargo.toml
    crates/flock-transcript/src/challenger.rs
    crates/flock-transcript/src/transcript_record.rs
    ```

20. Boolean Circuit interaction (2026-10-17):
    - `crates/flock-prover/src/prover.rs`: the interactive Circuit branch
      forks the wiring stream, runs Boolean protocols on the parent, then
      wiring on the child, then merges, matching the verifier. The SNARG
      branch retains its parallel join. Interactive element-class Circuit
      instances remain explicitly unsupported.
    - `crates/flock-core/src/union.rs`: interactive Circuit binding records
      the complete public words, their count and field sizes before L0.
      Native SNARG statement bytes are unchanged.
    - `crates/flock-core/src/verifier.rs`: the matching interactive
      element-class Circuit guard rejects rather than entering an unaudited
      execution path.
    The new `rmip-flock` Circuit tests compare native algebra under matched
    coins, sample retries, fresh OS execution/replay, transcript equality,
    public-word/wiring mutations and all wiring nonce slots. The pinned
    native SNARG digest remains 276372 bytes, fnv1a e0d65954522a2e38.
    Auxiliary chains use the existing challenger hooks and need no further
    vendor edits. Their conditional acceptance is not the complete
    relativized MNIST claim.

21. Native query capture (2026-10-17):
    - `crates/flock-transcript/src/challenger.rs`: a default no-op
      `native_query_tuple(oracle, positions)` notification. It draws no
      coins and records no transcript entry.
    - `crates/flock-core/src/pcs/ligerito/extension.rs`: notify once after
      sampling L0, each non-final recursive tuple and the final tuple,
      preserving every position and duplicate in native slot order.
    - `crates/flock-transcript/src/transcript_record.rs` and
      `crates/flock-core/src/zerocheck/ag_skip.rs`: their delegating
      challengers forward the notification.
    The MIP main party can now prepare its rows without rerunning the
    native verifier. A differential test compares captured tuples and all
    answers with the former verifier-based derivation. The SNARG default
    hook is inert; its digest remains the regression reference.

22. AArch64 field64 vector kernels (2026-10-17):
    - `crates/flock-field/src/f64_neon.rs` (new) and `src/lib.rs`:
      two independent GF(2^64) products per NEON register, with PMULL and
      vector polynomial reduction; compiled only for field64 + AArch64 + AES.
    - `crates/flock-field/src/f64_slice.rs`: paired adjacent folds with
      the existing scalar tail and portable fallback.
    - `crates/flock-core/src/ntt/additive_ntt_f128/kernels/aarch64_field64.rs`
      (new), `kernels.rs`, `kernels/portable.rs` and `additive_ntt_f128.rs`:
      paired one/two/three-layer butterflies, the zero-root shortcut,
      contiguous full-live blocks and fused small-transform dispatch.
      The portable fused-four-layer definition remains available under
      field64 even on x86 AVX512, matching its existing dispatcher.
    - `crates/flock-core/src/zerocheck/multilinear/kernels/field64.rs`
      (new), `multilinear/kernels.rs`, `multilinear.rs` and `ag_skip.rs`:
      paired AG lookup folds, pending one/two-variable folds and unreduced
      lookahead accumulation. Existing padding cleansing is retained.
    Independent polynomial references and scalar differential tests check
    arithmetic, guards, odd tails, partial chunks, NTT suffixes and dead
    lanes. These edits change execution kernels, not the field tower,
    query counts, transcript, security accounting or Boolean relation.
    The original-field dispatch is unchanged. Measurements and validation
    are recorded in the development repository.

23. leanVM-derived field64 arithmetic and encoder scheduling (2026-10-17):
    Upstream is `leanEthereum/leanVM` commit
    `a0ca7858f2533ce51abdc20bb635c007e19b26b3`. The exact source paths,
    correspondence and licenses are retained under
    [third-party/leanvm](third-party/leanvm/README.md). Copyright (c) 2026
    leanEthereum. Both upstream root license files are copied unchanged.
    Its GF64 source credits `binius-zk/binius64` under Apache-2.0; that
    attribution is explicitly preserved in the source and notice.
    - `crates/flock-field/src/f64_neon.rs`: immediate scalar and paired
      products reduce through two PMULL-by-0x1b folds while remaining in
      vector registers. The paired reduction retains upstream's empty
      compiler barrier. Three-input XOR is SHA3-gated with a two-XOR
      fallback. The deferred `mul2_unreduced` and `reduce2` interfaces and
      accumulator representation remain unchanged.
    - `crates/flock-field/src/gf2_64.rs`: ARM AES multiplication and
      squaring dispatch to the immediate kernel; other architectures keep
      their prior multiplication fallback. The Itoh–Tsujii inverse uses
      63 counted self-products and eight other products, reducing
      `MULS_PER_INV` from 127 to 71 while preserving zero-to-zero inversion
      and the existing standalone-square counter convention.
    - `crates/flock-core/src/ntt/additive_ntt_f128.rs`: joint radix-eight
      twiddle evaluation and deep three-layer fusion with two/one-layer
      tails, adapted within Flock's existing cache-blocked driver.
    - `crates/flock-core/src/ntt/additive_ntt_f128/kernels/aarch64_field64.rs`:
      eight independent row-lane products per iteration, with paired and
      scalar tails and the existing in-register fused butterflies.
    - `crates/flock-field/FIELD64.md` and `third-party/leanvm/{README.md,
      LICENSE-MIT,LICENSE-APACHE}`: kernel description and upstream notices.
    Diagnostic field64 switches select equivalent row widths, deep radix,
    joint-twiddle use and subgroup sizes; the retained subgroup default is
    2 MiB. No gathered upstream encoder driver, lane reversal, cubic field
    extension or WHIR protocol is imported. The default 128-bit field path,
    protocol messages, security parameters and audit geometry are unchanged
    by this item. Independent arithmetic references, inversion and operation
    counts, and scalar-versus-optimized NTT checks are included; execution
    results and end-to-end measurements are recorded separately by the
    validation driver rather than inferred from the kernel instruction count.

24. Direct native encoder-input hand-off (2026-10-17):
    - `crates/flock-transcript/src/challenger.rs` adds opt-in
      `wants_oracle_coefficients` and `post_oracle_with_coefficients`. The
      default ignores the local coefficients and posts the original table.
      This is an honest-execution optimization, not an extra proof field,
      remote prover claim, oracle or transcript message.
    - `crates/flock-transcript/src/transcript_record.rs` forwards the local
      interface through its existing recording wrapper.
    - `crates/flock-prover/src/prover.rs` hands off the L0 input already used
      by the encoder. For integer lanes it reproduces exactly the existing
      lane-major-to-degree-major transpose and omits the same zero tail;
      otherwise the input is already in coefficient order. Ordinary stores
      do not request or pay for this extra transpose.
    - `crates/flock-core/src/pcs/ligerito/extension.rs` retains `base_table`
      through each recursive posting and supplies that exact encoder input.
      Posting remains at the same point before cap observation and OOD coins.
    The MIP backend can therefore skip inverse recovery. Fixed coefficient
    dimensions are checked against the public plan. Encoded-only posting
    retains the inverse/high-degree-check fallback. A differential test in
    `rmip-flock/tests/native_coefficients.rs` checks every native encoding and
    complete proof/transcript/answer equality against that fallback in both
    field configurations. The original SNARG path does not invoke this seam.

25. Field64 paired round-one kernels (2026-10-17), confined to
    `crates/flock-core/src/genus95_curve_code/round1.rs`:
    - Two independent full-live 64-message blocks occupy the low/high halves
      of one 128-lane A/B transpose, Boolean SLP and product network. Their
      field contributions are accumulated with separate equality weights.
    - The paired C transpose uses eight byte rows instead of sixteen and
      folds both 64-bit halves independently into the even/odd banks.
    - Field64 unreduced accumulators use one vector instead of three.
      Dense and padded full pairs use the new path; partial and odd tails
      retain the single-block path. Dead blocks are never loaded.
    Independent scalar-product and unfused/fused comparisons remain. A new
    padded differential test covers nonadjacent live blocks, dirty dead
    bits, partial bit ranges and distinct weights. The original-field
    dispatch, algebraic messages and all security parameters are unchanged.
    End-to-end measurements and validation are recorded in the development repository.

26. Field64 C-claim/packed-row fold tiling (2026-10-17), in
    `crates/flock-core/src/lincheck.rs`: an eight-stripe/eight-column tile
    holds eight u64 result accumulators while applying the byte lookup tables.
    Each stripe loads one witness word per column tile. The tile bounds its
    lookup footprint to 16 KiB, preserves partial-row and useful-column tails,
    and avoids reading dropped columns. The full-row and sufficiently large
    partial-row field64 dispatches use it; tiny shapes retain the existing
    generic path. Original-field dispatch is unchanged. An independent
    Boolean-fold test covers full/partial/zero rows and dirty unused columns;
    an ignored standalone benchmark compares old and new kernels on the
    same data. This changes neither the C-claim nor its transcript schedule.

27. Lazy native oracles and independent dense-witness reconstruction
    (2026-10-17):
    - `crates/flock-transcript/src/challenger.rs` adds opt-in
      `uses_lazy_oracles`, `post_oracle_from_coefficients` and
      `read_oracle_rows`. The default remains eager. Coefficient-only posts
      carry the exact local encoder input and encoded shape; batch reads
      retain query order and duplicates and must neither absorb messages nor
      sample coins. `transcript_record.rs` forwards these local operations.
    - `crates/flock-core/src/pcs/commit.rs` adds `commit_lazy`, exported by
      `pcs.rs`: it validates the same ordinary or integer-lane message shape
      and retains only commitment metadata. The union path in
      `crates/flock-prover/src/prover.rs` uses it only when an interactive
      challenger opts in, posts the exact L0 coefficients at the existing
      posting boundary, and omits the L0 encoding and Merkle construction.
    - `crates/flock-core/src/pcs/ligerito/extension.rs` similarly retains
      recursive shapes without encoding, and posts each exact `base_table`
      input at the unchanged boundary before cap observation and OOD coins.
      L0, intermediate and final queried rows then come from the backend's
      batch-read hook. Bounds, tuple count and row widths are checked.
      `pcs/ligerito.rs` documents the empty matrix/tree representation of
      these opt-in coefficient-backed oracles.
    - `crates/flock-prover/src/r1cs_hashes/blake3.rs` adds
      `Blake3Setup::dense_witness_from_compressions` for the fixed single-slot
      BLAKE3 mixed binding. It runs the same eight-compression Boolean
      witness builder with small z/a/b group scratch, writes declared
      invocations directly to the union's compact column-major witness,
      and preserves the zero tail. It avoids full a/b tables and lincheck
      stripes and takes no transcript, challenge, proof or main-prover table.
    The containing repository's MIP backend fixes immutable native and
    auxiliary coefficients before their queries; temporary full encoding
    remains a local evaluation strategy when cheaper. Its specialized
    `QueryChecker` independently derives the dense witness and follows only
    witness folds, coordinate splits and pin projections from legal prefixes.
    Native proof messages are checked by the existing verifier, not re-proved
    by the checker. This division is documented in the development repository. The ordinary interactive and SNARG paths stay eager, and no
    field, target, query count, pin count or transcript schedule changes.
    Differential checks compare every native row with full encoding, complete
    eager/lazy proof and transcript bytes, direct dense witnesses with native
    compaction, and specialized answers with the independent cached checker.
    These are implementation checks; measured performance and theorem scope
    are recorded separately by the containing repository's validation driver.

28. Explicit deferred-row main party for trusted reset compilation
    (2026-10-17): `crates/flock-transcript/src/challenger.rs` adds
    `defers_oracle_reads`, default false, forwarded by `transcript_record.rs`.
    In `crates/flock-core/src/pcs/ligerito/extension.rs`, the two honest-prover
    presplit consistency sanity checks take an optional row-derived claim.
    Only a challenger explicitly enabling both lazy interactive oracles and
    deferred reads omits that claim: the main party supplies zero placeholder
    rows, while the verifier later fills those slots from independent checker
    snapshots. Native algebraic messages still use the witness and induced
    basis, and native verifier equations are unchanged. Ordinary interactive
    and SNARG paths keep the original check. This seam avoids relying on
    release builds dropping a debug assertion; it neither implements trusted
    reset isolation nor establishes ordinary two-prover MIP soundness. The
    containing repository's `rmip-flock::reset` module states the distinct
    reset-per-query model explicitly.

29. Boolean Circuit execution at the 64-bit base-field packing width
    (2026-10-17):
    - `crates/flock-core/src/circuit.rs` derives cell word addresses, gather
      claim dimensions, union compatibility and circuit-static table offsets
      from `F128::LOG_BITS`. The original 128-bit values are unchanged. A
      cross-field Boolean-only test compares every cell and Boolean claim
      point with the union's own slot addressing, including distinct slot
      widths and nonzero word columns.
    - `crates/flock-core/src/circuit/builder.rs` describes wire words using
      the active base field; its builder operations already used field values
      without a fixed packing width.
    - `crates/flock-prover/src/prover.rs` and
      `crates/flock-core/src/verifier.rs` admit Boolean Circuit binding in
      the existing interactive field64 AG path, including its native wiring
      proof. The interactive-only, AG-only and no-element-class restrictions
      remain, as does the verifier's Interactive40 parameter policy.
    This enables an experimental execution path, not a certified 40-bit
    Circuit or relativized MNIST theorem. The BLAKE3 statistical report is
    not reused as a Circuit report. Full Circuit/wiring accounting and
    external-oracle answer authentication retain their separate obligations.

30. Opt-in AG Circuit verifier phase timing (2026-10-17):
    `crates/flock-core/src/verifier.rs` exposes
    `verify_ligerito_union_circuit_ag_profiled` beside the unchanged public
    verifier entry. Both call the same verification body; only the profiled
    entry supplies scoped timers. It reports disjoint native statement binding,
    Boolean/element PIOP algebra, wiring (Product-GKR and sigma/permutation),
    deferred matrix assertions and the complete merged opening, plus total
    native verification time. The unclassified residual covers checks,
    orchestration, pool dispatch and profiling overhead. Challenger callbacks
    are nested inside these phases, so a separately measured transcript total
    must not be added to them. Caller-side binding and final reset transcript
    checks are outside this native API. Disabled timing does not read a clock;
    no challenge, proof, check, field, profile or protocol schedule changes.

The native oracle ids are the posting order: 0 for L0, then 1, 2, … for the
recursive levels, which is also the verifier's reading order; each posting
site asserts the id the store returned is the one the verifier will read
(items 5 and 7), so a store serving more than one proof, or a posting out
of order, fails at the prover. The
native statement binding still observes the cap, which is empty under
`Interactive`; the extracted interactive prefix additionally binds the
parameters (item 15) and the target/field instantiation (item 19).
