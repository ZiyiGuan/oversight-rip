# Optional 64-bit field tower

The `field64` feature selects GF(2^64) modulo `x^64+x^4+x^3+x+1`
and its quadratic extension GF(2^128), defined by `u^2+u+x^-1`.
The legacy public type names `F128` and `F256` remain at the integration
seam, but their actual sizes are 8 and 16 bytes. `BITS` and `BYTES` on
the types are authoritative; the names are not width guarantees.

The default feature selection retains the original 128/256-bit tower.
The optional field is not a security configuration by itself: all
protocol error terms, packing widths, sampling and query counts must
also use the selected tower.

The supported integration is the shared **AG BLAKE3 interactive oracle
reference**, selected with `rmip-flock/field64` and
`Blake3Setup::with_interactive_target(n_compressions, 40)`. The target
constructor selects `LigeritoProfile::Interactive40`, disables grinding,
and checks the shape's summed conditional native error bound using the
active base and quadratic field degrees. A rejected shape is not a
40-bit configuration. The report imports the native AG/IOP soundness
claims; it is not a proof of those claims.

The conditional native BLAKE3 two-prover pinning/checker backend also runs
at this width. It recomputes history and rank bounds using 64-bit base
elements and derives the actual native ladder, including six roots at
2^30 witness bits. Its canonical frames bind both field widths. A separate
42-bit chain-collision allocation is added to the reported native bound,
and the benchmark checks that their sum meets 40 bits. This remains
conditional on the active native IOP soundness imports. It does not complete
relativized MNIST, the recursive proof tower, or the older Reed--Solomon
zerocheck path in 64-bit mode. Default builds retain the original fields.

Integer packing is checked: `new(lo, hi)` rejects a nonzero high limb in
64-bit mode, and `from_u128` rejects any value that does not fit. The
separate `from_polynomial` API reduces a degree-below-128 polynomial by
the active modulus. It must not replace an injective packing operation.
Canonical `to_le_bytes`/`from_le_bytes` use the active byte width.

All witness packing uses `F128::BITS` Boolean coordinates per word.
The remaining in-word coordinates after the six-coordinate AG skip have
length `F128::LOG_BITS - 6`: one in the original field and zero in the
smaller field. The NTT uses the selected field's polynomial basis, and
128-bit SIMD memory layouts are excluded when they are not width
generic. Generic kernels continue to use hardware 64-bit field
multiplication where available. These are arithmetic/layout changes;
the AG Boolean constraint relation and its six-coordinate skip remain.

The repository's transcript source still returns 128 random bits per
coin. In the `rmip-flock` bridge, `up` selects the low 64 bits under this
feature. A uniform 128-bit source projects to an exactly uniform 64-bit
field element, with no rejection or bias; independent source draws
remain independent. The full source coins are retained by the outer
transcript for exact replay. This projection is deliberately confined
to coin transport. It is **not a field homomorphism** between the two
polynomial-basis fields and must not be used to translate a field
product or a witness field equation. The inverse-direction `down`
zero-extends a representation; it does not embed GF(2^64) as the
corresponding GF(2^128) subfield.

The proof's algebraic messages use active-width bytes. The OS challenger
exercises fresh coins; seeded sources in the tests and measurements are
for reproducibility. The quadratic challenges still have 128 bits in
small-field mode because they consist of two independent base-field
coordinates.

The AES-byte embedding is regenerated for this field. Its root is
`0xb5edb70665632cca`, a root of `t^8+t^4+t^3+t+1`. One deterministic way
to reproduce it is to exponentiate the first integer field element
whose projection `a^((2^64-1)/255)` has order 255, then select its first
positive power that is an AES-polynomial root. The resulting generator
is `0x0069db4862da3cdb`; the selected power is 55.

Validation includes an independent polynomial long-division arithmetic
reference, the degree-64 Rabin irreducibility criterion, exhaustive
65,536-pair byte-field homomorphism checks, quadratic irreducibility
through the trace-one criterion, Frobenius orders, inverse roundtrips,
storage widths, canonical byte roundtrips, and rejection of lossy
construction. AArch64 PMULL and the portable fallback are exercised
separately; the x86 PCLMUL branch requires testing on that architecture.

From the repository root:

```sh
cargo test --manifest-path vendor/flock/Cargo.toml --offline -p flock-field
cargo test --manifest-path vendor/flock/Cargo.toml --offline -p flock-field --features field64
RUSTFLAGS='-C target-feature=-aes' cargo test --manifest-path vendor/flock/Cargo.toml --offline -p flock-field --features field64
```

Run the shared target-specific integration suite once at each width:

```sh
cargo test --release --offline -p rmip-flock --test interactive_target
cargo test --release --offline -p rmip-flock --features field64 --test interactive_target
```

The suite checks the 40-bit profile and field sizes, honest AG proof
verification, transcript equality, rejection with other coins or a
modified row/profile, and exact OS-coin replay. The first proof test
uses 256 and 257 compressions to cover a partially filled last shape.
These tests validate execution and binding; they do not replace the
conditional soundness imports.

The matching-boundary benchmark includes witness generation, oracle
copies, interactive sampling and transcript recording. For example, 12, 14
and 16 compression-count bits correspond to 26, 28 and 30 padded
witness bits for the recorded BLAKE3 shape:

```sh
RAYON_NUM_THREADS=1 RMIP_BLAKE3_LOG2S=12,14,16 cargo test --release --offline -p rmip-flock --features field64 --test interactive_target -- --ignored --nocapture --exact interactive_target40_cost
```

Omit `--features field64` to measure the 128/256-bit reference at the
same 40-bit target. The emitted report identifies both field widths,
the per-level query counts, the conditional native error, the proving
and verification times, and proof/oracle storage. No timing is implied
by the field-operation count alone.

The field64 kernel suites pass 13 NTT tests (including direct polynomial
evaluation against the encoder), 22 multilinear tests (including both
pending-fold lookahead cases), and 19 lincheck tests (including an
independent Boolean-to-stripe reference). Witness tests exercise exact
group-store widths and untouched allocation/interior canaries, then
compare the batch-major BLAKE3 witness against the Boolean generator,
scalar packed generator and sparse matrix products at 8, 9 and 17
compressions. The latter two sizes check partially filled groups and
zero dummy instances; the emitted lincheck stripes are independently
checked against the Boolean witness. These two witness tests pass at
both field widths. The group staging remains eight
instances, while each stored field column uses 64 bytes in field64 mode
and 128 bytes in the default mode.

The shared 64-byte bit transpose is field-independent and retains its
original NEON implementation at both widths. The 64-bit NTT also keeps
fusing its final layers: the default field's three-layer reservation is
for a special half-width-twiddle kernel that the 64-bit field does not
use. These preserve the underlying computations; they do not alter any
protocol parameter or error term.

On AArch64 with PMULL (`target_feature="aes"`), the field64 path now has
two independent GF(2^64) lanes per NEON register. The vector helper keeps
carryless products and polynomial reduction in vector registers. It is
used by adjacent pair folds, one/two/three-layer NTT butterflies, AG
lookup folds and the pending-fold lookahead kernels. Fully live NTT
blocks flatten their contiguous row slices, and small recursive NTTs
use the fused kernels as well. Odd tails and other architectures retain
the portable implementation; original-field dispatch is unchanged.

The immediate ARM field products now use leanVM-derived PMULL reductions:
two further products by `0x1b` fold the high polynomial and its overflow,
without extracting the intermediate limbs into integer registers. The
paired kernel reduces each raw product before joining the result lanes.
Deferred dot-product accumulators keep their existing lane layout and
shift/XOR `reduce2`; they are not reduced after each multiplication.
SHA3-gated EOR3 is available in the immediate reducer, although the compiler
also recognizes EOR3 in the existing shift/XOR expressions. The inversion
chain uses 63 squarings and eight other products; native inversion counters
therefore charge 71 products, with the original zero and square conventions.

The field64 encoder also computes the seven radix-eight twiddles jointly,
fuses deep passes in groups of three layers, and offers an eight-lane ARM
row kernel with paired/scalar tails. These retain the original shifted
basis and Flock cache driver. Diagnostic choices are
`FLOCK_FIELD64_NTT_ROW_LANES=2|8` (default 8, fixed on first use),
`FLOCK_FIELD64_NTT_DEEP_RADIX=2|3` (default 3),
`FLOCK_FIELD64_NTT_JOINT_TWIDDLES=0|1` (default 1), and
`FLOCK_FIELD64_NTT_SUBGROUP_LOG_BYTES=21|22|23|24` (default 21).
The last three are read at transform entry. They change execution strategy,
not the encoded word or protocol parameters. Upstream source correspondence,
the pinned revision, both licenses and the preserved binius64 attribution
are in [third-party/leanvm](../../third-party/leanvm/README.md).

The added tests compare against independent software polynomial arithmetic
and scalar kernels, including the full 128-bit unreduced input range,
unaligned offsets, destination guards, odd lengths, partial chunks, dead
NTT lanes, suffix transforms and one/two-thread execution. The
measurements are recorded in the development repository. These optimizations do not
change the 40-bit target or remove any protocol step.
