#!/usr/bin/env bash
# Table 2: Flock, from an untouched checkout of upstream Flock with its own benchmark, and our compilation of the
# same IOP over GF(2^64), at each witness size; then Table 2's rows and every measured quantity.
#
#   bash scripts/reproduce_blake3.sh [OUT] [BITS ...]
#
# OUT defaults to results-rerun/blake3; BITS to 22 24 26 28 30 32 34 (2^BITS witness bits, 2^(BITS-14) BLAKE3
# compressions; even values only). aarch64 only: Flock's AG-skip prover exists for aarch64 only. Flock is cloned
# from https://github.com/succinctlabs/flock at FLOCK_REV into $FLOCK_SRC (default .flock-upstream), checked to
# have no modified file, and built and run with its own configuration and benchmark, as it ships. Every run is
# pinned to core $CORE (default 4) with taskset where it exists, one thread. OUT gets the layout of results/blake3.
set -euo pipefail
cd "$(dirname "$0")/.."
case "$(uname -m)" in
  aarch64 | arm64) ;;
  *) echo "the BLAKE3 measurements need aarch64 (Flock's AG-skip prover); this is $(uname -m)" >&2; exit 1 ;;
esac
OUT=${1:-results-rerun/blake3}
[ $# -gt 0 ] && shift
BITS=${*:-22 24 26 28 30 32 34}
FLOCK_REV=${FLOCK_REV:-b684b1258e4b1f202bec24afd660ace851b09e5e}
FLOCK_SRC=${FLOCK_SRC:-.flock-upstream}
export RAYON_NUM_THREADS=1
# Neither prover reads tuning or configuration from the environment here: every FLOCK_*, BLAKE3_* and RMIP_*
# variable of the caller is dropped, and only the ones below are set.
CLEAN="env $(env | { grep -oE '^(FLOCK|BLAKE3|RMIP)_[A-Z0-9_]+' || true; } | sed 's/^/-u /' | tr '\n' ' ') -u RUSTFLAGS"
PIN=""
command -v taskset > /dev/null && PIN="taskset -c ${CORE:-4}"
OURS_FLAGS="${RUSTFLAGS:--C target-cpu=native}"
T="cargo test --release --locked -q -p rmip-flock"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

echo "== upstream Flock at $FLOCK_REV in $FLOCK_SRC"
[ -d "$FLOCK_SRC/.git" ] || git clone -q https://github.com/succinctlabs/flock "$FLOCK_SRC"
git -C "$FLOCK_SRC" fetch -q origin "$FLOCK_REV" 2> /dev/null || true
git -C "$FLOCK_SRC" checkout -q "$FLOCK_REV"
[ "$(git -C "$FLOCK_SRC" rev-parse HEAD)" = "$FLOCK_REV" ] || { echo "Flock is not at $FLOCK_REV" >&2; exit 1; }
git -C "$FLOCK_SRC" diff --quiet HEAD || { echo "Flock's checkout has modified files; it must be untouched" >&2; exit 1; }
git -C "$FLOCK_SRC" rev-parse HEAD > "$OUT/flock-revision.txt"
(cd "$FLOCK_SRC" && $CLEAN cargo bench --locked --bench blake3_proof --no-run)

echo "== our compilation over GF(2^64): building before any timed run"
RUSTFLAGS="$OURS_FLAGS" $T --test reset --no-run

for m in $BITS; do
  mkdir -p "$OUT/m$m"
  lb=$((m - 14))
  echo "== 2^$m bits: Flock"
  (cd "$FLOCK_SRC" && $CLEAN BLAKE3_LOG2S=$lb BLAKE3_RUNS=3 RAYON_NUM_THREADS=1 $PIN cargo bench --locked --bench blake3_proof) \
    > "$OUT/m$m/flock-upstream-m$m.log" 2>&1 || { echo "Flock failed, see $OUT/m$m/flock-upstream-m$m.log" >&2; exit 1; }
  echo "== 2^$m bits: ours over GF(2^64)"
  log="$OUT/m$m/reset64-rewind-m$m.log"
  $CLEAN RUSTFLAGS="$OURS_FLAGS" RMIP_MIP_LOG_BLOCKS=$lb RAYON_NUM_THREADS=1 $PIN \
    $T --test reset reset_cost -- --ignored --exact --nocapture --test-threads=1 > "$log" 2>&1 || { echo "failed, see $log" >&2; exit 1; }
done
echo "== Table 2 from $OUT (the paper's: python3 scripts/make_blake3_table.py results/blake3 --latex)"
python3 scripts/make_blake3_table.py "$OUT" --latex
python3 scripts/make_blake3_table.py "$OUT"
