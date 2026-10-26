#!/usr/bin/env bash
# Table 1 and Figure 1: run the protocol and direct evaluation at each circuit size, then print Table 1's
# rows, Figure 1's coordinates and every per-size median.
#
#   bash scripts/reproduce_mnist.sh [OUT] [VIEWS ...]
#
# OUT defaults to results-rerun/mnist. VIEWS defaults to every size whose input bundle is installed
# (benchmarks/mnist-test/artifacts/R<views>/selected.json; the 256- and 1,024-view bundles ship with the
# repository, the README says how to install the others). Each size runs every image of its bundle, pinned to
# core $CORE (default 4) with taskset where it exists, one thread. OUT gets the layout of results/mnist:
# R<views>/ledger.json, direct-eval-R<views>.json, and each run's log.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results-rerun/mnist}
[ $# -gt 0 ] && shift
A=benchmarks/mnist-test/artifacts
if [ $# -gt 0 ]; then
  SIZES="$*"
else
  SIZES=$(for f in "$A"/R*/selected.json; do [ -f "$f" ] && basename "$(dirname "$f")" | cut -c2-; done | sort -n)
fi
[ -n "$SIZES" ] || { echo "no installed bundle under $A (see the README's MNIST inputs)" >&2; exit 1; }
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" RAYON_NUM_THREADS=1
PIN=""
command -v taskset > /dev/null && PIN="taskset -c ${CORE:-4}"
field() { python3 -c "import json, sys; m = json.load(open(sys.argv[1])); print(m$2)" "$1"; }

echo "== building"
cargo build --release --locked -q
cargo build --release --locked -q --manifest-path tools/mnist-direct-eval/Cargo.toml
mkdir -p "$OUT"
for R in $SIZES; do
  B=$A/R$R/selected.json
  [ -f "$B" ] || { echo "R=$R: bundle $B not installed (see the README's MNIST inputs)" >&2; exit 1; }
  SHA=$(field "$A/R$R/manifest.json" "['artifacts']['selected.json']['sha256']")
  N=$(field "$A/R$R/manifest.json" "['summary']['images']")
  echo "== $R views, $N images: protocol, then direct evaluation"
  $PIN target/release/mnist_survey --mode protocol --backend ladder --body packed64 --soundness single-shot-40 \
    --depth 0 --compilation rewind --images "$N" --selection "$B" --expected-selection-sha256 "$SHA" --output-dir "$OUT/R$R" \
    > "$OUT/survey-R$R.log" 2>&1 || { echo "mnist_survey failed at R=$R, see $OUT/survey-R$R.log" >&2; exit 1; }
  $PIN tools/mnist-direct-eval/target/release/mnist-direct-eval --images "$N" --probes 9 --selection "$B" \
    --expected-selection-sha256 "$SHA" --output "$OUT/direct-eval-R$R.json" \
    > "$OUT/direct-eval-R$R.log" 2>&1 || { echo "mnist-direct-eval failed at R=$R, see $OUT/direct-eval-R$R.log" >&2; exit 1; }
done
echo "== Table 1 and Figure 1 from $OUT (the paper's: python3 scripts/make_mnist_table.py results/mnist)"
python3 scripts/make_mnist_table.py "$OUT"
