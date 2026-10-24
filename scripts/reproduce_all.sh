#!/usr/bin/env bash
# Everything Section 7 of the paper reports from the two experiments, in one command:
#
#   bash scripts/reproduce_all.sh [SECTION7.tex]
#
# 1. Table 1 and Figure 1: `scripts/reproduce_mnist.sh`, every circuit size whose input bundle is installed under
#    benchmarks/mnist-test/artifacts/R<views>/selected.json (the 256- and 1,024-view bundles ship with the
#    repository; the README's "Inputs" says how to download or regenerate the other eight).
# 2. Table 2: `scripts/reproduce_blake3.sh`, all seven sizes (aarch64 only; skipped elsewhere, with a message).
# 3. `scripts/check_paper.py` on the rerun: with the paper's Section 7 LaTeX source as the argument, it compares
#    the regenerated table rows with the paper's for every size that ran, and with every size present also
#    Figure 1 and the text figures.
# Results go to results-rerun/, in the layout of results/, which is cleared first so that no stale size is mixed
# in. MNIST_VIEWS="256 1024" or BLAKE3_BITS="22 24" restrict the sizes. The scripts build first, pin to core
# $CORE (default 4) with taskset where it exists, and run one thread.
set -euo pipefail
cd "$(dirname "$0")/.."
PAPER=${1:-}
rm -rf results-rerun
mkdir -p results-rerun
echo "=== MNIST: Table 1 and Figure 1"
bash scripts/reproduce_mnist.sh results-rerun/mnist ${MNIST_VIEWS:-}
echo
echo "=== BLAKE3: Table 2"
case "$(uname -m)" in
  aarch64 | arm64) bash scripts/reproduce_blake3.sh results-rerun/blake3 ${BLAKE3_BITS:-} ;;
  *) echo "skipped: the BLAKE3 measurements need aarch64 (Flock's AG-skip prover); this is $(uname -m)" ;;
esac
echo
echo "=== the rerun against the paper"
status=0
python3 scripts/check_paper.py --results results-rerun ${PAPER:+--paper "$PAPER"} || status=$?
echo
echo "The shipped results' tables: python3 scripts/make_mnist_table.py results/mnist; python3 scripts/make_blake3_table.py results/blake3"
exit $status
