#!/usr/bin/env python3
"""Table 1 and Figure 1 of the paper from a directory of MNIST results.

    python3 scripts/make_mnist_table.py results/mnist [--json OUT.json]

The directory holds, per circuit size R, `R<R>/ledger.json` (written by `mnist_survey --output-dir`) and
`direct-eval-R<R>.json` (written by `mnist-direct-eval --output`); `scripts/reproduce_mnist.sh` writes this
layout. The direct-evaluation baseline is `tight_in_order_best_us`: an in-order scan of the classifier's
answers that counts the accepted label's votes and compares the count with the threshold, the best of nine
batched probes. `mnist-direct-eval` also times the same loop reading the answers through an index list
(`tight_best_us`, 2.5 to 2.7 times slower, the baseline of the paper's first draft) and the gate-by-gate
interpreter (`best_us`); both stay in the JSON and the summary lines. Sizes without both files are skipped. Prints Table 1 as LaTeX, the coordinates of Figure 1's two
plots, and one line per size with the medians the table is computed from. Every value is a median over the
size's images.

The protocol has one prover, which the verifier restores to its state at the posting before each query (the
paper's IOP-to-RIP compilation, `mnist_survey --compilation rewind`): the prover's time is its messages and
its answers, read from the table it posted (`main_prover`); there is no second role and no audit, and the
script refuses a ledger measured otherwise.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

# The shipped manifests: beside the scripts in the release (scripts/../benchmarks), two levels up in the
# development tree (release/scripts/../../benchmarks), or wherever --artifacts says. Every size's manifest ships,
# and a ledger is refused unless it matches the manifest of its size.
def artifacts_dir(explicit=None):
    here = Path(__file__).resolve().parent
    usable = lambda d: (d / "R256" / "manifest.json").exists()
    if explicit is not None:
        # An explicit directory is authoritative: an unusable one is an error, not a fallback.
        if not usable(explicit):
            raise SystemExit(f"--artifacts {explicit}: no R256/manifest.json there")
        return explicit
    for d in (here.parent / "benchmarks" / "mnist-test" / "artifacts",
              here.parent.parent / "benchmarks" / "mnist-test" / "artifacts"):
        if usable(d):
            return d
    raise SystemExit("the shipped manifests (benchmarks/mnist-test/artifacts/R<views>/manifest.json) were not found; pass --artifacts DIR")


ARTIFACTS = None

SIZES = [256, 1024, 4096, 16384, 65536, 262144, 1048576, 1 << 22, 1 << 24, 1 << 26]


def validate(R, j, d):
    """Refuse a ledger and a direct evaluation that are not one complete, matched run of the paper's configuration."""
    def need(ok, what):
        if not ok:
            raise SystemExit(f"R={R}: {what}")
    ims, dims = j["images"], d["images"]
    need(j.get("status") == "completed", f"ledger status is {j.get('status')!r}, not 'completed'")
    need(j.get("views") == R, f"the ledger is for {j.get('views')} views")
    need(j.get("requested_images") == j.get("completed_images") == j.get("accepted_images") == len(ims) > 0,
         "not every requested image completed and was accepted")
    need((j.get("mode"), j.get("body"), j.get("soundness"), j.get("compilation")) == ("protocol", "packed-64", "single-shot-40", "rewind"),
         f"configuration {j.get('mode')}/{j.get('body')}/{j.get('soundness')}/{j.get('compilation')}, not protocol/packed-64/single-shot-40/rewind")
    for im in ims:
        e, p = im["execution"], im["protocol_parameters"]
        need(im["status"] == e["status"] == "accepted", f"image {im['image_id']} was not accepted")
        need(p.get("compilation") == e.get("compilation") == "rewind" and p.get("audit") == e.get("audit_mode") == "rewind",
             f"image {im['image_id']} was not run under the rewinding compilation")
        need(p.get("pins") == 0 and p.get("eps_col") == 0 and "checker_prover_including_preparation" not in e["party_measurements"],
             f"image {im['image_id']} has pins or a checker, the two-prover compilation's")
        need(e["configured_repetitions"] == e["executed_repetitions"] == p["amplification_repetitions"] == 1,
             f"image {im['image_id']} ran more than one execution")
        need(p.get("ladder", {}).get("depth") == 0, f"image {im['image_id']} is not at ladder depth 0")
        need(not e["oracle_lookups_recorded"],
             f"image {im['image_id']} was measured with --record-lookups, which inflates the main prover")
    need(d.get("status") == "completed" and d.get("views") == R, "the direct evaluation is incomplete or for another size")
    need(d.get("requested_images") == d.get("completed_images") == len(dims), "the direct evaluation did not complete every image")
    need(d.get("selection_sha256") == j.get("selection_sha256"), "the protocol and the direct evaluation read different bundles")
    need([i["image_id"] for i in dims] == [im["image_id"] for im in ims], "the protocol and the direct evaluation cover different images")
    manifest = (ARTIFACTS or artifacts_dir()) / f"R{R}" / "manifest.json"
    need(manifest.exists(), f"no shipped manifest at {manifest}")
    m = json.loads(manifest.read_text())
    need(j["selection_sha256"] == m["artifacts"]["selected.json"]["sha256"], "the bundle differs from the shipped manifest's")
    need(len(ims) == m["summary"]["images"], f"{len(ims)} images, the manifest has {m['summary']['images']}")


def row(results, R):
    ledger, direct = results / f"R{R}" / "ledger.json", results / f"direct-eval-R{R}.json"
    if not (ledger.exists() and direct.exists()):
        return None
    j = json.loads(ledger.read_text())
    validate(R, j, json.loads(direct.read_text()))
    ims = j["images"]
    med = lambda f: statistics.median(f(im) for im in ims)
    ms = lambda im, name: im["execution"]["party_measurements"][name]["runtime_ms"]
    party = lambda name: med(lambda im: ms(im, name))
    d = json.loads(direct.read_text())["images"]
    return dict(
        R=R, images=len(ims),
        queries=med(lambda im: im["execution"]["actual_total_Q_O"]),
        prover_ms=party("main_prover"),
        verifier_ms=party("verifier"),
        robustness=med(lambda im: im["certificate"]["certified_flips"] / R),
        error_bits=min(-math.log2(im["protocol_parameters"]["s_one_execution"]) for im in ims),
        loop_us=statistics.median(i["tight_in_order_best_us"] for i in d),
    )


def fmt_time(us):
    if us < 1:
        return f"{us:.3f}\\,$\\mu$s"
    if us < 10:
        return f"{us:.2f}\\,$\\mu$s"
    if us < 1000:
        return f"{us:.1f}\\,$\\mu$s"
    return f"{us / 1000:.2f}\\,ms"


def fmt_ratio(x):
    if x >= 1000:
        return f"{x:,.0f}$\\times$".replace(",", "{,}")
    if x >= 100:
        return f"{x:.0f}$\\times$"
    if x >= 10:
        return f"{x:.1f}$\\times$"
    return f"{x:.2f}$\\times$"


def fmt_views(R):
    return f"$2^{{{R.bit_length() - 1}}}$"


def table_rows(rows):
    out = []
    for r in rows:
        b = r["loop_us"] * 1e-3
        out.append(f"    {fmt_views(r['R'])} & {r['queries']:.0f} & {fmt_time(r['loop_us'])} & {fmt_ratio(r['verifier_ms'] / b)} & "
                   f"{fmt_ratio(r['prover_ms'] / b)} \\\\")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--artifacts", type=Path, help="the directory holding R<views>/manifest.json (found beside the scripts or in the development tree otherwise)")
    a = ap.parse_args()
    global ARTIFACTS
    ARTIFACTS = artifacts_dir(a.artifacts)
    rows = [r for r in (row(a.results, R) for R in SIZES) if r]
    if not rows:
        raise SystemExit(f"no complete size under {a.results}")
    print("% Table 1 rows: views & verifier oracle queries & direct evaluation & verifier online, prover as multiples of it")
    print("\n".join(table_rows(rows)))
    print("\n% Figure 1, left: verifier online time and direct evaluation (ms); right: verifier oracle queries")
    print("% verifier:  " + " ".join(f"({r['R']},{r['verifier_ms']:.3g})" for r in rows))
    print("% direct:    " + " ".join(f"({r['R']},{r['loop_us'] * 1e-3:.3g})" for r in rows))
    print("% queries:   " + " ".join(f"({r['R']},{r['queries']:.0f})" for r in rows))
    print("\n% per size: images, medians (ms), ratios to direct evaluation, robustness, one-shot error")
    for r in rows:
        b = r["loop_us"] * 1e-3
        print(f"% R={r['R']:>9} images={r['images']} prover={r['prover_ms']:.3f} V={r['verifier_ms']:.3f} direct={b:.6f} "
              f"prover/direct={r['prover_ms'] / b:.2f} V/direct={r['verifier_ms'] / b:.3f} "
              f"robustness={r['robustness']:.4f} error=2^-{r['error_bits']:.2f}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
