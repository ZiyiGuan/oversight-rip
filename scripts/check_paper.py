#!/usr/bin/env python3
"""Check the paper's Section 7 against the shipped results.

    python3 scripts/check_paper.py [--results results] [--paper SECTION.tex]

Recomputes, from `results/` (or a rerun's directory), every number stated in the text of the two results and
compares it with the paper's statement; each statement is embedded below as the paper words it (Section 7
as revised on 2026-10-27 for the rewinding compilation), and must be updated if the text changes. With `--paper`, it also reads Table 1
(`tab:mnist-cost`, or the earlier `tab:mnist-cost-40`), Table 2 (`tab:blake3`) and Figure 1's coordinates (`fig:mnist-plots`) from the paper's
LaTeX source and compares them, row by row and matched by size, with what `make_mnist_table.py` and
`make_blake3_table.py` print; nothing of the tables is embedded here, so these comparisons are independent of
the scripts. A rerun of some sizes is compared on those sizes; the text figures need every size. Prints each
comparison and, last, how many were made and what was not compared. "all figures match the paper" (exit 0)
needs every table row, Figure 1 and every text figure compared; differences exit 1; a partial comparison
that matches says so and exits 0; nothing compared exits 2. Needs only Python 3.9 or later.
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_blake3_table as blake3  # noqa: E402
import make_mnist_table as mnist  # noqa: E402

def claims(m, b, results):
    """The paper's statements (Section 7 as revised on 2026-10-27 for the rewinding compilation): (where, the statement, the same quantity
    computed from the data)."""
    R = sorted(m)
    big = [r for r in R if r >= 1 << 20]

    def rng(xs, fmt):
        lo, hi = fmt(min(xs)), fmt(max(xs))
        return lo if lo == hi else f"{lo} to {hi}"

    f2 = lambda x: f"{x:.2f}"
    led = {r: json.loads((results / "mnist" / f"R{r}" / "ledger.json").read_text()) for r in R}
    params = [im["protocol_parameters"] for r in R for im in led[r]["images"]]
    v = [m[r]["verifier_ms"] for r in R]
    loop = [m[r]["loop_us"] * 1e-3 for r in R]
    vs_loop = {r: m[r]["verifier_ms"] / (m[r]["loop_us"] * 1e-3) for r in R}
    growth = [v[i + 1] / v[i] for i in range(len(v) - 1)]
    fall = [vs_loop[R[i]] / vs_loop[R[i + 1]] for i in range(len(R) - 1)]
    prover = {r: m[r]["prover_ms"] / (m[r]["loop_us"] * 1e-3) for r in R}
    B = sorted(b)
    o, f = {k: b[k]["ours"] for k in B}, {k: b[k]["flock"] for k in B}
    below, above = [r for r in R if vs_loop[r] >= 1], [r for r in R if vs_loop[r] < 1]
    crossing = f"between 2^{int(math.log2(max(below)))} and 2^{int(math.log2(min(above)))}" if below and above else "never"
    # The classifier's recorded accuracy ships beside the scripts; a results directory elsewhere may carry a copy.
    accuracy_file = next((d / "benchmarks" / "artifacts" / "sparse_judge_accuracy.json" for d in (HERE.parent, results.parent)
                          if (d / "benchmarks" / "artifacts" / "sparse_judge_accuracy.json").exists()), None)
    if accuracy_file is None:
        sys.exit(f"benchmarks/artifacts/sparse_judge_accuracy.json not found beside {HERE.parent} or {results.parent}")
    accuracy = json.loads(accuracy_file.read_text())
    return [
        ("7 classifier's top-1 accuracy on random six-pixel views (%)", "59.4", f"{accuracy['mean_top1_accuracy'] * 100:.1f}"),
        ("7.1 compilation: one prover restored to its posting state before each query, no pins", "rewind / 0",
         f"{'/'.join(sorted({p['compilation'] for p in params}))} / {rng([p['pins'] for p in params], str)}"),
        ("7.1 opened columns", "64", rng([p["opening_columns"] for p in params], str)),
        ("7.1 proximity error below", "2^-42", "2^-42" if max(p["eps_open"] for p in params) < 2 ** -42 else "not below 2^-42"),
        ("7.1 sampling miss (1-eps)^t at most", "2^-41", "2^-41" if max(p["local_miss_bound"] for p in params) <= 2 ** -41 else "above 2^-41"),
        ("7.1 single-execution soundness error at most, every size", "2^-40.53", f"2^-{math.floor(min(m[r]['error_bits'] for r in R) * 100) / 100:.2f}"),
        # t is derived per image from its certified robustness eps = flips/views: the least t with (1-eps)^t <= 2^-41.
        ("7.1 sampled gates: the smallest t with (1-eps)^t <= 2^-41, every image", "smallest",
         "smallest" if all((1 - im["certificate"]["certified_flips"] / im["certificate"]["r"]) ** im["protocol_parameters"]["base_t"] <= 2 ** -41
                           < (1 - im["certificate"]["certified_flips"] / im["certificate"]["r"]) ** (im["protocol_parameters"]["base_t"] - 1)
                           for r in R for im in led[r]["images"]) else "not the smallest"),
        # The 50 images were chosen at 2^8 views among those accepted with certified robustness at least 1/8.
        ("7.2 images chosen at 2^8 views with certified robustness at least", "1/8",
         "1/8" if min(im["certificate"]["certified_flips"] / im["certificate"]["r"] for im in led[256]["images"]) >= 1 / 8 else "below 1/8"),
        ("7.2 images: 50; omitted below 2^24 / 2^26", "1 / 3", f"{50 - m[1 << 24]['images']} / {50 - m[1 << 26]['images']}"),
        ("7.2 median robustness at every size, between", "0.30 and 0.32", rng([m[r]["robustness"] for r in R], f2).replace(" to ", " and ")),
        ("7.2 median verifier oracle queries, between", "74 and 80", rng([m[r]["queries"] for r in R], lambda x: f"{x:.0f}").replace(" to ", " and ")),
        ("7.2 fewer oracle queries than the circuit at 2^26", "840,000", f"{round((1 << 26) / m[1 << 26]['queries'], -4):,.0f}"),
        ("7.2 verifier online time (ms)", "0.45 to 1.9", f"{v[0]:.2f} to {v[-1]:.1f}"),
        ("7.2 direct evaluation", "0.06 us to 16 ms", f"{loop[0] * 1e3:.2f} us to {loop[-1]:.0f} ms"),
        ("7.2 direct evaluation costs about (ns per view)", "0.23", f"{statistics.median(m[r]['loop_us'] * 1e3 / r for r in R):.2f}"),
        ("7.2 verifier and direct evaluation cross", "between 2^22 and 2^24", crossing),
        ("7.2 verifier over direct evaluation at 2^26", "0.12", f"{vs_loop[1 << 26]:.2f}"),
        ("7.2 verifier growth per fourfold step", "1.1 to 1.3", rng(growth, lambda x: f"{x:.1f}")),
        ("7.2 fall of verifier/direct per step", "3.0 to 3.7", rng(fall, lambda x: f"{x:.1f}")),
        ("7.2 prover over direct evaluation from 2^20, about", "60", rng([prover[r] for r in big], lambda x: f"{round(x, -1):.0f}")),
        ("7.2 prover at 2^26 (s)", "0.9", f"{m[1 << 26]['prover_ms'] / 1000:.1f}"),
        ("7.4 prover over direct evaluation at the largest sizes, about", "60", f"{round(prover[1 << 26], -1):.0f}"),
        ("7.3 error of our compilation", "2^-42.4 to 2^-42.9",
         "2^-" + rng([o[k]["native_error_bits"] for k in B], lambda x: f"{x:.1f}").replace(" to ", " to 2^-")),
        ("7.3 our prover faster than Flock's SNARG prover, every size (x)", "1.8 to 2.1", rng([f[k]["prover_ms"] / o[k]["prover_ms"] for k in B], lambda x: f"{x:.1f}")),
        ("7.3 sizes (witness bits / compressions)", "2^22 to 2^34 / 2^8 to 2^20", f"2^{B[0]} to 2^{B[-1]} / 2^{B[0] - 14} to 2^{B[-1] - 14}"),
    ]


# --- the paper's LaTeX source -------------------------------------------------------------------------------
SIZE_CELL = re.compile(r"^(\$2\^\{\d+\}\$|\d[\d{},]*)$")


def tex_table_rows(tex, label):
    at = tex.find(f"\\label{{{label}}}")
    if at < 0:
        return None
    start = tex.rfind("\\begin{tabular}", 0, at)
    end = tex.find("\\end{tabular}", start)
    if start < 0 or end < 0:
        return None
    rows = {}
    for line in tex[start:end].splitlines():
        line = line.strip()
        cells = line.split(" & ")
        if line.endswith("\\\\") and len(cells) > 1 and SIZE_CELL.match(cells[0].strip()):
            rows[cells[0].strip()] = line
    return rows


def tex_figure_coordinates(tex, label):
    at = tex.find(f"\\label{{{label}}}")
    start = tex.rfind("\\begin{figure", 0, at)
    if at < 0 or start < 0:
        return None
    return [" ".join(c.split()) for c in re.findall(r"coordinates\s*\{([^}]*)\}", tex[start:at])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE.parent / "results")
    ap.add_argument("--paper", type=Path, help="the LaTeX source of the paper's Section 7")
    a = ap.parse_args()
    diffs = compared = 0
    skipped = []

    def compare(what, paper, data):
        nonlocal diffs, compared
        ok = paper == data
        diffs += not ok
        compared += 1
        print(f"ok   {what}: {paper}" if ok else f"DIFF {what}: paper {paper!r}, data {data!r}")

    m_rows = [r for r in (mnist.row(a.results / "mnist", R) for R in mnist.SIZES) if r]
    b_rows = blake3.load(a.results / "blake3") if (a.results / "blake3").is_dir() else {}
    if not (m_rows or b_rows):
        sys.exit(f"no results under {a.results}")
    data1 = {l.strip().split(" & ", 1)[0]: l.strip() for l in mnist.table_rows(m_rows)}
    data2 = {l.strip().split(" & ", 1)[0]: l.strip() for l in blake3.latex(b_rows).splitlines() if l.startswith("    ")} if b_rows else {}

    if a.paper:
        tex = a.paper.read_text()
        for name, labels, data in (("Table 1", ("tab:mnist-cost", "tab:mnist-cost-40"), data1), ("Table 2", ("tab:blake3",), data2)):
            paper = next((rows for rows in (tex_table_rows(tex, label) for label in labels) if rows is not None), None)
            if paper is None:
                compare(f"{name} ({' or '.join(labels)}) in {a.paper.name}", "present", "not found")
                continue
            for size, row in data.items():
                compare(f"{name} row {size}", paper.get(size), row)
        coords = tex_figure_coordinates(tex, "fig:mnist-plots")
        if len(data1) < len(mnist.SIZES) or len(data2) < len(blake3.SIZES):
            skipped.append(f"the table rows of the sizes not in {a.results}")
        if len(m_rows) != len(mnist.SIZES) or coords is None:
            skipped.append("Figure 1 (it needs every MNIST size and the figure in the source)")
        else:
            fig = [" ".join(f"({r['R']},{r['verifier_ms']:.3g})" for r in m_rows),
                   " ".join(f"({r['R']},{r['loop_us'] * 1e-3:.3g})" for r in m_rows),
                   " ".join(f"({r['R']},{r['queries']:.0f})" for r in m_rows)]
            fig.append(" ".join(f"({r['R']},{r['R']})" for r in m_rows))
            for i, name in enumerate(("verifier", "direct", "queries", "circuit queries")):
                compare(f"Figure 1 {name} coordinates", coords[i] if i < len(coords) else None, fig[i])
    else:
        skipped.append("Tables 1 and 2 and Figure 1 (pass --paper with the LaTeX source of Section 7)")

    if len(m_rows) == len(mnist.SIZES) and len(b_rows) == len(blake3.SIZES):
        for what, paper, data in claims({r["R"]: r for r in m_rows}, b_rows, a.results):
            compare(what, paper, data)
    else:
        skipped.append(f"the text figures ({len(m_rows)} of {len(mnist.SIZES)} MNIST sizes and {len(b_rows)} of "
                       f"{len(blake3.SIZES)} BLAKE3 sizes present; they need every size)")
    for what in skipped:
        print(f"not compared: {what}")
    if compared == 0:
        print("nothing compared with the paper")
        sys.exit(2)
    if diffs:
        print(f"{diffs} of {compared} compared figures differ from the paper")
        sys.exit(1)
    print("all figures match the paper" if not skipped else f"{compared} compared figures match the paper; not every figure was compared")
    sys.exit(0)


if __name__ == "__main__":
    main()
