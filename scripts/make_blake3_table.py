#!/usr/bin/env python3
"""Table 2 of the paper from a directory of BLAKE3 logs.

    python3 scripts/make_blake3_table.py results/blake3 [--latex] [--json OUT.json]

The directory holds one folder per witness size, `m<bits>/`, with:
- `flock-upstream-m<bits>.log`: Flock's own benchmark (`crates/flock-prover/benches/blake3_proof.rs`) run from an
  untouched checkout of upstream Flock, as it ships: the Fast profile at 128-bit security (one warm-up, three
  timed proofs, one verification, the proof size);
- `reset64-rewind-m<bits>.log`: our compilation of the same IOP over GF(2^64) as a single rewinding prover
  (`reset_cost`), one warm-up and three measured executions.
`scripts/reproduce_blake3.sh` writes this layout. Table 2 reports prover time only; the text reports our
compilation's error. The logs also record the verifiers' times, Flock's proof size and our messages, which
the paper does not use.

Our compilation has one prover, rewound before each query. When it posts each proof string it fixes its own
snapshot of it from the coefficients it posts; after its messages it answers every query from those snapshots.
The fixture times its messages with the snapshots built inside (`main_ms`) and its answers from them
(`checker_ms`); the prover's time is their sum per measured execution. Both provers' times are medians of three. Without --latex it prints the prover comparison as
Markdown; with --latex, Table 2's rows.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

SIZES = [22, 24, 26, 28, 30, 32, 34]
RESET_KEYS = ("prover_ms", "native_error_bits")
TIME = r"([0-9.]+) (ms|s)\b"


def to_ms(value, unit):
    return float(value) * (1000 if unit == "s" else 1)


def reset_runs(path, m):
    text = path.read_text(errors="replace")
    head = next((l[l.index("reset_cost_v2 "):] for l in text.splitlines() if "reset_cost_v2 " in l), "")
    if not all(f in head.split() for f in ("field=64", "fold_field=128", "native_target=40", "repetitions=1", "warmup=1",
                                            "measured=3", "rewinding_single_prover=true", f"witness_bits=2^{m}")):
        raise SystemExit(f"{path}: not the GF(2^64) single rewinding prover at the 40-bit target, one execution, "
                         f"one warm-up and three measured, at 2^{m} witness bits")
    if "test result: ok. 1 passed" not in text:
        raise SystemExit(f"{path}: the fixture did not pass (every measured proof must verify)")
    out = []
    for line in text.splitlines():
        if line.startswith("reset run="):
            r = {k: float(v) for k, v in (t.split("=", 1) for t in line.split()[1:] if "=" in t)
                 if re.fullmatch(r"-?[0-9.]+(e-?[0-9]+)?", v)}
            r["prover_ms"] = r["main_ms"] + r["checker_ms"]
            out.append(r)
    if [r.get("run") for r in out] != [1, 2, 3]:
        raise SystemExit(f"{path}: measured executions {[r.get('run') for r in out]}, expected runs 1, 2 and 3")
    if min(r["native_error_bits"] for r in out) < 40:
        raise SystemExit(f"{path}: the compilation's error is above 2^-40, not the 40-bit setting")
    return out


def reset_medians(path, m):
    rs = reset_runs(path, m)
    return {k: statistics.median(r[k] for r in rs) for k in RESET_KEYS if all(k in r for r in rs)}


def flock_bench(path, m):
    text = path.read_text(errors="replace")
    size = re.search(r"===\s+(\d+) compressions\s+\(m = (\d+),", text)
    if not size or (int(size.group(1)), int(size.group(2))) != (1 << (m - 14), m):
        raise SystemExit(f"{path}: not Flock's benchmark at 2^{m} witness bits ({1 << (m - 14)} compressions)")
    if "FLOCK_NO_GRIND" in text:
        raise SystemExit(f"{path}: run with FLOCK_NO_GRIND; Flock must run as it ships")
    for setting in ("proof timings (prove_fast)", "merkle hash: blake3   fs hash: blake3", "zerocheck: ag"):
        if setting not in text:
            raise SystemExit(f"{path}: not Flock's default configuration ({setting!r} missing)")
    if re.findall(r"\[run (\d)/3\]", text) != ["1", "2", "3"]:
        raise SystemExit(f"{path}: timed proofs are not runs 1, 2 and 3")
    runs = [to_ms(v, u) for v, u in re.findall(r"\[run \d/3\] prove_fast:\s+" + TIME, text)]
    if len(runs) != 3:
        raise SystemExit(f"{path}: {len(runs)} timed proofs, expected 3")
    if not re.search(r"verify:\s+" + TIME, text):
        raise SystemExit(f"{path}: Flock's benchmark did not verify its proof")
    return dict(prover_ms=statistics.median(runs), prover_runs_ms=runs)


def load(root):
    data = {}
    for m in SIZES:
        folder = root / f"m{m}"
        flock, ours = folder / f"flock-upstream-m{m}.log", folder / f"reset64-rewind-m{m}.log"
        if not (flock.exists() and ours.exists()):
            continue
        data[m] = dict(bits=m, flock=flock_bench(flock, m), ours=reset_medians(ours, m))
    return data


def tms(v):
    if v >= 1000:
        return f"{v / 1000:.2f}\\,s" if v < 10000 else f"{v / 1000:.1f}\\,s"
    return f"{v:.1f}\\,ms" if v >= 10 else f"{v:.2f}\\,ms"


def tbytes(b):
    if b < 1 << 20:
        return f"{b / 1024:.0f} KiB"
    return f"{b / (1 << 20):.1f} MiB"


def latex(data):
    lines = ["% bits & Flock's prover (untouched upstream, GF(2^128), 128-bit Fast profile) & our prover (GF(2^64), "
             "40-bit interactive, one rewinding prover) & Flock / ours"]
    for m, r in sorted(data.items()):
        f, o = r["flock"]["prover_ms"], r["ours"]["prover_ms"]
        lines.append(f"    $2^{{{m}}}$ & {tms(f)} & {tms(o)} & {f / o:.2f}$\\times$ \\\\")
    lines.append("% per size: error of our compilation (bits)")
    for m, r in sorted(data.items()):
        lines.append(f"%   2^{m}: 2^-{r['ours']['native_error_bits']:.2f}")
    return "\n".join(lines)


def markdown(data):
    lines = ["Table 2 compares prover time: Flock, untouched upstream with its own benchmark as it ships (GF(2^128), Fast "
             "profile at 128-bit security), against our compilation of the same IOP over GF(2^64) at our 40-bit interactive "
             "setting, one rewinding prover whose time is its messages (snapshots fixed at each posting included) plus its "
             "answers from them, per run; medians of three.", "",
             "| bits | compressions | Flock prover (ms) | our prover (ms) | Flock / ours |",
             "|---:|---:|---:|---:|---:|"]
    for m, r in sorted(data.items()):
        f, o = r["flock"]["prover_ms"], r["ours"]["prover_ms"]
        lines.append(f"| 2^{m} | {1 << (m - 14):,} | {f:,.1f} | {o:,.1f} | {f / o:.2f}x |")
    lines += ["", "| bits | our compilation's error |", "|---:|---|"]
    for m, r in sorted(data.items()):
        lines.append(f"| 2^{m} | 2^-{r['ours']['native_error_bits']:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    data = load(a.results)
    if not data:
        raise SystemExit(f"no complete size under {a.results}")
    print(latex(data) if a.latex else markdown(data))
    if a.json:
        a.json.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
