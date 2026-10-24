#!/usr/bin/env python3
"""Check a regenerated MNIST bundle against the one the published runs used.

    python3 scripts/check_bundle.py R NEW_DIR [--install]

NEW_DIR is the output directory of `tools/mnist/survey_sparse.py --views R`, or of its `--merge` for the
three largest sizes (selected.json, selected.csv, config.json, manifest.json). The reference is
benchmarks/mnist-test/artifacts/R<R>/ (manifest.json, config.json, selected.csv). Standard library only;
selected.json is streamed, so the 350 GB bundle at 2^26 views needs no memory to speak of.

A bundle records the SHA-256 of the survey tools that wrote it, and three revisions of survey_sparse.py
wrote the published ones: 0233721f... the bundles up to 65,536 views, 544d442e... those at 262,144 and
1,048,576, and ea914773..., the revision shipped here, those at 2^22, 2^24 and 2^26. So the test is:
  1. selected.csv is byte-identical to the recorded one (every vote of every image);
  2. selected.json hashes to the manifest's SHA-256 once the recorded tool hashes are put back in place
     of the new ones (for the three largest sizes nothing is put back: the raw hash must match);
  3. config.json equals the recorded configuration apart from those tool hashes and the torch thread
     count, which does not enter selected.json.
With --install, and only if all three hold, the restored selected.json (byte for byte the published
bundle) is written to benchmarks/mnist-test/artifacts/R<R>/selected.json, where mnist_survey and the
direct-evaluation tool read it, and its hash is checked again. Exit status 0 if the bundle reproduces.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 1 << 24


def restored_stream(path, swaps):
    """Yield the bytes of `path` with the first occurrence of each old pattern
    replaced by its new one (patterns of equal length, as SHA-256 hex strings are)."""
    pending = [(a, b) for a, b in swaps if a != b]
    keep = max((len(a) for a, _ in pending), default=1) - 1
    tail = b""
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            buf = tail + block
            for pair in list(pending):
                i = buf.find(pair[0])
                if i >= 0:
                    buf = buf[:i] + pair[1] + buf[i + len(pair[0]):]
                    pending.remove(pair)
            cut = max(len(buf) - keep, 0)
            yield buf[:cut]
            tail = buf[cut:]
    yield tail


def rows(path):
    with open(path, newline="") as stream:
        return {row["image_id"]: row for row in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("views", type=int, help="R, the number of views (256 ... 67108864)")
    parser.add_argument("new_dir", type=Path, help="the regenerated bundle directory")
    parser.add_argument("--install", action="store_true",
                        help="if it reproduces, write the restored selected.json into benchmarks/mnist-test/artifacts/R<R>/")
    args = parser.parse_args()
    reference = ROOT / "benchmarks/mnist-test/artifacts" / f"R{args.views}"
    recorded = json.loads((reference / "manifest.json").read_text())
    want = recorded["artifacts"]["selected.json"]["sha256"]
    old_tools = recorded["config"]["sources"]["tool_source_sha256"]
    config = json.loads((args.new_dir / "config.json").read_text())
    new_tools = config["sources"]["tool_source_sha256"]
    swaps = [(f'"{tool}": "{value}"'.encode(), f'"{tool}": "{old_tools[tool]}"'.encode())
             for tool, value in sorted(new_tools.items())]
    changed = [tool for tool in sorted(new_tools) if new_tools[tool] != old_tools.get(tool)]
    print(f"R={args.views}: tool hashes that differ from the recorded bundle: {', '.join(changed) or 'none'}")

    csv_ok = (args.new_dir / "selected.csv").read_bytes() == (reference / "selected.csv").read_bytes()
    print(f"  1. selected.csv byte-identical to the recorded one: {csv_ok}")
    if not csv_ok:
        a, b = rows(reference / "selected.csv"), rows(args.new_dir / "selected.csv")
        for image_id, row in a.items():
            other = b.get(image_id)
            if row != other:
                keys = [k for k in row if other is None or row[k] != other.get(k)]
                print(f"     image {image_id} differs in: {', '.join(keys)}")
                if other is not None:
                    print(f"       votes {[row[f'votes_{y}'] for y in range(10)]} -> "
                          f"{[other[f'votes_{y}'] for y in range(10)]}")

    h = hashlib.sha256()
    for chunk in restored_stream(args.new_dir / "selected.json", swaps):
        h.update(chunk)
    json_ok = h.hexdigest() == want
    print(f"  2. selected.json with the recorded tool hashes restored: {h.hexdigest()}")
    print(f"     manifest records                                    : {want}  -> {json_ok}")

    threads = (config.get("threads"), recorded["config"].get("threads"))
    config["sources"]["tool_source_sha256"], config["threads"] = old_tools, recorded["config"].get("threads")
    config_ok = config == recorded["config"]
    print(f"  3. config.json equal apart from tool hashes and threads (new/recorded {threads[0]}/{threads[1]}): "
          f"{config_ok}")

    ok = csv_ok and json_ok and config_ok
    print(f"R={args.views}: the regenerated bundle {'REPRODUCES' if ok else 'DOES NOT reproduce'} the published one")
    if ok and args.install:
        target = reference / "selected.json"
        temporary = target.with_suffix(".json.tmp")
        check = hashlib.sha256()
        with open(temporary, "wb") as out:
            for chunk in restored_stream(args.new_dir / "selected.json", swaps):
                out.write(chunk)
                check.update(chunk)
        if check.hexdigest() != want:
            temporary.unlink()
            print("install: the written bytes do not hash to the manifest; nothing installed")
            return 1
        temporary.replace(target)
        print(f"installed {target.relative_to(ROOT)} (sha256 {want})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
