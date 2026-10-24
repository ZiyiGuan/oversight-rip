#!/usr/bin/env python3
"""Generate one large MNIST bundle (2^22 views and above) for a subset of the images on one machine.

    python3 scripts/generate_large_bundle.py --views 4194304 --image-ids 0,1,2 [--workers N]

Runs the three stages of tools/mnist/survey_sparse.py: the registered views of each image ("masks"),
then the classifier on every view, split into ranges evaluated by a pool of single-threaded workers
("range"), then one shard, a one-image bundle, per image ("assemble"). Resumable: finished masks, ranges
and shards are kept. Images with fewer distinct six-pixel views than the size are skipped. When every
machine is done, merge the shards with the command this script prints; the merged bundle is
byte-identical to a single-process run and its hash must equal the size's manifest.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "mnist" / "survey_sparse.py"


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)


def run(cmd, logfile):
    with open(logfile, "ab") as out:
        return subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=REPO).returncode


def available_gb():
    try:
        line = next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
        return int(line.split()[1]) / 1e6
    except (OSError, StopIteration):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--views", type=int, required=True)
    p.add_argument("--image-ids", type=str, required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--chunk", type=int, default=1 << 18, help="views per range task")
    p.add_argument("--mask-parallel", type=int, default=12)
    p.add_argument("--assemble-parallel", type=int, default=6)
    p.add_argument("--work-dir", type=Path, help="default: benchmarks/mnist-test/generated/R<views>-split")
    a = p.parse_args()
    r, ids = a.views, [int(x) for x in a.image_ids.split(",")]
    work = a.work_dir or REPO / "benchmarks" / "mnist-test" / "generated" / f"R{r}-split"
    masks, parts, shards, logs = work / "masks", work / "parts", work / "shards", work / "logs"
    for d in (masks, parts, shards, logs):
        d.mkdir(parents=True, exist_ok=True)
    py, t_all = sys.executable, time.time()

    # Masks. At 2^26 views one masks process holds about 25 GB, so the parallelism is capped by memory.
    todo = [i for i in ids if not (masks / f"masks-R{r}-{i:05d}.json").exists()]
    gb = available_gb()
    mask_parallel = max(1, min(a.mask_parallel, int((gb - 20) // 30))) if (gb and r >= 1 << 26) else a.mask_parallel
    log(f"R={r}: masks for {len(todo)} images, {mask_parallel} at a time")
    skipped = []
    with ThreadPoolExecutor(mask_parallel) as ex:
        futs = {ex.submit(run, [py, str(TOOL), "--views", str(r), "--image-ids", str(i), "--threads", "1",
                                "--stage", "masks", "--masks-dir", str(masks)], logs / f"masks-{i:05d}.log"): i for i in todo}
        for f in as_completed(futs):
            if f.result() != 0:
                i = futs[f]
                if "insufficient distinct masks" in (logs / f"masks-{i:05d}.log").read_text(errors="replace"):
                    log(f"image {i} skipped: fewer than {r} distinct six-pixel views")
                    skipped.append(i)
                    continue
                sys.exit(f"masks failed for image {i}; see {logs}/masks-{i:05d}.log")
    ids = [i for i in ids if i not in skipped]

    # Ranges.
    tasks = [(i, lo, min(r, lo + a.chunk)) for i in ids for lo in range(0, r, a.chunk)
             if not (parts / f"part-R{r}-{i:05d}-{lo:012d}-{min(r, lo + a.chunk):012d}.json").exists()]
    log(f"R={r}: {len(tasks)} range tasks of {a.chunk} views on {a.workers} workers")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run, [py, str(TOOL), "--views", str(r), "--image-ids", str(i), "--threads", "1", "--stage", "range",
                                "--range", f"{lo}:{hi}", "--masks-dir", str(masks), "--parts-dir", str(parts)],
                          logs / f"range-{i:05d}.log"): (i, lo, hi) for (i, lo, hi) in tasks}
        for f in as_completed(futs):
            done += 1
            if f.result() != 0:
                sys.exit(f"range {futs[f]} failed; see {logs}")
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                log(f"ranges {done}/{len(tasks)} in {el:.0f} s, eta {el / done * (len(tasks) - done):.0f} s")

    # Shards.
    todo = [i for i in ids if not (shards / f"{i:05d}" / "manifest.json").exists()]
    log(f"R={r}: assembling {len(todo)} shards")
    with ThreadPoolExecutor(a.assemble_parallel) as ex:
        futs = {ex.submit(run, [py, str(TOOL), "--views", str(r), "--image-ids", str(i), "--threads", "1", "--stage", "assemble",
                                "--masks-dir", str(masks), "--parts-dir", str(parts), "--output", str(shards / f"{i:05d}")],
                          logs / f"assemble-{i:05d}.log"): i for i in todo}
        for f in as_completed(futs):
            if f.result() != 0:
                sys.exit(f"assemble failed for image {futs[f]}; see {logs}")
    log(f"done in {time.time() - t_all:.0f} s; images {ids}, skipped {skipped}; shards in {shards}")
    print("\nWhen every machine is done, gather all shard directories in one place, merge them into a new directory,\n"
          "then check the merged bundle against the shipped manifest and install it:\n"
          f"  python3 tools/mnist/survey_sparse.py --views {r} --image-ids <all generated ids, ascending> "
          f"--merge <shard directories> --output benchmarks/mnist-test/generated/R{r}\n"
          f"  python3 scripts/check_bundle.py {r} benchmarks/mnist-test/generated/R{r} --install")


if __name__ == "__main__":
    main()
