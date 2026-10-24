"""Evaluate the frozen MNIST judge on the fixed 50 selected test
images at one view count R, and write the replayable selection bundle.

The 50 image identifiers are the benchmark's fixed public instances. They were
originally chosen, at R = 256, as the first 50 test images in test-set order
whose correct digit wins the majority with a certified radius of at least R/8.
At every other R the same 50 identifiers are evaluated; nothing is re-selected.
For each image the script draws the registered six-pixel mask registry
(PCG64(20260827 + image_id)), evaluates the frozen judge's CPU top-1
prediction on every view twice (a live/table differential), and records the
exact answer-flip certificate for all ten candidate digits.

The images are independent, so a long run can be split across processes: run
disjoint `--image-ids` subsets into separate directories, then `--merge` them.
The merged selected.json, selected.csv and config.json are byte-identical to
one run over all the IDs with the same --threads; manifest.json lists the
shards, and its elapsed_seconds is their sum.

Only main() imports numpy and torch; certificate and file checks and --merge
are stdlib, so `test_survey_sparse.py` runs without the MNIST environment.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GLOBAL_SEED = 20260827
K = 6
DATASET_COUNT = 10000
# The sparsest fixed image (ID 40) has 46 nonzero pixels, so C(46, 6) =
# 9,366,819 distinct masks: every size here draws without running out, and
# main() still checks each image.
VIEW_COUNTS = (256, 1024, 4096, 16384, 65536, 262144, 1048576, 1 << 22, 1 << 24, 1 << 26)
TOOL_SOURCES = ("common.py", "prepare_sparse.py", "train_sparse_judge.py", "survey_sparse.py")
# prepare_sparse.judge_predictions batches the masks it is given by 256 from
# the start of its list, so evaluating the registry in slices of JUDGE_CHUNK
# masks (a multiple of 256) feeds the model exactly the batches of one call on
# all R masks, while its (views, 2, 28, 28) float32 input stays at 6272 bytes
# x 65536 = 411 MB instead of 6.6 GB at R = 1048576. At R <= 65536 it is the
# single call it always was.
JUDGE_CHUNK = 65536
# The benchmark's fixed instances: zero-based MNIST test-set identifiers.
SELECTED_IMAGE_IDS = (
    0, 1, 2, 4, 5, 10, 13, 14, 17, 19, 21, 23, 25, 26, 27, 29, 30,
    31, 32, 34, 35, 36, 37, 39, 40, 41, 46, 47, 49, 50, 56, 57, 60,
    67, 69, 70, 71, 74, 76, 79, 81, 82, 85, 86, 88, 89, 90, 91, 94, 96,
)
EXPECTED_IMAGES_SHA256 = "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7"
EXPECTED_LABELS_SHA256 = "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2"
EXPECTED_CHECKPOINT_SHA256 = "c3b2b2bce5cc572f782118798d89d0fddca6cef0c0e37029e82b4b87b4d330d9"
EXPECTED_JUDGE_HASH = "4f8eb534290fdbc905f20f3afbf19663289df8e369f04785da5a71a92a2c9a41"
SOURCE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
CSV_FIELDS = [
    "image_id", "true_label", "accepted_label", "correctly_accepted",
    "min_flip_distance", "certified_flips", "certified_fraction",
    "positive_whole_circuit", "strong_whole_circuit", "strong_correct",
] + [f"votes_{y}" for y in range(10)] + [f"flip_distance_{y}" for y in range(10)] + [
    "table_sha256", "mask_registry_sha256", "image_sha256", "canonical_replay_checked",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    """digest(path.read_bytes()) without holding the file in memory; the
    R = 1048576 bundle is about 5.8 GB."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tool_source_hashes():
    return {name: digest((Path(__file__).parent / name).read_bytes()) for name in TOOL_SOURCES}


def checked_int(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def certificate(table, true_label, r, threshold, minimum_radius):
    """Exact distance to any output change over the ten legal label inputs."""
    checked_int(r, "r", 1, max(VIEW_COUNTS))
    checked_int(threshold, "threshold", r // 2 + 1, r // 2 + 1)
    checked_int(true_label, "true_label", 0, 9)
    checked_int(minimum_radius, "minimum_radius", 1, r)
    if len(table) != r:
        raise ValueError(f"expected {r} predictions, received {len(table)}")
    votes = [0] * 10
    for label in table:
        votes[checked_int(label, "prediction", 0, 9)] += 1
    distances = [v - threshold + 1 if v >= threshold else threshold - v for v in votes]
    accepted = [y for y, count in enumerate(votes) if count >= threshold]
    if len(accepted) > 1:
        raise ValueError("strict majority cannot accept two labels")
    accepted_label = accepted[0] if accepted else -1
    radius = min(distances) - 1
    correct = accepted_label == true_label
    return {
        "true_label": true_label,
        "accepted_label": accepted_label,
        "vote_counts": votes,
        "flip_distances": distances,
        "min_flip_distance": radius + 1,
        "certified_flips": radius,
        "correctly_accepted": correct,
        "positive_whole_circuit": radius >= 1,
        "strong_whole_circuit": radius >= minimum_radius,
        "strong_correct": correct and radius >= minimum_radius,
    }


def canonical_table(table):
    return ("".join(str(checked_int(y, "prediction", 0, 9)) for y in table) + "\n").encode("ascii")


def canonical_masks(masks):
    return "".join(" ".join(str(p) for p in row) + "\n" for row in masks).encode("ascii")


def judge_table(predict, masks):
    """The judge table as ints: `predict` on consecutive slices of at most
    JUDGE_CHUNK masks, concatenated in registry order."""
    table = []
    for lo in range(0, len(masks), JUDGE_CHUNK):
        table.extend(int(y) for y in predict(masks[lo:lo + JUDGE_CHUNK]))
    return table


def validate_masks(masks, image, r):
    if len(image) != 784 or len(masks) != r:
        raise ValueError("wrong image or registry size")
    nonzero_count = sum(p > 0 for p in image)
    if nonzero_count < K or math.comb(nonzero_count, K) < r:
        raise ValueError("image has insufficient distinct six-pixel masks")
    seen = set()
    for row in masks:
        if len(row) != K:
            raise ValueError("mask must contain exactly six coordinates")
        coords = tuple(checked_int(p, "coordinate", 0, 783) for p in row)
        if tuple(sorted(set(coords))) != coords:
            raise ValueError("mask coordinates must be distinct and sorted")
        if any(image[p] == 0 for p in coords):
            raise ValueError("mask reveals a zero-valued pixel")
        if coords in seen:
            raise ValueError("duplicate registered mask")
        seen.add(coords)


def validate_idx(images_bytes, labels_bytes):
    if len(images_bytes) != 16 + DATASET_COUNT * 784:
        raise ValueError("test image IDX has the wrong size")
    if struct.unpack(">IIII", images_bytes[:16]) != (2051, DATASET_COUNT, 28, 28):
        raise ValueError("test image IDX header does not describe 10000 28x28 images")
    if len(labels_bytes) != 8 + DATASET_COUNT:
        raise ValueError("test label IDX has the wrong size")
    if struct.unpack(">II", labels_bytes[:8]) != (2049, DATASET_COUNT):
        raise ValueError("test label IDX header does not describe 10000 labels")
    if any(y > 9 for y in labels_bytes[8:]):
        raise ValueError("test label IDX contains an unknown label")
    if digest(images_bytes) != EXPECTED_IMAGES_SHA256 or digest(labels_bytes) != EXPECTED_LABELS_SHA256:
        raise ValueError("MNIST test bytes do not match the original dataset identity")


def parse_image_ids(text):
    """Strictly increasing, distinct zero-based test identifiers."""
    ids = [checked_int(int(part), "image_id", 0, DATASET_COUNT - 1)
           for part in text.split(",") if part.strip()]
    if not ids or ids != sorted(set(ids)):
        raise ValueError("image IDs must be non-empty, distinct and strictly increasing")
    return ids


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def selection_bundle(config, images):
    """The replayable bundle: every fixed image must be correctly accepted
    with a radius of at least the configured minimum (one flip)."""
    ids = [row["image_id"] for row in images]
    if ids != sorted(set(ids)):
        raise ValueError("selected image IDs must be strictly increasing")
    if ids != list(config["criterion"]["image_ids"]):
        raise ValueError("selected image IDs differ from the configured fixed identifiers")
    minimum = config["criterion"]["minimum_certified_flips"]
    if any(not row["correctly_accepted"] or row["certified_flips"] < minimum for row in images):
        raise ValueError("selection contains an image that does not meet the criterion")
    return {
        "schema": "mnist_sparse_selection_v1",
        "circuit": "sparse_mnist_majority_v1",
        "dataset_count": DATASET_COUNT,
        "r": config["r"], "threshold": config["threshold"],
        "k": K, "global_seed": GLOBAL_SEED,
        "criterion": config["criterion"], "sources": config["sources"],
        "selected_count": len(images), "images": images,
    }


def csv_row(row, r):
    result = {key: row[key] for key in CSV_FIELDS if key in row}
    result["certified_fraction"] = row["certified_flips"] / r
    result.update({f"votes_{y}": row["vote_counts"][y] for y in range(10)})
    result.update({f"flip_distance_{y}": row["flip_distances"][y] for y in range(10)})
    return result


def write_bundle(output, config, rows, failures, elapsed, environment, shards=None):
    """Everything written after evaluation: selected.csv, then failures.json
    (and raise) or selected.json and manifest.json. A single run and --merge
    both end here, which is what keeps a merged bundle byte-identical."""
    r = config["r"]
    with (output / "selected.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row, r))
    if failures:
        atomic_json(output / "failures.json", {"r": r, "failures": failures})
        raise ValueError(f"{len(failures)} fixed image(s) are not correctly accepted with a positive "
                         f"radius at R={r}; see failures.json")
    atomic_json(output / "selected.json", selection_bundle(config, rows))
    manifest = {
        "schema": "mnist_sparse_selected_v1", "circuit": "sparse_mnist_majority_v1",
        "config": config, "elapsed_seconds": elapsed, "environment": environment,
        "provenance": {"images_url": SOURCE_URL + "t10k-images-idx3-ubyte.gz",
                       "labels_url": SOURCE_URL + "t10k-labels-idx1-ubyte.gz",
                       "selection": "fixed benchmark identifiers, originally the first 50 correct "
                                    "strong-majority test images at R=256 in test order",
                       "robustness_domain": "arbitrary Boolean oracle-answer flips, simultaneously over "
                                            "all ten legal candidate labels; invalid labels always reject",
                       "live_table_differential": "all registered masks of every image evaluated twice on CPU"},
        "summary": {"images": len(rows),
                    "certified_flips": {"min": min(row["certified_flips"] for row in rows),
                                        "max": max(row["certified_flips"] for row in rows)},
                    "strong_correct": sum(row["strong_correct"] for row in rows)},
        "artifacts": {name: {"sha256": file_digest(output / name),
                             "bytes": (output / name).stat().st_size}
                      for name in ("selected.json", "selected.csv", "config.json")},
    }
    if shards is not None:
        manifest["shards"] = shards
    atomic_json(output / "manifest.json", manifest)
    return manifest


def verified_bytes(directory, manifest, name):
    data = (directory / name).read_bytes()
    if digest(data) != manifest["artifacts"][name]["sha256"]:
        raise ValueError(f"{directory / name} does not match its manifest")
    return data


def merge_shards(shard_dirs, r, ids, output):
    """Write into `output` the bundle one run over `ids` would write, from
    completed runs over disjoint subsets of them. Every shard must be intact
    (its three files match its manifest), be a run at this R, and agree with
    the others on everything but its image IDs: threads, sources (with tool
    hashes equal to this script's own) and environment. The shards' IDs must
    be exactly `ids`. Nothing is written unless all of that holds."""
    common, by_id, shards = None, {}, []
    for directory in shard_dirs:
        if not (directory / "manifest.json").is_file():
            raise ValueError(f"{directory} is not a completed survey run (no manifest.json)")
        manifest = json.loads((directory / "manifest.json").read_text())
        verified_bytes(directory, manifest, "selected.csv")
        config = json.loads(verified_bytes(directory, manifest, "config.json"))
        bundle = json.loads(verified_bytes(directory, manifest, "selected.json"))
        rows = bundle["images"]
        if (not rows or manifest["config"] != config or config["r"] != r
                or config["threshold"] != r // 2 + 1 or selection_bundle(config, rows) != bundle):
            raise ValueError(f"{directory} is not a consistent R={r} survey run")
        shared = (dict(config, criterion=dict(config["criterion"], image_ids=None)), manifest["environment"])
        if common is None:
            common = shared
        elif shared != common:
            raise ValueError(f"{directory} differs from {shard_dirs[0]} in configuration or environment")
        for row in rows:
            if row["image_id"] in by_id:
                raise ValueError(f"image {row['image_id']} appears in two shards")
            by_id[row["image_id"]] = row
        shards.append({"image_ids": config["criterion"]["image_ids"],
                       "elapsed_seconds": manifest["elapsed_seconds"],
                       "selected_json_sha256": manifest["artifacts"]["selected.json"]["sha256"]})
    if sorted(by_id) != list(ids):
        raise ValueError("the shards' image IDs are not exactly the requested image IDs")
    base, environment = common
    if base["sources"]["tool_source_sha256"] != tool_source_hashes():
        raise ValueError("the shards were written by other tool sources than this script's")
    config = dict(base, criterion=dict(base["criterion"], image_ids=list(ids)))
    atomic_json(output / "config.json", config)
    shards.sort(key=lambda shard: shard["image_ids"][0])
    return write_bundle(output, config, [by_id[i] for i in ids], [],
                        sum(shard["elapsed_seconds"] for shard in shards), environment, shards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=int, default=256, choices=VIEW_COUNTS)
    parser.add_argument("--image-ids", type=str, default=",".join(map(str, SELECTED_IMAGE_IDS)),
                        help="comma-separated zero-based test IDs; default: the benchmark's fixed 50 "
                             "(with --merge: the IDs the shards must cover exactly)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("full", "masks", "range", "assemble"), default="full",
                        help="full (default): evaluate every view of every image here and write the bundle; "
                             "masks: write each image's mask registry to --masks-dir and stop; "
                             "range: evaluate the views --range LO:HI of each image from --masks-dir and write "
                             "the partial tables to --parts-dir; assemble: read --parts-dir, check that the "
                             "parts cover every view once, and write the bundle a full run over these images "
                             "would write (byte-identical; the parts' --threads and environment must match)")
    parser.add_argument("--range", type=str, help="LO:HI, the views to evaluate (stage range)")
    parser.add_argument("--masks-dir", type=Path, help="where stage masks writes and stages range/assemble read")
    parser.add_argument("--parts-dir", type=Path, help="where stage range writes and stage assemble reads")
    parser.add_argument("--merge", type=Path, nargs="+", metavar="SHARD_DIR",
                        help="evaluate nothing; merge completed runs at this --views over disjoint "
                             "--image-ids subsets, all with the same --threads, into the bundle one run "
                             "would write (--threads is then read from the shards)")
    args = parser.parse_args()
    checked_int(args.threads, "threads", 1, 64)
    ids = parse_image_ids(args.image_ids)
    r, threshold = args.views, args.views // 2 + 1
    strong_radius = r // 8
    output = args.output or ROOT / "benchmarks" / "mnist-test" / "generated" / f"R{r}"
    if args.stage in ("full", "assemble") or args.merge:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError(f"output directory {output} must be empty")
    if args.stage in ("masks", "range") and args.masks_dir is None:
        raise ValueError(f"stage {args.stage} needs --masks-dir")
    if args.stage in ("range", "assemble") and args.parts_dir is None:
        raise ValueError(f"stage {args.stage} needs --parts-dir")
    if args.stage == "range" and args.range is None:
        raise ValueError("stage range needs --range LO:HI")
    if args.merge:
        manifest = merge_shards(args.merge, r, ids, output)
        print(json.dumps({"event": "complete", "manifest": str(output / "manifest.json"),
                          "summary": manifest["summary"], "shards": len(manifest["shards"]),
                          "elapsed_seconds": round(manifest["elapsed_seconds"], 3)}), flush=True)
        return
    images_bytes = (ROOT / "benchmarks/data/t10k-images-idx3-ubyte").read_bytes()
    labels_bytes = (ROOT / "benchmarks/data/t10k-labels-idx1-ubyte").read_bytes()
    validate_idx(images_bytes, labels_bytes)
    checkpoint = ROOT / "benchmarks/checkpoints/sparse_judge_v1.pt"
    if digest(checkpoint.read_bytes()) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen judge checkpoint identity mismatch; do not retrain")
    report = json.loads((ROOT / "benchmarks/artifacts/sparse_judge_report.json").read_text())
    if report["judge_model_hash"] != EXPECTED_JUDGE_HASH or report["checkpoint_hash"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("original judge report identity mismatch")
    sources = {
        "test_images_sha256": digest(images_bytes),
        "test_labels_sha256": digest(labels_bytes),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "judge_model_hash": EXPECTED_JUDGE_HASH,
        "tool_source_sha256": tool_source_hashes(),
    }
    config = {
        "schema": "mnist_sparse_selected_config_v1", "r": r, "threshold": threshold,
        "threads": args.threads,
        "criterion": {"minimum_certified_flips": 1, "require_correct": True,
                      "ordering": "ascending_test_image_id",
                      "selection": "fixed_image_ids", "image_ids": ids,
                      "strong_radius": strong_radius},
        "sources": sources,
    }
    if args.stage in ("full", "assemble"):
        atomic_json(output / "config.json", config)

    import numpy as np
    import torch
    from prepare_sparse import masks_for, judge_predictions
    from train_sparse_judge import SparseJudgeV1

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model = SparseJudgeV1()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    judge_digest = hashlib.sha256()
    for _, parameter in sorted(model.state_dict().items()):
        judge_digest.update(parameter.cpu().numpy().astype("<f4").tobytes())
    if judge_digest.hexdigest() != EXPECTED_JUDGE_HASH:
        raise ValueError("frozen judge parameter stream mismatch")
    images = np.frombuffer(images_bytes, dtype=np.uint8, offset=16).reshape(DATASET_COUNT, 784)
    labels = list(labels_bytes[8:])
    environment = {"python": sys.version.split()[0], "numpy": np.__version__,
                   "torch": torch.__version__, "device": "cpu", "threads": args.threads,
                   "deterministic_algorithms": True}

    if args.stage == "masks":
        # The registry of every image, as the full run computes it, saved so
        # that the range workers of one image need not each regenerate it.
        args.masks_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            image = images[image_id]
            nonzero_count = int((image > 0).sum())
            if nonzero_count < K or math.comb(nonzero_count, K) < r:
                raise ValueError(f"image {image_id} has insufficient distinct masks")
            t0 = time.perf_counter()
            masks = [list(row) for row in masks_for(image, image_id, r)]
            validate_masks(masks, image, r)
            np.save(args.masks_dir / f"masks-R{r}-{image_id:05d}.npy", np.array(masks, dtype=np.uint16))
            atomic_json(args.masks_dir / f"masks-R{r}-{image_id:05d}.json",
                        {"schema": "mnist_sparse_masks_v1", "image_id": image_id, "r": r,
                         "mask_registry_sha256": digest(canonical_masks(masks)),
                         "image_sha256": digest(image.tobytes()),
                         "elapsed_seconds": round(time.perf_counter() - t0, 3)})
            print(json.dumps({"event": "masks", "image_id": image_id, "r": r,
                              "elapsed_seconds": round(time.perf_counter() - t0, 3)}), flush=True)
        return

    if args.stage == "range":
        # One slice of one image's views, evaluated twice as the full run
        # evaluates the whole table twice, written as a part.
        lo, hi = (checked_int(int(x), "range bound", 0, r) for x in args.range.split(":"))
        if not lo < hi:
            raise ValueError("--range needs LO < HI")
        args.parts_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            image = images[image_id]
            meta = json.loads((args.masks_dir / f"masks-R{r}-{image_id:05d}.json").read_text())
            if meta["image_sha256"] != digest(image.tobytes()) or meta["r"] != r:
                raise ValueError(f"masks of image {image_id} are not this image's at R={r}")
            registry = np.load(args.masks_dir / f"masks-R{r}-{image_id:05d}.npy", mmap_mode="r")
            part = [list(map(int, row)) for row in registry[lo:hi]]
            t0 = time.perf_counter()
            table = judge_table(lambda rows: judge_predictions(model, image, rows), part)
            again = judge_table(lambda rows: judge_predictions(model, image, rows), part)
            if table != again:
                raise ValueError(f"live/table differential failed at image {image_id} views {lo}:{hi}")
            elapsed_part = time.perf_counter() - t0
            atomic_json(args.parts_dir / f"part-R{r}-{image_id:05d}-{lo:012d}-{hi:012d}.json",
                        {"schema": "mnist_sparse_part_v1", "image_id": image_id, "r": r, "lo": lo, "hi": hi,
                         "mask_registry_sha256": meta["mask_registry_sha256"], "table": table,
                         "elapsed_seconds": elapsed_part, "threads": args.threads, "environment": environment})
            print(json.dumps({"event": "range", "image_id": image_id, "lo": lo, "hi": hi,
                              "elapsed_seconds": round(elapsed_part, 3)}), flush=True)
        return

    if args.stage == "assemble":
        # The bundle a full run over `ids` would write, from each image's
        # registry and the parts covering its views exactly once. The row is
        # built by the same statements as the full run's loop below.
        started = time.perf_counter()
        rows, failures, elapsed = [], [], 0.0
        for image_id in ids:
            image = images[image_id]
            meta = json.loads((args.masks_dir / f"masks-R{r}-{image_id:05d}.json").read_text())
            if meta["image_sha256"] != digest(image.tobytes()) or meta["r"] != r:
                raise ValueError(f"masks of image {image_id} are not this image's at R={r}")
            masks = [list(map(int, row)) for row in np.load(args.masks_dir / f"masks-R{r}-{image_id:05d}.npy")]
            validate_masks(masks, image, r)
            if digest(canonical_masks(masks)) != meta["mask_registry_sha256"]:
                raise ValueError(f"the saved registry of image {image_id} does not match its digest")
            parts = sorted(args.parts_dir.glob(f"part-R{r}-{image_id:05d}-*.json"))
            table, cursor = [], 0
            for path in parts:
                part = json.loads(path.read_text())
                if (part["image_id"], part["r"], part["mask_registry_sha256"]) != (image_id, r, meta["mask_registry_sha256"]):
                    raise ValueError(f"{path} is not a part of image {image_id} at R={r}")
                if part["lo"] != cursor or part["hi"] <= part["lo"] or len(part["table"]) != part["hi"] - part["lo"]:
                    raise ValueError(f"{path} does not continue the table at view {cursor}")
                if part["threads"] != args.threads or part["environment"] != environment:
                    raise ValueError(f"{path} was evaluated with other threads or another environment")
                table.extend(part["table"])
                elapsed += part["elapsed_seconds"]
                cursor = part["hi"]
            if cursor != r:
                raise ValueError(f"the parts of image {image_id} cover {cursor} of {r} views")
            row = certificate(table, labels[image_id], r, threshold, strong_radius)
            row.update({
                "image_id": image_id, "table": canonical_table(table).decode("ascii").rstrip("\n"),
                "table_sha256": digest(canonical_table(table)),
                "mask_registry_sha256": digest(canonical_masks(masks)),
                "image_sha256": digest(image.tobytes()),
                "canonical_replay_checked": True,
                "masks": masks, "image_pixels": image.astype(int).tolist(),
            })
            original_manifest = ROOT / f"benchmarks/manifests/sparse_instance_R{r}.json"
            if image_id == 0 and original_manifest.exists():
                original = json.loads(original_manifest.read_text())
                if row["table_sha256"] != original["table_hash"] or row["mask_registry_sha256"] != original["mask_registry_hash"]:
                    raise ValueError("original image-zero table/mask identity does not reproduce")
            if not row["correctly_accepted"] or row["certified_flips"] < 1:
                failures.append({"image_id": image_id, "accepted_label": row["accepted_label"],
                                 "true_label": row["true_label"], "certified_flips": row["certified_flips"],
                                 "vote_counts": row["vote_counts"]})
            rows.append(row)
            print(json.dumps({"event": "image", "image_id": image_id, "true_label": row["true_label"],
                              "accepted_label": row["accepted_label"], "certified_flips": row["certified_flips"],
                              "elapsed_seconds": round(time.perf_counter() - started, 3), "parts": len(parts)}), flush=True)
        manifest = write_bundle(output, config, rows, failures, elapsed, environment)
        print(json.dumps({"event": "complete", "manifest": str(output / "manifest.json"),
                          "summary": manifest["summary"], "elapsed_seconds": round(elapsed, 3)}), flush=True)
        return

    started = time.perf_counter()
    rows, failures = [], []
    for image_id in ids:
        image = images[image_id]
        nonzero_count = int((image > 0).sum())
        if nonzero_count < K or math.comb(nonzero_count, K) < r:
            raise ValueError(f"image {image_id} has insufficient distinct masks")
        masks = [list(row) for row in masks_for(image, image_id, r)]
        validate_masks(masks, image, r)
        table = judge_table(lambda part: judge_predictions(model, image, part), masks)
        again = judge_table(lambda part: judge_predictions(model, image, part), masks)
        if table != again:
            raise ValueError(f"live/table differential failed at image {image_id}")
        row = certificate(table, labels[image_id], r, threshold, strong_radius)
        row.update({
            "image_id": image_id, "table": canonical_table(table).decode("ascii").rstrip("\n"),
            "table_sha256": digest(canonical_table(table)),
            "mask_registry_sha256": digest(canonical_masks(masks)),
            "image_sha256": digest(image.tobytes()),
            "canonical_replay_checked": True,
            "masks": masks, "image_pixels": image.astype(int).tolist(),
        })
        original_manifest = ROOT / f"benchmarks/manifests/sparse_instance_R{r}.json"
        if image_id == 0 and original_manifest.exists():
            original = json.loads(original_manifest.read_text())
            if row["table_sha256"] != original["table_hash"] or row["mask_registry_sha256"] != original["mask_registry_hash"]:
                raise ValueError("original image-zero table/mask identity does not reproduce")
        if not row["correctly_accepted"] or row["certified_flips"] < 1:
            failures.append({"image_id": image_id, "accepted_label": row["accepted_label"],
                             "true_label": row["true_label"], "certified_flips": row["certified_flips"],
                             "vote_counts": row["vote_counts"]})
        rows.append(row)
        print(json.dumps({"event": "image", "image_id": image_id, "true_label": row["true_label"],
                          "accepted_label": row["accepted_label"], "certified_flips": row["certified_flips"],
                          "elapsed_seconds": round(time.perf_counter() - started, 3)}), flush=True)
    elapsed = time.perf_counter() - started
    manifest = write_bundle(output, config, rows, failures, elapsed, environment)
    print(json.dumps({"event": "complete", "manifest": str(output / "manifest.json"),
                      "summary": manifest["summary"], "elapsed_seconds": round(elapsed, 3)}), flush=True)


if __name__ == "__main__":
    main()
