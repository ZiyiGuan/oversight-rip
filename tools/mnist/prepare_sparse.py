"""Sparse-majority instance preparation (spec sections 14, 15, 20).

For each configured R: deterministic mask registry for candidate images
(PCG64(GLOBAL_SEED + image_id)), the frozen judge's top-1 prediction on every
registered mask (the MANDATORY table oracle, computed on CPU so the canonical
table does not depend on accelerator arithmetic), selection of the first test
image with S_{y*} >= T + t_target, and a live-vs-table differential re-run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import mnist, sha256_file
from train_sparse_judge import SparseJudgeV1, K

ART = Path(__file__).resolve().parents[2] / "benchmarks" / "artifacts"
MAN = Path(__file__).resolve().parents[2] / "benchmarks" / "manifests"
CKPT = Path(__file__).resolve().parents[2] / "benchmarks" / "checkpoints"
GLOBAL_SEED = 20260827
R_VALUES = [256, 1024, 4096]


def masks_for(image_flat, image_id, r):
    nonzero = sorted(int(i) for i in np.flatnonzero(image_flat > 0))
    assert len(nonzero) >= K, "image has fewer than six nonzero pixels"
    rng = np.random.Generator(np.random.PCG64(GLOBAL_SEED + image_id))
    masks, seen = [], set()
    while len(masks) < r:
        mask = tuple(sorted(rng.choice(nonzero, size=K, replace=False).tolist()))
        if mask not in seen:
            seen.add(mask)
            masks.append(mask)
    for m in masks:
        assert len(set(m)) == K
        assert all(image_flat[p] > 0 for p in m)
    assert len(set(masks)) == r
    return masks


def judge_predictions(model, image_flat, masks):
    x = np.zeros((len(masks), 2, 28, 28), dtype=np.float32)
    for i, m in enumerate(masks):
        coords = np.array(m)
        r, c = coords // 28, coords % 28
        x[i, 0, r, c] = 1.0
        x[i, 1, r, c] = image_flat[coords].astype(np.float32) / 255.0
    with torch.no_grad():
        out = []
        for lo in range(0, len(masks), 256):
            logits = model(torch.tensor(x[lo:lo + 256]))
            # smallest-index argmax: torch argmax returns the first maximum.
            out.append(logits.argmax(1).numpy())
    return np.concatenate(out).astype(np.uint8)


def main():
    torch.use_deterministic_algorithms(True)
    model = SparseJudgeV1()
    model.load_state_dict(torch.load(CKPT / "sparse_judge_v1.pt", map_location="cpu"))
    model.eval()
    judge_hash = json.loads((ART / "sparse_judge_report.json").read_text())["judge_model_hash"]

    test_imgs, test_labels = mnist("test")
    flat = test_imgs.reshape(len(test_imgs), -1)

    for r in R_VALUES:
        threshold = r // 2 + 1
        t_target = r // 8
        for image_id in range(len(test_imgs)):
            masks = masks_for(flat[image_id], image_id, r)
            table = judge_predictions(model, flat[image_id], masks)
            votes = np.bincount(table, minlength=10)
            y_star = int(test_labels[image_id])
            if votes[y_star] >= threshold + t_target:
                break
            print(f"R={r} image {image_id}: S_y*={votes[y_star]} < {threshold + t_target}, next",
                  flush=True)
        else:
            raise SystemExit(f"R={r}: no image met the vote target")

        # d_y per label, and the whole-circuit radius d_min.
        d = [int(votes[y] - threshold + 1) if votes[y] >= threshold else int(threshold - votes[y])
             for y in range(10)]
        d_min = min(d)

        # Live-vs-table differential: a second, independent pass must agree on
        # every registered mask.
        again = judge_predictions(model, flat[image_id], masks)
        assert np.array_equal(table, again), "live and table oracles disagree"

        masks_path = ART / f"sparse_masks_R{r}.txt"
        with open(masks_path, "w") as f:
            for m in masks:
                f.write(" ".join(str(p) for p in m) + "\n")
        table_path = ART / f"sparse_table_R{r}.txt"
        table_path.write_text("".join(str(int(p)) for p in table) + "\n")

        manifest = {
            "circuit": "sparse_mnist_majority_v1",
            "image_id": int(image_id),
            "true_label": y_star,
            "k": K,
            "R": r,
            "threshold": threshold,
            "dataset_hash": sha256_file(Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "t10k-images-idx3-ubyte"),
            "judge_model_hash": judge_hash,
            "mask_registry_hash": sha256_file(masks_path),
            "table_hash": sha256_file(table_path),
            "vote_counts": [int(v) for v in votes],
            "accepted_label": y_star,
            "min_flip_distance": d_min,
            "certified_flips": d_min - 1,
            "t_target": t_target,
            "global_seed": GLOBAL_SEED,
        }
        (MAN / f"sparse_instance_R{r}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with open(ART / f"sparse_instance_R{r}.txt", "w") as f:
            f.write(f"image_id {image_id}\n")
            f.write(f"true_label {y_star}\n")
            f.write(f"r {r}\n")
            f.write(f"threshold {threshold}\n")
            f.write("votes " + " ".join(str(int(v)) for v in votes) + "\n")
            f.write(f"d_min {d_min}\n")
            f.write("table " + "".join(str(int(p)) for p in table) + "\n")
        print(f"R={r}: image {image_id} (digit {y_star}), votes {list(votes)}, "
              f"T={threshold}, d_min={d_min}", flush=True)


if __name__ == "__main__":
    main()
