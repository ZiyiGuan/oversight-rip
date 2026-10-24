"""Train the frozen sparse-MNIST judge (SparseJudgeV1) per the spec:
28x28, two input planes (revealed-position mask, revealed values/255),
k=6 revealed nonzero pixels, Adam 1e-4, batch 128, 30,000 steps.

The circuit never evaluates this CNN; it lives entirely inside the oracle.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import mnist, sha256_file

CKPT = Path(__file__).resolve().parents[2] / "benchmarks" / "checkpoints"
OUT = Path(__file__).resolve().parents[2] / "benchmarks" / "artifacts"
SEED = 20260827
STEPS = 30000
BATCH = 128
LR = 1e-4
K = 6


class SparseJudgeV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.net(x)


def sparse_views(imgs_u8, idx, rng):
    """Batch of two-plane sparse views: k=6 uniformly sampled nonzero coords."""
    b = len(idx)
    x = np.zeros((b, 2, 28, 28), dtype=np.float32)
    for i, j in enumerate(idx):
        img = imgs_u8[j].reshape(-1)
        nonzero = np.flatnonzero(img > 0)
        coords = rng.choice(nonzero, size=K, replace=False)
        r, c = coords // 28, coords % 28
        x[i, 0, r, c] = 1.0
        x[i, 1, r, c] = img[coords].astype(np.float32) / 255.0
    return x


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    CKPT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    train_imgs, train_labels = mnist("train")
    test_imgs, test_labels = mnist("test")
    # Images with fewer than K nonzero pixels cannot be sampled; MNIST has
    # none, but assert rather than assume.
    assert min((train_imgs.reshape(len(train_imgs), -1) > 0).sum(axis=1).min(),
               (test_imgs.reshape(len(test_imgs), -1) > 0).sum(axis=1).min()) >= K

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SparseJudgeV1().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    for step in range(STEPS):
        idx = rng.integers(0, len(train_imgs), size=BATCH)
        x = torch.tensor(sparse_views(train_imgs, idx, rng), device=dev)
        y = torch.tensor(train_labels[idx], dtype=torch.long, device=dev)
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            model.eval()
            with torch.no_grad():
                vi = rng.integers(0, len(test_imgs), size=1024)
                xv = torch.tensor(sparse_views(test_imgs, vi, rng), device=dev)
                acc = (model(xv).argmax(1).cpu().numpy() == test_labels[vi]).mean()
            model.train()
            print(f"step {step} loss {loss.item():.4f} sparse_test_acc {acc:.4f}", flush=True)

    model.eval()
    torch.save(model.state_dict(), CKPT / "sparse_judge_v1.pt")
    # The judge hash of record: the flat little-endian f32 parameter stream,
    # independent of torch's serialization format.
    h = hashlib.sha256()
    for _, p in sorted(model.state_dict().items()):
        h.update(p.cpu().numpy().astype("<f4").tobytes())
    judge_hash = h.hexdigest()
    report = {
        "judge_model_hash": judge_hash,
        "checkpoint_hash": sha256_file(CKPT / "sparse_judge_v1.pt"),
        "seed": SEED, "steps": STEPS, "batch": BATCH, "lr": LR, "k": K,
    }
    (OUT / "sparse_judge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
