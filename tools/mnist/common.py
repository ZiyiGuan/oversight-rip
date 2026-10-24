"""Shared MNIST loading and canonical preprocessing.

The idx files in benchmarks/data are the canonical MNIST distribution
(ossci-datasets mirror); their sha256 hashes are recorded in the manifests.
"""
import hashlib
import struct
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "benchmarks" / "data"


def load_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, path
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, path
        return np.frombuffer(f.read(), dtype=np.uint8)


def mnist(split):
    prefix = {"train": "train", "test": "t10k"}[split]
    imgs = load_idx_images(DATA / f"{prefix}-images-idx3-ubyte")
    labels = load_idx_labels(DATA / f"{prefix}-labels-idx1-ubyte")
    return imgs, labels


def sha256_file(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()
