"""
FAISS vector index: build, persist, and load.

Index type: IndexFlatIP (exact inner-product search).
Because embeddings are L2-normalized, inner product == cosine similarity,
scores are in [-1, 1] where 1.0 means identical.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

INDEX_PATH = "data/vector_index/products.index"
META_PATH = "data/vector_index/products_meta.pkl"


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a flat inner-product index from a (N, dim) float32 array."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal:,} vectors, dim={dim}")
    return index


def save_index(
    index: faiss.Index,
    metadata: pd.DataFrame,
    index_path: str = INDEX_PATH,
    meta_path: str = META_PATH,
) -> None:
    """Persist the FAISS index and product metadata to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata.reset_index(drop=True), f)
    print(f"Index saved  -> {index_path}")
    print(f"Metadata saved -> {meta_path}")


def load_index(
    index_path: str = INDEX_PATH,
    meta_path: str = META_PATH,
):
    """Load a persisted FAISS index and its metadata DataFrame."""
    if not index_exists(index_path, meta_path):
        raise FileNotFoundError(
            f"Index not found at {index_path}. "
            "Run scripts/build_index.py first."
        )
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    print(f"Index loaded: {index.ntotal:,} vectors from {index_path}")
    return index, metadata


def index_exists(
    index_path: str = INDEX_PATH,
    meta_path: str = META_PATH,
) -> bool:
    return Path(index_path).exists() and Path(meta_path).exists()
