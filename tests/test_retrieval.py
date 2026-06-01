"""
Unit tests for Member 2: RAG Core — Vector store.
Tests run without a real FAISS index by mocking heavy dependencies.
"""

import numpy as np
import pandas as pd
import pytest


# ─── embedder ────────────────────────────────────────────────────────────────

def test_build_product_texts_format():
    from src.retrieval.embedder import build_product_texts

    df = pd.DataFrame([{
        "name_clean": "silk blouse",
        "colors_extracted": "blue, white",
        "fulldescription_clean": "A lightweight silk blouse with button front.",
    }])
    texts = build_product_texts(df)
    assert len(texts) == 1
    assert "Product: silk blouse" in texts[0]
    assert "Color: blue, white" in texts[0]
    assert "Description:" in texts[0]


def test_embed_texts_shape(monkeypatch):
    from src.retrieval import embedder

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.random.rand(len(texts), 384).astype(np.float32)

    embs = embedder.embed_texts(FakeModel(), ["hello", "world"], show_progress=False)
    assert embs.shape == (2, 384)
    assert embs.dtype == np.float32


def test_embed_texts_normalized(monkeypatch):
    """Embeddings should be L2-normalized (norm ≈ 1.0)."""
    from src.retrieval import embedder

    class FakeModel:
        def encode(self, texts, **kwargs):
            # simulate unit vectors
            vecs = np.ones((len(texts), 4), dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    embs = embedder.embed_texts(FakeModel(), ["a", "b"], show_progress=False)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ─── vector_store ─────────────────────────────────────────────────────────────

def test_build_index_ntotal():
    import faiss
    from src.retrieval.vector_store import build_index

    embeddings = np.random.rand(50, 384).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = build_index(embeddings)
    assert index.ntotal == 50


def test_save_and_load_index(tmp_path):
    from src.retrieval.vector_store import build_index, load_index, save_index

    embeddings = np.random.rand(10, 384).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = build_index(embeddings)

    meta = pd.DataFrame({
        "name_clean": [f"product_{i}" for i in range(10)],
        "colors_extracted": ["red"] * 10,
        "fulldescription_clean": ["desc"] * 10,
    })

    idx_path = str(tmp_path / "test.index")
    meta_path = str(tmp_path / "test_meta.pkl")
    save_index(index, meta, index_path=idx_path, meta_path=meta_path)

    loaded_index, loaded_meta = load_index(index_path=idx_path, meta_path=meta_path)
    assert loaded_index.ntotal == 10
    assert list(loaded_meta.columns) == ["name_clean", "colors_extracted", "fulldescription_clean"]


def test_index_exists_false(tmp_path):
    from src.retrieval.vector_store import index_exists

    assert not index_exists(
        index_path=str(tmp_path / "no.index"),
        meta_path=str(tmp_path / "no.pkl"),
    )


# ─── retriever ────────────────────────────────────────────────────────────────

def _make_retriever(tmp_path):
    """Helper: build a tiny real index and return a ProductRetriever over it."""
    import faiss
    from src.retrieval.retriever import ProductRetriever
    from src.retrieval.vector_store import build_index, save_index

    n, dim = 20, 384
    embeddings = np.random.rand(n, dim).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = build_index(embeddings)

    meta = pd.DataFrame({
        "name_clean": [f"product_{i}" for i in range(n)],
        "colors_extracted": ["blue"] * n,
        "fulldescription_clean": [f"description number {i}" for i in range(n)],
    })

    idx_path = str(tmp_path / "test.index")
    meta_path = str(tmp_path / "test_meta.pkl")
    save_index(index, meta, index_path=idx_path, meta_path=meta_path)

    # Patch the model so we don't download sentence-transformers in CI
    class FakeModel:
        def encode(self, texts, **kwargs):
            vecs = np.random.rand(len(texts), dim).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    retriever = ProductRetriever.__new__(ProductRetriever)
    retriever.model = FakeModel()
    from src.retrieval.vector_store import load_index
    retriever.index, retriever._metadata = load_index(idx_path, meta_path)
    retriever._records = retriever._metadata.to_dict(orient="records")
    return retriever


def test_retrieve_returns_top_k(tmp_path):
    retriever = _make_retriever(tmp_path)
    results = retriever.retrieve("casual shirt", top_k=5)
    assert len(results) == 5


def test_retrieve_has_expected_keys(tmp_path):
    retriever = _make_retriever(tmp_path)
    results = retriever.retrieve("dress", top_k=3)
    for r in results:
        assert "name_clean" in r
        assert "colors_extracted" in r
        assert "fulldescription_clean" in r
        assert "similarity_score" in r


def test_retrieve_for_prompt(tmp_path):
    retriever = _make_retriever(tmp_path)
    results = retriever.retrieve_for_prompt(name="blouse", color="red", top_k=3)
    assert len(results) == 3
    assert all("similarity_score" in r for r in results)


def test_retrieve_similarity_score_range(tmp_path):
    retriever = _make_retriever(tmp_path)
    results = retriever.retrieve("jacket", top_k=5)
    for r in results:
        assert -1.0 <= r["similarity_score"] <= 1.0
