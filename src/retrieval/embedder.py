"""
Embedding model setup and product text encoding.
Uses sentence-transformers (all-MiniLM-L6-v2) to produce
normalized 384-dim vectors suitable for cosine similarity via FAISS IndexFlatIP.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def build_product_texts(df: pd.DataFrame) -> list:
    """Combine product fields into a single string per row for embedding."""
    texts = []
    for _, row in df.iterrows():
        text = (
            f"Product: {row['name_clean']}\n"
            f"Color: {row['colors_extracted']}\n"
            f"Description: {row['fulldescription_clean']}"
        )
        texts.append(text)
    return texts


def embed_texts(
    model: SentenceTransformer,
    texts: list,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings into L2-normalized float32 embeddings.
    Normalized vectors allow cosine similarity via inner product (IndexFlatIP).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)
