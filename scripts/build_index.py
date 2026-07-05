"""
One-time script to embed the full product dataset and persist the FAISS index.

Run from the project root:
    python scripts/build_index.py

Reads:  data/processed/cleaned_features.csv   (output of feature_engineering)
Writes: data/vector_index/products.index
        data/vector_index/products_meta.pkl

On a CPU with ~24k products this takes roughly 2-5 minutes.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.retrieval.embedder import build_product_texts, embed_texts, load_embedding_model
from src.retrieval.vector_store import META_PATH, build_index, index_exists, save_index

DATA_PATH = "data/processed/cleaned_features.csv"
REQUIRED_COLS = {"name_clean", "fulldescription_clean", "colors_extracted"}


def main():
    if index_exists():
        print("Index already exists. Delete data/vector_index/ to rebuild.")
        return

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Run the data preprocessing pipeline first (scripts/run_train.py or feature_engineering).")
        sys.exit(1)

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns in CSV: {missing}")
        sys.exit(1)
    print(f"  {len(df):,} products loaded")

    print("\nLoading embedding model ...")
    model = load_embedding_model()

    print("\nBuilding product text representations ...")
    texts = build_product_texts(df)

    print(f"\nEmbedding {len(texts):,} products (batch_size=64) ...")
    embeddings = embed_texts(model, texts, batch_size=64, show_progress=True)
    print(f"  Embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    print("\nBuilding FAISS index ...")
    index = build_index(embeddings)

    meta_cols = ["name_clean", "colors_extracted", "fulldescription_clean"]
    metadata = df[meta_cols].reset_index(drop=True)

    print("\nSaving index and metadata ...")
    save_index(index, metadata)

    print("\nDone! Index is ready for retrieval.")


if __name__ == "__main__":
    main()
