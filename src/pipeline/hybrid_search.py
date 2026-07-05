"""
src/pipeline/hybrid_search.py
==============================
Hybrid search: combines dense (FAISS) retrieval with BM25 keyword search.
Improves results when the user's query shares exact keywords with products
that the embedding model might rank lower.

Requires: pip install rank-bm25
"""
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from src.retrieval.retriever import ProductRetriever


class HybridRetriever:
    """
    Wraps ProductRetriever (dense) and adds BM25 (keyword) search on top.
    Results from both are merged and deduplicated.

    Usage:
        retriever = ProductRetriever()
        hybrid = HybridRetriever(retriever)
        results = hybrid.search("Running Shoes", "Black", top_k=3)
    """

    def __init__(self, retriever: ProductRetriever) -> None:
        self.retriever = retriever

        # build BM25 index from the same metadata the FAISS index uses
        self._records = retriever._records
        corpus = [
            (r.get("name_clean", "") + " " + r.get("colors_extracted", "")).lower()
            for r in self._records
        ]
        tokenized = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[hybrid] BM25 index built over {len(self._records):,} products.")

    def search(
        self,
        name: str,
        color: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Run dense + BM25 search and merge results.

        Returns top_k deduplicated results. Dense results are prioritized
        since they carry similarity_score; BM25 results that weren't in
        the dense set are appended after.
        """
        # -- dense retrieval (fetch more so merging has material to work with)
        dense_results = self.retriever.retrieve_for_prompt(
            name, color, top_k=top_k * 2
        )

        # -- BM25 keyword search
        query_tokens = (name + " " + color).lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_indices = bm25_scores.argsort()[::-1][: top_k * 2]
        bm25_results = []
        for idx in top_indices:
            record = dict(self._records[idx])
            record["similarity_score"] = round(float(bm25_scores[idx]), 4)
            bm25_results.append(record)

        # -- merge: dense first, then BM25 extras, deduplicate by name
        seen = set()
        merged = []
        for r in dense_results + bm25_results:
            key = r.get("name_clean", "")
            if key not in seen:
                seen.add(key)
                merged.append(r)

        return merged[:top_k]