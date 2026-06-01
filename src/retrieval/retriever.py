"""
Clean retrieval API for semantic product search.
This is the interface that the prompt builder (Member 1) consumes.

Usage:
    retriever = ProductRetriever()
    results = retriever.retrieve("lightweight summer dress", top_k=5)
    results = retriever.retrieve_for_prompt(name="silk blouse", color="blue", top_k=3)
"""

from typing import Any, Dict, List

from .embedder import embed_texts, load_embedding_model
from .vector_store import INDEX_PATH, META_PATH, load_index

MODEL_NAME = "all-MiniLM-L6-v2"


class ProductRetriever:
    """
    Loads the FAISS index once and exposes a simple retrieve() interface.
    Thread-safe for read-only queries after construction.
    """

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        meta_path: str = META_PATH,
        model_name: str = MODEL_NAME,
    ):
        self.model = load_embedding_model(model_name)
        self.index, self._metadata = load_index(index_path, meta_path)
        self._records = self._metadata.to_dict(orient="records")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return top_k products most similar to query.

        Args:
            query: Free-text search query.
            top_k: Number of results to return.

        Returns:
            List of dicts, each containing product fields plus
            'similarity_score' (cosine similarity, higher is better).
        """
        query_emb = embed_texts(self.model, [query], show_progress=False)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = dict(self._records[idx])
            record["similarity_score"] = round(float(score), 4)
            results.append(record)
        return results

    def retrieve_for_prompt(
        self,
        name: str,
        color: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper for Member 1's planner agent.
        Builds a structured query from product name + color
        to find similar products with rich descriptions.

        Args:
            name:   Product name from the planner's parsed input.
            color:  Optional color hint from the planner.
            top_k:  Number of similar products to return.

        Returns:
            Same format as retrieve().
        """
        if color:
            query = f"Product: {name}\nColor: {color}"
        else:
            query = f"Product: {name}"
        return self.retrieve(query, top_k=top_k)
