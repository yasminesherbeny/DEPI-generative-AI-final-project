"""
src/pipeline/filters.py
=======================
Post-retrieval filters for the RAG pipeline.
Called after ProductRetriever returns results to narrow them
by color or category before passing to the prompt builder.
"""
from typing import Any, Dict, List


def filter_by_color(
    results: List[Dict[str, Any]],
    color: str,
) -> List[Dict[str, Any]]:
    """
    Keep only results where the product color matches the query color.
    If color is empty or no results match, returns the original list unfiltered
    so the pipeline never gets an empty retrieval.
    """
    if not color:
        return results

    filtered = [
        r for r in results
        if color.lower() in r.get("colors_extracted", "").lower()
    ]
    # fallback: if nothing matched, return original so pipeline doesn't break
    return filtered if filtered else results


def filter_by_category(
    results: List[Dict[str, Any]],
    category: str,
) -> List[Dict[str, Any]]:
    """
    Keep only results where the product name contains the category keyword.
    Same fallback logic as filter_by_color.
    """
    if not category:
        return results

    filtered = [
        r for r in results
        if category.lower() in r.get("name_clean", "").lower()
    ]
    return filtered if filtered else results


def apply_all_filters(
    results: List[Dict[str, Any]],
    color: str = "",
    category: str = "",
) -> List[Dict[str, Any]]:
    """
    Convenience function: apply all filters in sequence.
    This is what pipeline.py calls.
    """
    results = filter_by_color(results, color)
    results = filter_by_category(results, category)
    return results