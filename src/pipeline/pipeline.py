import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.retriever import ProductRetriever
from src.inference.generate import load_generator, generate_text
from src.pipeline.filters import apply_all_filters
from src.pipeline.prompt_builder import build_prompt

_retriever = None
_generator = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = ProductRetriever()
    return _retriever


def _get_generator():
    global _generator
    if _generator is None:
        _generator = load_generator("lora_model")
    return _generator


def run_pipeline(
    name: str,
    color: str = "",
    category: str = "",
    top_k: int = 3,
) -> dict:
    retriever = _get_retriever()
    generator = _get_generator()

    # retrieve similar products
    raw_results = retriever.retrieve_for_prompt(name, color, top_k=top_k)

    # apply your filters
    filtered = apply_all_filters(raw_results, color=color, category=category)
    # build enriched prompt
    prompt = build_prompt(name, color)

    # generate description
    description = generate_text(generator, prompt)

    return {
        "name": name,
        "color": color,
        "description": description,
        "similar_products": filtered,
    }