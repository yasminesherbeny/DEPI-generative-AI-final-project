"""
src/pipeline/prompt_builder.py
================================
Python prompt builder for the pipeline.
needed to wire the pipeline together.

The format mirrors src/inference/infer.py's build_inference_prompt()
which is what the LoRA model was actually trained on.
"""
from typing import Any, Dict, List


def build_prompt(
    product_name: str,
    color: str,
    retrieved_examples: List[Dict[str, Any]] = None,
) -> str:
    """
    Build the final prompt to feed into the fine-tuned model.

    Uses the format the model was trained on:
        Product: <name>
            Color: <color>
            Description:

    Optionally prepends retrieved similar products as context
    so the model has examples to draw from (RAG enhancement).
    """
    lines = []

    # add retrieved examples as context if available
    if retrieved_examples:
        lines.append("Here are similar products for reference:\n")
        for i, example in enumerate(retrieved_examples, 1):
            name = example.get("name_clean", "")
            color_ex = example.get("colors_extracted", "")
            desc = example.get("fulldescription_clean", "")
            lines.append(f"Example {i}:")
            lines.append(f"Product: {name}")
            if color_ex:
                lines.append(f"    Color: {color_ex}")
            lines.append(f"    Description: {desc}\n")
        lines.append("---\n")
        lines.append("Now generate a description for this product:\n")

    # the actual product to generate for
    lines.append(f"Product: {product_name}")
    lines.append(f"    Color: {color}")
    lines.append(f"    Description: ")

    return "\n".join(lines)