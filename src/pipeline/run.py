"""src/pipeline/run.py
===================
End-to-end test script for the full pipeline.
Run from the project root:
    python src/pipeline/run.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline.pipeline import run_pipeline
from src.pipeline.pipeline import run_pipeline


def main():
    test_cases = [
        {"name": "Running Shoes",     "color": "Black", "category": "shoes"},
        {"name": "Floral Maxi Dress", "color": "Blue",  "category": "dress"},
        {"name": "Leather Wallet",    "color": "",       "category": ""},
    ]

    for tc in test_cases:
        print(f"\n{'='*55}")
        print(f"Input: {tc['name']} | {tc['color'] or 'no color'}")
        print('='*55)

        result = run_pipeline(
            name=tc["name"],
            color=tc["color"],
            category=tc["category"],
        )

        print(f"Generated description:\n  {result['description']}")
        print(f"\nTop similar products after filtering:")
        for p in result["similar_products"]:
            print(f"  [{p['similarity_score']:.3f}] {p['name_clean']} | {p.get('colors_extracted','')}")
        


if __name__ == "__main__":
    main()