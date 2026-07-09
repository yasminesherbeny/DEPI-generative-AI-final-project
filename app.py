import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from src.pipeline.pipeline import run_pipeline

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    if not data or "name" not in data:
        return jsonify({"error": "Missing required field: name"}), 400

    name     = data.get("name", "").strip()
    color    = data.get("color", "").strip()
    category = data.get("category", "").strip()
    top_k    = data.get("top_k", 3)

    if not name:
        return jsonify({"error": "Product name cannot be empty"}), 400

    try:
        result = run_pipeline(
            name=name,
            color=color,
            category=category,
            top_k=top_k,
        )
        return jsonify({
            "name": result["name"],
            "color": result["color"],
            "description": result["description"],
            "similar_products": [
                {
                    "name": p.get("name_clean", ""),
                    "color": p.get("colors_extracted", ""),
                    "score": p.get("similarity_score", 0),
                }
                for p in result["similar_products"]
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)