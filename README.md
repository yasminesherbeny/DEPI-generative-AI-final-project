# 🛍️ Intelligent Product Description Generation System

An AI-powered system that automatically generates high-quality e-commerce product descriptions using a fine-tuned GPT-2 model enhanced with Retrieval-Augmented Generation (RAG) and an LLM-based Planner Agent.

---

## 📌 Project Overview

This system transforms structured product information (name, color, category) into professional marketing copy. It combines:

- **LoRA fine-tuned GPT-2** for domain-specific text generation
- **FAISS vector search + BM25 hybrid retrieval** for contextual grounding
- **LLM-based Planner Agent (Ollama)** for intelligent retrieval strategy decisions
- **Flask REST API** for programmatic access
- **Streamlit web interface** for user interaction

### Pipeline Flow

```
User Input
    ↓
Flask API (POST /generate)
    ↓
Planner Agent (Ollama LLM)
    ↓ decides: top_k, strategy, keywords
Retriever (FAISS dense or hybrid BM25+dense)
    ↓
Filters (color + category)
    ↓
Prompt Builder
    ↓
LoRA GPT-2 Model
    ↓
Generated Description
```

---

## 🗂️ Project Structure

```
DEPI-generative-AI-final-project/
├── app.py                          # Flask REST API
├── streamlit_app.py                # Streamlit web interface
├── requirements.txt
├── lora_model/                     # LoRA adapter (not in repo — share manually)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── data/
│   ├── processed/
│   │   └── cleaned_features.csv    # Preprocessed dataset
│   └── vector_index/               # Built at runtime by build_index.py
│       ├── products.index
│       └── products_meta.pkl
├── scripts/
│   ├── build_index.py              # One-time FAISS index builder
│   └── run_train.py                # Full training pipeline
└── src/
    ├── agents/
    │   └── planner.py              # LLM-based Planner Agent (Ollama)
    ├── data/
    │   ├── load_data.py            # Kaggle dataset download
    │   ├── preprocess.py           # Data cleaning
    │   └── feature_engineering.py  # Feature extraction
    ├── inference/
    │   └── generate.py             # LoRA model loading and generation
    ├── nlp/
    │   └── prompt_builder.py       # Training prompt formatter
    ├── pipeline/
    │   ├── pipeline.py             # Main orchestrator
    │   ├── filters.py              # Color + category filtering
    │   ├── hybrid_search.py        # BM25 + dense hybrid retrieval
    │   ├── prompt_builder.py       # Inference prompt builder
    │   └── run.py                  # End-to-end test script
    ├── retrieval/
    │   ├── embedder.py             # Sentence-transformer embeddings
    │   ├── vector_store.py         # FAISS index management
    │   └── retriever.py            # Product retrieval API
    ├── training/
    │   ├── dataset_preparation.py  # Tokenization + HF dataset
    │   └── training_lora.py        # LoRA fine-tuning
    └── utils/
        └── logger.py               # Shared logging utility
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- Git
- The `lora_model/` folder (obtain from team — not stored in repo)
- Ollama (optional, for LLM-based planner) — [https://ollama.com](https://ollama.com)

### Steps

**1. Clone the repo and switch to the project branch**
```bash
git clone https://github.com/yasminesherbeny/DEPI-generative-AI-final-project
cd DEPI-generative-AI-final-project
git checkout yasmine
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
pip install flask streamlit
```

**4. Place the `lora_model/` folder in the project root**

The folder should contain:
```
lora_model/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

**5. Build the FAISS index (first time only)**
```bash
python scripts/build_index.py
```
This takes 2–5 minutes and creates `data/vector_index/` automatically.

---

## 🚀 Running the System

Open **two terminals**:

**Terminal 1 — Start the Flask API**
```bash
python app.py
```
API will be available at `http://localhost:7860`

**Terminal 2 — Start the Streamlit interface**
```bash
python -m streamlit run streamlit_app.py
```
Interface will open at `http://localhost:8501`

---

## 🔌 API Reference

### `GET /health`
Check if the API is running.

**Response:**
```json
{
  "status": "ok",
  "message": "API is running"
}
```

---

### `POST /generate`
Generate a product description.

**Request body:**
```json
{
  "name": "Running Shoes",
  "color": "Black",
  "category": "shoes",
  "tone": "sporty",
  "top_k": 3
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✅ | Product name |
| `color` | string | ❌ | Product color |
| `category` | string | ❌ | Category keyword for filtering |
| `tone` | string | ❌ | Desired tone (e.g. sporty, luxury) |
| `top_k` | integer | ❌ | Number of similar products to retrieve (default: 3) |

**Response:**
```json
{
  "name": "Running Shoes",
  "color": "Black",
  "description": "Classic style running shoes perfect for everyday runners...",
  "keywords": ["running", "shoes", "black"],
  "retrieval_strategy": "dense",
  "similar_products": [
    {
      "name": "men's new balance 1080v8 running shoes",
      "color": "steel/black",
      "score": 0.6248
    }
  ]
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{"name": "Running Shoes", "color": "Black", "category": "shoes"}'
```

**Example with Python:**
```python
import requests

response = requests.post(
    "http://localhost:7860/generate",
    json={"name": "Running Shoes", "color": "Black", "top_k": 3}
)
print(response.json()["description"])
```

---

## 🧠 Planner Agent (Ollama)

The Planner Agent uses a local Ollama LLM to intelligently decide:
- Which **keywords** to extract from the product name
- The **top_k** value (2–5 based on input richness)
- The **retrieval strategy** (dense for short names, hybrid for 3+ word names)
- The **tone** (inferred from product type if not provided)

### Optional Setup

```bash
# Install Ollama from https://ollama.com/download
ollama pull llama3
ollama serve
```

The agent automatically detects Ollama and uses it. If Ollama is not running, it falls back to rule-based planning — **no configuration change needed**.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language Model | GPT-2 (Hugging Face) |
| Fine-Tuning | LoRA via PEFT |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Search | FAISS IndexFlatIP |
| Keyword Search | rank-bm25 |
| Planner LLM | Ollama (llama3 / mistral) |
| API | Flask |
| Interface | Streamlit |
| Dataset | Kaggle E-commerce Products (~24k rows) |

---

## 👥 Team

| Member | Role |
|---|---|
| Member 1 | Planner Agent |
| Member 2 | RAG Core — Vector Store |
| Member 3 | Prompt Builder |
| Member 4 | Model Inference |
| Member 5 (Yasmine) | RAG Enhancement & Pipeline Integration |

---

## 📋 Notes

- The `lora_model/` folder and `data/vector_index/` are excluded from the repo (too large for git). Share `lora_model/` directly with teammates and run `build_index.py` locally.
- The FAISS index must be rebuilt on each new machine by running `python scripts/build_index.py`.
- The system runs on CPU — no GPU required for inference.
- Generation takes 5–15 seconds per product on CPU.
