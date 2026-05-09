"""
evaluate_model.py
=================
Evaluation script for the fine-tuned LoRA GPT-2 Product Description Generator.

Metrics computed:
  - Perplexity          (how well the model predicts the validation text)
  - BLEU Score          (n-gram overlap between generated and reference descriptions)
  - ROUGE-1/2/L         (recall-oriented overlap)
  - Distinct-1/2        (diversity of generated text)
  - Avg Generation Time (seconds per sample)

Usage:
  python evaluate_model.py

Requirements (already in requirements.txt):
  transformers, torch, nltk, rouge-score, pandas, numpy, peft
"""

import os
import time
import math
import warnings
import pandas as pd
import numpy as np
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from peft import PeftModel

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG  –  adjust paths if needed
# ─────────────────────────────────────────────
MODEL_PATH       = "./models/fine_tuned_lora"   # LoRA adapter folder
BASE_MODEL_NAME  = "gpt2"
VAL_CSV          = "./data/processed/val_prompts.csv"
NUM_SAMPLES      = 50        # how many val rows to evaluate (set None for all)
MAX_GEN_LENGTH   = 150       # tokens to generate
MAX_EVAL_LENGTH  = 256       # tokens used for perplexity window
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# 1. LOAD MODEL & TOKENIZER
# ─────────────────────────────────────────────
def load_model_and_tokenizer():
    print(f"\n{'='*55}")
    print("  Loading model & tokenizer ...")
    print(f"{'='*55}")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME)
    model      = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    model.to(DEVICE)

    print(f"  Device : {DEVICE}")
    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# ─────────────────────────────────────────────
# 2. LOAD VALIDATION DATA
# ─────────────────────────────────────────────
def load_val_data():
    print(f"\n  Loading validation data from: {VAL_CSV}")
    df = pd.read_csv(VAL_CSV)

    if NUM_SAMPLES:
        df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42).reset_index(drop=True)

    print(f"  Samples : {len(df)}")

    return df["prompt"].tolist()


# ─────────────────────────────────────────────
# 3. PERPLEXITY
# ─────────────────────────────────────────────
def compute_perplexity(model, tokenizer, prompts):
    """
    Average per-token perplexity on the validation prompts.
    Lower is better.
    """
    print("\n  Computing Perplexity ...")
    total_loss = 0.0
    count      = 0

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_EVAL_LENGTH
        ).to(DEVICE)

        input_ids = enc["input_ids"]

        with torch.no_grad():
            outputs = model(**enc, labels=input_ids)
            loss    = outputs.loss

        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss.item()
            count      += 1

    avg_loss   = total_loss / count if count else float("inf")
    perplexity = math.exp(avg_loss)
    return perplexity


# ─────────────────────────────────────────────
# 4. EXTRACT INPUT vs REFERENCE
#    Prompt format:
#      "...Description:\n<reference text>"
# ─────────────────────────────────────────────
def split_prompt_reference(prompt):
    """
    Returns (input_part, reference_description).
    input_part  = everything up to and including 'Description:\n'
    reference   = the actual description text that follows
    """
    marker = "Description:\n"  # matches "...\n\nDescription:\n<reference>"
    idx    = prompt.find(marker)
    if idx == -1:
        return prompt, ""
    split_at   = idx + len(marker)
    input_part = prompt[:split_at]
    reference  = prompt[split_at:].strip()
    return input_part, reference


# ─────────────────────────────────────────────
# 5. GENERATE DESCRIPTIONS
# ─────────────────────────────────────────────
def generate_descriptions(model, tokenizer, prompts):
    print("\n  Generating descriptions ...")

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1
    )

    inputs, references, generated, gen_times = [], [], [], []

    for i, prompt in enumerate(prompts):
        inp, ref = split_prompt_reference(prompt)
        if not ref:
            continue

        start = time.time()
        out   = gen_pipeline(
            inp,
            max_length=len(tokenizer(inp)["input_ids"]) + MAX_GEN_LENGTH,
            do_sample=True,
            temperature=0.6,
            top_k=40,
            top_p=0.85,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        elapsed = time.time() - start

        full_text = out[0]["generated_text"]
        # strip the input prefix to get only the generated part
        gen_text  = full_text[len(inp):].strip()

        inputs.append(inp)
        references.append(ref)
        generated.append(gen_text)
        gen_times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(prompts)} done ...")

    return inputs, references, generated, gen_times


# ─────────────────────────────────────────────
# 6. BLEU SCORE
# ─────────────────────────────────────────────
def compute_bleu(references, generated):
    """Corpus BLEU-1/2/3/4 with smoothing."""
    nltk.download("punkt",      quiet=True)
    nltk.download("punkt_tab",  quiet=True)

    smooth  = SmoothingFunction().method1
    ref_tok = [[word_tokenize(r.lower())] for r in references]
    gen_tok = [word_tokenize(g.lower())   for g in generated]

    bleu1 = corpus_bleu(ref_tok, gen_tok, weights=(1,0,0,0), smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_tok, gen_tok, weights=(.5,.5,0,0), smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_tok, gen_tok, weights=(.33,.33,.33,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tok, gen_tok, weights=(.25,.25,.25,.25), smoothing_function=smooth)

    return {"BLEU-1": bleu1, "BLEU-2": bleu2, "BLEU-3": bleu3, "BLEU-4": bleu4}


# ─────────────────────────────────────────────
# 7. ROUGE SCORE
# ─────────────────────────────────────────────
def compute_rouge(references, generated):
    """ROUGE-1, ROUGE-2, ROUGE-L (F1)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for ref, gen in zip(references, generated):
            scores = scorer.score(ref, gen)
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rl.append(scores["rougeL"].fmeasure)
        return {
            "ROUGE-1": np.mean(r1),
            "ROUGE-2": np.mean(r2),
            "ROUGE-L": np.mean(rl),
        }
    except ImportError:
        print("  [SKIP] rouge_score not installed → skipping ROUGE")
        return {}


# ─────────────────────────────────────────────
# 8. DIVERSITY  (Distinct-1 / Distinct-2)
# ─────────────────────────────────────────────
def compute_diversity(generated):
    """
    Distinct-1 : unique unigrams / total unigrams
    Distinct-2 : unique bigrams  / total bigrams
    Higher = more diverse outputs.
    """
    all_tokens  = []
    all_bigrams = []

    for text in generated:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
        all_bigrams.extend(zip(tokens, tokens[1:]))

    d1 = len(set(all_tokens))  / len(all_tokens)  if all_tokens  else 0
    d2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0

    return {"Distinct-1": d1, "Distinct-2": d2}


# ─────────────────────────────────────────────
# 9. PRINT RESULTS + SAMPLE OUTPUTS
# ─────────────────────────────────────────────
def print_results(perplexity, bleu, rouge, diversity, gen_times, inputs, references, generated):
    print(f"\n{'='*55}")
    print("  EVALUATION RESULTS")
    print(f"{'='*55}")

    print(f"\n  Perplexity              : {perplexity:.4f}")
    print(f"  Avg Generation Time/s   : {np.mean(gen_times):.3f}s")

    print("\n  ── BLEU ──")
    for k, v in bleu.items():
        print(f"  {k:<10}: {v:.4f}")

    if rouge:
        print("\n  ── ROUGE ──")
        for k, v in rouge.items():
            print(f"  {k:<10}: {v:.4f}")

    print("\n  ── DIVERSITY ──")
    for k, v in diversity.items():
        print(f"  {k:<12}: {v:.4f}")

    # 3 sample outputs
    print(f"\n{'='*55}")
    print("  SAMPLE OUTPUTS (first 3)")
    print(f"{'='*55}")
    for i in range(min(3, len(generated))):
        print(f"\n  ── Sample {i+1} ──")
        # show only the first line of input (product name)
        first_line = [l for l in inputs[i].split('\n') if l.strip()]
        print(f"  Input   : {first_line[1] if len(first_line)>1 else inputs[i][:80]}")
        print(f"  Ref     : {references[i][:200]} ...")
        print(f"  Generated: {generated[i][:200]} ...")

    print(f"\n{'='*55}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    prompts          = load_val_data()

    # Perplexity (uses all loaded prompts)
    perplexity = compute_perplexity(model, tokenizer, prompts)

    # Generation-based metrics
    inputs, references, generated, gen_times = generate_descriptions(model, tokenizer, prompts)

    bleu      = compute_bleu(references, generated)
    rouge     = compute_rouge(references, generated)
    diversity = compute_diversity(generated)

    print_results(perplexity, bleu, rouge, diversity, gen_times, inputs, references, generated)
