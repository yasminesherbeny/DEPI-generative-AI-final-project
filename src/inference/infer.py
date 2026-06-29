"""
src/inference/infer.py
======================
Model inference for the fine-tuned LoRA GPT-2 Product Description Generator.

This is the inference counterpart to the training pipeline in
`src/training/training_lora.py`. Training saves a *LoRA adapter* (not a full
model) to `lora_model/`, so inference must load the base GPT-2 model first and
then attach the adapter on top of it with PEFT — exactly how `evaluate_model.py`
does it. Loading the adapter folder directly through a plain `pipeline(...)`
(as the older `generate.py` does) is fragile and can silently fall back to an
untrained model.

The prompt format here mirrors `src/nlp/prompt_builder.build_prompt`, which is
what the model was actually trained on:

    Product: <name>
        Color: <color>
        Description: <description>

At inference we feed everything up to and including "Description: " and let the
model generate the description.

Usage
-----
As a script (single product):
    python -m src.inference.infer --name "Casual Cotton T-Shirt" --color "White"

Interactive mode (keeps the model loaded, ask repeatedly):
    python -m src.inference.infer --interactive

As a module:
    from src.inference.infer import ProductDescriptionGenerator
    gen = ProductDescriptionGenerator()            # loads model once
    text = gen.generate("Leather Wallet", "Brown")

Requirements (already in requirements.txt):
    transformers, torch, peft
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel


# ─────────────────────────────────────────────
# CONFIG  –  adjust if your paths differ
# ─────────────────────────────────────────────
DEFAULT_ADAPTER_PATH = "lora_model"   # where save_model() writes the LoRA adapter
DEFAULT_BASE_MODEL = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Generation defaults
# These mirror the decoding settings used in evaluate_model.py so that
# inference behaves consistently with how the model was evaluated.
# ─────────────────────────────────────────────
@dataclass
class GenerationConfig:
    max_new_tokens: int = 120
    do_sample: bool = True
    temperature: float = 0.6
    top_k: int = 40
    top_p: float = 0.85
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3
    num_return_sequences: int = 1


# ─────────────────────────────────────────────
# Prompt construction
# Mirrors src/nlp/prompt_builder.build_prompt (including the 4-space indentation
# on the Color / Description lines) so the input distribution matches training.
# ─────────────────────────────────────────────
def build_inference_prompt(product_name: str, color: str) -> str:
    """Build the input prompt the model expects, ending at 'Description: '."""
    return (
        f"Product: {product_name}\n"
        f"    Color: {color}\n"
        f"    Description: "
    )


def _clean_generated_text(text: str) -> str:
    """
    Tidy up the raw generated continuation:
      - stop at the start of a new 'Product:' block (model sometimes rambles on)
      - drop the trailing indentation artifact from the training format
      - collapse leftover whitespace
    """
    # If the model starts a brand-new product entry, cut it off there.
    for marker in ("\nProduct:", "\n    Product:", "Product:"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break

    # Remove the trailing 4-space line that the training format appended.
    text = text.strip()
    return text


# ─────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────
@dataclass
class ProductDescriptionGenerator:
    """
    Loads the base GPT-2 model + LoRA adapter once, then generates descriptions.

    Example
    -------
    >>> gen = ProductDescriptionGenerator()
    >>> gen.generate("Leather Wallet", "Brown")
    'A genuine brown leather wallet ...'
    """

    adapter_path: str = DEFAULT_ADAPTER_PATH
    base_model: str = DEFAULT_BASE_MODEL
    device: str = DEVICE
    gen_config: GenerationConfig = field(default_factory=GenerationConfig)

    model: GPT2LMHeadModel = field(init=False, repr=False, default=None)
    tokenizer: GPT2Tokenizer = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._load()

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        if not os.path.isdir(self.adapter_path):
            raise FileNotFoundError(
                f"Adapter folder not found: '{self.adapter_path}'. "
                f"Train the model first (scripts/run_train.py) or pass "
                f"--adapter-path pointing to a folder with adapter_config.json."
            )

        print(f"[infer] Loading tokenizer from '{self.adapter_path}' ...")
        # The tokenizer was saved alongside the adapter by save_model().
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(self.adapter_path)
        except Exception:
            # Fall back to the base tokenizer if not saved with the adapter.
            tokenizer = GPT2Tokenizer.from_pretrained(self.base_model)
        # GPT-2 has no pad token by default — match training/eval setup.
        tokenizer.pad_token = tokenizer.eos_token

        print(f"[infer] Loading base model '{self.base_model}' ...")
        base = GPT2LMHeadModel.from_pretrained(self.base_model)

        print(f"[infer] Attaching LoRA adapter from '{self.adapter_path}' ...")
        model = PeftModel.from_pretrained(base, self.adapter_path)
        model.eval()
        model.to(self.device)

        self.model = model
        self.tokenizer = tokenizer
        print(f"[infer] Ready on device: {self.device}")

    # -- generation --------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        product_name: str,
        color: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        return_full_text: bool = False,
    ) -> str:
        """
        Generate a product description for a given name + color.

        Parameters
        ----------
        product_name, color : str
            Inputs that get formatted into the training-style prompt.
        max_new_tokens, temperature :
            Optional per-call overrides of the defaults in GenerationConfig.
        seed :
            Optional int for reproducible sampling.
        return_full_text :
            If True, return prompt + generation. If False (default), return only
            the generated description.
        """
        if seed is not None:
            torch.manual_seed(seed)

        cfg = self.gen_config
        prompt = build_inference_prompt(product_name, color)

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens or cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=temperature or cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            num_return_sequences=cfg.num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if return_full_text:
            return full_text

        # Strip the prompt prefix to keep only the generated description.
        generated = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        return _clean_generated_text(generated)

    def generate_batch(self, items: list[tuple[str, str]], **kwargs) -> list[dict]:
        """
        Generate for a list of (product_name, color) tuples.
        Returns a list of dicts: {"name", "color", "description"}.
        """
        results = []
        for name, color in items:
            desc = self.generate(name, color, **kwargs)
            results.append({"name": name, "color": color, "description": desc})
        return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _run_interactive(gen: ProductDescriptionGenerator) -> None:
    print("\nInteractive mode — type a product name and color (Ctrl-C to quit).\n")
    try:
        while True:
            name = input("Product name : ").strip()
            if not name:
                continue
            color = input("Color        : ").strip()
            print("\nGenerating ...\n")
            desc = gen.generate(name, color)
            print(f"Description:\n{desc}\n")
            print("-" * 55)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate product descriptions with the fine-tuned LoRA GPT-2 model."
    )
    parser.add_argument("--name", type=str, help="Product name.")
    parser.add_argument("--color", type=str, default="", help="Product color.")
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH,
                        help=f"Path to the LoRA adapter folder (default: {DEFAULT_ADAPTER_PATH}).")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"Base model name (default: {DEFAULT_BASE_MODEL}).")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override the number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override sampling temperature.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--interactive", action="store_true",
                        help="Run an interactive prompt loop.")
    args = parser.parse_args()

    gen = ProductDescriptionGenerator(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
    )

    if args.interactive:
        _run_interactive(gen)
        return

    if not args.name:
        # Fall back to a demo example matching run_train.py's smoke test.
        args.name = "Casual Cotton T-Shirt"
        args.color = args.color or "White"
        print("[infer] No --name given; using demo example.")

    description = gen.generate(
        args.name,
        args.color,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    print("\n" + "=" * 55)
    print(f"  Product : {args.name}")
    print(f"  Color   : {args.color}")
    print("=" * 55)
    print(f"\n{description}\n")


if __name__ == "__main__":
    main()
