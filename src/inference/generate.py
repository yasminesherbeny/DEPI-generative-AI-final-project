"""
src/inference/generate.py
=========================
Lightweight text-generation helpers used by scripts/run_train.py for a quick
post-training smoke test.

NOTE: training saves a *LoRA adapter* (not a full model) to `lora_model/`.
Pointing a plain `pipeline("text-generation", model="lora_model")` at that
folder is fragile — it can silently load an untrained base model. So here we
load the base GPT-2 model first and attach the LoRA adapter with PEFT, the same
correct approach used in `evaluate_model.py` and `src/inference/infer.py`.

The public functions keep their original signatures so existing imports
(`from src.inference.generate import load_generator, generate_text`) keep working.

For richer inference (CLI, batch, interactive, clean output post-processing),
see `src/inference/infer.py`.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from peft import PeftModel


def load_generator(model_path="lora_model", base_model="gpt2"):
    """
    Build a text-generation pipeline backed by base GPT-2 + the LoRA adapter.

    Parameters
    ----------
    model_path : str
        Folder containing the saved LoRA adapter (adapter_config.json + weights)
        and tokenizer files. Defaults to "lora_model".
    base_model : str
        Base model the adapter was trained on. Defaults to "gpt2".
    """
    device = 0 if torch.cuda.is_available() else -1

    # Tokenizer (saved alongside the adapter by save_model); fall back to base.
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Base model + LoRA adapter on top.
    base = GPT2LMHeadModel.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return generator


def generate_text(generator, prompt, max_new_tokens=120):
    """
    Generate text from a prompt using the pipeline returned by load_generator().

    Uses decoding settings consistent with evaluate_model.py / infer.py.
    """
    tokenizer = generator.tokenizer
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return output[0]["generated_text"]
