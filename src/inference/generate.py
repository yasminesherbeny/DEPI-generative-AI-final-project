import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_generator(adapter_path="lora_model", base_model="gpt2"):
    print(f"[inference] Loading tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(adapter_path)
    except Exception:
        tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[inference] Loading base model...")
    base = GPT2LMHeadModel.from_pretrained(base_model)

    print(f"[inference] Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    model.to(DEVICE)

    print(f"[inference] Ready on {DEVICE}")
    return model, tokenizer


@torch.no_grad()
def generate_text(generator, prompt, max_new_tokens=120):
    model, tokenizer = generator
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    return generated.strip()