from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


# -----------------------------
# 1. Load base model
# -----------------------------
def load_model(model_name="gpt2"):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model


# -----------------------------
# 2. Apply LoRA
# -----------------------------
def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],  # GPT-2 specific
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# -----------------------------
# 3. Training setup
# -----------------------------
def get_training_args(output_dir="./results"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        learning_rate=2e-4,
        warmup_steps=100
    )


# -----------------------------
# 4. Trainer
# -----------------------------
def train_model(model, tokenized_train, tokenized_val, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
    )

    trainer.train()

    return trainer


# -----------------------------
# 5. Save model
# -----------------------------
def save_model(model, tokenizer, path="lora_model"):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    print(f"Model saved to {path} ✓")