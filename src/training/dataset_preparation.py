

from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
from datasets import Dataset


# -----------------------------
# 1. Train / Validation Split
# -----------------------------
def split_dataset(df, test_size=0.1, random_state=42):
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train size: {train_df.shape}")
    print(f"Validation size: {val_df.shape}")

    return train_df, val_df


# -----------------------------
# 2. Save prompts (optional)
# -----------------------------
def save_splits(train_df, val_df, train_path="data/processed/train_prompts.csv", val_path="data/processed/val_prompts.csv"):
    train_df[['prompt']].to_csv(train_path, index=False)
    val_df[['prompt']].to_csv(val_path, index=False)

    print("Train/Validation files saved ✓")


# -----------------------------
# 3. Tokenizer
# -----------------------------
def load_tokenizer(model_name="gpt2"):
    print("Loading GPT-2 Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # GPT-2 does not have pad token by default
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# -----------------------------
# 4. Convert pandas → HF Dataset
# -----------------------------
def to_hf_dataset(df):
    return Dataset.from_pandas(df[['prompt']])


# -----------------------------
# 5. Tokenization
# -----------------------------
def tokenize_dataset(dataset, tokenizer, max_length=256):
    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized = dataset.map(tokenize_function, batched=True)

    # Add labels (for causal LM)
    tokenized = tokenized.map(
        lambda x: {'labels': x['input_ids']},
        batched=True
    )

    tokenized.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    return tokenized


# -----------------------------
# 6. Full preparation pipeline
# -----------------------------
def prepare_datasets(df):
    # Split
    train_df, val_df = split_dataset(df)

    # Convert to HuggingFace datasets
    train_dataset = to_hf_dataset(train_df)
    val_dataset = to_hf_dataset(val_df)

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Tokenize
    print("Tokenizing training data...")
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)

    print("Tokenizing validation data...")
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)

    print("Tokenization complete ✓")
    print(f"Sample train input shape: {tokenized_train[0]['input_ids'].shape}")

    return tokenized_train, tokenized_val, tokenizer