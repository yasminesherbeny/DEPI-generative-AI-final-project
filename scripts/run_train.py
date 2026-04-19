# scripts/run_train.py

from src.data.load_data import load_dataset
from src.data.preprocess import full_preprocessing_pipeline
from src.data.feature_engineering import feature_pipeline
from src.nlp.prompt_builder import prompt_pipeline
from src.training.dataset_preparation import prepare_datasets
from src.training.training_lora import (
    load_model,
    apply_lora,
    get_training_args,
    train_model,
    save_model
)
from src.inference.generate import load_generator, generate_text


def run():
    # -------------------------
    # 1. Data pipeline
    # -------------------------
    df = load_dataset()
    df = full_preprocessing_pipeline(df)
    df = feature_pipeline(df)
    df = prompt_pipeline(df)

    # -------------------------
    # 2. Dataset preparation
    # -------------------------
    train_data, val_data, tokenizer = prepare_datasets(df)

    # -------------------------
    # 3. Model + LoRA
    # -------------------------
    model = load_model()
    model = apply_lora(model)

    # -------------------------
    # 4. Training
    # -------------------------
    training_args = get_training_args()
    trainer = train_model(model, train_data, val_data, training_args)

    # -------------------------
    # 5. Save model
    # -------------------------
    save_model(model, tokenizer)

    # -------------------------
    # 6. Inference test
    # -------------------------
    generator = load_generator()

    prompt = """Product: Casual Cotton T-Shirt
Color: White
Description:"""

    output = generate_text(generator, prompt)
    print("\nGenerated Output:\n", output)


if __name__ == "__main__":
    run()