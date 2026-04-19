from transformers import pipeline


def load_generator(model_path="lora_model"):
    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path
    )
    return generator


def generate_text(generator, prompt, max_length=120):
    output = generator(prompt, max_length=max_length)
    return output[0]["generated_text"]