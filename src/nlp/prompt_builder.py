# src/nlp/prompt_builder.py

def build_prompt(row):
    """
    Build a structured prompt for GPT-style training from a dataframe row.
    """
    return f"""Product: {row['name_clean']}
    Color: {row['colors_extracted']}
    Description: {row['fulldescription_clean']}
    """


def generate_prompts(df):
    """
    Apply prompt construction to the entire dataframe.
    """
    df["prompt"] = df.apply(build_prompt, axis=1)
    return df


def add_prompt_length(df):
    """
    Add a column for prompt length (in words).
    Useful for analysis and filtering.
    """
    df["prompt_length"] = df["prompt"].apply(lambda x: len(x.split()))
    return df


def prompt_pipeline(df):
    """
    Full prompt generation pipeline.
    """
    df = generate_prompts(df)
    df = add_prompt_length(df)

    # Basic stats (optional but useful)
    print("Prompt length stats:")
    print(df["prompt_length"].describe())

    return df