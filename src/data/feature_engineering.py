import ast

def add_text_features(df):
    df['desc_word_count'] = df['fulldescription_clean'].apply(lambda x: len(x.split()))
    return df

def extract_colors(val):
    try:
        parsed = ast.literal_eval(val) if val != 'unknown' else []
        if isinstance(parsed, list):
            return ', '.join([c.strip().lower() for c in parsed])
    except:
        pass
    return 'unknown'

def add_color_features(df):
    df['colors_extracted'] = df['colorname'].apply(extract_colors)
    return df

def filter_short_descriptions(df, min_words=20):
    before = df.shape[0]
    df = df[df['desc_word_count'] >= min_words].reset_index(drop=True)

    print(f"Removed {before - df.shape[0]} short entries | Remaining: {df.shape[0]}")

    return df

def save_features(df, path="data/processed/cleaned_features.csv"):
    output_cols = [
        'name_clean',
        'fulldescription_clean',
        'colors_extracted',
        'desc_word_count'
    ]

    df[output_cols].to_csv(path, index=False)
    print(f"Saved: {path} ✓")

def feature_pipeline(df):
    df = add_text_features(df)
    df = add_color_features(df)
    df = filter_short_descriptions(df)
    return df

