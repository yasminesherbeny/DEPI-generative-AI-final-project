import numpy as np
import pandas as pd
import re
import ast
import nltk
from nltk.tokenize import word_tokenize

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('punkt_tab')

def basic_cleaning(df):
    df = df[['name', 'fulldescription', 'colorname']]

    df.replace("[]", np.nan, inplace=True)
    df = df.dropna(subset=['fulldescription'])

    df['colorname'] = df['colorname'].fillna('unknown')

    return df

def parse_list_column(val):
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return ' '.join([str(i) for i in parsed])
        return str(parsed)
    except:
        return str(val)
    
def parse_columns(df):
    df['fulldescription_clean'] = df['fulldescription'].apply(parse_list_column)
    df['colorname_clean']       = df['colorname'].apply(parse_list_column)
    df['name_clean']            = df['name'].str.lower().str.strip()

    return df

def clean_text(text):
    text = re.sub(r'#[\w-]+\s*\{[^}]*\}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def apply_text_cleaning(df):
    df['fulldescription_clean'] = df['fulldescription_clean'].apply(clean_text)
    df['name_clean']            = df['name_clean'].apply(clean_text)

    return df

light_stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'its', 'this', 'that'}
def remove_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join([w for w in tokens if w not in light_stopwords])

def apply_stopword_removal(df):
    df['fulldescription_clean'] = df['fulldescription_clean'].apply(remove_stopwords)
    return df

def finalize_dataframe(df):
    df.drop_duplicates(subset=['fulldescription_clean'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Rows after cleaning: {df.shape[0]}")

    return df

def full_preprocessing_pipeline(df):
    download_nltk_resources()
    df = basic_cleaning(df)
    df = parse_columns(df)
    df = apply_text_cleaning(df)
    df = apply_stopword_removal(df)
    df = finalize_dataframe(df)

    return df