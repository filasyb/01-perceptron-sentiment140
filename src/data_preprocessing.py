import re
import pandas as pd
import yaml

def load_config(path="config/local.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

def clean_text(text):
    """Limpieza básica de tweets según config."""
    if config["preprocessing"]["lowercase"]:
        text = text.lower()

    if config["preprocessing"]["remove_urls"]:
        text = re.sub(r"http\S+|www\S+", "", text)

    if config["preprocessing"]["remove_mentions"]:
        text = re.sub(r"@\w+", "", text)

    if config["preprocessing"]["remove_hashtags"]:
        text = re.sub(r"#\w+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)  # solo letras
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess_dataset(df):
    df["text"] = df["text"].apply(clean_text)
    return df
