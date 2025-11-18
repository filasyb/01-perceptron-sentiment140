import numpy as np
import yaml
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from data_preprocessing import clean_text, load_config
from model import Perceptron

def load_artifacts(features_dir):
    """
    Carga vocabulario, pesos y bias del modelo entrenado.
    """
    vocab = np.load(f"{features_dir}/vocab.npy", allow_pickle=True)
    weights = np.load(f"{features_dir}/weights.npy")
    bias = np.load(f"{features_dir}/bias.npy")
    return vocab, weights, bias

def build_vectorizer(vocab):
    """
    Construye un CountVectorizer usando un vocabulario ya entrenado.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    return vectorizer

def predict_text(text, vectorizer, weights, bias):
    """
    Realiza predicción binaria para un texto nuevo.
    """
    processed = clean_text(text)
    X = vectorizer.transform([processed]).toarray()
    linear = np.dot(X, weights) + bias
    return 1 if linear >= 0 else 0

if __name__ == "__main__":
    config = load_config()
    features_dir = config["paths"]["features_dir"]

    vocab, weights, bias = load_artifacts(features_dir)
    vectorizer = build_vectorizer(vocab)

    user_text = " ".join(sys.argv[1:])
    prediction = predict_text(user_text, vectorizer, weights, bias)

    label = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(label)
