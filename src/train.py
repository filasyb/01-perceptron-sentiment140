import pandas as pd
import numpy as np
import yaml
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model import Perceptron
from data_preprocessing import preprocess_dataset, load_config


config = load_config()
data_path = config["paths"]["preprocessed_data"]
features_dir = config["paths"]["features_dir"]

os.makedirs(features_dir, exist_ok=True)

print(f"\n=== ENTRENAMIENTO PERCEPTRÓN SENTIMENT140 ===")
print(f"Cargando dataset preprocesado desde: {data_path}\n")


df = pd.read_csv(data_path)


X_text = df["text"].values
y = df["target"].values


print("Vectorizando texto con CountVectorizer...\n")

X_text = df["text"].fillna("").astype(str).values

vectorizer = CountVectorizer(
    max_features=5000,        
    stop_words="english"      
)

X = vectorizer.fit_transform(X_text).toarray()


np.save(os.path.join(features_dir, "vocab.npy"), vectorizer.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=config["model"]["random_seed"],
    stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")


print("Entrenando Perceptrón...\n")

model = Perceptron()
model.fit(X_train, y_train)


np.save(os.path.join(features_dir, "weights.npy"), model.w)
np.save(os.path.join(features_dir, "bias.npy"), model.b)

print("Pesos del modelo guardados:")
print(f"- {features_dir}/weights.npy")
print(f"- {features_dir}/bias.npy\n")


y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== RESULTADOS DEL MODELO ===")
print(f"Accuracy: {acc:.4f}\n")

print("Matriz de confusión:")
print(cm, "\n")

print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))


metrics_path = os.path.join(features_dir, "metrics.txt")

with open(metrics_path, "w") as f:
    f.write("=== METRICAS PERCEPTRON ===\n\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Matriz de confusión:\n")
    f.write(str(cm) + "\n\n")
    f.write("Reporte de clasificación:\n")
    f.write(classification_report(y_test, y_pred))

print(f"Métricas guardadas en: {metrics_path}\n")
print("Entrenamiento completado.\n")
