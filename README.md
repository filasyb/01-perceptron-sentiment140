# Perceptron Sentiment Classifier — Sentiment140  
Machine Learning Mini-Project · By Juan David Jaramillo

---

## Python Version
This project was developed using **Python 3.12.12**.

## Overview

This project implements a **binary sentiment classifier** using a **Perceptron built from scratch**, trained on a curated subset of the public **Sentiment140** dataset.  
The objective is to demonstrate a **complete, lightweight and fully reproducible ML pipeline**, following a clean project structure inspired by industry standards.

The model classifies tweets as:

- **Positive (1)**
- **Negative (0)**

The implementation avoids frameworks like PyTorch/TensorFlow to emphasize **core ML understanding, explainability, and mathematical grounding**.

---

## Key Features

- Perceptron implemented manually in `model.py`
- Clean NLP preprocessing pipeline (URLs, mentions, hashtags, symbols)
- Text vectorization using Bag-of-Words (CountVectorizer)
- Train/test split with reproducibility from configuration
- Metric reporting (accuracy, confusion matrix, classification report)
- Model artifacts saved for reuse (weights, bias, vocabulary)
- Inference script (`predict.py`) for single-sentence predictions
- Modular structure aligned with professional ML project practices

---

## Project Structure

01-perceptron-sentiment140/
│
├── config/
│ └── local.yaml
│
├── data/
│ ├── 01_raw/
│ ├── 02_preprocessed/
│ └── 03_features/
│
├── notebooks/
│ └── 01_sentiment_perceptron_exploration.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py (optional)
│ ├── model.py
│ ├── train.py
│ └── predict.py
│
├── README.md
└── requirements.txt

This structure ensures **clarity, decoupling, and maintainability**, inspired by templates used in production ML workflows.

---

## Dataset — Sentiment140

Source: https://www.kaggle.com/datasets/kazanova/sentiment140

A 1.6M tweet corpus labeled as:

- `0` → Negative  
- `4` → Positive  

For this project, a balanced sample is generated via `generate_sample.py`:

- 5,000 positive tweets  
- 5,000 negative tweets  
- Preprocessed and saved to `data/02_preprocessed/sentiment140_sample.csv`

---

## Pipeline Overview

### **1. Preprocessing**
- Lowercasing  
- URL removal  
- Mention removal  
- Hashtag removal  
- Non-alphabetic filtering  

Configured in `local.yaml`.

### **2. Vectorization**
- Bag-of-Words (5,000 max features)
- English stopwords removal

### **3. Model Training**
The Perceptron updates weights using Rosenblatt’s rule:

w = w + lr * (y_true - y_pred) * x
b = b + lr * (y_true - y_pred)

Hyperparameters defined in `local.yaml`:

```yaml
learning_rate: 0.01
epochs: 10
random_seed: 42
```

### **4. Evaluation**

Metrics produced include:

- Accuracy
- Confusion Matrix
- Precision, Recall, F1

Saved to:
data/03_features/metrics.txt

### **5. Artifacts Saved**

- weights.npy
- bias.npy
- vocab.npy
These enable consistent inference later.

## How to Run the Project

### **1. Install dependencies**
- pip install -r requirements.txt

### **2. Generate a reduced dataset (optional)**
If you downloaded the full Sentiment140 dataset:
python src/generate_sample.py

### **3. Train the Perceptron**
python src/train.py

### **4. Run inference on new text**
python src/predict.py "I love this movie"

## Results (Typical)

### Using a 10k tweet sample:

- Accuracy: ~62–70%
- Stronger performance on polarized sentiment
- Interpretable weights showing positive/negative terms
- PCA visualization included in the notebook

### Interpretability Highlights

The notebook extracts the top positive and negative words learned by the Perceptron by inspecting the weights:

```
Top Positive: ["love", "great", "awesome", ...]
Top Negative: ["bad", "worst", "hate", ...]
```

This adds transparency to the model and demonstrates understanding of linear classifiers.

### **Roadmap**

- Add batch inference (predict_batch.py)
- Improve preprocessing (stemming, lemmatization)
- Add n-gram features
- Compare with Logistic Regression baseline
- Export as a simple REST API (FastAPI)
- Replace BoW with TF-IDF for improved signa

# Author
Juan David Jaramillo
Data Scientist · ML Engineer
Focused on: ML systems, NLP, fraud detection, AI for contact centers, GCP & Generative AI.

### Final Notes

This project is intentionally simple, interpretable, and structured like a real-world ML module, making it ideal for:
- Demonstrating core ML understanding
- Presenting clean code and reproducible pipelines
- Showcasing the ability to build from scratch
- Serving as the first entry in your ML mini-project portfolio