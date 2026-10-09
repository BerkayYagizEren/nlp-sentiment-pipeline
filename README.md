# 📊 NLP Sentiment Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Pipeline-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **Natural Language Processing (NLP)** pipeline for text preprocessing, n-gram TF-IDF vectorization, and sentiment classification. Built with clean OOP architecture and comprehensive evaluation metrics.

---

## ⚡ Pipeline Architecture

```mermaid
graph LR
    Raw[Raw Text] --> Clean[Regex Cleaning & Lowercasing]
    Clean --> Token[Tokenization & Stopword Filter]
    Token --> Ngram[N-Gram Generator]
    Ngram --> TFIDF[TF-IDF Vectorizer]
    TFIDF --> Classifier[Naive Bayes / Linear Classifier]
    Classifier --> Metrics[Precision, Recall, F1 Evaluation]
```

---

## 🚀 Features

- **Text Normalization:** Handles punctuation stripping, contraction expansion, and stopword pruning.
- **Configurable N-Gram Extraction:** Unigram, bigram, and trigram feature extraction.
- **TF-IDF Weighting:** Inverse document frequency scaling with sublinear term frequency.
- **Evaluation Suite:** Generates confusion matrices, accuracy, precision, recall, and F1-scores.

---

## 🛠️ Usage

```python
from nlp.pipeline import SentimentPipeline

pipeline = SentimentPipeline()
pipeline.train([
    ("The product was fantastic and exceeded expectations!", "positive"),
    ("Terrible experience, completely broke on day one.", "negative")
])

pred, confidence = pipeline.predict("Absolutely wonderful service!")
print(f"Sentiment: {pred} (Confidence: {confidence:.2f})")
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📄 License
MIT License.
