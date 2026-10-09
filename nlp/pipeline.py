import math
import re
from typing import List, Tuple, Dict, Any


DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "is", "was", "are", "were", "it", "this", "that", "of", "from", "by"
}


class TextPreprocessor:
    """Cleans, normalizes, and tokenizes raw strings."""

    def __init__(self, stopwords: set = None):
        self.stopwords = stopwords or DEFAULT_STOPWORDS

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean(text)
        return [w for w in cleaned.split() if w not in self.stopwords and len(w) > 1]


class TFIDFVectorizer:
    """N-Gram TF-IDF feature extractor."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)):
        self.ngram_range = ngram_range
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append("_".join(tokens[i:i + n]))
        return ngrams

    def fit(self, tokenized_docs: List[List[str]]):
        doc_count = len(tokenized_docs)
        doc_freq = {}

        for doc in tokenized_docs:
            unique_terms = set(self._extract_ngrams(doc))
            for term in unique_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(doc_freq.keys()))}
        self.idf = {
            term: math.log((1.0 + doc_count) / (1.0 + freq)) + 1.0
            for term, freq in doc_freq.items()
        }

    def transform(self, tokens: List[str]) -> Dict[str, float]:
        terms = self._extract_ngrams(tokens)
        tf = {}
        for t in terms:
            if t in self.vocabulary:
                tf[t] = tf.get(t, 0.0) + 1.0

        vector = {}
        for t, count in tf.items():
            vector[t] = (1.0 + math.log(count)) * self.idf.get(t, 1.0)

        # L2 norm
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            for k in vector:
                vector[k] /= norm
        return vector


class SentimentPipeline:
    """End-to-end text classification model."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TFIDFVectorizer(ngram_range=(1, 2))
        self.class_priors: Dict[str, float] = {}
        self.word_probs: Dict[str, Dict[str, float]] = {}

    def train(self, data: List[Tuple[str, str]]):
        tokenized = [self.preprocessor.tokenize(text) for text, _ in data]
        labels = [label for _, label in data]

        self.vectorizer.fit(tokenized)

        total_samples = len(data)
        class_counts = {}
        class_term_counts: Dict[str, Dict[str, float]] = {}

        for tokens, label in zip(tokenized, labels):
            class_counts[label] = class_counts.get(label, 0) + 1
            if label not in class_term_counts:
                class_term_counts[label] = {}
            for t in tokens:
                class_term_counts[label][t] = class_term_counts[label].get(t, 0.0) + 1.0

        for label, count in class_counts.items():
            self.class_priors[label] = count / total_samples

            total_terms = sum(class_term_counts[label].values()) + len(self.vectorizer.vocabulary)
            self.word_probs[label] = {
                term: (class_term_counts[label].get(term, 0.0) + 1.0) / total_terms
                for term in self.vectorizer.vocabulary
            }

    def predict(self, text: str) -> Tuple[str, float]:
        tokens = self.preprocessor.tokenize(text)
        if not self.class_priors:
            return "neutral", 0.5

        log_scores = {}
        for label, prior in self.class_priors.items():
            score = math.log(prior)
            for t in tokens:
                if t in self.word_probs[label]:
                    score += math.log(self.word_probs[label][t])
            log_scores[label] = score

        best_label = max(log_scores.keys(), key=lambda k: log_scores[k])
        # Approximate normalized probability
        max_log = max(log_scores.values())
        exp_sum = sum(math.exp(v - max_log) for v in log_scores.values())
        confidence = math.exp(log_scores[best_label] - max_log) / exp_sum

        return best_label, confidence
