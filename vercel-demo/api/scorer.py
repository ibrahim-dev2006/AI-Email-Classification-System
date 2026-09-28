"""Dependency-free reimplementation of the trained TF-IDF + logistic-regression pipeline.

Mirrors scikit-learn's TfidfVectorizer (sublinear TF, smoothed IDF, L2 norm, 1-2 grams,
English stop words) and LogisticRegression scoring using only the standard library.
`build_model.py` asserts this matches scikit-learn to within 1e-6 before deployment.
"""

from __future__ import annotations

import math
import re
import unicodedata

# Cleaning must stay byte-identical to src/preprocessing.clean_text, which produced
# the text the exported model was trained on. Duplicated so this bundle has no imports.
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_HTML_ENTITY = re.compile(r"&[a-z]{2,8};|&#\d{1,5};", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY = re.compile(
    r"[$£€¥]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|gbp|eur|dollars?|pounds?)\b",
    re.IGNORECASE,
)
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_NON_WORD = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")

# scikit-learn's default token_pattern.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def clean_text(text: str) -> str:
    """Normalise raw email text exactly as the training pipeline did."""
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = _HTML_ENTITY.sub(" ", cleaned)
    cleaned = _URL.sub(" urltoken ", cleaned)
    cleaned = _EMAIL.sub(" emailtoken ", cleaned)
    cleaned = _MONEY.sub(" moneytoken ", cleaned)
    cleaned = _PHONE.sub(" phonetoken ", cleaned)
    cleaned = _NUMBER.sub(" numtoken ", cleaned)
    cleaned = _NON_WORD.sub(" ", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


class Scorer:
    """Scores email text against the exported model weights."""

    def __init__(self, model: dict):
        self.index = {term: i for i, term in enumerate(model["terms"])}
        self.idf = model["idf"]
        self.coef = model["coef"]
        self.intercept = model["intercept"]
        self.stop_words = frozenset(model["stop_words"])
        self.threshold = model.get("threshold", 0.5)
        self.metrics = model.get("metrics", {})

    def _ngrams(self, text: str) -> list[str]:
        """Tokenise, drop stop words, then emit unigrams and bigrams."""
        tokens = [t for t in _TOKEN.findall(text) if t not in self.stop_words]
        grams = list(tokens)
        for i in range(len(tokens) - 1):
            grams.append(tokens[i] + " " + tokens[i + 1])
        return grams

    def vectorise(self, text: str) -> dict[int, float]:
        """Build the L2-normalised TF-IDF vector as {column index: weight}."""
        counts: dict[int, int] = {}
        for gram in self._ngrams(clean_text(text)):
            column = self.index.get(gram)
            if column is not None:
                counts[column] = counts.get(column, 0) + 1

        # sublinear_tf=True -> 1 + ln(count), then multiply by the IDF weight.
        vector = {c: (1.0 + math.log(n)) * self.idf[c] for c, n in counts.items()}

        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            vector = {c: v / norm for c, v in vector.items()}
        return vector

    def score(self, text: str) -> float:
        """Return the probability that the text is spam."""
        vector = self.vectorise(text)
        z = self.intercept + sum(weight * self.coef[c] for c, weight in vector.items())
        return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0

    def explain(self, text: str, n: int = 8) -> list[dict]:
        """Return the terms that contributed most to this score."""
        vector = self.vectorise(text)
        contributions = [
            {
                "term": self.terms_by_index(c),
                "contribution": round(weight * self.coef[c], 5),
            }
            for c, weight in vector.items()
        ]
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        for item in contributions[:n]:
            item["direction"] = "spam" if item["contribution"] > 0 else "legitimate"
        return contributions[:n]

    def terms_by_index(self, column: int) -> str:
        """Reverse-lookup a term from its column index."""
        if not hasattr(self, "_terms"):
            self._terms = [""] * len(self.index)
            for term, i in self.index.items():
                self._terms[i] = term
        return self._terms[column]

    def classify(self, text: str) -> dict:
        """Score the text and return the full verdict payload."""
        probability = self.score(text)
        if probability < 0.35:
            decision = "deliver"
        elif probability > 0.75:
            decision = "quarantine"
        else:
            decision = "review"

        return {
            "label": "spam" if probability >= self.threshold else "ham",
            "spam_probability": round(probability, 6),
            "confidence": round(abs(probability - 0.5) * 2, 6),
            "decision": decision,
            "threshold": self.threshold,
            "top_terms": self.explain(text),
        }
