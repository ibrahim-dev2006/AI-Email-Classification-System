"""Export a dependency-free version of the classifier for serverless deployment.

Trains a logistic-regression pipeline, then writes its vocabulary, IDF weights, and
coefficients to model.json so the serverless function can score with the standard
library alone - no numpy, scipy, or scikit-learn in the deployment bundle.

Run with: python vercel-demo/build_model.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import prepare_dataset  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.preprocessing import TextCleaner  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "api" / "model.json"
MAX_FEATURES = 20_000


def build() -> dict:
    """Train the exportable model and return its serialisable weights."""
    print("Loading dataset...")
    frame = prepare_dataset()
    X = frame["text"].to_numpy()
    y = (frame["label"] == "spam").astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("cleaner", TextCleaner()),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=MAX_FEATURES,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    stop_words="english",
                    strip_accents="unicode",
                ),
            ),
            ("classifier", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )

    print(f"Training on {len(X_train):,} emails...")
    pipeline.fit(X_train, y_train)

    scores = pipeline.predict_proba(X_test)[:, 1]
    result = evaluate(y_test, (scores >= 0.5).astype(int), scores)
    print(
        f"  accuracy {result.accuracy:.4f}  precision {result.precision:.4f}  "
        f"recall {result.recall:.4f}  F1 {result.f1:.4f}  ROC AUC {result.roc_auc:.4f}"
    )

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    # vocabulary_ maps term -> column index; invert it so terms line up with idf and coef.
    terms = [""] * len(vectorizer.vocabulary_)
    for term, index in vectorizer.vocabulary_.items():
        terms[index] = term

    return {
        "terms": terms,
        "idf": [round(float(v), 6) for v in vectorizer.idf_],
        "coef": [round(float(v), 6) for v in classifier.coef_.ravel()],
        "intercept": float(classifier.intercept_[0]),
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "threshold": 0.5,
        "metrics": {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "roc_auc": result.roc_auc,
            "test_rows": result.support,
        },
    }, pipeline, X_test


def verify(model: dict, pipeline: Pipeline, samples) -> None:
    """Confirm the pure-Python scorer reproduces scikit-learn's probabilities."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "api"))
    from scorer import Scorer  # noqa: PLC0415

    scorer = Scorer(model)
    subset = list(samples[:400])
    expected = pipeline.predict_proba(subset)[:, 1]
    actual = np.array([scorer.score(t) for t in subset])

    diff = np.abs(expected - actual)
    print(f"\nVerification against scikit-learn on {len(subset)} emails:")
    print(f"  max absolute difference: {diff.max():.3e}")
    print(f"  mean absolute difference: {diff.mean():.3e}")

    if diff.max() > 1e-6:
        worst = int(diff.argmax())
        raise SystemExit(
            f"FAILED: pure-Python scorer diverges from scikit-learn "
            f"(worst case expected {expected[worst]:.8f}, got {actual[worst]:.8f})"
        )
    print("  PASSED - the serverless scorer matches scikit-learn")


if __name__ == "__main__":
    model, pipeline, X_test = build()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(f"\nWrote {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  {len(model['terms']):,} terms, {len(model['stop_words'])} stop words")

    verify(model, pipeline, X_test)
