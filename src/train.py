"""Training pipeline: model selection, threshold tuning, evaluation, and persistence."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import DATA_DIR, LABELS, METRICS_PATH, MODEL_DIR, MODEL_PATH, TRAINING, TrainingConfig
from .dataset import prepare_dataset, write_sample
from .evaluation import best_threshold, evaluate
from .exceptions import DatasetError
from .logger import get_logger
from .predictor import extract_coefficients
from .preprocessing import TextCleaner

logger = get_logger(__name__)

HOLDOUT_PREDICTIONS_CSV = DATA_DIR / "holdout_predictions.csv"


def build_vectorizer(config: TrainingConfig) -> TfidfVectorizer:
    """Create the TF-IDF vectoriser shared by every candidate model."""
    return TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        stop_words="english",
        strip_accents="unicode",
    )


def build_pipeline(name: str, config: TrainingConfig) -> Pipeline:
    """Assemble a named candidate pipeline from cleaning through classification."""
    classifiers = {
        "logistic_regression": LogisticRegression(
            C=4.0, max_iter=2000, class_weight="balanced", random_state=config.random_state
        ),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", random_state=config.random_state),
            cv=3,
            method="sigmoid",
        ),
        "naive_bayes": MultinomialNB(alpha=0.2),
    }
    if name not in classifiers:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(classifiers)}.")

    return Pipeline(
        [
            ("cleaner", TextCleaner()),
            ("tfidf", build_vectorizer(config)),
            ("classifier", classifiers[name]),
        ]
    )


def _scores(pipeline: Pipeline, X) -> np.ndarray:  # noqa: N803 - sklearn API
    """Return spam probabilities, falling back to a squashed decision function."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    margins = pipeline.decision_function(X)
    return 1.0 / (1.0 + np.exp(-margins))


def top_indicative_terms(pipeline: Pipeline, n: int = 25) -> dict[str, list[dict]]:
    """Extract the terms pushing hardest toward each class, when the model exposes them."""
    names = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    weights = extract_coefficients(pipeline.named_steps["classifier"])
    if weights is None:
        return {"spam": [], "ham": []}

    order = np.argsort(weights)
    return {
        "spam": [
            {"term": str(names[i]), "weight": float(weights[i])} for i in order[::-1][:n]
        ],
        "ham": [{"term": str(names[i]), "weight": float(weights[i])} for i in order[:n]],
    }


def train(
    config: TrainingConfig = TRAINING,
    *,
    force_download: bool = False,
    models: tuple[str, ...] | None = None,
    skip_cv: bool = False,
) -> dict:
    """Run the full training workflow and persist the winning model and its metrics."""
    started = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    frame = prepare_dataset(force_download=force_download, max_rows=config.max_rows)
    if len(frame) < 200:
        raise DatasetError(f"Only {len(frame)} rows available; need at least 200 to train.")

    X = frame["text"].to_numpy()
    y = (frame["label"] == "spam").astype(int).to_numpy()
    logger.info("Dataset ready: %s rows (%.1f%% spam)", len(X), y.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    candidates = models or config.candidate_models
    comparison: list[dict] = []
    fitted: dict[str, Pipeline] = {}

    for name in candidates:
        logger.info("Training candidate: %s", name)
        pipeline = build_pipeline(name, config)
        candidate_started = time.perf_counter()

        cv_mean = cv_std = float("nan")
        if not skip_cv:
            folds = StratifiedKFold(
                n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
            )
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="f1", n_jobs=-1)
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
            logger.info("  cross-validated F1: %.4f (+/- %.4f)", cv_mean, cv_std)

        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline

        scores = _scores(pipeline, X_test)
        result = evaluate(y_test, (scores >= 0.5).astype(int), scores, split="test")
        comparison.append(
            {
                "model": name,
                "cv_f1_mean": cv_mean,
                "cv_f1_std": cv_std,
                "test_f1": result.f1,
                "test_accuracy": result.accuracy,
                "test_precision": result.precision,
                "test_recall": result.recall,
                "test_roc_auc": result.roc_auc,
                "fit_seconds": round(time.perf_counter() - candidate_started, 2),
            }
        )

    selection_key = "cv_f1_mean" if not skip_cv else "test_f1"
    best = max(comparison, key=lambda row: (row[selection_key], row["test_f1"]))
    best_name = best["model"]
    best_pipeline = fitted[best_name]
    logger.info("Selected model: %s", best_name)

    test_scores = _scores(best_pipeline, X_test)
    default_result = evaluate(y_test, (test_scores >= 0.5).astype(int), test_scores, split="test")
    tuned_threshold = best_threshold(default_result.threshold_sweep, objective=config.threshold_objective)
    tuned_result = evaluate(
        y_test, (test_scores >= tuned_threshold).astype(int), test_scores, split="test_tuned"
    )

    train_scores = _scores(best_pipeline, X_train)
    train_result = evaluate(
        y_train, (train_scores >= tuned_threshold).astype(int), train_scores, split="train"
    )

    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "threshold": tuned_threshold,
        "labels": list(LABELS),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    logger.info("Saved model to %s (%.1f KB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1024)

    pd.DataFrame(
        {
            "text": [t[:500] for t in X_test],
            "true_label": np.where(y_test == 1, "spam", "ham"),
            "spam_probability": test_scores,
            "predicted_label": np.where(test_scores >= tuned_threshold, "spam", "ham"),
        }
    ).to_csv(HOLDOUT_PREDICTIONS_CSV, index=False)

    write_sample(frame)

    metrics = {
        "model_name": best_name,
        "threshold": tuned_threshold,
        "trained_at": artifact["trained_at"],
        "training_seconds": round(time.perf_counter() - started, 2),
        "dataset": {
            "source": str(frame["source"].iloc[0]),
            "total_rows": int(len(frame)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "spam_ratio": float(y.mean()),
            "vocabulary_size": int(len(best_pipeline.named_steps["tfidf"].vocabulary_)),
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "model_comparison": comparison,
        "train": train_result.to_dict(),
        "test_default_threshold": default_result.to_dict(),
        "test": tuned_result.to_dict(),
        "top_terms": top_indicative_terms(best_pipeline),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", METRICS_PATH)
    logger.info(
        "Done in %.1fs - accuracy %.4f, F1 %.4f, ROC AUC %.4f",
        metrics["training_seconds"],
        tuned_result.accuracy,
        tuned_result.f1,
        tuned_result.roc_auc,
    )
    return metrics


def main() -> None:
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(description="Train the email spam classifier.")
    parser.add_argument("--force-download", action="store_true", help="Re-download the dataset.")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap the number of rows used.")
    parser.add_argument("--cv-folds", type=int, default=TRAINING.cv_folds, help="Cross-validation folds.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation (faster).")
    parser.add_argument(
        "--models", nargs="+", default=None, choices=list(TRAINING.candidate_models),
        help="Restrict which candidate models are trained.",
    )
    args = parser.parse_args()

    config = TrainingConfig(max_rows=args.max_rows, cv_folds=args.cv_folds)
    train(
        config,
        force_download=args.force_download,
        models=tuple(args.models) if args.models else None,
        skip_cv=args.skip_cv,
    )


if __name__ == "__main__":
    main()
