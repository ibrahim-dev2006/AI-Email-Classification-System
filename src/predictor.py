"""Model loading and inference with per-prediction explanations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import (
    LABEL_DISPLAY,
    MODEL_PATH,
    REVIEW_BAND_HIGH,
    REVIEW_BAND_LOW,
)
from .exceptions import ModelLoadError, ModelNotTrainedError, PredictionError
from .logger import get_logger
from .preprocessing import validate_email_text

logger = get_logger(__name__)

DECISIONS = {
    "deliver": "Deliver to inbox",
    "review": "Route to human review",
    "quarantine": "Quarantine as spam",
}


@dataclass
class PredictionResult:
    """One classified email with its score, routing decision, and evidence."""

    label: str
    spam_probability: float
    confidence: float
    decision: str
    threshold: float
    model_name: str
    latency_ms: float
    top_terms: list[dict] = field(default_factory=list)
    source: str = "local_model"

    @property
    def display_label(self) -> str:
        return LABEL_DISPLAY.get(self.label, self.label)

    @property
    def decision_text(self) -> str:
        return DECISIONS.get(self.decision, self.decision)

    @property
    def is_spam(self) -> bool:
        return self.label == "spam"


def decide(probability: float) -> str:
    """Map a spam probability onto a routing decision with a human-review band."""
    if probability < REVIEW_BAND_LOW:
        return "deliver"
    if probability > REVIEW_BAND_HIGH:
        return "quarantine"
    return "review"


def extract_coefficients(classifier) -> np.ndarray | None:
    """Recover per-feature weights across the classifier types we train."""
    if hasattr(classifier, "coef_"):
        return np.asarray(classifier.coef_).ravel()
    if hasattr(classifier, "feature_log_prob_"):
        return classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

    # CalibratedClassifierCV hides the linear model behind one wrapper per CV fold.
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if calibrated:
        weights = [
            np.asarray(inner.coef_).ravel()
            for c in calibrated
            if (inner := getattr(c, "estimator", None)) is not None and hasattr(inner, "coef_")
        ]
        if weights:
            return np.mean(weights, axis=0)
    return None


class EmailPredictor:
    """Wraps the trained pipeline with validation, routing, and explanations."""

    def __init__(self, artifact: dict):
        self.pipeline: Pipeline = artifact["pipeline"]
        self.model_name: str = artifact.get("model_name", "unknown")
        self.threshold: float = float(artifact.get("threshold", 0.5))
        self.trained_at: str = artifact.get("trained_at", "unknown")
        self._vectorizer = self.pipeline.named_steps["tfidf"]
        self._coef = extract_coefficients(self.pipeline.named_steps["classifier"])
        self._feature_names = np.asarray(self._vectorizer.get_feature_names_out())

    def _scores(self, texts) -> np.ndarray:
        classifier = self.pipeline.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            return self.pipeline.predict_proba(texts)[:, 1]
        return 1.0 / (1.0 + np.exp(-self.pipeline.decision_function(texts)))

    def explain(self, text: str, n: int = 8) -> list[dict]:
        """Return the terms that contributed most to this specific score."""
        if self._coef is None:
            return []
        try:
            cleaned = self.pipeline.named_steps["cleaner"].transform([text])
            vector = self._vectorizer.transform(cleaned)
            _, indices = vector.nonzero()
            if indices.size == 0:
                return []

            contributions = vector.toarray().ravel()[indices] * self._coef[indices]
            order = np.argsort(np.abs(contributions))[::-1][:n]
            return [
                {
                    "term": str(self._feature_names[indices[i]]),
                    "contribution": float(contributions[i]),
                    "direction": "spam" if contributions[i] > 0 else "legitimate",
                }
                for i in order
            ]
        except Exception as exc:  # noqa: BLE001 - explanations must never break a prediction
            logger.warning("Explanation failed: %s", exc)
            return []

    def predict(self, subject: str = "", body: str = "", *, explain: bool = True) -> PredictionResult:
        """Classify a single email, validating the input first."""
        text = validate_email_text(subject, body)
        started = time.perf_counter()

        try:
            probability = float(self._scores([text])[0])
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a friendly error
            logger.exception("Inference failed")
            raise PredictionError(f"Inference failed: {exc}") from exc

        return PredictionResult(
            label="spam" if probability >= self.threshold else "ham",
            spam_probability=probability,
            confidence=abs(probability - 0.5) * 2,
            decision=decide(probability),
            threshold=self.threshold,
            model_name=self.model_name,
            latency_ms=(time.perf_counter() - started) * 1000,
            top_terms=self.explain(text) if explain else [],
        )

    def predict_frame(self, frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Score a DataFrame of emails and append prediction columns."""
        if text_column not in frame.columns:
            raise PredictionError(f"Column '{text_column}' is missing from the uploaded file.")

        texts = frame[text_column].fillna("").astype(str)
        usable = texts.str.strip().str.len() > 0
        if not usable.any():
            raise PredictionError(f"Column '{text_column}' contains no readable text.")

        probabilities = np.full(len(frame), np.nan)
        try:
            probabilities[usable.to_numpy()] = self._scores(texts[usable].tolist())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batch inference failed")
            raise PredictionError(f"Batch inference failed: {exc}") from exc

        result = frame.copy()
        result["spam_probability"] = probabilities
        result["predicted_label"] = np.where(
            np.isnan(probabilities), "unscored", np.where(probabilities >= self.threshold, "spam", "ham")
        )
        result["decision"] = [
            "unscored" if np.isnan(p) else decide(p) for p in probabilities
        ]
        result["confidence"] = np.abs(probabilities - 0.5) * 2
        return result


@lru_cache(maxsize=1)
def load_predictor(model_mtime: float | None = None) -> EmailPredictor:
    """Load the trained model, cached against the artifact's modification time."""
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError()
    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - corrupted or version-mismatched artifact
        logger.exception("Model load failed")
        raise ModelLoadError(f"Could not load {MODEL_PATH.name}: {exc}") from exc

    if "pipeline" not in artifact:
        raise ModelLoadError("Model artifact is missing its pipeline.")

    logger.info("Loaded model '%s' trained at %s", artifact.get("model_name"), artifact.get("trained_at"))
    return EmailPredictor(artifact)


def get_predictor() -> EmailPredictor:
    """Return the current predictor, reloading automatically after retraining."""
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError()
    return load_predictor(MODEL_PATH.stat().st_mtime)


def model_is_available() -> bool:
    """Report whether a trained artifact exists on disk."""
    return MODEL_PATH.exists()
