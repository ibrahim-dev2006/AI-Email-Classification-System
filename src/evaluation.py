"""Model evaluation metrics, curves, and operating-point analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import LABELS

# A legitimate email sent to quarantine costs more than a spam that slips through.
FALSE_POSITIVE_COST = 5.0
FALSE_NEGATIVE_COST = 1.0


@dataclass
class EvaluationResult:
    """Full evaluation of one model on one dataset split."""

    split: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    balanced_accuracy: float
    specificity: float
    mcc: float
    log_loss: float
    support: int
    confusion: list[list[int]]
    report: dict = field(default_factory=dict)
    roc_curve: dict = field(default_factory=dict)
    pr_curve: dict = field(default_factory=dict)
    threshold_sweep: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "balanced_accuracy": self.balanced_accuracy,
            "specificity": self.specificity,
            "mcc": self.mcc,
            "log_loss": self.log_loss,
            "support": self.support,
            "confusion": self.confusion,
            "report": self.report,
            "roc_curve": self.roc_curve,
            "pr_curve": self.pr_curve,
            "threshold_sweep": self.threshold_sweep,
        }

    @property
    def headline(self) -> dict[str, float]:
        return {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1,
            "ROC AUC": self.roc_auc,
        }


def _thin(array: np.ndarray, max_points: int = 300) -> list[float]:
    """Subsample a curve so serialised metrics stay small."""
    if len(array) <= max_points:
        return [float(v) for v in array]
    idx = np.linspace(0, len(array) - 1, max_points).astype(int)
    return [float(v) for v in array[idx]]


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, *, split: str = "test"
) -> EvaluationResult:
    """Compute the full metric suite for one set of predictions."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_score)

    return EvaluationResult(
        split=split,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        average_precision=float(average_precision_score(y_true, y_score)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        specificity=float(tn / (tn + fp)) if (tn + fp) else 0.0,
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        log_loss=float(log_loss(y_true, np.clip(y_score, 1e-9, 1 - 1e-9))),
        support=int(len(y_true)),
        confusion=matrix.tolist(),
        report=classification_report(
            y_true, y_pred, target_names=list(LABELS), output_dict=True, zero_division=0
        ),
        roc_curve={"fpr": _thin(fpr), "tpr": _thin(tpr)},
        pr_curve={"precision": _thin(precision_pts), "recall": _thin(recall_pts)},
        threshold_sweep=threshold_sweep(y_true, y_score),
    )


def threshold_sweep(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray | None = None
) -> list[dict]:
    """Score a range of decision thresholds so an operating point can be chosen."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)

    rows = []
    for threshold in thresholds:
        predictions = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "weighted_cost": float(fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST),
            }
        )
    return rows


def best_threshold(sweep: list[dict], objective: str = "weighted_cost") -> float:
    """Pick the threshold that optimises the chosen objective."""
    if not sweep:
        return 0.5
    if objective == "weighted_cost":
        return min(sweep, key=lambda row: row["weighted_cost"])["threshold"]
    return max(sweep, key=lambda row: row[objective])["threshold"]


def confusion_frame(matrix: list[list[int]]) -> pd.DataFrame:
    """Render a confusion matrix as a labelled DataFrame for display."""
    return pd.DataFrame(
        matrix,
        index=[f"Actual {LABELS[0]}", f"Actual {LABELS[1]}"],
        columns=[f"Predicted {LABELS[0]}", f"Predicted {LABELS[1]}"],
    )


def error_cost(matrix: list[list[int]]) -> dict[str, float]:
    """Summarise the business cost implied by a confusion matrix."""
    (tn, fp), (fn, tp) = matrix
    return {
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_cost": fp * FALSE_POSITIVE_COST,
        "false_negative_cost": fn * FALSE_NEGATIVE_COST,
        "total_cost": fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST,
        "emails_reviewed": int(tp + fp),
        "spam_caught_pct": float(tp / (tp + fn) * 100) if (tp + fn) else 0.0,
        "legitimate_blocked_pct": float(fp / (tn + fp) * 100) if (tn + fp) else 0.0,
    }
