"""Model evaluation: metrics, curves, threshold analysis, and error inspection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src.config import METRICS_PATH
from src.evaluation import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    confusion_frame,
    error_cost,
    threshold_sweep,
)
from src.train import HOLDOUT_PREDICTIONS_CSV
from src.ui import (
    configure_page,
    format_percent,
    hero,
    metric_cards,
    require_login,
    sidebar_account,
)


@st.cache_data(ttl=30)
def _metrics() -> dict | None:
    """Load saved evaluation metrics."""
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=30)
def _holdout() -> pd.DataFrame | None:
    """Load the held-out predictions used for interactive threshold analysis."""
    if not HOLDOUT_PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(HOLDOUT_PREDICTIONS_CSV)


def render() -> None:
    """Render the performance evaluation page."""
    configure_page()
    user = require_login()
    sidebar_account(user)

    hero("Model performance", "How the trained classifier scores on data it never saw during training.")

    metrics = _metrics()
    if metrics is None:
        st.warning("No evaluation metrics found. Train the model from the **Admin** page first.")
        return

    test = metrics["test"]
    dataset = metrics["dataset"]

    st.caption(
        f"Model **{metrics['model_name'].replace('_', ' ').title()}** · "
        f"trained {metrics['trained_at'][:19].replace('T', ' ')} UTC · "
        f"{dataset['train_rows']:,} training / {dataset['test_rows']:,} test emails · "
        f"decision threshold {metrics['threshold']:.2f}"
    )

    metric_cards(
        [
            ("Accuracy", format_percent(test["accuracy"]), "Overall correctness"),
            ("Precision", format_percent(test["precision"]), "Of those flagged spam, how many were"),
            ("Recall", format_percent(test["recall"]), "Of all spam, how much was caught"),
            ("F1 score", f"{test['f1']:.4f}", "Harmonic mean of the two"),
            ("ROC AUC", f"{test['roc_auc']:.4f}", "Ranking quality"),
        ]
    )

    tabs = st.tabs(
        ["Confusion & cost", "Curves", "Threshold explorer", "Model comparison", "What the model learned", "Errors"]
    )

    with tabs[0]:
        _render_confusion(test)
    with tabs[1]:
        _render_curves(test)
    with tabs[2]:
        _render_threshold_explorer(metrics)
    with tabs[3]:
        _render_comparison(metrics)
    with tabs[4]:
        _render_learned(metrics)
    with tabs[5]:
        _render_errors(metrics)


def _render_confusion(test: dict) -> None:
    """Show the confusion matrix and its operational cost."""
    left, right = st.columns([3, 2])

    with left:
        st.altair_chart(charts.confusion_heatmap(test["confusion"]), width="stretch")

    with right:
        cost = error_cost(test["confusion"])
        st.markdown("#### Operational impact")
        st.markdown(
            f"""
            - **{cost['spam_caught_pct']:.2f}%** of all spam was caught.
            - **{cost['legitimate_blocked_pct']:.2f}%** of legitimate email was wrongly blocked.
            - **{cost['false_negatives']:,}** spam messages reached the inbox.
            - **{cost['false_positives']:,}** legitimate messages were quarantined.
            """
        )
        st.caption(
            f"Weighting a false positive at {FALSE_POSITIVE_COST:.0f}x a false negative "
            f"(blocking real mail is worse than missing spam), the total error cost on this "
            f"test set is **{cost['total_cost']:,.0f}**."
        )

    st.markdown("#### Per-class breakdown")
    report = pd.DataFrame(test["report"]).T
    report = report.loc[["ham", "spam", "macro avg", "weighted avg"]]
    st.dataframe(
        report.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:,.0f}"}),
        width="stretch",
    )

    st.markdown("#### Raw counts")
    st.dataframe(confusion_frame(test["confusion"]), width="stretch")


def _render_curves(test: dict) -> None:
    """Show ROC and precision-recall curves."""
    left, right = st.columns(2)

    with left:
        st.markdown("#### ROC curve")
        st.altair_chart(
            charts.roc_chart(test["roc_curve"]["fpr"], test["roc_curve"]["tpr"], test["roc_auc"]),
            width="stretch",
        )
        st.caption(
            "Plots the true positive rate against the false positive rate at every threshold. "
            "The dashed diagonal is what random guessing would achieve."
        )

    with right:
        st.markdown("#### Precision-recall curve")
        st.altair_chart(
            charts.pr_chart(
                test["pr_curve"]["precision"], test["pr_curve"]["recall"], test["average_precision"]
            ),
            width="stretch",
        )
        st.caption(
            "More informative than ROC when the cost of false positives is high, since it ignores "
            "the large pool of correctly-delivered legitimate mail."
        )

    st.markdown("#### Additional metrics")
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": [
                    "Balanced accuracy",
                    "Specificity (true negative rate)",
                    "Matthews correlation coefficient",
                    "Log loss",
                    "Average precision",
                ],
                "Value": [
                    f"{test['balanced_accuracy']:.4f}",
                    f"{test['specificity']:.4f}",
                    f"{test['mcc']:.4f}",
                    f"{test['log_loss']:.4f}",
                    f"{test['average_precision']:.4f}",
                ],
                "Interpretation": [
                    "Mean of per-class recall; unaffected by class imbalance.",
                    "Share of legitimate email correctly delivered.",
                    "Correlation between predictions and truth; 1.0 is perfect.",
                    "Penalises confident wrong answers; lower is better.",
                    "Area under the precision-recall curve.",
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_threshold_explorer(metrics: dict) -> None:
    """Let the user move the decision threshold and see the consequences live."""
    holdout = _holdout()
    if holdout is None:
        st.info("Held-out predictions are unavailable. Retrain the model to enable this view.")
        return

    st.markdown(
        """
        The threshold decides how confident the model must be before calling an email spam.
        Lower it to catch more spam at the cost of blocking real mail; raise it to protect
        legitimate mail at the cost of letting spam through.
        """
    )

    trained_threshold = metrics["threshold"]
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(trained_threshold),
        step=0.01,
    )

    y_true = (holdout["true_label"] == "spam").astype(int).to_numpy()
    y_score = holdout["spam_probability"].to_numpy()
    predictions = (y_score >= threshold).astype(int)

    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)

    metric_cards(
        [
            ("Accuracy", format_percent(accuracy), "At this threshold"),
            ("Precision", format_percent(precision), "Of flagged mail"),
            ("Recall", format_percent(recall), "Of all spam"),
            ("Spam reaching inbox", f"{fn:,}", "False negatives"),
            ("Legitimate blocked", f"{fp:,}", "False positives"),
        ]
    )

    if abs(threshold - trained_threshold) > 1e-6:
        st.caption(
            f"The deployed model uses **{trained_threshold:.2f}**, chosen to maximise F1 during training. "
            "This slider previews alternatives; it does not change the live model."
        )

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Metrics across thresholds")
        st.altair_chart(
            charts.threshold_chart(threshold_sweep(y_true, y_score), threshold), width="stretch"
        )
    with right:
        st.markdown("#### Weighted error cost")
        st.altair_chart(charts.cost_chart(threshold_sweep(y_true, y_score), threshold), width="stretch")
        st.caption(
            f"False positives weighted {FALSE_POSITIVE_COST:.0f}x, false negatives {FALSE_NEGATIVE_COST:.0f}x."
        )

    st.markdown("#### Score separation")
    st.altair_chart(charts.probability_distribution(holdout), width="stretch")
    st.caption(
        "Well-separated peaks at each end mean the model is decisive. Overlap in the middle is "
        "where the human-review band earns its place."
    )


def _render_comparison(metrics: dict) -> None:
    """Compare the candidate algorithms evaluated during training."""
    st.markdown(
        "Three algorithms were trained on identical features and splits. The winner was chosen by "
        "cross-validated F1 on the training set, so the test set stayed untouched during selection."
    )

    comparison = pd.DataFrame(metrics["model_comparison"])
    st.altair_chart(charts.model_comparison_chart(metrics["model_comparison"]), width="stretch")

    display = comparison.copy()
    display["model"] = display["model"].str.replace("_", " ").str.title()
    display = display.rename(
        columns={
            "model": "Model",
            "cv_f1_mean": "CV F1 (mean)",
            "cv_f1_std": "CV F1 (std)",
            "test_accuracy": "Test accuracy",
            "test_precision": "Test precision",
            "test_recall": "Test recall",
            "test_f1": "Test F1",
            "test_roc_auc": "Test ROC AUC",
            "fit_seconds": "Fit time (s)",
        }
    )
    st.dataframe(
        display.style.format(
            {
                "CV F1 (mean)": "{:.4f}",
                "CV F1 (std)": "{:.4f}",
                "Test accuracy": "{:.4f}",
                "Test precision": "{:.4f}",
                "Test recall": "{:.4f}",
                "Test F1": "{:.4f}",
                "Test ROC AUC": "{:.4f}",
                "Fit time (s)": "{:.1f}",
            }
        ).highlight_max(subset=["CV F1 (mean)", "Test F1"], color="rgba(48,163,108,.25)"),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Overfitting check")
    train, test = metrics["train"], metrics["test"]
    gap = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Training set": [train["accuracy"], train["precision"], train["recall"], train["f1"]],
            "Test set": [test["accuracy"], test["precision"], test["recall"], test["f1"]],
        }
    )
    gap["Gap"] = gap["Training set"] - gap["Test set"]
    st.dataframe(gap.style.format({"Training set": "{:.4f}", "Test set": "{:.4f}", "Gap": "{:+.4f}"}),
                 width="stretch", hide_index=True)
    st.caption(
        f"A gap of {gap['Gap'].mean():+.4f} on average indicates the model generalises well rather "
        "than memorising the training set."
    )


def _render_learned(metrics: dict) -> None:
    """Show the terms the model weights most heavily for each class."""
    terms = metrics.get("top_terms", {})
    if not terms.get("spam") and not terms.get("ham"):
        st.info("This model type does not expose per-term weights.")
        return

    st.markdown(
        "These are the terms carrying the most weight in the trained model. They make the model's "
        "behaviour auditable — you can see exactly what it keys on."
    )

    left, right = st.columns(2)
    with left:
        st.altair_chart(
            charts.term_importance_chart(terms["spam"][:20], "Strongest spam indicators", charts.SPAM_COLOR),
            width="stretch",
        )
    with right:
        st.altair_chart(
            charts.term_importance_chart(
                terms["ham"][:20], "Strongest legitimate indicators", charts.HAM_COLOR
            ),
            width="stretch",
        )

    st.warning(
        "**Corpus bias worth noting:** the legitimate half of this dataset is Enron's internal mail, "
        "so company-specific terms (names, 'gas', 'energy', 'houston') rank as strong legitimacy "
        "signals. The model would need retraining on in-domain mail before deployment elsewhere."
    )


def _render_errors(metrics: dict) -> None:
    """Let the user inspect the emails the model got wrong."""
    holdout = _holdout()
    if holdout is None:
        st.info("Held-out predictions are unavailable. Retrain the model to enable this view.")
        return

    errors = holdout[holdout["true_label"] != holdout["predicted_label"]].copy()
    errors["error_type"] = np.where(
        errors["true_label"] == "ham", "False positive (legitimate blocked)", "False negative (spam missed)"
    )
    errors = errors.sort_values("spam_probability", ascending=False)

    st.markdown(
        f"The model misclassified **{len(errors):,}** of **{len(holdout):,}** held-out emails "
        f"({len(errors) / len(holdout):.2%}). Reading the failures is usually more informative "
        "than reading the successes."
    )

    error_type = st.radio(
        "Show", ["All errors", "False positives", "False negatives"], horizontal=True
    )
    if error_type == "False positives":
        errors = errors[errors["true_label"] == "ham"]
    elif error_type == "False negatives":
        errors = errors[errors["true_label"] == "spam"]

    if errors.empty:
        st.success("No errors of this type on the held-out set.")
        return

    st.dataframe(
        errors[["error_type", "true_label", "predicted_label", "spam_probability", "text"]],
        width="stretch",
        height=460,
        hide_index=True,
        column_config={
            "error_type": "Error type",
            "true_label": "Actual",
            "predicted_label": "Predicted",
            "spam_probability": st.column_config.ProgressColumn(
                "Spam score", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "text": st.column_config.TextColumn("Email text (truncated)", width="large"),
        },
    )
