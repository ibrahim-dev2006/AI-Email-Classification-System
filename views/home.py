"""Landing page with system status and headline model figures."""

from __future__ import annotations

import json

import streamlit as st

from src import history, llm_classifier
from src.config import METRICS_PATH
from src.predictor import model_is_available
from src.ui import configure_page, format_percent, hero, metric_cards, require_login, sidebar_account


@st.cache_data(ttl=30)
def _load_metrics() -> dict | None:
    """Read the saved evaluation metrics, if training has been run."""
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def render() -> None:
    """Render the home page."""
    configure_page()
    user = require_login()
    sidebar_account(user)

    hero(
        f"Welcome back, {user.username}",
        "Classify email as spam or legitimate using a trained model, with optional AI-assisted review.",
    )

    metrics = _load_metrics()
    if metrics:
        test = metrics["test"]
        metric_cards(
            [
                ("Model", metrics["model_name"].replace("_", " ").title(), "Selected by cross-validation"),
                ("Accuracy", format_percent(test["accuracy"]), "On the held-out test set"),
                ("F1 score", f"{test['f1']:.4f}", "Balance of precision and recall"),
                ("ROC AUC", f"{test['roc_auc']:.4f}", "Ranking quality across thresholds"),
                ("Training emails", f"{metrics['dataset']['total_rows']:,}", "Enron-Spam corpus"),
            ]
        )
    else:
        st.warning(
            "No trained model found yet. An administrator can train one from the **Admin** page, "
            "or run `python -m src.train` from the project root."
        )

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        st.subheader("What this system does")
        st.markdown(
            """
            **1. Preprocessing** normalises raw email text — stripping HTML, folding Unicode, and
            replacing URLs, addresses, amounts, and numbers with stable tokens so the model learns
            patterns rather than specific values.

            **2. Classification** scores the email with a TF-IDF pipeline trained on 30,000+ real
            emails. Three candidate algorithms compete during training and the best is selected by
            cross-validated F1.

            **3. Decision output** converts the score into an action — deliver, route to human
            review, or quarantine — using a calibrated threshold plus an uncertainty band.

            **4. AI-assisted analysis** can optionally send the email to a large language model for
            a second opinion with written reasoning, which is useful for the borderline cases the
            local model is least certain about.
            """
        )

    with right:
        st.subheader("System status")
        _status_row("Trained model", model_is_available(), "Ready", "Not trained yet")
        _status_row(
            f"AI analysis ({llm_classifier.provider_label()})",
            llm_classifier.api_key_available(),
            "Enabled",
            "No API key configured (optional)",
        )
        _status_row("Evaluation metrics", metrics is not None, "Available", "Run training first")

        stats = history.summary_stats()
        st.write("")
        st.caption("Activity on this deployment")
        st.metric("Emails classified", f"{stats['total']:,}")
        if stats["total"]:
            st.metric("Flagged as spam", f"{stats['spam']:,}", f"{format_percent(stats['spam_rate'], 1)} of total")


def _status_row(label: str, ok: bool, ok_text: str, bad_text: str) -> None:
    """Render one status line with a coloured indicator."""
    icon = ":green_circle:" if ok else ":red_circle:"
    st.markdown(f"{icon} **{label}** — {ok_text if ok else bad_text}")
