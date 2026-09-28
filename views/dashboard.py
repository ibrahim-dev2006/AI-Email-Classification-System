"""Interactive dashboard covering live activity and the training corpus."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts, history
from src.dataset import load_sample
from src.ui import (
    configure_page,
    error_boundary,
    format_percent,
    hero,
    metric_cards,
    require_login,
    sidebar_account,
)

FEATURE_LABELS = {
    "char_count": "Character count",
    "word_count": "Word count",
    "avg_word_length": "Average word length",
    "uppercase_ratio": "Uppercase ratio",
    "digit_ratio": "Digit ratio",
    "exclamation_count": "Exclamation marks",
    "url_count": "URLs",
    "currency_count": "Currency mentions",
    "unique_word_ratio": "Unique word ratio",
    "spam_trigger_hits": "Spam trigger words",
}


@st.cache_data(ttl=300, show_spinner="Loading the email corpus...")
def _corpus() -> pd.DataFrame:
    """Load the bundled dataset sample with engineered features."""
    return load_sample()


def render() -> None:
    """Render the dashboard page."""
    configure_page()
    user = require_login()
    sidebar_account(user)

    hero("Dashboard", "Live classification activity and an explorer for the training corpus.")

    activity_tab, corpus_tab = st.tabs(["Classification activity", "Training corpus"])

    with activity_tab:
        _render_activity(user)

    with corpus_tab:
        with error_boundary("corpus loading"):
            _render_corpus()


def _render_activity(user) -> None:
    """Summarise what this deployment has classified so far."""
    scope = st.radio(
        "Scope",
        options=["My classifications", "Everyone"],
        horizontal=True,
        label_visibility="collapsed",
    )
    username = user.username if scope == "My classifications" else None

    stats = history.summary_stats(username=username)
    if stats["total"] == 0:
        st.info("No classifications recorded yet. Score an email on the **Classify** page to populate this view.")
        return

    accuracy = stats["accuracy_from_feedback"]
    metric_cards(
        [
            ("Total classified", f"{stats['total']:,}", "Emails scored"),
            ("Flagged as spam", f"{stats['spam']:,}", format_percent(stats["spam_rate"], 1) + " of total"),
            ("Awaiting review", f"{stats['needs_review']:,}", "Inside the uncertainty band"),
            ("Mean confidence", format_percent(stats["avg_confidence"], 1), "Distance from the boundary"),
            (
                "Analyst-verified accuracy",
                format_percent(accuracy, 1) if accuracy is not None else "No feedback yet",
                f"From {stats['feedback_given']:,} reviewed verdicts",
            ),
        ]
    )

    frame = history.load_history(username=username, limit=5_000)
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Volume over time")
        st.altair_chart(charts.timeline_chart(frame), width="stretch")
    with right:
        st.subheader("Routing decisions")
        st.altair_chart(charts.decision_breakdown_chart(frame), width="stretch")

    st.subheader("Recent classifications")
    display = frame[
        ["created_at", "username", "subject", "predicted_label", "spam_probability", "decision", "feedback"]
    ].head(200)
    st.dataframe(
        display,
        width="stretch",
        height=380,
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn("When", format="YYYY-MM-DD HH:mm"),
            "username": "User",
            "subject": "Subject",
            "predicted_label": "Verdict",
            "spam_probability": st.column_config.ProgressColumn(
                "Spam probability", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "decision": "Decision",
            "feedback": "Analyst feedback",
        },
    )


def _render_corpus() -> None:
    """Explore the labelled dataset the model was trained on."""
    frame = _corpus()

    spam_share = float((frame["label"] == "spam").mean())
    metric_cards(
        [
            ("Emails in sample", f"{len(frame):,}", "Stratified from the full corpus"),
            ("Spam", f"{int((frame['label'] == 'spam').sum()):,}", format_percent(spam_share, 1) + " of sample"),
            ("Legitimate", f"{int((frame['label'] == 'ham').sum()):,}", "Real Enron business mail"),
            ("Median length", f"{int(frame['char_count'].median()):,} chars", "Across both classes"),
        ]
    )

    st.write("")
    left, right = st.columns([2, 3])

    with left:
        st.subheader("Class balance")
        st.altair_chart(charts.class_balance_chart(frame["label"].value_counts()), width="stretch")
        st.caption(
            "A near-balanced corpus means accuracy is a meaningful headline metric rather than an "
            "artefact of class imbalance."
        )

    with right:
        st.subheader("How the classes differ")
        feature = st.selectbox(
            "Feature",
            options=list(FEATURE_LABELS),
            format_func=lambda key: FEATURE_LABELS[key],
            index=list(FEATURE_LABELS).index("spam_trigger_hits"),
        )
        st.altair_chart(charts.feature_comparison_chart(frame, feature), width="stretch")

    st.subheader("Feature summary by class")
    summary = frame.groupby("label")[list(FEATURE_LABELS)].mean().T
    summary.index = [FEATURE_LABELS[i] for i in summary.index]
    summary["difference"] = summary["spam"] - summary["ham"]
    summary = summary.reindex(summary["difference"].abs().sort_values(ascending=False).index)
    st.dataframe(
        summary.style.format("{:.3f}"),
        width="stretch",
        column_config={
            "ham": "Legitimate (mean)",
            "spam": "Spam (mean)",
            "difference": st.column_config.NumberColumn(
                "Difference", help="Spam mean minus legitimate mean", format="%.3f"
            ),
        },
    )
    st.caption("Sorted by the size of the gap between classes — the strongest separators appear first.")

    st.subheader("Browse the corpus")
    label_filter = st.radio("Show", ["All", "Spam only", "Legitimate only"], horizontal=True)
    filtered = frame
    if label_filter == "Spam only":
        filtered = frame[frame["label"] == "spam"]
    elif label_filter == "Legitimate only":
        filtered = frame[frame["label"] == "ham"]

    search = st.text_input("Search the text", placeholder="e.g. invoice, winner, meeting")
    if search:
        filtered = filtered[filtered["text"].str.contains(search, case=False, na=False, regex=False)]
        st.caption(f"{len(filtered):,} matching emails")

    st.dataframe(
        filtered[["label", "text", "char_count", "spam_trigger_hits"]].head(200),
        width="stretch",
        height=380,
        hide_index=True,
        column_config={
            "label": "Class",
            "text": st.column_config.TextColumn("Email text", width="large"),
            "char_count": "Length",
            "spam_trigger_hits": "Trigger words",
        },
    )
