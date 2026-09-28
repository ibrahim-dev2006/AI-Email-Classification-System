"""Bulk classification of an uploaded CSV file."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src import charts
from src.predictor import get_predictor, model_is_available
from src.ui import (
    configure_page,
    error_boundary,
    format_percent,
    hero,
    metric_cards,
    require_login,
    sidebar_account,
)

MAX_ROWS = 5_000
LIKELY_TEXT_COLUMNS = ("text", "body", "message", "content", "email", "email_text")


def render() -> None:
    """Render the batch classification page."""
    configure_page()
    user = require_login()
    sidebar_account(user)

    hero("Batch classification", "Upload a CSV of emails and score every row in one pass.")

    if not model_is_available():
        st.warning("No trained model is available. Train one from the **Admin** page first.")
        return

    st.markdown(
        f"""
        Upload a CSV containing your emails. The file needs a column of email text — the app will
        try to detect it automatically, and you can override the choice. A maximum of
        **{MAX_ROWS:,} rows** are scored per upload.
        """
    )

    upload = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
    if upload is None:
        _show_template()
        return

    with error_boundary("file upload"):
        frame = _read_csv(upload)

        st.success(f"Loaded {len(frame):,} rows and {len(frame.columns)} columns.")
        with st.expander("Preview the uploaded file"):
            st.dataframe(frame.head(20), width="stretch")

        text_column = st.selectbox(
            "Which column holds the email text?",
            options=list(frame.columns),
            index=_guess_column(frame),
        )
        label_column = st.selectbox(
            "Ground-truth label column (optional, enables accuracy scoring)",
            options=["(none)"] + list(frame.columns),
            index=0,
        )

        if not st.button("Classify all rows", type="primary"):
            return

        if len(frame) > MAX_ROWS:
            st.info(f"Only the first {MAX_ROWS:,} rows will be scored.")
            frame = frame.head(MAX_ROWS)

        with st.spinner(f"Scoring {len(frame):,} emails..."):
            scored = get_predictor().predict_frame(frame, text_column)

        _render_results(scored, label_column if label_column != "(none)" else None)


def _read_csv(upload) -> pd.DataFrame:
    """Parse an uploaded CSV, tolerating common encodings."""
    raw = upload.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The uploaded file is empty.") from exc
    else:
        raise ValueError("Could not decode the file. Save it as UTF-8 and try again.")

    if frame.empty:
        raise ValueError("The uploaded file contains no rows.")
    return frame


def _guess_column(frame: pd.DataFrame) -> int:
    """Pick the most likely email-text column."""
    lowered = [c.lower().strip() for c in frame.columns]
    for candidate in LIKELY_TEXT_COLUMNS:
        if candidate in lowered:
            return lowered.index(candidate)

    # Otherwise assume the widest text column holds the message body.
    text_widths = {
        i: frame[c].astype(str).str.len().mean()
        for i, c in enumerate(frame.columns)
        if frame[c].dtype == object
    }
    return max(text_widths, key=text_widths.get) if text_widths else 0


def _render_results(scored: pd.DataFrame, label_column: str | None) -> None:
    """Summarise, visualise, and offer a download of the scored rows."""
    st.divider()
    st.subheader("Results")

    total = len(scored)
    spam_count = int((scored["predicted_label"] == "spam").sum())
    review_count = int((scored["decision"] == "review").sum())
    unscored = int((scored["predicted_label"] == "unscored").sum())

    metric_cards(
        [
            ("Emails scored", f"{total - unscored:,}", f"{unscored:,} skipped as empty"),
            ("Flagged as spam", f"{spam_count:,}", format_percent(spam_count / total, 1) + " of rows"),
            ("Needs human review", f"{review_count:,}", "Scores inside the uncertainty band"),
            ("Mean confidence", format_percent(scored["confidence"].mean(), 1), "Distance from the boundary"),
        ]
    )

    st.write("")
    left, right = st.columns(2)
    with left:
        st.altair_chart(charts.decision_breakdown_chart(scored), width="stretch")
    with right:
        counts = scored["predicted_label"].value_counts()
        st.altair_chart(charts.class_balance_chart(counts), width="stretch")

    if label_column:
        _render_accuracy(scored, label_column)

    st.subheader("Scored rows")
    display_columns = [
        c for c in scored.columns if c not in ("clean_text",)
    ]
    st.dataframe(
        scored[display_columns],
        width="stretch",
        height=420,
        column_config={
            "spam_probability": st.column_config.ProgressColumn(
                "Spam probability", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
        },
    )

    st.download_button(
        "Download results as CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="classified_emails.csv",
        mime="text/csv",
        type="primary",
    )


def _render_accuracy(scored: pd.DataFrame, label_column: str) -> None:
    """Score the batch against supplied ground-truth labels."""
    truth = scored[label_column].astype(str).str.strip().str.lower()
    valid = truth.isin(("spam", "ham")) & (scored["predicted_label"] != "unscored")

    if not valid.any():
        st.info(
            f"Column '{label_column}' holds no usable labels. "
            "Values must be 'spam' or 'ham' to score accuracy."
        )
        return

    correct = (truth[valid] == scored.loc[valid, "predicted_label"]).sum()
    total = int(valid.sum())
    st.success(
        f"Accuracy against the supplied labels: **{correct / total:.2%}** "
        f"({correct:,} of {total:,} rows correct)."
    )


def _show_template() -> None:
    """Offer a starter CSV so the expected format is obvious."""
    template = pd.DataFrame(
        {
            "text": [
                "Subject: Team offsite agenda\n\nHi all, attaching the draft agenda for next month.",
                "WIN A FREE IPHONE NOW!!! Click http://example.com/claim to collect your prize.",
            ],
            "label": ["ham", "spam"],
        }
    )
    st.caption("Expected format")
    st.dataframe(template, width="stretch")
    st.download_button(
        "Download a template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="email_template.csv",
        mime="text/csv",
    )
