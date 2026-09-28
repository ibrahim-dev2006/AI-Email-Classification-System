"""Persistent log of classifications, powering the dashboard and feedback loop."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from .config import PREDICTION_LOG_PATH
from .logger import get_logger

logger = get_logger(__name__)


@contextmanager
def _connect():
    """Open a SQLite connection that commits on success and always closes."""
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(PREDICTION_LOG_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Create the predictions table if it does not yet exist."""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                username TEXT NOT NULL,
                subject TEXT,
                snippet TEXT,
                char_count INTEGER,
                predicted_label TEXT NOT NULL,
                spam_probability REAL NOT NULL,
                confidence REAL,
                decision TEXT NOT NULL,
                model_name TEXT,
                source TEXT NOT NULL,
                latency_ms REAL,
                feedback TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)"
        )


def log_prediction(
    *,
    username: str,
    subject: str,
    body: str,
    predicted_label: str,
    spam_probability: float,
    confidence: float,
    decision: str,
    model_name: str,
    source: str,
    latency_ms: float,
) -> int:
    """Record one classification and return its row id."""
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at, username, subject, snippet, char_count, predicted_label,
                spam_probability, confidence, decision, model_name, source, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                username,
                (subject or "")[:200],
                (body or "")[:300],
                len(body or ""),
                predicted_label,
                float(spam_probability),
                float(confidence),
                decision,
                model_name,
                source,
                float(latency_ms),
            ),
        )
        return cursor.lastrowid


def record_feedback(prediction_id: int, feedback: str) -> None:
    """Attach an analyst's correct/incorrect verdict to a logged prediction."""
    with _connect() as connection:
        connection.execute(
            "UPDATE predictions SET feedback = ? WHERE id = ?", (feedback, prediction_id)
        )


def load_history(username: str | None = None, limit: int = 1_000) -> pd.DataFrame:
    """Return recent predictions, optionally scoped to one user."""
    query = "SELECT * FROM predictions"
    params: tuple = ()
    if username:
        query += " WHERE username = ?"
        params = (username,)
    query += " ORDER BY created_at DESC LIMIT ?"
    params += (limit,)

    with _connect() as connection:
        frame = pd.read_sql_query(query, connection, params=params)

    if not frame.empty:
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", format="mixed")
    return frame


def summary_stats(username: str | None = None) -> dict:
    """Aggregate headline counters for the dashboard."""
    frame = load_history(username=username, limit=100_000)
    if frame.empty:
        return {
            "total": 0,
            "spam": 0,
            "ham": 0,
            "spam_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0,
            "needs_review": 0,
            "feedback_given": 0,
            "accuracy_from_feedback": None,
        }

    feedback = frame[frame["feedback"].notna()]
    return {
        "total": int(len(frame)),
        "spam": int((frame["predicted_label"] == "spam").sum()),
        "ham": int((frame["predicted_label"] == "ham").sum()),
        "spam_rate": float((frame["predicted_label"] == "spam").mean()),
        "avg_confidence": float(frame["confidence"].mean()),
        "avg_latency_ms": float(frame["latency_ms"].mean()),
        "needs_review": int((frame["decision"] == "review").sum()),
        "feedback_given": int(len(feedback)),
        "accuracy_from_feedback": (
            float((feedback["feedback"] == "correct").mean()) if len(feedback) else None
        ),
    }


def clear_history(username: str | None = None) -> int:
    """Delete logged predictions, returning how many rows were removed."""
    with _connect() as connection:
        if username:
            cursor = connection.execute("DELETE FROM predictions WHERE username = ?", (username,))
        else:
            cursor = connection.execute("DELETE FROM predictions")
        return cursor.rowcount
