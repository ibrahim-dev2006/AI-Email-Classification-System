"""Central configuration for the AI Email Classification System."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

RAW_DATASET_CSV = DATA_DIR / "enron_spam_data.csv"
SAMPLE_DATASET_CSV = DATA_DIR / "sample_emails.csv"

MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
AUTH_DB_PATH = DATA_DIR / "users.db"
PREDICTION_LOG_PATH = DATA_DIR / "predictions.db"

DATASET_SOURCES = (
    (
        "enron",
        "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip",
    ),
    (
        "sms",
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    ),
)

LABELS = ("ham", "spam")
LABEL_DISPLAY = {"ham": "Legitimate", "spam": "Spam"}

# Below LOW, auto-deliver; above HIGH, auto-quarantine; between, route to human review.
REVIEW_BAND_LOW = 0.35
REVIEW_BAND_HIGH = 0.75


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and split settings for the supervised pipeline."""

    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.9
    sublinear_tf: bool = True
    cv_folds: int = 5
    max_rows: int | None = None
    candidate_models: tuple[str, ...] = ("logistic_regression", "linear_svc", "naive_bayes")
    threshold_objective: str = "f1"


@dataclass(frozen=True)
class LLMConfig:
    """Settings for the optional API-based classifier (Groq or Anthropic)."""

    groq_model: str = "openai/gpt-oss-120b"
    # temperature/top_p are rejected with a 400 on Claude Opus 5; determinism comes from effort + schema.
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: str = "low"
    max_tokens: int = 2048
    timeout_seconds: float = 45.0
    max_retries: int = 3


TRAINING = TrainingConfig()
LLM = LLMConfig()


@dataclass
class AppSettings:
    """Runtime settings resolved from Streamlit secrets or environment variables."""

    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    allow_self_signup: bool = True
    admin_bootstrap_username: str = "admin"
    admin_bootstrap_password: str = field(default="ChangeMe!123", repr=False)


def _secret(name: str) -> str | None:
    """Read a secret from Streamlit secrets, falling back to the environment."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def load_settings() -> AppSettings:
    """Build application settings from the current environment."""
    return AppSettings(
        groq_api_key=_secret("GROQ_API_KEY"),
        anthropic_api_key=_secret("ANTHROPIC_API_KEY"),
        allow_self_signup=(_secret("ALLOW_SELF_SIGNUP") or "true").lower() == "true",
        admin_bootstrap_username=_secret("ADMIN_USERNAME") or "admin",
        admin_bootstrap_password=_secret("ADMIN_PASSWORD") or "ChangeMe!123",
    )


def ensure_directories() -> None:
    """Create the writable directories the app expects at startup."""
    for directory in (DATA_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
