"""Text cleaning and feature engineering for raw email content."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_HTML_ENTITY = re.compile(r"&[a-z]{2,8};|&#\d{1,5};", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY = re.compile(r"[$£€¥]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|gbp|eur|dollars?|pounds?)\b", re.IGNORECASE)
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_NON_WORD = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")

SPAM_TRIGGER_WORDS = (
    "free", "winner", "win", "cash", "prize", "urgent", "click", "offer",
    "guarantee", "credit", "loan", "viagra", "pills", "investment", "bonus",
    "limited", "act now", "risk free", "unsubscribe", "congratulations",
    "money back", "no obligation", "cheap", "discount", "earn", "income",
)


def normalise_unicode(text: str) -> str:
    """Fold accents and unify look-alike characters used to evade filters."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_text(text: object) -> str:
    """Normalise raw email text into the token stream the vectoriser expects."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    cleaned = normalise_unicode(str(text)).lower()
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = _HTML_ENTITY.sub(" ", cleaned)
    cleaned = _URL.sub(" urltoken ", cleaned)
    cleaned = _EMAIL.sub(" emailtoken ", cleaned)
    cleaned = _MONEY.sub(" moneytoken ", cleaned)
    cleaned = _PHONE.sub(" phonetoken ", cleaned)
    cleaned = _NUMBER.sub(" numtoken ", cleaned)
    cleaned = _NON_WORD.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned


def combine_subject_body(subject: object, body: object) -> str:
    """Join a subject line and body into the single string the model scores."""
    subject_text = "" if subject is None or pd.isna(subject) else str(subject)
    body_text = "" if body is None or pd.isna(body) else str(body)
    return f"{subject_text}\n\n{body_text}".strip()


class TextCleaner(BaseEstimator, TransformerMixin):
    """Pipeline step that applies `clean_text`, keeping train and serve identical."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        return pd.Series(list(X), dtype="object").map(clean_text).tolist()

    def get_feature_names_out(self, input_features=None):
        return input_features


def engineer_features(texts: pd.Series) -> pd.DataFrame:
    """Derive interpretable numeric features used by the dashboard and EDA."""
    raw = texts.fillna("").astype(str)
    lowered = raw.str.lower()
    words = raw.str.split()

    char_count = raw.str.len()
    word_count = words.map(len)
    letters = raw.str.count(r"[A-Za-z]")
    uppercase = raw.str.count(r"[A-Z]")
    digits = raw.str.count(r"\d")

    features = pd.DataFrame(
        {
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_length": (char_count / word_count.replace(0, pd.NA)).fillna(0.0),
            "uppercase_ratio": (uppercase / letters.replace(0, pd.NA)).fillna(0.0),
            "digit_ratio": (digits / char_count.replace(0, pd.NA)).fillna(0.0),
            "exclamation_count": raw.str.count(r"!"),
            "question_count": raw.str.count(r"\?"),
            "url_count": raw.str.count(_URL),
            "email_count": raw.str.count(_EMAIL),
            "currency_count": raw.str.count(_MONEY),
            "unique_word_ratio": words.map(lambda w: len(set(w)) / len(w) if w else 0.0),
            "spam_trigger_hits": sum(lowered.str.count(re.escape(w)) for w in SPAM_TRIGGER_WORDS),
        }
    )
    return features.astype(float)


def validate_email_text(subject: str, body: str, *, min_chars: int = 3, max_chars: int = 200_000) -> str:
    """Validate and return user-submitted email text, raising on unusable input."""
    from .exceptions import ValidationError

    combined = combine_subject_body(subject, body)
    stripped = combined.strip()

    if len(stripped) < min_chars:
        raise ValidationError(
            user_message=f"Please enter at least {min_chars} characters of email content."
        )
    if len(stripped) > max_chars:
        raise ValidationError(
            user_message=f"Email is too long ({len(stripped):,} characters). The limit is {max_chars:,}."
        )
    if not clean_text(stripped):
        raise ValidationError(
            user_message="The email contains no readable text after cleaning. Check the content and try again."
        )
    return combined
