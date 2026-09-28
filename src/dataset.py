"""Dataset acquisition, validation, and preparation."""

from __future__ import annotations

import io
import random
import zipfile
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, DATASET_SOURCES, RAW_DATASET_CSV, SAMPLE_DATASET_CSV
from .exceptions import DatasetError
from .logger import get_logger
from .preprocessing import clean_text, combine_subject_body, engineer_features

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("text", "label")


def _download(url: str, timeout: int = 180) -> bytes:
    """Fetch a URL and return its bytes, raising DatasetError on any failure."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "ai-email-classifier/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise DatasetError(f"Could not download {url}: {exc}") from exc


def _load_enron(force_download: bool = False) -> pd.DataFrame:
    """Load the Enron-Spam corpus, downloading and extracting it if absent."""
    if force_download or not RAW_DATASET_CSV.exists():
        logger.info("Downloading Enron-Spam corpus...")
        payload = _download(DATASET_SOURCES[0][1])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".csv"))
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            RAW_DATASET_CSV.write_bytes(archive.read(name))
        logger.info("Saved corpus to %s", RAW_DATASET_CSV)

    frame = pd.read_csv(RAW_DATASET_CSV)
    expected = {"Subject", "Message", "Spam/Ham"}
    if not expected.issubset(frame.columns):
        raise DatasetError(f"Enron CSV is missing columns: {expected - set(frame.columns)}")

    return pd.DataFrame(
        {
            "subject": frame["Subject"].fillna(""),
            "body": frame["Message"].fillna(""),
            "text": [
                combine_subject_body(s, b)
                for s, b in zip(frame["Subject"], frame["Message"], strict=True)
            ],
            "label": frame["Spam/Ham"].str.strip().str.lower(),
            "date": pd.to_datetime(frame.get("Date"), errors="coerce"),
            "source": "enron",
        }
    )


def _load_sms() -> pd.DataFrame:
    """Load the SMS Spam Collection as a smaller secondary corpus."""
    logger.info("Falling back to the SMS Spam Collection...")
    payload = _download(DATASET_SOURCES[1][1], timeout=60)
    frame = pd.read_csv(io.BytesIO(payload), sep="\t", header=None, names=["label", "body"])
    return pd.DataFrame(
        {
            "subject": "",
            "body": frame["body"].fillna(""),
            "text": frame["body"].fillna(""),
            "label": frame["label"].str.strip().str.lower(),
            "date": pd.NaT,
            "source": "sms",
        }
    )


def _synthetic_corpus(n_per_class: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a labelled corpus so the app stays usable with no network access."""
    logger.warning("Using the synthetic fallback corpus; metrics will be optimistic.")
    rng = random.Random(seed)

    spam_subjects = [
        "CONGRATULATIONS! You have won ${amount}",
        "Urgent: claim your free {item} today",
        "Re: Your {amount}$ payout is waiting",
        "Limited time offer - {pct}% off everything",
        "Your account will be suspended - act now",
    ]
    spam_bodies = [
        "Click here {url} to claim your prize of ${amount}. No obligation, risk free! Act now before this offer expires.",
        "Dear winner, you have been selected to receive a free {item}. Reply with your bank details to process the payment.",
        "Cheap {item} available now. Guaranteed lowest price, {pct}% discount, order today and earn extra income.",
        "URGENT!!! Your credit is approved for ${amount}. Call {phone} immediately to release the funds.",
    ]
    ham_subjects = [
        "Meeting notes from {day}",
        "Re: Q{q} budget review",
        "Updated schedule for the {item} project",
        "Lunch on {day}?",
        "Follow-up on the vendor contract",
    ]
    ham_bodies = [
        "Hi, attached are the notes from our {day} meeting. Let me know if I missed anything before I circulate them.",
        "Thanks for sending the report. I reviewed the Q{q} figures and they look consistent with the forecast.",
        "Can we move the {item} review to {day}? I have a conflict with the client call at the original time.",
        "Please find the revised contract attached. Legal has signed off, so we can proceed once you approve.",
    ]

    def fill(template: str) -> str:
        return (
            template.replace("{amount}", f"{rng.randrange(500, 500_000):,}")
            .replace("{item}", rng.choice(["laptop", "iphone", "vacation", "gift card", "watch"]))
            .replace("{pct}", str(rng.randrange(20, 90)))
            .replace("{url}", "http://claim-now.example.com/offer")
            .replace("{phone}", "555-012-3456")
            .replace("{day}", rng.choice(["Monday", "Tuesday", "Thursday", "Friday"]))
            .replace("{q}", str(rng.randrange(1, 5)))
        )

    rows = []
    for label, subjects, bodies in (
        ("spam", spam_subjects, spam_bodies),
        ("ham", ham_subjects, ham_bodies),
    ):
        for _ in range(n_per_class):
            subject = fill(rng.choice(subjects))
            body = " ".join(fill(rng.choice(bodies)) for _ in range(rng.randrange(1, 4)))
            rows.append(
                {
                    "subject": subject,
                    "body": body,
                    "text": combine_subject_body(subject, body),
                    "label": label,
                    "date": pd.NaT,
                    "source": "synthetic",
                }
            )

    rng.shuffle(rows)
    return pd.DataFrame(rows)


def load_raw_dataset(force_download: bool = False) -> pd.DataFrame:
    """Load the best available corpus, degrading gracefully through fallbacks."""
    for loader in (lambda: _load_enron(force_download), _load_sms, _synthetic_corpus):
        try:
            frame = loader()
            logger.info("Loaded %s rows from source '%s'", len(frame), frame["source"].iloc[0])
            return validate_dataset(frame)
        except DatasetError as exc:
            logger.warning("Dataset source failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 - any loader failure must fall through
            logger.warning("Unexpected dataset error: %s", exc)

    raise DatasetError("Every dataset source failed, including the offline fallback.")


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and assert the schema the training pipeline relies on."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {missing}")

    cleaned = frame.copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str)
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()

    cleaned = cleaned[cleaned["label"].isin(("ham", "spam"))]
    cleaned = cleaned[cleaned["text"].str.strip().str.len() > 0]
    cleaned = cleaned.drop_duplicates(subset="text").reset_index(drop=True)

    if len(cleaned) < 100:
        raise DatasetError(f"Only {len(cleaned)} usable rows after validation; need at least 100.")
    if cleaned["label"].nunique() < 2:
        raise DatasetError("Dataset contains a single class; both 'ham' and 'spam' are required.")

    return cleaned


def stratified_sample(frame: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """Draw up to `per_class` rows from each label, preserving every column."""
    parts = [
        group.sample(min(len(group), per_class), random_state=seed)
        for _, group in frame.groupby("label", sort=True)
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_dataset(
    force_download: bool = False, max_rows: int | None = None, seed: int = 42
) -> pd.DataFrame:
    """Return a validated dataset with cleaned text and engineered features attached."""
    frame = load_raw_dataset(force_download=force_download)

    if max_rows and len(frame) > max_rows:
        frame = stratified_sample(frame, max_rows // 2, seed)

    frame["clean_text"] = frame["text"].map(clean_text)
    frame = frame[frame["clean_text"].str.len() > 0].reset_index(drop=True)
    return pd.concat([frame, engineer_features(frame["text"])], axis=1)


def write_sample(frame: pd.DataFrame, n: int = 2_000, seed: int = 42) -> Path:
    """Persist a small stratified sample that ships with the deployed app."""
    sample = stratified_sample(frame, n // 2, seed)
    columns = [c for c in ("subject", "body", "text", "label", "date", "source") if c in sample.columns]
    SAMPLE_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(SAMPLE_DATASET_CSV, index=False)
    return SAMPLE_DATASET_CSV


def load_sample() -> pd.DataFrame:
    """Load the bundled sample, falling back to a fresh prepare on a cold deploy."""
    if SAMPLE_DATASET_CSV.exists():
        frame = pd.read_csv(SAMPLE_DATASET_CSV)
        frame["clean_text"] = frame["text"].map(clean_text)
        return pd.concat([frame, engineer_features(frame["text"])], axis=1)
    return prepare_dataset(max_rows=4_000)
