"""API-based email analysis, backed by either the Groq or the Anthropic API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .config import LLM, load_settings
from .exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)
from .logger import get_logger
from .preprocessing import combine_subject_body

logger = get_logger(__name__)

MAX_INPUT_CHARS = 40_000

PROVIDER_LABELS = {"groq": "Groq", "anthropic": "Claude"}

SYSTEM_PROMPT = """You are an email security analyst. Classify an email as spam or legitimate.

Treat as spam: unsolicited bulk mail, phishing, advance-fee fraud, deceptive marketing,
credential harvesting, and malware lures.
Treat as legitimate: genuine business correspondence, transactional notices, newsletters the
recipient plausibly opted into, and personal mail.

Judge intent and structure rather than individual keywords. Legitimate marketing exists;
urgency alone is not proof of spam. Weigh sender-recipient mismatch, implausible offers,
credential or payment requests, obfuscated links, and pressure tactics.

Set spam_probability to your calibrated belief the email is spam, not a rounded guess."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["spam", "ham"]},
        "spam_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
        "indicators": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "signal": {"type": "string"},
                    "direction": {"type": "string", "enum": ["spam", "legitimate"]},
                },
                "required": ["signal", "direction"],
                "additionalProperties": False,
            },
        },
        "recommended_action": {
            "type": "string",
            "enum": ["deliver", "review", "quarantine"],
        },
    },
    "required": [
        "label",
        "spam_probability",
        "confidence",
        "reasoning",
        "indicators",
        "recommended_action",
    ],
    "additionalProperties": False,
}

# Groq exposes JSON mode rather than schema-constrained decoding, so the shape goes in the prompt.
_JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else, matching this schema exactly:\n"
    + json.dumps(RESPONSE_SCHEMA, indent=2)
)


@dataclass
class LLMResult:
    """Structured verdict returned by the API-based classifier."""

    label: str
    spam_probability: float
    confidence: str
    reasoning: str
    indicators: list[dict] = field(default_factory=list)
    recommended_action: str = "review"
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False
    source: str = "llm"

    @property
    def is_spam(self) -> bool:
        return self.label == "spam"

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider.title())


def active_provider() -> str | None:
    """Return the provider backed by a configured key, preferring Groq."""
    settings = load_settings()
    if settings.groq_api_key:
        return "groq"
    if settings.anthropic_api_key:
        return "anthropic"
    return None


def provider_label() -> str:
    """Human-readable name of the active provider, for UI labels."""
    provider = active_provider()
    return PROVIDER_LABELS.get(provider, "AI") if provider else "AI"


def api_key_available() -> bool:
    """Report whether any AI provider is configured."""
    return active_provider() is not None


def _prepare(subject: str, body: str) -> tuple[str, bool]:
    """Join and, if necessary, truncate the email for the request."""
    text = combine_subject_body(subject, body)
    if len(text) > MAX_INPUT_CHARS:
        logger.warning("Email of %s chars truncated to %s", len(text), MAX_INPUT_CHARS)
        return text[:MAX_INPUT_CHARS], True
    return text, False


def _parse(raw: str) -> dict:
    """Parse and sanity-check the model's JSON verdict."""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMResponseError() from exc

    missing = {"label", "spam_probability", "reasoning"} - set(payload)
    if missing:
        raise LLMResponseError(
            user_message=f"The AI service returned an incomplete response (missing {', '.join(sorted(missing))})."
        )

    label = str(payload["label"]).strip().lower()
    if label not in ("spam", "ham"):
        raise LLMResponseError(user_message=f"The AI service returned an unknown label '{label}'.")

    try:
        probability = min(max(float(payload["spam_probability"]), 0.0), 1.0)
    except (TypeError, ValueError) as exc:
        raise LLMResponseError(user_message="The AI service returned a non-numeric probability.") from exc

    payload["label"] = label
    payload["spam_probability"] = probability
    return payload


def _classify_groq(text: str) -> tuple[dict, str, int, int]:
    """Send the email to Groq and return the parsed verdict plus usage."""
    import groq

    api_key = load_settings().groq_api_key
    client = groq.Groq(api_key=api_key, timeout=LLM.timeout_seconds, max_retries=LLM.max_retries)

    try:
        response = client.chat.completions.create(
            model=LLM.groq_model,
            temperature=0,
            max_tokens=LLM.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + _JSON_INSTRUCTION},
                {"role": "user", "content": f"Classify this email:\n\n{text}"},
            ],
        )
    except groq.AuthenticationError as exc:
        raise LLMConfigurationError(
            user_message="The Groq API key was rejected. Check GROQ_API_KEY in your secrets."
        ) from exc
    except groq.NotFoundError as exc:
        raise LLMError(
            user_message=f"Model '{LLM.groq_model}' is not available to this Groq key."
        ) from exc
    except groq.RateLimitError as exc:
        raise LLMRateLimitError() from exc
    except groq.APIStatusError as exc:
        raise LLMError(
            user_message=f"Groq returned an error ({exc.status_code}). Try again shortly."
        ) from exc
    except groq.APIConnectionError as exc:
        raise LLMError(user_message="Could not reach Groq. Check your network connection.") from exc

    usage = response.usage
    return (
        _parse(response.choices[0].message.content),
        response.model,
        getattr(usage, "prompt_tokens", 0),
        getattr(usage, "completion_tokens", 0),
    )


def _classify_anthropic(text: str) -> tuple[dict, str, int, int]:
    """Send the email to the Anthropic API and return the parsed verdict plus usage."""
    import anthropic

    api_key = load_settings().anthropic_api_key
    client = anthropic.Anthropic(
        api_key=api_key, timeout=LLM.timeout_seconds, max_retries=LLM.max_retries
    )

    try:
        response = client.messages.create(
            model=LLM.anthropic_model,
            max_tokens=LLM.max_tokens,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": LLM.anthropic_effort,
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            messages=[{"role": "user", "content": f"Classify this email:\n\n{text}"}],
        )
    except anthropic.AuthenticationError as exc:
        raise LLMConfigurationError(
            user_message="The Anthropic API key was rejected. Check ANTHROPIC_API_KEY."
        ) from exc
    except anthropic.NotFoundError as exc:
        raise LLMError(
            user_message=f"Model '{LLM.anthropic_model}' is unavailable to this API key."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise LLMRateLimitError() from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(
            user_message=f"The AI service returned an error ({exc.status_code}). Try again shortly."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(
            user_message="Could not reach the AI service. Check your network connection."
        ) from exc

    if response.stop_reason == "refusal":
        raise LLMResponseError(user_message="The AI service declined to analyse this content.")

    try:
        raw = next(b.text for b in response.content if b.type == "text")
    except StopIteration as exc:
        raise LLMResponseError() from exc

    return _parse(raw), response.model, response.usage.input_tokens, response.usage.output_tokens


def classify(subject: str = "", body: str = "") -> LLMResult:
    """Classify one email through the configured AI provider."""
    provider = active_provider()
    if provider is None:
        raise LLMConfigurationError(
            user_message="No AI provider is configured. Add GROQ_API_KEY or ANTHROPIC_API_KEY to enable AI analysis."
        )

    text, truncated = _prepare(subject, body)
    started = time.perf_counter()

    backend = _classify_groq if provider == "groq" else _classify_anthropic
    payload, model, input_tokens, output_tokens = backend(text)

    return LLMResult(
        label=payload["label"],
        spam_probability=payload["spam_probability"],
        confidence=payload.get("confidence", "medium"),
        reasoning=payload["reasoning"],
        indicators=[
            i for i in payload.get("indicators", []) if isinstance(i, dict) and "signal" in i
        ],
        recommended_action=payload.get("recommended_action", "review"),
        model=model,
        provider=provider,
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        truncated=truncated,
    )


def classify_safe(subject: str = "", body: str = "") -> tuple[LLMResult | None, str | None]:
    """Classify without raising, returning either a result or a user-facing message."""
    try:
        return classify(subject, body), None
    except LLMError as exc:
        logger.warning("LLM classification unavailable: %s", exc)
        return None, exc.user_message
    except Exception as exc:  # noqa: BLE001 - the local model must still be usable
        logger.exception("Unexpected LLM failure")
        return None, f"Unexpected AI service error: {exc}"
