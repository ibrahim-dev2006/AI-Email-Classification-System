"""Single-email classification with explanations and optional AI review."""

from __future__ import annotations

import streamlit as st

from src import charts, history, llm_classifier
from src.predictor import get_predictor, model_is_available
from src.ui import (
    configure_page,
    error_boundary,
    format_percent,
    hero,
    metric_cards,
    require_login,
    sidebar_account,
    verdict_banner,
)

SAMPLES = {
    "Phishing attempt": (
        "Urgent: Your account will be suspended",
        "Dear Customer,\n\nWe detected unusual activity on your account. Your access will be "
        "permanently suspended within 24 hours unless you verify your identity immediately.\n\n"
        "Click here to verify: http://secure-verify-account.example.net/login\n\n"
        "You will need your full card number, PIN and online banking password to complete "
        "verification. Failure to act now will result in permanent closure.\n\nSecurity Department",
    ),
    "Advance-fee fraud": (
        "CONGRATULATIONS!!! YOU HAVE WON $2,500,000",
        "ATTENTION WINNER,\n\nYour email address was selected in our international lottery draw. "
        "You have won TWO MILLION FIVE HUNDRED THOUSAND DOLLARS ($2,500,000.00 USD)!\n\n"
        "To claim your prize, send a processing fee of $250 via wire transfer along with your "
        "full name, address, bank account details and a copy of your passport.\n\n"
        "This offer is 100% risk free and guaranteed. Act now, this is a limited time offer!\n\n"
        "Dr. James Okafor, Claims Agent",
    ),
    "Legitimate business email": (
        "Re: Q3 budget review meeting",
        "Hi Sarah,\n\nThanks for sending through the draft figures yesterday. I've reviewed the "
        "Q3 numbers and they line up with what finance forecast back in June.\n\n"
        "Two things before Thursday's meeting:\n"
        "- The marketing line item looks about 8% over. Can you confirm whether that includes "
        "the conference spend?\n"
        "- I'd like to add a short section on headcount planning for Q4.\n\n"
        "I've attached my annotated version. Happy to walk through it beforehand if useful.\n\n"
        "Best,\nMichael",
    ),
    "Legitimate newsletter": (
        "Your weekly engineering digest",
        "Hello,\n\nHere's what shipped this week:\n\n"
        "- The search reindexing job now runs incrementally, cutting nightly runtime from "
        "4 hours to about 25 minutes.\n"
        "- We upgraded the Postgres cluster to 16.2. No action needed on your side.\n"
        "- Three flaky integration tests were fixed in the payments suite.\n\n"
        "Next week we begin the staged rollout of the new notification service.\n\n"
        "You're receiving this because you're on the platform team. "
        "Manage your preferences in your account settings.",
    ),
}


def render() -> None:
    """Render the classification page."""
    configure_page()
    user = require_login()
    sidebar_account(user)

    hero("Classify an email", "Score a single message, see why the model decided, and record the outcome.")

    if not model_is_available():
        st.warning(
            "No trained model is available. An administrator can train one from the **Admin** page, "
            "or run `python -m src.train` from the project root."
        )
        return

    _sample_picker()

    with st.form("classify_form"):
        subject = st.text_input("Subject", key="subject_input", placeholder="Enter the email subject line")
        body = st.text_area(
            "Message body", key="body_input", height=260, placeholder="Paste the full email body here"
        )

        columns = st.columns([2, 1])
        with columns[0]:
            provider = llm_classifier.provider_label()
            use_llm = st.checkbox(
                f"Also analyse with {provider} (AI second opinion)",
                value=False,
                disabled=not llm_classifier.api_key_available(),
                help=(
                    f"Sends the email to {provider} for written reasoning."
                    if llm_classifier.api_key_available()
                    else "Unavailable: no GROQ_API_KEY or ANTHROPIC_API_KEY is configured."
                ),
            )
        with columns[1]:
            submitted = st.form_submit_button("Classify email", type="primary", width="stretch")

    if submitted:
        _classify(user.username, subject, body, use_llm)

    if st.session_state.get("last_result"):
        _render_results()


def _sample_picker() -> None:
    """Offer one-click sample emails to populate the form."""
    st.caption("Load a sample email")
    columns = st.columns(len(SAMPLES))
    for column, (name, (subject, body)) in zip(columns, SAMPLES.items(), strict=True):
        with column:
            if st.button(name, width="stretch", key=f"sample_{name}"):
                st.session_state["subject_input"] = subject
                st.session_state["body_input"] = body
                st.session_state.pop("last_result", None)
                st.rerun()


def _classify(username: str, subject: str, body: str, use_llm: bool) -> None:
    """Run the local model, optionally the API, and persist the outcome."""
    with error_boundary("classification"):
        predictor = get_predictor()
        with st.spinner("Scoring with the trained model..."):
            result = predictor.predict(subject, body)

        prediction_id = history.log_prediction(
            username=username,
            subject=subject,
            body=body,
            predicted_label=result.label,
            spam_probability=result.spam_probability,
            confidence=result.confidence,
            decision=result.decision,
            model_name=result.model_name,
            source="local_model",
            latency_ms=result.latency_ms,
        )

        llm_result = llm_error = None
        if use_llm:
            with st.spinner(f"Requesting AI analysis from {llm_classifier.provider_label()}..."):
                llm_result, llm_error = llm_classifier.classify_safe(subject, body)

        st.session_state["last_result"] = result
        st.session_state["last_llm"] = llm_result
        st.session_state["last_llm_error"] = llm_error
        st.session_state["last_prediction_id"] = prediction_id
        st.session_state.pop("feedback_given", None)


def _render_results() -> None:
    """Render the verdict, score, explanation, and AI comparison."""
    result = st.session_state["last_result"]
    llm_result = st.session_state.get("last_llm")
    llm_error = st.session_state.get("last_llm_error")

    st.divider()

    kind = {"deliver": "ham", "quarantine": "spam", "review": "review"}[result.decision]
    verdict_banner(
        f"{result.display_label} — {result.decision_text}",
        f"The model scored this email {format_percent(result.spam_probability)} likely to be spam, "
        f"against a decision threshold of {result.threshold:.2f}.",
        kind,
    )

    metric_cards(
        [
            ("Spam probability", format_percent(result.spam_probability), "Calibrated model output"),
            ("Confidence", format_percent(result.confidence), "Distance from the decision boundary"),
            ("Decision", result.decision.title(), "Routing action"),
            ("Latency", f"{result.latency_ms:.1f} ms", "Local inference time"),
        ]
    )

    st.write("")
    st.altair_chart(charts.probability_bar(result.spam_probability, result.threshold), width="stretch")

    left, right = st.columns(2)

    with left:
        st.subheader("Why the model decided this")
        if result.top_terms:
            st.altair_chart(charts.contribution_chart(result.top_terms), width="stretch")
            st.caption(
                "Bars to the right pushed the score toward spam; bars to the left pushed it toward "
                "legitimate. Contribution is the term's TF-IDF weight multiplied by its model coefficient."
            )
        else:
            st.info("This model does not expose per-term weights, so no breakdown is available.")

    with right:
        st.subheader("AI second opinion")
        if llm_error:
            st.warning(llm_error)
            st.caption("The local model's verdict above is unaffected.")
        elif llm_result:
            agreement = "agrees with" if llm_result.label == result.label else "disagrees with"
            st.markdown(
                f"**{llm_result.provider_label} says: {llm_result.label.upper()}** "
                f"({format_percent(llm_result.spam_probability)} spam, {llm_result.confidence} confidence) "
                f"— this {agreement} the local model."
            )
            st.markdown(f"> {llm_result.reasoning}")

            if llm_result.indicators:
                st.caption("Signals identified")
                for indicator in llm_result.indicators[:6]:
                    marker = ":red_circle:" if indicator["direction"] == "spam" else ":green_circle:"
                    st.markdown(f"{marker} {indicator['signal']}")

            st.caption(
                f"Model: {llm_result.model} · {llm_result.latency_ms:.0f} ms · "
                f"{llm_result.input_tokens:,} in / {llm_result.output_tokens:,} out tokens"
            )
            if llm_result.truncated:
                st.caption("Note: the email was truncated to fit the analysis request.")
        else:
            st.info(
                f"Enable **Also analyse with {llm_classifier.provider_label()}** before classifying "
                "to get written reasoning alongside the model's score."
            )

    _feedback_controls()


def _feedback_controls() -> None:
    """Let the analyst confirm or correct the verdict, feeding the dashboard."""
    st.divider()
    prediction_id = st.session_state.get("last_prediction_id")
    if prediction_id is None:
        return

    if st.session_state.get("feedback_given"):
        st.success("Thanks — your feedback was recorded and appears on the Dashboard.")
        return

    st.caption("Was this classification correct?")
    yes, no, _ = st.columns([1, 1, 4])
    with yes:
        if st.button("Correct", width="stretch"):
            history.record_feedback(prediction_id, "correct")
            st.session_state["feedback_given"] = True
            st.rerun()
    with no:
        if st.button("Incorrect", width="stretch"):
            history.record_feedback(prediction_id, "incorrect")
            st.session_state["feedback_given"] = True
            st.rerun()
