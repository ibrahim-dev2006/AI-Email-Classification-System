"""Test suite. Run with: python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import auth, history, llm_classifier  # noqa: E402
from src.dataset import stratified_sample, validate_dataset  # noqa: E402
from src.evaluation import best_threshold, error_cost, evaluate, threshold_sweep  # noqa: E402
from src.exceptions import (  # noqa: E402
    AccountLockedError,
    DatasetError,
    InvalidCredentialsError,
    LLMResponseError,
    UserExistsError,
    ValidationError,
    WeakPasswordError,
)
from src.predictor import decide, get_predictor, model_is_available  # noqa: E402
from src.preprocessing import (  # noqa: E402
    TextCleaner,
    clean_text,
    combine_subject_body,
    engineer_features,
    validate_email_text,
)


class TestTextCleaning(unittest.TestCase):
    """Normalisation must be stable, lossless of signal, and injection-proof."""

    def test_lowercases_and_collapses_whitespace(self):
        self.assertEqual(clean_text("  Hello   WORLD  "), "hello world")

    def test_strips_html_tags_and_entities(self):
        self.assertEqual(clean_text("<b>Buy</b>&nbsp;now"), "buy now")

    def test_replaces_urls_with_a_stable_token(self):
        self.assertIn("urltoken", clean_text("Visit https://evil.example.com/path?a=1"))
        self.assertIn("urltoken", clean_text("Go to www.example.com now"))

    def test_replaces_emails_and_amounts(self):
        cleaned = clean_text("Contact bob@example.com to claim $5,000")
        self.assertIn("emailtoken", cleaned)
        self.assertIn("moneytoken", cleaned)

    def test_replaces_bare_numbers(self):
        self.assertIn("numtoken", clean_text("Call in 24 hours"))

    def test_folds_unicode_evasion(self):
        self.assertEqual(clean_text("VIAGRA"), clean_text("ViAgRa"))
        self.assertIn("cafe", clean_text("Café"))

    def test_handles_missing_and_empty_input(self):
        for value in (None, "", "   ", float("nan")):
            self.assertEqual(clean_text(value), "")

    def test_is_deterministic(self):
        text = "URGENT!! Claim $1000 at http://x.example.com <b>now</b>"
        self.assertEqual(clean_text(text), clean_text(text))

    def test_transformer_matches_the_function(self):
        texts = ["Free MONEY!!!", "Meeting at 3pm", ""]
        self.assertEqual(TextCleaner().fit_transform(texts), [clean_text(t) for t in texts])


class TestCombineAndValidate(unittest.TestCase):
    """Subject and body join predictably, and bad input is rejected early."""

    def test_combines_subject_and_body(self):
        self.assertEqual(combine_subject_body("Hi", "There"), "Hi\n\nThere")

    def test_tolerates_missing_parts(self):
        self.assertEqual(combine_subject_body(None, "Body"), "Body")
        self.assertEqual(combine_subject_body("Subject", None), "Subject")

    def test_rejects_too_short_input(self):
        with self.assertRaises(ValidationError):
            validate_email_text("", "")

    def test_rejects_unreadable_input(self):
        with self.assertRaises(ValidationError):
            validate_email_text("", "!!!!!!!!")

    def test_accepts_valid_input(self):
        self.assertIn("Quarterly report", validate_email_text("Quarterly report", "Please review."))


class TestFeatureEngineering(unittest.TestCase):
    """Engineered features must separate obvious spam from obvious ham."""

    def setUp(self):
        self.frame = engineer_features(
            pd.Series(
                [
                    "FREE MONEY!!! WIN CASH NOW!!! Click http://x.example.com",
                    "Hi Sarah, attached are the notes from the meeting on Thursday.",
                ]
            )
        )

    def test_produces_one_row_per_input(self):
        self.assertEqual(len(self.frame), 2)

    def test_detects_higher_uppercase_ratio_in_shouting(self):
        self.assertGreater(self.frame["uppercase_ratio"][0], self.frame["uppercase_ratio"][1])

    def test_counts_exclamations_and_urls(self):
        self.assertGreater(self.frame["exclamation_count"][0], 0)
        self.assertEqual(self.frame["url_count"][0], 1)
        self.assertEqual(self.frame["url_count"][1], 0)

    def test_counts_spam_trigger_words(self):
        self.assertGreater(self.frame["spam_trigger_hits"][0], self.frame["spam_trigger_hits"][1])

    def test_all_features_are_numeric(self):
        self.assertTrue(all(np.issubdtype(dtype, np.floating) for dtype in self.frame.dtypes))


class TestDatasetValidation(unittest.TestCase):
    """The schema contract the training pipeline depends on is enforced."""

    def _frame(self, n=200):
        return pd.DataFrame(
            {
                "text": [f"message number {i}" for i in range(n)],
                "label": ["spam" if i % 2 else "ham" for i in range(n)],
            }
        )

    def test_accepts_a_well_formed_frame(self):
        self.assertEqual(len(validate_dataset(self._frame())), 200)

    def test_rejects_missing_columns(self):
        with self.assertRaises(DatasetError):
            validate_dataset(pd.DataFrame({"text": ["a"]}))

    def test_rejects_a_single_class(self):
        frame = self._frame()
        frame["label"] = "spam"
        with self.assertRaises(DatasetError):
            validate_dataset(frame)

    def test_rejects_too_few_rows(self):
        with self.assertRaises(DatasetError):
            validate_dataset(self._frame(n=10))

    def test_drops_duplicates_and_blank_rows(self):
        frame = self._frame()
        frame = pd.concat([frame, frame.head(20)], ignore_index=True)
        frame.loc[len(frame)] = {"text": "   ", "label": "ham"}
        self.assertEqual(len(validate_dataset(frame)), 200)

    def test_normalises_label_case(self):
        frame = self._frame()
        frame["label"] = frame["label"].str.upper()
        self.assertSetEqual(set(validate_dataset(frame)["label"]), {"ham", "spam"})

    def test_stratified_sample_preserves_all_columns(self):
        sample = stratified_sample(self._frame(), per_class=20)
        self.assertIn("label", sample.columns)
        self.assertEqual(len(sample), 40)


class TestEvaluation(unittest.TestCase):
    """Metrics, sweeps, and cost accounting behave as expected."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.y_true = np.array([0] * 100 + [1] * 100)
        self.y_score = np.clip(
            np.concatenate([rng.normal(0.2, 0.1, 100), rng.normal(0.8, 0.1, 100)]), 0.001, 0.999
        )
        self.y_pred = (self.y_score >= 0.5).astype(int)

    def test_computes_a_sensible_metric_suite(self):
        result = evaluate(self.y_true, self.y_pred, self.y_score)
        self.assertGreater(result.accuracy, 0.9)
        self.assertGreater(result.roc_auc, 0.95)
        self.assertEqual(result.support, 200)
        self.assertEqual(np.array(result.confusion).sum(), 200)

    def test_perfect_predictions_score_perfectly(self):
        result = evaluate(self.y_true, self.y_true, self.y_true.astype(float))
        self.assertAlmostEqual(result.accuracy, 1.0)
        self.assertAlmostEqual(result.f1, 1.0)

    def test_sweep_covers_the_threshold_range(self):
        sweep = threshold_sweep(self.y_true, self.y_score)
        self.assertGreater(len(sweep), 10)
        self.assertTrue(all(0 <= row["threshold"] <= 1 for row in sweep))

    def test_recall_falls_as_the_threshold_rises(self):
        sweep = threshold_sweep(self.y_true, self.y_score)
        recalls = [row["recall"] for row in sweep]
        self.assertGreaterEqual(recalls[0], recalls[-1])

    def test_best_threshold_optimises_the_requested_objective(self):
        sweep = threshold_sweep(self.y_true, self.y_score)
        chosen = best_threshold(sweep, objective="f1")
        self.assertEqual(chosen, max(sweep, key=lambda r: r["f1"])["threshold"])

    def test_cost_weights_false_positives_more_heavily(self):
        heavy_fp = error_cost([[90, 10], [0, 100]])
        heavy_fn = error_cost([[100, 0], [10, 90]])
        self.assertGreater(heavy_fp["total_cost"], heavy_fn["total_cost"])


class TestDecisionRouting(unittest.TestCase):
    """Scores map onto deliver / review / quarantine bands."""

    def test_low_scores_are_delivered(self):
        self.assertEqual(decide(0.01), "deliver")

    def test_high_scores_are_quarantined(self):
        self.assertEqual(decide(0.99), "quarantine")

    def test_middling_scores_go_to_review(self):
        self.assertEqual(decide(0.5), "review")


class TestAuthentication(unittest.TestCase):
    """Account lifecycle, password policy, and lockout."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original = auth.AUTH_DB_PATH
        auth.AUTH_DB_PATH = Path(self._tmp.name) / "users.db"
        auth.init_db()

    def tearDown(self):
        auth.AUTH_DB_PATH = self._original
        self._tmp.cleanup()

    def test_registers_and_authenticates_a_user(self):
        auth.register_user("alice", "Password123")
        user = auth.authenticate("alice", "Password123")
        self.assertEqual(user.username, "alice")
        self.assertEqual(user.role, "analyst")

    def test_username_is_case_insensitive(self):
        auth.register_user("Bob", "Password123")
        self.assertEqual(auth.authenticate("bob", "Password123").username, "Bob")

    def test_rejects_a_wrong_password(self):
        auth.register_user("carol", "Password123")
        with self.assertRaises(InvalidCredentialsError):
            auth.authenticate("carol", "WrongPassword1")

    def test_rejects_an_unknown_user(self):
        with self.assertRaises(InvalidCredentialsError):
            auth.authenticate("nobody", "Password123")

    def test_rejects_duplicate_usernames(self):
        auth.register_user("dave", "Password123")
        with self.assertRaises(UserExistsError):
            auth.register_user("dave", "Password123")

    def test_enforces_the_password_policy(self):
        for weak in ("short1", "alllettersonly", "12345678"):
            with self.assertRaises(WeakPasswordError):
                auth.register_user(f"user_{weak}", weak)

    def test_passwords_are_never_stored_in_plaintext(self):
        auth.register_user("erin", "Password123")
        raw = auth.AUTH_DB_PATH.read_bytes()
        self.assertNotIn(b"Password123", raw)

    def test_hashes_are_salted_per_user(self):
        first, _ = auth.hash_password("SamePassword1")
        second, _ = auth.hash_password("SamePassword1")
        self.assertNotEqual(first, second)

    def test_locks_an_account_after_repeated_failures(self):
        auth.register_user("frank", "Password123")
        for _ in range(auth.MAX_FAILED_ATTEMPTS):
            with self.assertRaises((InvalidCredentialsError, AccountLockedError)):
                auth.authenticate("frank", "WrongPassword1")
        with self.assertRaises(AccountLockedError):
            auth.authenticate("frank", "Password123")

    def test_the_failure_counter_persists_across_attempts(self):
        auth.register_user("grace", "Password123")
        remaining = []
        for _ in range(3):
            with self.assertRaises(InvalidCredentialsError) as caught:
                auth.authenticate("grace", "WrongPassword1")
            remaining.append(caught.exception.user_message)
        for attempt, message in enumerate(remaining, start=1):
            self.assertIn(f"{auth.MAX_FAILED_ATTEMPTS - attempt} attempt", message)

    def test_a_successful_login_resets_the_failure_counter(self):
        auth.register_user("grace", "Password123")
        for _ in range(2):
            with self.assertRaises(InvalidCredentialsError):
                auth.authenticate("grace", "WrongPassword1")
        auth.authenticate("grace", "Password123")
        with self.assertRaises(InvalidCredentialsError) as caught:
            auth.authenticate("grace", "WrongPassword1")
        self.assertIn(f"{auth.MAX_FAILED_ATTEMPTS - 1} attempt", caught.exception.user_message)

    def test_changes_a_password(self):
        auth.register_user("heidi", "Password123")
        auth.change_password("heidi", "Password123", "NewPassword456")
        auth.authenticate("heidi", "NewPassword456")
        with self.assertRaises(InvalidCredentialsError):
            auth.authenticate("heidi", "Password123")

    def test_bootstraps_an_admin_account(self):
        self.assertTrue(any(u.is_admin for u in auth.list_users()))

    def test_refuses_to_delete_the_last_admin(self):
        admins = [u for u in auth.list_users() if u.is_admin]
        auth.register_user("ivan", "Password123", role="admin")
        actor = auth.authenticate("ivan", "Password123")
        for admin in admins:
            auth.delete_user(admin.username, acting_user=actor)
        with self.assertRaises(Exception):
            auth.delete_user("ivan", acting_user=actor)


class TestHistoryLog(unittest.TestCase):
    """Predictions are persisted and aggregated correctly."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original = history.PREDICTION_LOG_PATH
        history.PREDICTION_LOG_PATH = Path(self._tmp.name) / "predictions.db"
        history.init_db()

    def tearDown(self):
        history.PREDICTION_LOG_PATH = self._original
        self._tmp.cleanup()

    def _log(self, label="spam", probability=0.9, decision="quarantine"):
        return history.log_prediction(
            username="tester",
            subject="Test subject",
            body="Test body",
            predicted_label=label,
            spam_probability=probability,
            confidence=abs(probability - 0.5) * 2,
            decision=decision,
            model_name="linear_svc",
            source="local_model",
            latency_ms=1.5,
        )

    def test_starts_empty(self):
        self.assertEqual(history.summary_stats()["total"], 0)

    def test_logs_and_counts_predictions(self):
        self._log("spam", 0.95)
        self._log("ham", 0.05, "deliver")
        stats = history.summary_stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["spam"], 1)
        self.assertEqual(stats["ham"], 1)

    def test_records_analyst_feedback(self):
        history.record_feedback(self._log(), "correct")
        stats = history.summary_stats()
        self.assertEqual(stats["feedback_given"], 1)
        self.assertEqual(stats["accuracy_from_feedback"], 1.0)

    def test_scopes_history_by_user(self):
        self._log()
        self.assertEqual(len(history.load_history(username="tester")), 1)
        self.assertEqual(len(history.load_history(username="someone_else")), 0)

    def test_clears_the_log(self):
        self._log()
        self.assertEqual(history.clear_history(), 1)
        self.assertEqual(history.summary_stats()["total"], 0)


class TestLLMResponseParsing(unittest.TestCase):
    """Malformed provider responses are rejected rather than trusted."""

    VALID = (
        '{"label":"spam","spam_probability":0.91,"confidence":"high",'
        '"reasoning":"Requests bank details.","indicators":[],'
        '"recommended_action":"quarantine"}'
    )

    def test_parses_a_valid_payload(self):
        payload = llm_classifier._parse(self.VALID)
        self.assertEqual(payload["label"], "spam")
        self.assertAlmostEqual(payload["spam_probability"], 0.91)

    def test_rejects_invalid_json(self):
        with self.assertRaises(LLMResponseError):
            llm_classifier._parse("not json at all")

    def test_rejects_a_missing_field(self):
        with self.assertRaises(LLMResponseError):
            llm_classifier._parse('{"label":"spam"}')

    def test_rejects_an_unknown_label(self):
        with self.assertRaises(LLMResponseError):
            llm_classifier._parse(
                '{"label":"phishing","spam_probability":0.9,"reasoning":"x"}'
            )

    def test_rejects_a_non_numeric_probability(self):
        with self.assertRaises(LLMResponseError):
            llm_classifier._parse(
                '{"label":"spam","spam_probability":"very high","reasoning":"x"}'
            )

    def test_clamps_an_out_of_range_probability(self):
        payload = llm_classifier._parse(
            '{"label":"spam","spam_probability":1.7,"reasoning":"x"}'
        )
        self.assertEqual(payload["spam_probability"], 1.0)

    def test_normalises_label_case(self):
        payload = llm_classifier._parse(
            '{"label":"SPAM","spam_probability":0.8,"reasoning":"x"}'
        )
        self.assertEqual(payload["label"], "spam")


@unittest.skipUnless(llm_classifier.api_key_available(), "no AI provider key configured")
class TestLLMIntegration(unittest.TestCase):
    """Live call against the configured provider. Skipped when no key is present."""

    def test_reports_an_active_provider(self):
        self.assertIn(llm_classifier.active_provider(), ("groq", "anthropic"))

    def test_classifies_obvious_spam(self):
        result, error = llm_classifier.classify_safe(
            "URGENT: verify your account now",
            "Your account will be suspended. Confirm your full card number, PIN and online "
            "banking password at http://secure-verify.example.net immediately.",
        )
        self.assertIsNone(error, f"provider call failed: {error}")
        self.assertEqual(result.label, "spam")
        self.assertGreater(result.spam_probability, 0.5)
        self.assertTrue(result.reasoning)
        self.assertGreater(result.input_tokens, 0)

    def test_classifies_obvious_legitimate_mail(self):
        result, error = llm_classifier.classify_safe(
            "Re: Q3 budget review",
            "Hi Sarah, thanks for the draft figures. The Q3 numbers match what finance "
            "forecast in June. I've attached my annotated version ahead of Thursday.",
        )
        self.assertIsNone(error, f"provider call failed: {error}")
        self.assertEqual(result.label, "ham")
        self.assertLess(result.spam_probability, 0.5)


@unittest.skipUnless(model_is_available(), "no trained model artifact present")
class TestTrainedModel(unittest.TestCase):
    """End-to-end checks against the real trained artifact."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()

    def test_flags_obvious_spam(self):
        result = self.predictor.predict(
            "CONGRATULATIONS you have WON $1,000,000",
            "Click http://claim-prize.example.com now and send your bank details to claim "
            "your free cash prize. Limited time offer, act now, 100% risk free guarantee!",
        )
        self.assertEqual(result.label, "spam")
        self.assertGreater(result.spam_probability, 0.5)

    def test_passes_obvious_legitimate_mail(self):
        result = self.predictor.predict(
            "Re: Thursday budget meeting",
            "Hi Daren, thanks for sending the gas nomination figures through. I've attached "
            "my notes from the Houston meeting. Let me know if you have questions.",
        )
        self.assertEqual(result.label, "ham")
        self.assertLess(result.spam_probability, 0.5)

    def test_produces_a_routing_decision(self):
        result = self.predictor.predict("Hello", "This is a short test message about a meeting.")
        self.assertIn(result.decision, ("deliver", "review", "quarantine"))

    def test_explains_its_prediction(self):
        result = self.predictor.predict("Free money", "Click here to win a free prize now!")
        self.assertTrue(result.top_terms)
        self.assertIn(result.top_terms[0]["direction"], ("spam", "legitimate"))

    def test_rejects_empty_input(self):
        with self.assertRaises(ValidationError):
            self.predictor.predict("", "")

    def test_scores_a_dataframe(self):
        frame = pd.DataFrame({"text": ["Win free cash now!!!", "Meeting notes attached", ""]})
        scored = self.predictor.predict_frame(frame)
        self.assertIn("spam_probability", scored.columns)
        self.assertEqual(scored["predicted_label"].iloc[2], "unscored")

    def test_probabilities_stay_in_range(self):
        frame = pd.DataFrame({"text": ["a" * 50, "Free money now", "Quarterly report attached"]})
        scored = self.predictor.predict_frame(frame)
        valid = scored["spam_probability"].dropna()
        self.assertTrue(((valid >= 0) & (valid <= 1)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
