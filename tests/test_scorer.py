"""Tests for the dependency-free serverless scorer used by the Vercel demo.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_API = PROJECT_ROOT / "vercel-demo" / "api"
MODEL_JSON = DEMO_API / "model.json"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DEMO_API))

from src.preprocessing import clean_text as reference_clean  # noqa: E402


@unittest.skipUnless(MODEL_JSON.exists(), "vercel-demo/api/model.json not built")
class TestServerlessScorer(unittest.TestCase):
    """The exported model must behave like the trained pipeline."""

    @classmethod
    def setUpClass(cls):
        from scorer import Scorer  # noqa: PLC0415

        cls.model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
        cls.scorer = Scorer(cls.model)

    def test_model_export_is_internally_consistent(self):
        self.assertEqual(len(self.model["terms"]), len(self.model["idf"]))
        self.assertEqual(len(self.model["terms"]), len(self.model["coef"]))
        self.assertGreater(len(self.model["terms"]), 1000)

    def test_flags_obvious_spam(self):
        result = self.scorer.classify(
            "CONGRATULATIONS you have WON $1,000,000\n\nClick http://claim-prize.example.com "
            "and send your bank details to claim your free cash prize. Act now, risk free!"
        )
        self.assertEqual(result["label"], "spam")
        self.assertGreater(result["spam_probability"], 0.5)
        self.assertEqual(result["decision"], "quarantine")

    def test_passes_obvious_legitimate_mail(self):
        result = self.scorer.classify(
            "Re: Thursday budget meeting\n\nHi Daren, thanks for sending the gas nomination "
            "figures. I have attached my notes from the Houston meeting."
        )
        self.assertEqual(result["label"], "ham")
        self.assertLess(result["spam_probability"], 0.5)
        self.assertEqual(result["decision"], "deliver")

    def test_probability_is_always_in_range(self):
        for text in ("", "a", "free money now", "quarterly report attached", "!!!!!", "x" * 5000):
            with self.subTest(text=text[:20]):
                self.assertTrue(0.0 <= self.scorer.score(text) <= 1.0)

    def test_vector_is_l2_normalised(self):
        vector = self.scorer.vectorise("free money click here to win a prize now")
        norm = math.sqrt(sum(v * v for v in vector.values()))
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_unknown_text_produces_an_empty_vector(self):
        self.assertEqual(self.scorer.vectorise("zzzqqq xxxyyy wwwvvv"), {})

    def test_scoring_is_deterministic(self):
        text = "Win a free prize now, click here!"
        self.assertEqual(self.scorer.score(text), self.scorer.score(text))

    def test_explanations_are_ranked_by_magnitude(self):
        terms = self.scorer.explain("free money click here to win a cash prize now")
        self.assertTrue(terms)
        magnitudes = [abs(t["contribution"]) for t in terms]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        for term in terms:
            self.assertIn(term["direction"], ("spam", "legitimate"))

    def test_decision_bands_match_the_streamlit_app(self):
        from src.config import REVIEW_BAND_HIGH, REVIEW_BAND_LOW  # noqa: PLC0415

        self.assertEqual(REVIEW_BAND_LOW, 0.35)
        self.assertEqual(REVIEW_BAND_HIGH, 0.75)


class TestCleanTextParity(unittest.TestCase):
    """The scorer duplicates clean_text; the two copies must not drift apart."""

    CASES = (
        "",
        "   ",
        "Hello WORLD",
        "<b>Buy</b>&nbsp;now",
        "Visit https://evil.example.com/path?a=1 today",
        "Go to www.example.com now",
        "Contact bob@example.com to claim $5,000",
        "Call 555-012-3456 within 24 hours",
        "Café naïve résumé",
        "Price: £250 or 300 EUR",
        "MIXED case W1TH d1g1ts and !!! punctuation ???",
        "line one\nline two\n\n\nline three",
    )

    def test_matches_the_training_implementation(self):
        from scorer import clean_text as serverless_clean  # noqa: PLC0415

        for text in self.CASES:
            with self.subTest(text=text[:30]):
                self.assertEqual(serverless_clean(text), reference_clean(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
