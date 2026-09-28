"""Headless UI tests driving each Streamlit page with AppTest.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src import auth, history  # noqa: E402
from src.auth import User  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIEWS = ("login", "home", "classify", "batch", "dashboard", "performance", "admin")

ADMIN = User(id=1, username="admin", role="admin", created_at="2026-01-01T00:00:00", last_login=None)
ANALYST = User(id=2, username="analyst", role="analyst", created_at="2026-01-01T00:00:00", last_login=None)


class ViewTestCase(unittest.TestCase):
    """Base class that builds a runnable script per view against temp databases."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)

        cls._auth_path, cls._history_path = auth.AUTH_DB_PATH, history.PREDICTION_LOG_PATH
        auth.AUTH_DB_PATH = tmp / "users.db"
        history.PREDICTION_LOG_PATH = tmp / "predictions.db"
        auth.init_db()
        history.init_db()
        auth.register_user("analyst", "Password123", role="analyst")

        cls._scripts = {}
        for name in VIEWS:
            script = tmp / f"run_{name}.py"
            script.write_text(
                "import sys\n"
                f"sys.path.insert(0, r'{PROJECT_ROOT}')\n"
                f"from views import {name}\n"
                f"{name}.render()\n",
                encoding="utf-8",
            )
            cls._scripts[name] = str(script)

    @classmethod
    def tearDownClass(cls):
        auth.AUTH_DB_PATH = cls._auth_path
        history.PREDICTION_LOG_PATH = cls._history_path
        cls._tmp.cleanup()

    def run_view(self, name: str, user: User | None = ADMIN, **session) -> AppTest:
        """Render one view and assert it raised no uncaught exception."""
        app = AppTest.from_file(self._scripts[name])
        if user is not None:
            app.session_state["user"] = user
        for key, value in session.items():
            app.session_state[key] = value
        app.run(timeout=180)
        self.assertFalse(
            app.exception,
            f"{name} raised: " + "; ".join(f"{e.type}: {e.message}" for e in app.exception),
        )
        return app


class TestAllViewsRender(ViewTestCase):
    """Every page renders without an uncaught exception or a surfaced error."""

    def test_every_view_renders_cleanly(self):
        for name in VIEWS:
            with self.subTest(view=name):
                app = self.run_view(name, user=None if name == "login" else ADMIN)
                self.assertEqual([e.value for e in app.error], [], f"{name} surfaced an error")


class TestAppRouter(ViewTestCase):
    """app.py itself must route without colliding page paths or raising."""

    APP = str(PROJECT_ROOT / "app.py")

    def run_app(self, user: User | None) -> AppTest:
        app = AppTest.from_file(self.APP)
        if user is not None:
            app.session_state["user"] = user
        app.run(timeout=180)
        self.assertFalse(
            app.exception,
            "app.py raised: " + "; ".join(f"{e.type}: {e.message}" for e in app.exception),
        )
        return app

    def test_shows_the_login_page_when_signed_out(self):
        app = self.run_app(None)
        self.assertTrue(app.text_input, "the sign-in form should render")

    def test_routes_an_analyst_without_error(self):
        app = self.run_app(ANALYST)
        self.assertEqual([e.value for e in app.error], [])

    def test_routes_an_admin_without_error(self):
        app = self.run_app(ADMIN)
        self.assertEqual([e.value for e in app.error], [])


class TestAuthenticationGate(ViewTestCase):
    """Signed-out and under-privileged users are handled correctly."""

    def test_protected_views_stop_without_a_user(self):
        for name in ("home", "classify", "batch", "dashboard", "performance"):
            with self.subTest(view=name):
                app = self.run_view(name, user=None)
                self.assertTrue(app.warning, f"{name} should warn when signed out")

    def test_admin_page_rejects_a_non_admin(self):
        app = self.run_view("admin", user=ANALYST)
        self.assertTrue(any("restricted" in e.value.lower() for e in app.error))

    def test_admin_page_accepts_an_admin(self):
        app = self.run_view("admin", user=ADMIN)
        self.assertEqual([e.value for e in app.error], [])
        self.assertTrue(app.tabs)


class TestLoginFlow(ViewTestCase):
    """The sign-in form authenticates against the user database."""

    def test_rejects_wrong_credentials(self):
        app = self.run_view("login", user=None)
        app.text_input[0].set_value("analyst")
        app.text_input[1].set_value("TotallyWrong9")
        app.button[0].click().run(timeout=60)

        self.assertTrue(app.error)
        self.assertNotIn("user", app.session_state)

    def test_accepts_correct_credentials(self):
        app = self.run_view("login", user=None)
        app.text_input[0].set_value("analyst")
        app.text_input[1].set_value("Password123")
        app.button[0].click().run(timeout=60)

        self.assertEqual(app.session_state["user"].username, "analyst")

    def test_requires_both_fields(self):
        app = self.run_view("login", user=None)
        app.text_input[0].set_value("analyst")
        app.button[0].click().run(timeout=60)
        self.assertTrue(app.error)


class TestClassifyFlow(ViewTestCase):
    """Submitting an email produces a verdict and logs it."""

    def test_classifies_a_spam_email_end_to_end(self):
        app = self.run_view("classify", user=ANALYST)
        app.text_input[0].set_value("CONGRATULATIONS you have WON $1,000,000")
        app.text_area[0].set_value(
            "Click http://claim-prize.example.com now and send your bank details to claim your "
            "free cash prize. Limited time offer, act now, 100% risk free guarantee!"
        )
        submit = next(b for b in app.button if "Classify" in b.label)
        submit.click().run(timeout=120)

        self.assertFalse(app.exception)
        self.assertIn("last_result", app.session_state)
        self.assertEqual(app.session_state["last_result"].label, "spam")

    def test_classifies_a_legitimate_email_end_to_end(self):
        app = self.run_view("classify", user=ANALYST)
        app.text_input[0].set_value("Re: Thursday budget meeting")
        app.text_area[0].set_value(
            "Hi Daren, thanks for sending the gas nomination figures through. I have attached my "
            "notes from the Houston meeting. Let me know if you have any questions."
        )
        submit = next(b for b in app.button if "Classify" in b.label)
        submit.click().run(timeout=120)

        self.assertEqual(app.session_state["last_result"].label, "ham")

    def test_rejects_an_empty_submission(self):
        app = self.run_view("classify", user=ANALYST)
        submit = next(b for b in app.button if "Classify" in b.label)
        submit.click().run(timeout=60)

        self.assertTrue(app.error)
        self.assertNotIn("last_result", app.session_state)

    def test_a_sample_button_populates_the_form(self):
        app = self.run_view("classify", user=ANALYST)
        sample = next(b for b in app.button if b.label == "Phishing attempt")
        sample.click().run(timeout=60)

        self.assertIn("suspended", app.session_state["subject_input"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
