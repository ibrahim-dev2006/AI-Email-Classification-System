"""Vercel serverless function exposing the spam classifier over HTTP."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scorer import Scorer  # noqa: E402

MAX_BODY_BYTES = 200_000
MAX_TEXT_CHARS = 100_000

_scorer: Scorer | None = None


def get_scorer() -> Scorer:
    """Load the exported model once per warm container."""
    global _scorer
    if _scorer is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
        with open(path, encoding="utf-8") as handle:
            _scorer = Scorer(json.load(handle))
    return _scorer


class handler(BaseHTTPRequestHandler):
    """Handles POST /api/classify and the CORS preflight."""

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Report model metadata so the page can show its accuracy."""
        try:
            scorer = get_scorer()
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"Model unavailable: {exc}"})
            return
        self._send(200, {"status": "ok", "metrics": scorer.metrics, "threshold": scorer.threshold})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            self._send(400, {"error": "Invalid Content-Length header."})
            return

        if length <= 0:
            self._send(400, {"error": "Send a JSON body containing an email."})
            return
        if length > MAX_BODY_BYTES:
            self._send(413, {"error": "That email is too large to analyse."})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": "Request body must be valid UTF-8 JSON."})
            return

        if not isinstance(payload, dict):
            self._send(400, {"error": "Request body must be a JSON object."})
            return

        subject = str(payload.get("subject") or "")
        body = str(payload.get("body") or "")
        text = f"{subject}\n\n{body}".strip()

        if len(text) < 3:
            self._send(400, {"error": "Enter at least a few characters of email content."})
            return
        if len(text) > MAX_TEXT_CHARS:
            self._send(413, {"error": "That email is too long to analyse."})
            return

        try:
            scorer = get_scorer()
            result = scorer.classify(text)
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the client
            self._send(500, {"error": f"Classification failed: {exc}"})
            return

        if not result["top_terms"]:
            result["note"] = "No recognised terms in this email, so the score reflects the model's prior."

        result["metrics"] = scorer.metrics
        self._send(200, result)
