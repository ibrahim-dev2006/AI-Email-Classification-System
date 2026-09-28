"""Shared Streamlit helpers: page setup, styling, session state, and error boundaries."""

from __future__ import annotations

import html
import traceback
from contextlib import contextmanager

import streamlit as st
from streamlit.errors import StreamlitAPIException

from .auth import User
from .exceptions import EmailClassifierError
from .logger import get_logger

logger = get_logger(__name__)

APP_TITLE = "Winnow - AI Email Classification"
APP_ICON = "M"

_CSS = """
<style>
  .block-container { padding-top: 2.4rem; max-width: 1180px; }
  [data-testid="stMetricValue"] { font-size: 1.7rem; }

  .hero {
    border-radius: 16px;
    padding: 26px 30px;
    background: linear-gradient(135deg, rgba(91,124,250,.16), rgba(48,163,108,.12));
    border: 1px solid rgba(128,138,157,.22);
    margin-bottom: 1.4rem;
  }
  .hero h1 { margin: 0 0 .35rem 0; font-size: 1.85rem; letter-spacing: -.02em; }
  .hero p { margin: 0; color: #7c8291; font-size: .97rem; }

  .card {
    border: 1px solid rgba(128,138,157,.25);
    border-radius: 12px;
    padding: 16px 18px;
    background: rgba(128,138,157,.06);
    height: 100%;
  }
  .card-label {
    font-size: .74rem; text-transform: uppercase; letter-spacing: .07em;
    color: #8b8f9e; font-weight: 600; margin-bottom: .3rem;
  }
  .card-value { font-size: 1.55rem; font-weight: 700; line-height: 1.15; }
  .card-help { font-size: .78rem; color: #8b8f9e; margin-top: .25rem; }

  .verdict {
    border-radius: 14px; padding: 20px 24px; margin: .4rem 0 1rem 0;
    border: 1px solid; border-left-width: 6px;
  }
  .verdict-spam { background: rgba(229,72,77,.10); border-color: rgba(229,72,77,.45); }
  .verdict-ham { background: rgba(48,163,108,.10); border-color: rgba(48,163,108,.45); }
  .verdict-review { background: rgba(245,165,36,.12); border-color: rgba(245,165,36,.5); }
  .verdict h2 { margin: 0 0 .25rem 0; font-size: 1.5rem; }
  .verdict p { margin: 0; color: #7c8291; font-size: .93rem; }

  .pill {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: .76rem; font-weight: 600; margin-right: .4rem;
  }
  .pill-spam { background: rgba(229,72,77,.16); color: #e5484d; }
  .pill-ham { background: rgba(48,163,108,.16); color: #30a46c; }
  .pill-review { background: rgba(245,165,36,.18); color: #b87503; }
  .pill-muted { background: rgba(128,138,157,.16); color: #8b8f9e; }

  .login-note { color: #8b8f9e; font-size: .85rem; }
</style>
"""


def configure_page() -> None:
    """Apply page config and shared styling; safe to call from every view."""
    try:
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon=APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except StreamlitAPIException:
        # Already set earlier in this script run - the router calls this first.
        pass
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    """Render the page header banner."""
    st.markdown(
        f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    """Render a row of labelled metric cards."""
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=True):
        with column:
            st.markdown(
                f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
                f'<div class="card-value">{html.escape(str(value))}</div>'
                f'<div class="card-help">{html.escape(help_text)}</div></div>',
                unsafe_allow_html=True,
            )


def verdict_banner(title: str, detail: str, kind: str) -> None:
    """Render the large classification verdict card."""
    st.markdown(
        f'<div class="verdict verdict-{kind}"><h2>{html.escape(title)}</h2>'
        f"<p>{html.escape(detail)}</p></div>",
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "muted") -> str:
    """Return an inline coloured pill for use inside markdown."""
    return f'<span class="pill pill-{kind}">{html.escape(text)}</span>'


def current_user() -> User | None:
    """Return the signed-in user, if any."""
    return st.session_state.get("user")


def set_user(user: User) -> None:
    """Store the signed-in user in session state."""
    st.session_state["user"] = user


def logout() -> None:
    """Clear every key in the session."""
    st.session_state.clear()


def require_login() -> User:
    """Return the current user or halt rendering with a prompt to sign in."""
    user = current_user()
    if user is None:
        st.warning("Please sign in to continue.")
        st.stop()
    return user


def require_admin_ui() -> User:
    """Return the current user or halt rendering unless they are an admin."""
    user = require_login()
    if not user.is_admin:
        st.error("This page is restricted to administrator accounts.")
        st.stop()
    return user


@contextmanager
def error_boundary(context: str = "operation"):
    """Convert exceptions into friendly Streamlit messages instead of a stack trace."""
    try:
        yield
    except EmailClassifierError as exc:
        st.error(exc.user_message)
        logger.warning("%s failed: %s", context, exc)
    except Exception as exc:  # noqa: BLE001 - the UI must never crash on the user
        st.error(f"Something went wrong during {context}. The details below can help diagnose it.")
        logger.exception("Unhandled error during %s", context)
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", language="text")


def sidebar_account(user: User) -> None:
    """Render the account panel and sign-out control in the sidebar."""
    with st.sidebar:
        st.markdown(f"**{user.username}**")
        st.caption(f"Role: {user.role}")
        if user.last_login:
            st.caption(f"Signed in: {user.last_login[:19].replace('T', ' ')} UTC")
        if st.button("Sign out", width="stretch"):
            logout()
            st.rerun()
        st.divider()


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a 0-1 ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"
