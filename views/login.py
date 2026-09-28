"""Sign-in and registration page."""

from __future__ import annotations

import streamlit as st

from src import auth
from src.config import load_settings
from src.exceptions import AuthError
from src.ui import configure_page, error_boundary, hero, set_user


def render() -> None:
    """Render the authentication gate."""
    configure_page()
    settings = load_settings()

    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        hero(
            "Winnow",
            "AI email classification - machine-learning spam detection with AI-assisted analysis. "
            "Sign in to continue.",
        )

        sign_in_tab, register_tab = (
            st.tabs(["Sign in", "Create account"]) if settings.allow_self_signup else (st.container(), None)
        )

        with sign_in_tab:
            _sign_in_form()

        if register_tab is not None:
            with register_tab:
                _register_form()

        with st.expander("First time here?"):
            st.markdown(
                f"""
                An administrator account is created automatically on first run:

                - **Username:** `{settings.admin_bootstrap_username}`
                - **Password:** set via the `ADMIN_PASSWORD` secret
                  (defaults to `ChangeMe!123` when unset)

                Change this password immediately after signing in, and set `ADMIN_PASSWORD`
                in `.streamlit/secrets.toml` before deploying.
                """
            )


def _sign_in_form() -> None:
    """Username and password form."""
    with st.form("sign_in", clear_on_submit=False):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")

    if not submitted:
        return

    if not username or not password:
        st.error("Enter both a username and a password.")
        return

    try:
        user = auth.authenticate(username, password)
    except AuthError as exc:
        st.error(exc.user_message)
        return

    set_user(user)
    st.rerun()


def _register_form() -> None:
    """New account registration form."""
    with st.form("register", clear_on_submit=False):
        username = st.text_input("Choose a username", help="3-32 letters, numbers, dots, hyphens, or underscores.")
        password = st.text_input(
            "Choose a password", type="password", help="At least 8 characters, including a letter and a number."
        )
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account", width="stretch")

    if not submitted:
        return

    if password != confirm:
        st.error("The two passwords do not match.")
        return

    with error_boundary("registration"):
        auth.register_user(username, password, role="analyst")
        st.success("Account created. Switch to the Sign in tab to continue.")
