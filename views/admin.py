"""Administration: user management, retraining, and system information."""

from __future__ import annotations

import platform
import sys

import pandas as pd
import sklearn
import streamlit as st

from src import auth, history, llm_classifier
from src.config import LLM, METRICS_PATH, MODEL_PATH, TrainingConfig
from src.exceptions import AuthError, EmailClassifierError
from src.predictor import load_predictor, model_is_available
from src.train import train
from src.ui import (
    configure_page,
    error_boundary,
    hero,
    require_admin_ui,
    sidebar_account,
)


def render() -> None:
    """Render the admin page."""
    configure_page()
    user = require_admin_ui()
    sidebar_account(user)

    hero("Administration", "Manage accounts, retrain the classifier, and inspect system state.")

    users_tab, training_tab, account_tab, system_tab = st.tabs(
        ["User accounts", "Model training", "My account", "System"]
    )

    with users_tab:
        _render_users(user)
    with training_tab:
        _render_training()
    with account_tab:
        _render_account(user)
    with system_tab:
        _render_system()


def _render_users(current) -> None:
    """List, create, and manage user accounts."""
    st.subheader("Registered accounts")

    users = auth.list_users()
    frame = pd.DataFrame(
        [
            {
                "Username": u.username,
                "Role": u.role,
                "Created": (u.created_at or "")[:19].replace("T", " "),
                "Last sign-in": (u.last_login or "Never")[:19].replace("T", " "),
            }
            for u in users
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("#### Create an account")
        with st.form("create_user", clear_on_submit=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", options=list(auth.ROLES))
            if st.form_submit_button("Create account", type="primary"):
                with error_boundary("account creation"):
                    auth.register_user(username, password, role=role)
                    st.success(f"Created '{username}' with the {role} role.")
                    st.rerun()

    with right:
        st.markdown("#### Modify an account")
        others = [u.username for u in users if u.username != current.username]
        if not others:
            st.caption("No other accounts to manage.")
            return

        target = st.selectbox("Account", options=others)
        new_role = st.selectbox("Set role to", options=list(auth.ROLES), key="role_change")

        change, delete = st.columns(2)
        with change:
            if st.button("Update role", width="stretch"):
                with error_boundary("role change"):
                    auth.set_role(target, new_role)
                    st.success(f"'{target}' is now {new_role}.")
                    st.rerun()
        with delete:
            if st.button("Delete account", width="stretch", type="secondary"):
                try:
                    auth.delete_user(target, acting_user=current)
                    st.success(f"Deleted '{target}'.")
                    st.rerun()
                except AuthError as exc:
                    st.error(exc.user_message)


def _render_training() -> None:
    """Trigger retraining and show the current artifact's state."""
    st.subheader("Retrain the classifier")

    if model_is_available():
        predictor = load_predictor(MODEL_PATH.stat().st_mtime)
        st.success(
            f"Current model: **{predictor.model_name.replace('_', ' ').title()}** · "
            f"threshold {predictor.threshold:.2f} · trained {predictor.trained_at[:19].replace('T', ' ')} UTC"
        )
    else:
        st.warning("No model artifact exists yet. Train one to enable classification.")

    st.markdown(
        """
        Training downloads the Enron-Spam corpus if needed, compares three algorithms by
        cross-validated F1, tunes the decision threshold, and writes a new model artifact.
        **The full run takes roughly three minutes** and this page will be unresponsive while it runs.
        """
    )

    with st.form("retrain"):
        quick = st.checkbox(
            "Quick run (8,000 emails, no cross-validation)",
            value=True,
            help="Finishes in well under a minute. Use the full run for final metrics.",
        )
        models = st.multiselect(
            "Algorithms to compare",
            options=list(TrainingConfig().candidate_models),
            default=list(TrainingConfig().candidate_models),
        )
        force_download = st.checkbox("Re-download the dataset", value=False)
        submitted = st.form_submit_button("Start training", type="primary")

    if not submitted:
        return

    if not models:
        st.error("Select at least one algorithm to train.")
        return

    config = TrainingConfig(max_rows=8_000 if quick else None)
    with st.spinner("Training in progress. This can take several minutes..."):
        try:
            metrics = train(
                config, force_download=force_download, models=tuple(models), skip_cv=quick
            )
        except EmailClassifierError as exc:
            st.error(exc.user_message)
            return
        except Exception as exc:  # noqa: BLE001 - training touches network and disk
            st.error(f"Training failed: {exc}")
            return

    load_predictor.cache_clear()
    st.cache_data.clear()

    test = metrics["test"]
    st.success(
        f"Training complete in {metrics['training_seconds']:.0f}s. "
        f"Selected **{metrics['model_name']}** — accuracy {test['accuracy']:.2%}, F1 {test['f1']:.4f}."
    )
    st.balloons()


def _render_account(user) -> None:
    """Let the signed-in admin rotate their own password."""
    st.subheader("Change your password")

    with st.form("change_password", clear_on_submit=True):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update password", type="primary")

    if not submitted:
        return

    if new != confirm:
        st.error("The new passwords do not match.")
        return

    with error_boundary("password change"):
        auth.change_password(user.username, current, new)
        st.success("Password updated.")


def _render_system() -> None:
    """Show environment details and maintenance actions."""
    provider = llm_classifier.active_provider()
    model = {"groq": LLM.groq_model, "anthropic": LLM.anthropic_model}.get(provider, "-")

    st.subheader("Environment")
    st.dataframe(
        pd.DataFrame(
            {
                "Component": [
                    "Python", "scikit-learn", "Streamlit", "Platform", "AI provider", "AI model",
                ],
                "Value": [
                    platform.python_version(),
                    sklearn.__version__,
                    st.__version__,
                    platform.platform(),
                    llm_classifier.provider_label() if provider else "Not configured",
                    model,
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Artifacts")
    for label, path in (("Model", MODEL_PATH), ("Metrics", METRICS_PATH)):
        if path.exists():
            st.markdown(f"- **{label}**: `{path.name}` ({path.stat().st_size / 1024:,.0f} KB)")
        else:
            st.markdown(f"- **{label}**: not present")

    st.subheader("Maintenance")
    stats = history.summary_stats()
    st.caption(f"{stats['total']:,} classifications are currently logged.")

    if st.button("Clear the classification log", type="secondary"):
        removed = history.clear_history()
        st.cache_data.clear()
        st.success(f"Removed {removed:,} logged classifications.")
        st.rerun()
