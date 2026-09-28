"""Streamlit entry point for the AI Email Classification System."""

from __future__ import annotations

import streamlit as st

from src import auth, history
from src.config import ensure_directories
from src.logger import configure_logging
from src.ui import configure_page, current_user
from views import admin, batch, classify, dashboard, home, login, performance


@st.cache_resource
def _bootstrap() -> bool:
    """Prepare directories, logging, and databases once per server process."""
    ensure_directories()
    configure_logging()
    auth.init_db()
    history.init_db()
    return True


def main() -> None:
    """Route to the correct page based on authentication state and role."""
    configure_page()
    _bootstrap()

    user = current_user()
    if user is None:
        login.render()
        return

    # Every view's entry point is named `render`, so url_path must be set explicitly -
    # Streamlit otherwise infers the same pathname for all of them and rejects the collision.
    pages = [
        st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
        st.Page(classify.render, title="Classify", icon=":material/mark_email_unread:", url_path="classify"),
        st.Page(batch.render, title="Batch", icon=":material/table_rows:", url_path="batch"),
        st.Page(dashboard.render, title="Dashboard", icon=":material/monitoring:", url_path="dashboard"),
        st.Page(performance.render, title="Performance", icon=":material/analytics:", url_path="performance"),
    ]
    if user.is_admin:
        pages.append(
            st.Page(admin.render, title="Admin", icon=":material/settings:", url_path="admin")
        )

    st.navigation(pages).run()


if __name__ == "__main__":
    main()
