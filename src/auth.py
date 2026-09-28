"""User authentication backed by SQLite with salted PBKDF2 password hashing."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import AUTH_DB_PATH, load_settings
from .exceptions import (
    AccountLockedError,
    InvalidCredentialsError,
    PermissionDeniedError,
    UserExistsError,
    WeakPasswordError,
)
from .logger import get_logger

logger = get_logger(__name__)

PBKDF2_ITERATIONS = 260_000
SALT_BYTES = 16
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
ROLES = ("admin", "analyst")

_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{3,32}$")


@dataclass(frozen=True)
class User:
    """An authenticated account."""

    id: int
    username: str
    role: str
    created_at: str
    last_login: str | None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    """Open a SQLite connection that commits on success and always closes."""
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(AUTH_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Create the users table and bootstrap the configured admin account."""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'analyst',
                created_at TEXT NOT NULL,
                last_login TEXT,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT
            )
            """
        )

    settings = load_settings()
    try:
        if not user_exists(settings.admin_bootstrap_username):
            register_user(
                settings.admin_bootstrap_username,
                settings.admin_bootstrap_password,
                role="admin",
                enforce_policy=False,
            )
            logger.info("Bootstrapped admin account '%s'", settings.admin_bootstrap_username)
    except UserExistsError:
        pass


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-HMAC-SHA256, returning hex digest and salt."""
    salt = salt or os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return digest.hex(), salt.hex()


def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    """Check a password against a stored hash in constant time."""
    candidate, _ = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, password_hash)


def validate_password(password: str) -> None:
    """Enforce the password strength policy."""
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise WeakPasswordError()


def validate_username(username: str) -> str:
    """Normalise and validate a username."""
    cleaned = (username or "").strip()
    if not _USERNAME_PATTERN.match(cleaned):
        raise UserExistsError(
            user_message="Username must be 3-32 characters using letters, numbers, dots, hyphens, or underscores."
        )
    return cleaned


def user_exists(username: str) -> bool:
    """Report whether a username is already registered."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)
        ).fetchone()
    return row is not None


def register_user(
    username: str, password: str, role: str = "analyst", *, enforce_policy: bool = True
) -> User:
    """Create a new account, rejecting duplicates and weak passwords."""
    username = validate_username(username)
    if enforce_policy:
        validate_password(password)
    if role not in ROLES:
        raise PermissionDeniedError(user_message=f"Unknown role '{role}'.")

    password_hash, salt = hash_password(password)
    try:
        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, password_hash, salt, role, _now()),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise UserExistsError() from exc

    logger.info("Registered user '%s' with role '%s'", username, role)
    return User(id=user_id, username=username, role=role, created_at=_now(), last_login=None)


def authenticate(username: str, password: str) -> User:
    """Verify credentials, applying lockout after repeated failures."""
    username = (username or "").strip()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
        ).fetchone()

        if row is None:
            # Spend comparable time so a missing user is not distinguishable by timing.
            hash_password(password)
            raise InvalidCredentialsError()

        if row["locked_until"]:
            locked_until = datetime.fromisoformat(row["locked_until"])
            if locked_until > datetime.now(timezone.utc):
                remaining = int((locked_until - datetime.now(timezone.utc)).total_seconds() // 60) + 1
                raise AccountLockedError(
                    user_message=f"Account locked. Try again in {remaining} minute(s)."
                )

        if not verify_password(password, row["password_hash"], row["salt"]):
            attempts = row["failed_attempts"] + 1
            locked_until = (
                (datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
                if attempts >= MAX_FAILED_ATTEMPTS
                else None
            )
            connection.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                (attempts, locked_until, row["id"]),
            )
            # Commit before raising: the context manager only commits on a clean exit,
            # so without this the attempt counter and lockout would be rolled back.
            connection.commit()
            logger.warning("Failed login for '%s' (attempt %s)", username, attempts)
            if locked_until:
                raise AccountLockedError(
                    user_message=f"Too many failed attempts. Account locked for {LOCKOUT_MINUTES} minutes."
                )
            raise InvalidCredentialsError(
                user_message=f"Incorrect username or password. "
                f"{MAX_FAILED_ATTEMPTS - attempts} attempt(s) remaining."
            )

        login_time = _now()
        connection.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login = ? WHERE id = ?",
            (login_time, row["id"]),
        )

    logger.info("User '%s' signed in", row["username"])
    return User(
        id=row["id"],
        username=row["username"],
        role=row["role"],
        created_at=row["created_at"],
        last_login=login_time,
    )


def change_password(username: str, current_password: str, new_password: str) -> None:
    """Rotate a user's password after verifying the current one."""
    authenticate(username, current_password)
    validate_password(new_password)
    password_hash, salt = hash_password(new_password)
    with _connect() as connection:
        connection.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE username = ? COLLATE NOCASE",
            (password_hash, salt, username.strip()),
        )
    logger.info("Password changed for '%s'", username)


def list_users() -> list[User]:
    """Return every registered account."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, username, role, created_at, last_login FROM users ORDER BY id"
        ).fetchall()
    return [User(**dict(row)) for row in rows]


def set_role(username: str, role: str) -> None:
    """Change a user's role."""
    if role not in ROLES:
        raise PermissionDeniedError(user_message=f"Unknown role '{role}'.")
    with _connect() as connection:
        connection.execute(
            "UPDATE users SET role = ? WHERE username = ? COLLATE NOCASE", (role, username.strip())
        )


def delete_user(username: str, *, acting_user: User) -> None:
    """Delete an account, refusing to remove the last remaining admin."""
    if not acting_user.is_admin:
        raise PermissionDeniedError()
    if acting_user.username.lower() == username.strip().lower():
        raise PermissionDeniedError(user_message="You cannot delete your own account.")

    with _connect() as connection:
        target = connection.execute(
            "SELECT role FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)
        ).fetchone()
        if target is None:
            raise PermissionDeniedError(user_message="That user no longer exists.")

        if target["role"] == "admin":
            admin_count = connection.execute(
                "SELECT COUNT(*) AS n FROM users WHERE role = 'admin'"
            ).fetchone()["n"]
            if admin_count <= 1:
                raise PermissionDeniedError(
                    user_message="Cannot delete the last administrator account."
                )

        connection.execute("DELETE FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),))
    logger.info("User '%s' deleted by '%s'", username, acting_user.username)


def require_admin(user: User | None) -> User:
    """Assert that the current user holds the admin role."""
    if user is None:
        raise PermissionDeniedError(user_message="Please sign in to continue.")
    if not user.is_admin:
        raise PermissionDeniedError()
    return user
