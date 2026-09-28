"""Exception hierarchy used across the application for predictable error handling."""

from __future__ import annotations


class EmailClassifierError(Exception):
    """Base class for every error raised by this project."""

    user_message = "An unexpected error occurred. Please try again."

    def __init__(self, message: str = "", *, user_message: str | None = None):
        super().__init__(message or self.user_message)
        if user_message:
            self.user_message = user_message
        elif message:
            self.user_message = message


class DatasetError(EmailClassifierError):
    """Raised when the dataset cannot be downloaded, read, or validated."""

    user_message = "The email dataset could not be loaded. Check your connection or run the training script."


class ModelNotTrainedError(EmailClassifierError):
    """Raised when a prediction is requested before a model artifact exists."""

    user_message = "No trained model was found. Train the model from the Admin page before classifying."


class ModelLoadError(EmailClassifierError):
    """Raised when a model artifact exists but cannot be deserialised."""

    user_message = "The saved model is corrupted or incompatible. Please retrain the model."


class PredictionError(EmailClassifierError):
    """Raised when inference fails on otherwise valid input."""

    user_message = "Classification failed for this email. Please try again."


class ValidationError(EmailClassifierError):
    """Raised when user-supplied input fails validation."""

    user_message = "The submitted input is not valid."


class LLMError(EmailClassifierError):
    """Base class for failures in the API-based classifier."""

    user_message = "The AI service is unavailable. The local model was used instead."


class LLMConfigurationError(LLMError):
    """Raised when the API key is missing or malformed."""

    user_message = "No Anthropic API key is configured. Add ANTHROPIC_API_KEY to enable AI-assisted analysis."


class LLMRateLimitError(LLMError):
    """Raised when the provider rejects the request due to rate limiting."""

    user_message = "The AI service is rate limited right now. Please retry in a moment."


class LLMResponseError(LLMError):
    """Raised when the provider returns a response that cannot be parsed."""

    user_message = "The AI service returned an unreadable response. Falling back to the local model."


class AuthError(EmailClassifierError):
    """Base class for authentication and authorisation failures."""

    user_message = "Authentication failed."


class InvalidCredentialsError(AuthError):
    """Raised when a username/password pair does not match a stored account."""

    user_message = "Incorrect username or password."


class UserExistsError(AuthError):
    """Raised when registering a username that is already taken."""

    user_message = "That username is already registered."


class WeakPasswordError(AuthError):
    """Raised when a proposed password fails the strength policy."""

    user_message = "Password must be at least 8 characters and include a letter and a number."


class AccountLockedError(AuthError):
    """Raised when an account is temporarily locked after repeated failures."""

    user_message = "Too many failed attempts. This account is locked temporarily."


class PermissionDeniedError(AuthError):
    """Raised when a user lacks the role required for an action."""

    user_message = "Your account does not have permission to perform this action."
