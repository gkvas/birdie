"""
Birdie-level exceptions raised by the agent layer.

These wrap provider-specific errors so callers (CLI, programmatic) can handle
them without importing any vendor SDK.
"""


class BirdieError(Exception):
    """Base class for all Birdie exceptions."""


class BirdieRateLimitError(BirdieError):
    """Raised when a provider rate limit is hit and all retries are exhausted."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
