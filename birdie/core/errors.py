"""
Birdie-level exceptions raised by the agent layer.

These wrap provider-specific errors so callers (CLI, programmatic) can handle
them without importing any vendor SDK.
"""


class BirdieError(Exception):
    """Base class for all Birdie exceptions."""


class BirdieTransientError(BirdieError):
    """Base for provider failures that were retried and still failed.

    ``retry_after`` is the back-off (seconds) the agent would have waited
    next, as a hint for callers that want to try again later.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BirdieRateLimitError(BirdieTransientError):
    """Raised when a provider rate limit is hit and all retries are exhausted."""


class BirdieProviderUnavailableError(BirdieTransientError):
    """Raised when the provider could not be reached (connection failures,
    timeouts, dropped connections) and all retries are exhausted."""
