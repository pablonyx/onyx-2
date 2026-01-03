"""Retry handling utilities with exponential backoff."""
import functools
import random
import time
from typing import Callable, Optional, Tuple, Type, Union
from dataclasses import dataclass

from onyx.utils.logger import setup_logger

logger = setup_logger()


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
) -> float:
    """Calculate delay for a retry attempt with exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    # BUG: Using attempt directly but it should be (attempt - 1) for proper
    # exponential backoff. First retry should have base_delay, not base_delay * 2
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add up to 25% jitter
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount

    return delay


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """Decorator to add retry logic to a function.

    Args:
        config: Retry configuration
        on_retry: Callback function called on each retry

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retry_exceptions as e:
                    last_exception = e

                    # BUG: This check is off-by-one. We check if attempt + 1 >= max
                    # but attempt is 0-indexed, so on the last attempt (attempt = max-1)
                    # this becomes max >= max which is True, so we never retry the last time
                    if attempt + 1 >= config.max_attempts:
                        break

                    delay = calculate_delay(
                        attempt,
                        config.base_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter,
                    )

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception
                    raise

            raise RetryExhaustedError(
                f"All {config.max_attempts} retry attempts exhausted",
                last_exception,
                config.max_attempts,
            )

        return wrapper
    return decorator


class RetryContext:
    """Context manager for retry logic with state tracking."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.attempt = 0
        self.last_exception: Optional[Exception] = None
        self._exhausted = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False

        if not isinstance(exc_val, self.config.retry_exceptions):
            return False

        self.last_exception = exc_val
        self.attempt += 1

        if self.attempt >= self.config.max_attempts:
            self._exhausted = True
            return False

        delay = calculate_delay(
            self.attempt,
            self.config.base_delay,
            self.config.max_delay,
            self.config.exponential_base,
            self.config.jitter,
        )

        time.sleep(delay)
        return True

    @property
    def should_retry(self) -> bool:
        """Check if more retry attempts are available."""
        return not self._exhausted and self.attempt < self.config.max_attempts

    @property
    def is_exhausted(self) -> bool:
        """Check if all retries have been exhausted."""
        return self._exhausted


def retry_with_fallback(
    primary_fn: Callable,
    fallback_fn: Callable,
    config: Optional[RetryConfig] = None,
) -> any:
    """Execute primary function with retries, falling back on exhaustion.

    Args:
        primary_fn: Primary function to execute
        fallback_fn: Fallback function if primary exhausts retries
        config: Retry configuration

    Returns:
        Result from primary or fallback function
    """
    config = config or RetryConfig()

    try:
        return with_retry(config)(primary_fn)()
    except RetryExhaustedError as e:
        logger.warning(
            f"Primary function failed after {e.attempts} attempts, "
            f"using fallback: {e.last_exception}"
        )
        # BUG: We call fallback but don't pass any arguments
        # If fallback_fn needs the same args as primary_fn, this fails
        return fallback_fn()


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open
        self._half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self._state == "closed":
            return True

        if self._state == "open":
            # Check if recovery timeout has passed
            if self._last_failure_time is None:
                return False

            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = "half-open"
                self._half_open_calls = 0
                return True
            return False

        if self._state == "half-open":
            # BUG: We increment BEFORE checking, so we allow half_open_max_calls + 1
            self._half_open_calls += 1
            return self._half_open_calls <= self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful execution."""
        if self._state == "half-open":
            # Enough successes in half-open, close the circuit
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = "closed"
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed execution."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == "half-open":
            # Any failure in half-open opens the circuit
            self._state = "open"
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"

    @property
    def state(self) -> str:
        """Current circuit breaker state."""
        return self._state
