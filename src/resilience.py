"""Rate limiting and resilience mechanisms from first principles."""

import time
from dataclasses import dataclass
import random


class RateLimitExceeded(Exception):
    """Raised when a request exceeds available token/request capacity."""


@dataclass
class TokenBucket:
    """Token Bucket rate limiter for controlling request/token throughput."""

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill_timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill_timestamp = time.monotonic()

    def _refill(self, now: float) -> None:
        """Refill tokens based on elapsed time up to maximum capacity."""
        time_elapsed = now - self.last_refill_timestamp
        self.tokens = min(self.capacity,self.tokens + time_elapsed * self.refill_rate)
        self.last_refill_timestamp = now

    def consume(self, tokens: float, now: float | None = None) -> bool:
        """Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume.
            now: Explicit timestamp used for fast, deterministic testing.
                 Defaults to real-time `time.monotonic()` if omitted.

        Returns:
            True if tokens were available and consumed, False otherwise.
        """
        if tokens <= 0:
            raise ValueError("Tokens must be positive")
        if now is None:
            now = time.monotonic()
        self._refill(now)
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float | None = None,
) -> float:
    """Calculate exponential backoff delay with optional deterministic jitter.

    Formula:
        raw_delay = min(max_delay, base_delay * (2 ** attempt))
        delay = raw_delay * jitter_factor (or random uniform [0, 1] if None)

    Args:
        attempt: Current retry attempt index (0-indexed: 0, 1, 2...).
        base_delay: Initial delay in seconds for attempt 0.
        max_delay: Upper cap on the exponential delay.
        jitter_factor: Optional float in [0.0, 1.0] used for deterministic testing.
                       If None, generate random float between 0.0 and 1.0.

    Returns:
        Delay in seconds to wait before next retry.
    """
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    if base_delay <= 0:
        raise ValueError("base_delay must be positive")
    if jitter_factor is None:
        jitter_factor = random.uniform(0.0,1.0)
    if jitter_factor > 1.0 or jitter_factor < 0:
        raise ValueError("jitter_factor must be between 0.0 and 1.0")
    
    raw_delay = min(max_delay,base_delay * (2 ** attempt))
    delay = raw_delay * jitter_factor

    return delay