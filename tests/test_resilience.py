import pytest
from src.resilience import TokenBucket


def test_initial_bucket_is_full():
    bucket = TokenBucket(capacity=100.0, refill_rate=10.0)
    assert bucket.tokens == 100.0


def test_consume_within_capacity():
    bucket = TokenBucket(capacity=100.0, refill_rate=10.0)
    now = bucket.last_refill_timestamp

    # Consume 40 tokens
    assert bucket.consume(40.0, now=now) is True
    assert bucket.tokens == 60.0


def test_consume_exceeding_capacity_fails():
    bucket = TokenBucket(capacity=50.0, refill_rate=10.0)
    now = bucket.last_refill_timestamp

    # Try consuming 60 tokens from 50 token bucket
    assert bucket.consume(60.0, now=now) is False
    # Tokens shouldn't be deducted on failure
    assert bucket.tokens == 50.0


def test_token_bucket_refills_over_time():
    bucket = TokenBucket(capacity=100.0, refill_rate=10.0)  # 10 tokens per sec
    t0 = bucket.last_refill_timestamp

    # Drain completely
    assert bucket.consume(100.0, now=t0) is True
    assert bucket.tokens == 0.0

    # 2.5 seconds later -> +25 tokens
    t1 = t0 + 2.5
    assert bucket.consume(20.0, now=t1) is True
    assert pytest.approx(bucket.tokens, rel=1e-3) == 5.0


def test_bucket_does_not_overflow_capacity():
    bucket = TokenBucket(capacity=100.0, refill_rate=10.0)
    t0 = bucket.last_refill_timestamp

    # 100 seconds later -> should cap at capacity (100.0), not 1000.0
    t1 = t0 + 100.0
    bucket.consume(10.0, now=t1)
    assert pytest.approx(bucket.tokens, rel=1e-3) == 90.0


def test_consume_negative_or_zero_tokens():
    bucket = TokenBucket(capacity=100.0, refill_rate=10.0)
    with pytest.raises(ValueError, match="positive"):
        bucket.consume(0.0)

    with pytest.raises(ValueError, match="positive"):
        bucket.consume(-5.0)


from src.resilience import calculate_backoff


def test_exponential_delay_growth():
    # Attempt 0: 1.0 * (2^0) = 1.0
    assert calculate_backoff(attempt=0, base_delay=1.0, jitter_factor=1.0) == 1.0
    # Attempt 1: 1.0 * (2^1) = 2.0
    assert calculate_backoff(attempt=1, base_delay=1.0, jitter_factor=1.0) == 2.0
    # Attempt 2: 1.0 * (2^2) = 4.0
    assert calculate_backoff(attempt=2, base_delay=1.0, jitter_factor=1.0) == 4.0
    # Attempt 3: 1.0 * (2^3) = 8.0
    assert calculate_backoff(attempt=3, base_delay=1.0, jitter_factor=1.0) == 8.0


def test_max_delay_cap():
    # Attempt 10: 1.0 * 1024 would be 1024, but capped at 30.0
    assert calculate_backoff(attempt=10, base_delay=1.0, max_delay=30.0, jitter_factor=1.0) == 30.0


def test_jitter_scaling():
    # raw_delay = 4.0. Jitter factor 0.5 -> 2.0
    assert calculate_backoff(attempt=2, base_delay=1.0, jitter_factor=0.5) == 2.0
    # Jitter factor 0.0 -> 0.0
    assert calculate_backoff(attempt=2, base_delay=1.0, jitter_factor=0.0) == 0.0


def test_random_jitter_bounds():
    # When jitter_factor is None, delay must be in [0.0, raw_delay]
    for _ in range(20):
        delay = calculate_backoff(attempt=2, base_delay=1.0, jitter_factor=None)
        assert 0.0 <= delay <= 4.0


def test_invalid_negative_arguments():
    with pytest.raises(ValueError, match="attempt"):
        calculate_backoff(attempt=-1)

    with pytest.raises(ValueError, match="base_delay"):
        calculate_backoff(attempt=0, base_delay=0.0)

    with pytest.raises(ValueError, match="jitter_factor"):
        calculate_backoff(attempt=0, jitter_factor=1.5)
