# rate_limiter.py
import time
import threading
class TokenBucket:
    """
    Simple in-process token bucket.
    Good for single-instance deployments.
    """
    def __init__(self, rate_per_minute: int, burst: int):
        self.rate_per_sec = rate_per_minute / 60.0
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.last = time.time()
        self.lock = threading.Lock()

    def allow(self, tokens: float = 1.0) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def retry_after_seconds(self, tokens: float = 1.0) -> int:
        missing = max(0.0, tokens - self.tokens)
        if self.rate_per_sec <= 0:
            return 60
        return int(missing / self.rate_per_sec) + 1


# One shared limiter for ALL OpenRouter calls using your single API key
OPENROUTER_LIMITER = TokenBucket(rate_per_minute=15, burst=3)