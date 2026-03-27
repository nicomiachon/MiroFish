"""
Global token usage tracker.
Thread-safe singleton that accumulates input/output token counts across all LLM calls.
"""

import threading


class TokenTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._input_tokens = 0
                    cls._instance._output_tokens = 0
                    cls._instance._requests = 0
                    cls._instance._counter_lock = threading.Lock()
        return cls._instance

    def add(self, input_tokens: int, output_tokens: int):
        with self._counter_lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._requests += 1

    def get_total(self) -> dict:
        with self._counter_lock:
            return {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
                "requests": self._requests,
            }

    def reset(self):
        with self._counter_lock:
            self._input_tokens = 0
            self._output_tokens = 0
            self._requests = 0


# Module-level singleton
tracker = TokenTracker()
