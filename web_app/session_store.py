"""Lightweight in-memory session storage with TTL support."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass
class _Entry:
    value: Any
    created_at: float


class SessionStore:
    """In-memory key/value store with optional TTL eviction."""

    def __init__(self, ttl_seconds: Optional[int] = None) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def _expired(self, entry: _Entry) -> bool:
        if self._ttl is None:
            return False
        return (time.time() - entry.created_at) > self._ttl

    def _purge_locked(self) -> None:
        if not self._store:
            return
        expired_keys = [key for key, entry in self._store.items() if self._expired(entry)]
        for key in expired_keys:
            self._store.pop(key, None)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = _Entry(value=value, created_at=time.time())
            self._purge_locked()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                raise KeyError(key)
            if self._expired(entry):
                self._store.pop(key, None)
                raise KeyError(key)
            return entry.value

    def pop(self, key: str) -> Any:
        with self._lock:
            entry = self._store.pop(key, None)
            if entry is None or self._expired(entry):
                if entry and self._expired(entry):
                    # Entry expired but still removed from store.
                    self._purge_locked()
                raise KeyError(key)
            return entry.value

    def contains(self, key: str) -> bool:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if self._expired(entry):
                self._store.pop(key, None)
                return False
            return True

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return self.contains(key) if isinstance(key, str) else False

    def keys(self) -> Iterable[str]:
        with self._lock:
            self._purge_locked()
            return list(self._store.keys())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        self.pop(key)
