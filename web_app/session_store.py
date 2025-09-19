"""Lightweight in-memory session storage with TTL support."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    value: Any
    created_at: float


class SessionStore:
    """In-memory key/value store with optional TTL eviction."""

    def __init__(self, ttl_seconds: Optional[int] = None, on_evict: Optional[Callable[[str, Any], None]] = None) -> None:
        self._ttl = ttl_seconds
        self._on_evict = on_evict
        self._store: Dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def _expired(self, entry: _Entry) -> bool:
        if self._ttl is None:
            return False
        return (time.time() - entry.created_at) > self._ttl

    def _purge_locked(self) -> List[Tuple[str, Any]]:
        evicted: List[Tuple[str, Any]] = []
        if not self._store:
            return evicted
        expired_keys = [key for key, entry in self._store.items() if self._expired(entry)]
        for key in expired_keys:
            entry = self._store.pop(key, None)
            if entry is not None:
                evicted.append((key, entry.value))
        return evicted

    def _run_callbacks(self, evicted: List[Tuple[str, Any]]) -> None:
        if not evicted or self._on_evict is None:
            return
        for key, value in evicted:
            try:
                self._on_evict(key, value)
            except Exception:  # pragma: no cover - defensive
                logger.exception("SessionStore eviction callback failed for key %s", key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = _Entry(value=value, created_at=time.time())
            evicted = self._purge_locked()
        self._run_callbacks(evicted)

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                raise KeyError(key)
            if self._expired(entry):
                self._store.pop(key, None)
                evicted = [(key, entry.value)]
            else:
                evicted = []
                value = entry.value
        self._run_callbacks(evicted)
        if evicted:
            raise KeyError(key)
        return value

    def pop(self, key: str) -> Any:
        with self._lock:
            entry = self._store.pop(key, None)
            if entry is None or self._expired(entry):
                evicted = []
                if entry and self._expired(entry):
                    evicted = [(key, entry.value)]
                    evicted.extend(self._purge_locked())
                else:
                    evicted = self._purge_locked()
            else:
                value = entry.value
                return value
        self._run_callbacks(evicted)
        raise KeyError(key)

    def contains(self, key: str) -> bool:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if self._expired(entry):
                value = entry.value
                self._store.pop(key, None)
                evicted = [(key, value)]
            else:
                evicted = []
                result = True
        self._run_callbacks(evicted)
        return False if evicted else result

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return self.contains(key) if isinstance(key, str) else False

    def keys(self) -> Iterable[str]:
        with self._lock:
            evicted = self._purge_locked()
            keys_snapshot = list(self._store.keys())
        self._run_callbacks(evicted)
        return keys_snapshot

    def clear(self) -> None:
        with self._lock:
            evicted = [(key, entry.value) for key, entry in self._store.items()]
            self._store.clear()
        self._run_callbacks(evicted)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        self.pop(key)
