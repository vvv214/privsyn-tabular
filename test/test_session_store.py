import time
import tempfile
import os

import pytest

from web_app.session_store import SessionStore


def test_session_store_eviction_callback_on_ttl(tmp_path):
    removed = []

    def on_evict(key, payload):
        removed.append((key, payload.get('marker')))

    store = SessionStore(ttl_seconds=0.01, on_evict=on_evict)
    store.set('foo', {'marker': 'expired'})

    time.sleep(0.02)
    # Trigger purge
    store.keys()

    assert removed == [('foo', 'expired')]


def test_session_store_pop_does_not_trigger_callback(tmp_path):
    marker_path = tempfile.mkdtemp(dir=tmp_path)
    callbacks = []

    def on_evict(key, payload):
        callbacks.append(key)
        path = payload.get('temp_dir')
        if path and os.path.isdir(path):
            os.rmdir(path)

    store = SessionStore(ttl_seconds=1, on_evict=on_evict)
    store.set('keep', {'temp_dir': marker_path})

    payload = store.pop('keep')
    assert payload == {'temp_dir': marker_path}
    assert callbacks == []
    assert os.path.isdir(marker_path)
