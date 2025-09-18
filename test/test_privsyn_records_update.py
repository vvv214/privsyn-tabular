import numpy as np
import pandas as pd
import pytest

from method.synthesis.privsyn.lib_dataset.domain import Domain
from method.synthesis.privsyn.lib_synthesize.records_update import RecordUpdate
from method.synthesis.privsyn.lib_synthesize.update_config import UpdateConfig
from method.synthesis.privsyn.lib_synthesize.GUM import GUM_Mechanism


class DummyMarginal:
    def __init__(self, attributes_index, encode_num, actual, synth, tuple_key):
        self.attributes_index = attributes_index
        self.encode_num = encode_num
        self.actual = actual
        self.synth = synth
        self.tuple_key = tuple_key

    def calculate_tuple_key(self):
        pass

    def calculate_normalize_count(self):
        return self.actual

    def count_records_general(self, records):
        encoded = np.matmul(records[:, self.attributes_index], self.encode_num)
        counts = np.bincount(encoded, minlength=len(self.actual))
        return counts.astype(float)

    def calculate_normalize_count_general(self, counts):
        total = counts.sum()
        return counts / total if total > 0 else counts


def _build_update(records, domain):
    updater = RecordUpdate(domain, num_records=records.shape[0])
    updater.records = records.copy()
    updater.df = pd.DataFrame(records, columns=domain.attrs)
    updater.error_tracker = pd.DataFrame()
    return updater


def test_normal_rounding_clips_negative():
    domain = Domain(["a"], [3])
    updater = _build_update(np.zeros((4, 1), dtype=np.uint32), domain)
    result = updater.normal_rounding(np.array([0.2, -0.7, 1.4]))
    assert result.tolist() == [0, 0, 1]


def test_update_records_main_and_throw_indices(monkeypatch):
    domain = Domain(["x", "y"], [3, 2])
    records = np.array([[0, 0], [1, 0], [2, 1], [2, 1]], dtype=np.uint32)
    updater = _build_update(records, domain)

    actual = np.array([0.1, 0.6, 0.3])
    synth = np.array([0.5, 0.25, 0.25])
    marg = DummyMarginal(np.array([0], dtype=int), np.array([1]), actual, synth, tuple_key=[np.array([0]), np.array([1]), np.array([2])])

    updater.actual_marginal = actual
    updater.synthesize_marginal = synth
    updater.update_records_main(marg, alpha=0.5)
    assert updater.cell_over_indices.size > 0

    monkeypatch.setattr(np.random, "choice", lambda arr, size, replace: arr[:size])
    updater.determine_throw_indices()
    assert updater.records_throw_indices.size > 0


def test_find_optimal_beta_zero():
    domain = Domain(["x"], [2])
    updater = _build_update(np.array([[0], [1]], dtype=np.uint32), domain)
    assert updater.find_optimal_beta(0, np.array([], dtype=int)) == 0.0
    assert updater.find_optimal_beta(2, np.array([0], dtype=int)) == 1.0


def test_handle_zero_cells_and_complete_ratio():
    domain = Domain(["x", "y"], [3, 2])
    records = np.array([[0, 0], [1, 0], [2, 1], [2, 1]], dtype=np.uint32)
    updater = _build_update(records, domain)

    actual = np.array([0.0, 0.5, 0.5])
    synth = np.array([0.4, 0.6, 0.0])
    tuple_key = [np.array([0]), np.array([1]), np.array([2])]
    marg = DummyMarginal(np.array([0], dtype=int), np.array([1]), actual, synth, tuple_key)

    updater.actual_marginal = actual
    updater.synthesize_marginal = synth
    updater.num_add = np.array([0, 0, 0])
    updater.num_add_zero = np.array([0, 0, 2])
    updater.cell_zero_indices = np.array([2])
    updater.cell_under_indices = np.array([], dtype=int)
    updater.records_throw_indices = np.array([0, 1], dtype=np.uint32)

    updater.complete_partial_ratio(marg, ratio=1.0)
    # ensure destinations were touched for the second attribute index
    assert np.all(updater.records[0:2, 0] == np.array([0, 1]))

    updater.records_throw_indices = np.array([0], dtype=np.uint32)
    updater.handle_zero_cells(marg)
    assert updater.records_throw_indices.size <= 1


def test_update_records_before_and_after_tracks_errors():
    domain = Domain(["x"], [2])
    updater = _build_update(np.array([[0], [1]], dtype=np.uint32), domain)
    marg = DummyMarginal(np.array([0], dtype=int), np.array([1]), np.array([0.5, 0.5]), np.array([0.5, 0.5]), tuple_key=[np.array([0]), np.array([1])])

    updater.update_records_before(marg, ("x",), iteration=0)
    assert "x" in updater.error_tracker.index

    updater.update_records_after(marg, ("x",), iteration=0)
    assert "0-after" in updater.error_tracker.columns


def test_update_order_normalizes_string_labels():
    domain = Domain(["a", "b"], [2, 2])
    config = UpdateConfig(
        domain,
        num_records=4,
        update_config={"alpha": 1.0, "alpha_update_method": "U8", "update_method": "S5", "threshold": 0.0},
    )

    class StubUpdate:
        def __init__(self):
            self.error_tracker = pd.DataFrame()

        def update_records_before(self, marg, key, iteration, mute=False):
            key_tuple = tuple(key) if not isinstance(key, tuple) else key
            label = "::".join(key_tuple)
            col = f"{iteration}-before"
            self.error_tracker.loc[label, col] = 1.0

    stub = StubUpdate()
    config.update = stub

    marginals = {("a", "b"): object()}
    iterate_keys = [("a", "b")]

    ordered = config.update_order(0, marginals, iterate_keys)
    assert ordered == [("a", "b")]


def test_normalize_key_handles_serialised_forms():
    assert GUM_Mechanism._normalize_key("age") == ("age",)
    assert GUM_Mechanism._normalize_key("education::occupation") == ("education", "occupation")
    assert GUM_Mechanism._normalize_key(["x", "y"]) == ("x", "y")
