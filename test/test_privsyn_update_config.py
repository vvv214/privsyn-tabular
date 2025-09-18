import pandas as pd
import pytest
import math

from method.synthesis.privsyn.lib_synthesize import update_config as update_config_module


class StubRecordUpdate:
    def __init__(self, domain, num_records, init_df=None):
        self.domain = domain
        self.num_records = num_records
        self.init_df = init_df
        self.error_tracker = pd.DataFrame()
        self.before_calls = []
        self.after_calls = []
        self.main_calls = []
        self.complete_ratios = []

    def update_records_before(self, marg, marg_key, iteration, mute=False):
        key_label = "::".join(map(str, marg_key))
        col = f"{iteration}-before"
        self.error_tracker.loc[key_label, col] = getattr(marg, "score", 0)
        self.before_calls.append((key_label, iteration))

    def update_records_main(self, marg, alpha):
        self.main_calls.append((marg, alpha))

    def determine_throw_indices(self):
        self.throw_determined = True

    def handle_zero_cells(self, marg):
        self.handled_zero = True

    def complete_partial_ratio(self, marg, ratio):
        self.complete_ratios.append(ratio)

    def update_records_after(self, marg, marg_key, iteration, mute=False):
        key_label = "::".join(map(str, marg_key))
        col = f"{iteration}-after"
        self.error_tracker.loc[key_label, col] = getattr(marg, "score_after", 0)
        self.after_calls.append((key_label, iteration))


class DummyMarginal:
    def __init__(self, score=0, score_after=0):
        self.score = score
        self.score_after = score_after


@pytest.fixture(autouse=True)
def patch_record_update(monkeypatch):
    monkeypatch.setattr(update_config_module, "RecordUpdate", StubRecordUpdate)


@pytest.mark.parametrize(
    "method,iteration",
    [
        ("U1", 5),
        ("U2", 3),
        ("U3", 120),
        ("U4", 40),
        ("U5", 2),
        ("U6", 10),
        ("U7", 25),
        ("U8", 7),
        ("U9", 7),
        ("U10", 7),
        ("U11", 7),
        ("U12", 7),
    ],
)
def test_update_alpha_branches(method, iteration):
    config = {
        "alpha": 1.0,
        "alpha_update_method": method,
        "update_method": "S1",
        "threshold": 0.0,
    }
    updater = update_config_module.UpdateConfig(domain=None, num_records=10, update_config=config)
    updater.update_alpha(iteration)
    if method == "U1":
        expected = 1.0
    elif method == "U2":
        expected = 0.98
    elif method == "U3":
        expected = 0.99
    elif method == "U4":
        expected = 1.0 * 0.84 ** (iteration // 20)
    elif method == "U5":
        expected = math.exp(-0.008 * iteration)
    elif method == "U6":
        expected = 1.0 / (1.0 + 0.02 * iteration)
    elif method == "U7":
        expected = 1.0 / math.sqrt(0.12 * iteration + 1.0)
    elif method == "U8":
        expected = 1.0
    elif method == "U9":
        expected = 0.2
    elif method == "U10":
        expected = 0.3
    elif method == "U11":
        expected = 0.5
    elif method == "U12":
        expected = 0.1
    else:  # pragma: no cover - unreachable in this parametrization
        expected = None
    assert updater.alpha == pytest.approx(expected)


def test_update_alpha_invalid_method():
    config = {
        "alpha": 1.0,
        "alpha_update_method": "invalid",
        "update_method": "S1",
        "threshold": 0.0,
    }
    updater = update_config_module.UpdateConfig(domain=None, num_records=5, update_config=config)
    with pytest.raises(Exception):
        updater.update_alpha(1)


@pytest.mark.parametrize(
    "update_method,iteration,expected_ratio",
    [
        ("S1", 0, 0.0),
        ("S2", 0, 1.0),
        ("S3", 0, 0.5),
        ("S4", 1, 1.0),
        ("S4", 2, 0.0),
        ("S5", 1, 1.0),
        ("S5", 2, 0.5),
        ("S6", 1, 0.0),
        ("S6", 2, 0.5),
    ],
)
def test_update_records_branches(update_method, iteration, expected_ratio):
    config = {
        "alpha": 0.7,
        "alpha_update_method": "U1",
        "update_method": update_method,
        "threshold": 0.0,
    }
    updater = update_config_module.UpdateConfig(domain=None, num_records=5, update_config=config)
    original = DummyMarginal(score=iteration, score_after=iteration + 1)
    updater.update_records(original, ("a", "b"), iteration)
    assert updater.update.complete_ratios[-1] == expected_ratio
    assert updater.update.before_calls[-1][1] == iteration
    assert updater.update.after_calls[-1][1] == iteration


def test_update_records_invalid_method():
    config = {
        "alpha": 0.7,
        "alpha_update_method": "U1",
        "update_method": "bad",
        "threshold": 0.0,
    }
    updater = update_config_module.UpdateConfig(domain=None, num_records=5, update_config=config)
    with pytest.raises(Exception):
        updater.update_records(DummyMarginal(), ("a",), 0)


def test_update_order_sorts_descending():
    config = {
        "alpha": 0.5,
        "alpha_update_method": "U1",
        "update_method": "S1",
        "threshold": 0.0,
    }
    updater = update_config_module.UpdateConfig(domain=None, num_records=5, update_config=config)
    marginals = {
        ("x",): DummyMarginal(score=1),
        ("y",): DummyMarginal(score=5),
    }
    order = updater.update_order(0, marginals, list(marginals.keys()))
    assert order == [("y",), ("x",)]
