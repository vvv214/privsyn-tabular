import pandas as pd

from web_app.data_comparison import calculate_tvd_metrics


def test_numeric_tvd_with_domain_edges():
    original = pd.DataFrame({"age": [10, 20, 30, 40]})
    synth = pd.DataFrame({"age": [12, 22, 32, 42]})

    domain = {
        "age": {
            "type": "numerical",
            "binning": {"edges": [0, 20, 40, 60]},
        }
    }

    result = calculate_tvd_metrics(original, synth, domain_data=domain)
    tvd_age = result["tvd_per_column"]["age"]
    assert tvd_age < 0.5


def test_numeric_tvd_defaults_to_bins():
    original = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    synth = pd.DataFrame({"value": [5, 4, 3, 2, 1]})
    domain = {"value": {"type": "numerical"}}
    result = calculate_tvd_metrics(original, synth, domain_data=domain)
    assert result["tvd_per_column"]["value"] >= 0.0


def test_categorical_tvd_remains_default():
    original = pd.DataFrame({"city": ["NY", "SF", "NY"]})
    synth = pd.DataFrame({"city": ["NY", "NY", "NY"]})
    domain = {"city": {"type": "categorical"}}
    result = calculate_tvd_metrics(original, synth, domain_data=domain)
    assert result["tvd_per_column"]["city"] > 0
