import os

import numpy as np
import pandas as pd
import pytest

from web_app.synthesis_service import Args, run_synthesis


@pytest.mark.asyncio
async def test_run_synthesis_forwards_advanced_config(monkeypatch, tmp_path):
    captured = {}

    def _fake_dispatch(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"value": [42]})

    monkeypatch.setattr("web_app.synthesis_service.dispatch_synthesize", _fake_dispatch)
    monkeypatch.setattr("web_app.synthesis_service.project_root", str(tmp_path))

    args = Args(
        method="privsyn",
        dataset="demo",
        epsilon=1.23,
        delta=5e-6,
        num_preprocess="uniform_kbins",
        rare_threshold=0.1,
        n_sample=2,
        consist_iterations=7,
        non_negativity="N9",
        append=False,
        sep_syn=True,
        initialize_method="random",
        update_method="S9",
        update_rate_method="U9",
        update_rate_initial=0.5,
        update_iterations=4,
        degree=3,
        max_cells=128,
        max_iters=33,
        max_model_size=12,
        num_marginals=5,
    )

    domain = {
        "num": {"type": "numerical", "size": 10},
        "cat": {"type": "categorical", "size": 3},
    }
    info = {"num_columns": ["num"], "cat_columns": ["cat"]}

    x_num = np.array([[1.0], [2.0]])
    x_cat = np.array([["a"], ["b"]])

    synth_path, _ = await run_synthesis(
        args=args,
        data_dir=str(tmp_path),
        X_num_raw=x_num,
        X_cat_raw=x_cat,
        confirmed_domain_data=domain,
        confirmed_info_data=info,
    )

    assert captured["method"] == "privsyn"
    assert captured["n_sample"] == 2
    config = captured["config"]
    assert config["epsilon"] == pytest.approx(1.23)
    assert config["delta"] == pytest.approx(5e-6)
    assert config["consist_iterations"] == 7
    assert config["degree"] == 3
    assert config["max_cells"] == 128
    assert config["num_marginals"] == 5

    assert os.path.exists(synth_path)
    saved = pd.read_csv(synth_path)
    assert "value" in saved.columns
