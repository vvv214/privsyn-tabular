"""
Quick local benchmark for the PrivSyn backend synthesis path.

It builds a small synthetic dataset, writes minimal domain/info JSON to a
temporary directory, and invokes web_app.synthesis_service.run_synthesis.
It prints timing and a simple cProfile summary to help spot hotspots.
"""
import asyncio
import json
import os
import random
import sys
import tempfile
import time
import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_app.synthesis_service import run_synthesis, Args


def make_synthetic(n_rows: int = 2000, n_num: int = 3, n_cat: int = 2):
    rng = np.random.default_rng(42)

    # Numerical features
    num_cols = [f"num_{i}" for i in range(n_num)]
    X_num = rng.normal(0, 1, size=(n_rows, n_num))

    # Categorical features
    cat_cols = [f"cat_{i}" for i in range(n_cat)]
    cats = [np.array(["A", "B", "C"]) for _ in range(n_cat)]
    X_cat = np.column_stack([rng.choice(c, size=n_rows) for c in cats])

    # Build DataFrame (for domain sizing convenience)
    df = pd.DataFrame(np.concatenate([X_num, X_cat], axis=1), columns=num_cols + cat_cols)
    # Ensure categorical columns are str dtype
    for c in cat_cols:
        df[c] = df[c].astype(str)

    # Domain and info in the same shape as web inference
    domain_data = {}
    for c in num_cols:
        domain_data[c] = {"type": "numerical", "size": int(pd.to_numeric(df[c]).nunique())}
    for c in cat_cols:
        domain_data[c] = {"type": "categorical", "size": int(df[c].nunique())}

    info_data = {
        "name": "bench_local",
        "id": "bench-local",
        "task_type": "unknown",
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "train_size": n_rows,
        "test_size": 0,
        "val_size": 0,
        "num_columns": num_cols,
        "cat_columns": cat_cols,
    }

    return (X_num.astype(float), X_cat.astype(str)), domain_data, info_data


async def bench_once(n_rows=50, n_sample=20):
    (X_num, X_cat), domain_data, info_data = make_synthetic(n_rows=n_rows)

    # Prepare temp dir with domain.json and info.json for run_synthesis
    local_root = os.path.join(PROJECT_ROOT, "temp_bench_runs")
    os.makedirs(local_root, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="privsyn-bench-", dir=local_root)
    with open(os.path.join(tmpdir, "domain.json"), "w") as f:
        json.dump(domain_data, f)
    with open(os.path.join(tmpdir, "info.json"), "w") as f:
        json.dump(info_data, f)

    # Minimal args mirroring web app
    args = Args(
        method="privsyn",
        dataset="bench_local",
        epsilon=1.0,
        delta=1e-5,
        num_preprocess="uniform_kbins",
        rare_threshold=0.002,
        n_sample=n_sample,
        consist_iterations=10,
        non_negativity="N3",
        append=True,
        sep_syn=False,
        initialize_method="singleton",
        update_method="S5",
        update_rate_method="U4",
        update_rate_initial=1.0,
        update_iterations=5,
        device="cpu",
        sample_device="cpu",
    )

    t0 = time.perf_counter()
    synthesized_csv_path, _ = await run_synthesis(
        args=args,
        data_dir=tmpdir,
        X_num_raw=X_num,
        X_cat_raw=X_cat,
        confirmed_domain_data=domain_data,
        confirmed_info_data=info_data,
    )
    t1 = time.perf_counter()

    return synthesized_csv_path, t1 - t0


def main():
    # Run with cProfile to capture hotspots
    pr = cProfile.Profile()
    pr.enable()
    try:
        synthesized_csv_path, dt = asyncio.run(bench_once())
    finally:
        pr.disable()

    print(f"\nSynthesis output: {synthesized_csv_path}")
    print(f"Total runtime: {dt:.2f} s")

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print("\nTop functions by cumulative time:\n")
    print(s.getvalue())


if __name__ == "__main__":
    main()
