import os
import io
import time
import zipfile
import argparse
import pandas as pd

from web_app.data_inference import infer_data_metadata
from method.preprocess_common.load_data_common import data_preporcesser_common
from method.privsyn.privsyn import privsyn_main
from util.rho_cdp import cdp_rho


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="adult")
    parser.add_argument("--zip_path", default=os.path.join("sample_data", "adult.csv.zip"))
    parser.add_argument("--csv_name", default="adult.csv")
    parser.add_argument("--n_sample", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()

    print(f"[bench] Loading {args.zip_path} -> {args.csv_name}")
    t0 = time.time()
    with zipfile.ZipFile(args.zip_path, 'r') as zf:
        with zf.open(args.csv_name) as f:
            df_orig = pd.read_csv(io.StringIO(f.read().decode('utf-8')))
    t1 = time.time()
    print(f"[bench] Loaded CSV: shape={df_orig.shape}, dt={t1-t0:.2f}s")

    print("[bench] Inferring metadata")
    inf0 = time.time()
    meta = infer_data_metadata(df_orig.copy(), target_column='income')
    X_num, X_cat = meta['X_num'], meta['X_cat']
    domain_data, info_data = meta['domain_data'], meta['info_data']
    inf1 = time.time()
    print(f"[bench] Inferred: X_num={None if X_num is None else X_num.shape}, X_cat={None if X_cat is None else X_cat.shape}, dt={inf1-inf0:.2f}s")

    print("[bench] Preprocessing")
    pre_args = Args(method='privsyn', num_preprocess='uniform_kbins', epsilon=args.epsilon, delta=args.delta, rare_threshold=0.002, dataset=args.dataset_name)
    pre = data_preporcesser_common(pre_args)
    p0 = time.time()
    df_proc, domain_proc, _ = pre.load_data(X_num_raw=X_num, X_cat_raw=X_cat, rho=0.1, user_domain_data=domain_data, user_info_data=info_data)
    p1 = time.time()
    print(f"[bench] Preprocessed: shape={df_proc.shape}, dt={p1-p0:.2f}s")

    print("[bench] PrivSyn: select marginals")
    rho = cdp_rho(args.epsilon, args.delta)
    ps_args = Args(method='privsyn', dataset=args.dataset_name, epsilon=args.epsilon, delta=args.delta,
                   num_preprocess='uniform_kbins', rare_threshold=0.002,
                   is_cal_marginals=True, is_cal_depend=True, is_combine=True,
                   marg_add_sensitivity=1.0, marg_sel_threshold=20000,
                   non_negativity='N3', consist_iterations=5, initialize_method='singleton', update_method='S5',
                   append=True, sep_syn=False, update_rate_method='U4', update_rate_initial=1.0, update_iterations=5)
    s0 = time.time()
    res = privsyn_main(ps_args, df_proc, domain_proc, rho)
    gen = res["privsyn_generator"]
    s1 = time.time()
    print(f"[bench] Selected marginals, dt={s1-s0:.2f}s")

    print(f"[bench] Synthesizing n={args.n_sample}")
    y0 = time.time()
    gen.syn(args.n_sample, pre, None)
    y1 = time.time()
    print(f"[bench] Synthesized df: shape={gen.synthesized_df.shape}, dt={y1-y0:.2f}s")


if __name__ == "__main__":
    main()
