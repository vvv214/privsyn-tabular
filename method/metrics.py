import itertools
import numpy as np
import pandas as pd

from method.privsyn.lib_dataset.dataset import Dataset
from method.privsyn.lib_dataset.domain import Domain
from method.privsyn.lib_marginal.marg import Marginal


def calculate_marginal_error(original_df: pd.DataFrame, synth_df: pd.DataFrame, max_way: int = 2) -> dict:
    """
    Black-box evaluation of synthetic data quality.
    Calculates the total variation distance on all k-way marginals up to max_way.
    """
    # For this black-box metric, we need to infer the domain from the data.
    # We will assume that all columns are categorical.
    domain_config = {col: {'size': original_df[col].nunique()} for col in original_df.columns}
    domain = Domain(domain_config.keys(), [d['size'] for d in domain_config.values()])

    # We need to convert the data to numerical format.
    # We will use a simple factorize for each column.
    original_df_processed = original_df.copy()
    synth_df_processed = synth_df.copy()
    for col in original_df.columns:
        original_df_processed[col] = pd.factorize(original_df[col].astype(str))[0]
        synth_df_processed[col] = pd.factorize(synth_df[col].astype(str))[0]

    original_dataset = Dataset(original_df_processed, domain)
    synth_dataset = Dataset(synth_df_processed, domain)

    errors = {}
    for k in range(1, max_way + 1):
        for marg_spec in itertools.combinations(original_df.columns, k):
            original_marg = Marginal(original_dataset.domain.project(marg_spec), original_dataset.domain)
            original_marg.count_records(original_dataset.df.values)
            original_marg_norm = original_marg.calculate_normalize_count()

            synth_marg = Marginal(synth_dataset.domain.project(marg_spec), synth_dataset.domain)
            synth_marg.count_records(synth_dataset.df.values)
            synth_marg_norm = synth_marg.calculate_normalize_count()

            error = np.abs(original_marg_norm - synth_marg_norm).sum() / 2.0
            errors[str(marg_spec)] = error

    return errors