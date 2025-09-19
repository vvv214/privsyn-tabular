#####################################################################
#                                                                   #
#           function marginal_selection's hepler function           #
#                                                                   #
#####################################################################

import logging
import pickle
import copy
import math

import numpy as np
import pandas as pd
from method.util.dp_noise import gaussian_noise

from method.synthesis.privsyn.lib_marginal.marg import Marginal
import method.synthesis.privsyn.config as config

def transform_records_distinct_value(logger, df, dataset_domain):
    '''

    Replaces the attribute value with the index of the unique value on that attribute.
    Fix the domain size at the same time.
    
    It's vital for marginal caculation to function normally.

    '''
    logger.info("transforming records")

    distinct_shape = []
    for attr_index, attr in enumerate(dataset_domain.attrs):
        record = np.copy(df.loc[:, attr])
        unique_value = np.unique(record)
        distinct_shape.append(unique_value.size)

        for index, value in enumerate(unique_value):
            indices = np.where(record == value)[0]
            df.iloc[indices, attr_index] = index
    dataset_domain.shape = tuple(distinct_shape)
    dataset_domain.config = dict(
        zip(dataset_domain.attrs, distinct_shape))
    logger.info("transformed records")

    return df


def calculate_indif(logger, dataset, dataset_name, rho):
    """Calculate InDif on a Dataset object and persist to dependency path.

    Args:
        dataset: object with fields .df (DataFrame) and .domain (Domain)
    Returns:
        indif_df DataFrame with columns [first_attr, second_attr, num_cells, error]
    """
    df = dataset.df
    n_records = dataset.df.shape[0]
    dataset_domain = dataset.domain
    original_domain = dataset.domain

    logger.info("calculating pair indif")
    indif_df = pd.DataFrame(columns=["first_attr", "second_attr", "num_cells", "error"])
    indif_index = 0

    for first_index, first_attr in enumerate(dataset_domain.attrs[:-1]):
        first_marg = Marginal(dataset_domain.project(first_attr), dataset_domain)
        first_marg.count_records(df.values)
        first_histogram = first_marg.calculate_normalize_count()

        for second_attr in dataset_domain.attrs[first_index + 1:]:
            logger.info("calculating [%s, %s]" % (first_attr, second_attr))

            second_marg = Marginal(dataset_domain.project(second_attr), dataset_domain)
            second_marg.count_records(df.values)
            second_histogram = second_marg.calculate_normalize_count()

            pair_marg = Marginal(dataset_domain.project((first_attr, second_attr)), dataset_domain)
            pair_marg.count_records(df.values)
            pair_marg.calculate_count_matrix()

            independent_pair_distribution = np.outer(first_histogram, second_histogram)
            normalize_pair_marg_count = pair_marg.count_matrix / np.sum(pair_marg.count_matrix)
            error = np.sum(np.absolute(normalize_pair_marg_count - independent_pair_distribution))
            
            error_counts = error * n_records

            num_cells = original_domain.config[first_attr] * original_domain.config[second_attr]
            indif_df.loc[indif_index] = [first_attr, second_attr, num_cells, error_counts]
            indif_index += 1

    if rho > 0.0:
        noise = gaussian_noise(scale=np.sqrt(8 * indif_df.shape[0] / rho), size=indif_df.shape[0])
        indif_df.error += noise
    elif rho < 0.0:
        raise ValueError("Privacy budget rho cannot be negative.")

    try:
        pickle.dump(indif_df, open(config.DEPENDENCY_PATH + dataset_name, "wb"))
    except Exception:
        pass
    logger.info("calculated pair indif")
    return indif_df

def handle_isolated_attrs(dataset_domain, selected_attrs, indif_df, marginals, method="isolate", sort=False):
    # find attrs that does not appear in any of the pairwise marginals
    missing_attrs = set(dataset_domain.attrs) - selected_attrs

    if sort:
        indif_df.sort_values(
            by="error", ascending=False, inplace=True)
        indif_df.reset_index(drop=True, inplace=True)

    for attr in missing_attrs:
        if method == "isolate":
            marginals.append((attr,))
        elif method == "connect":
            # pick the best partner attribute by minimal error
            mask = (indif_df["first_attr"] == attr) | (indif_df["second_attr"] == attr)
            if not mask.any():
                marginals.append((attr,))
                continue
            sub = indif_df.loc[mask]
            best = sub.sort_values(by="error", ascending=True).iloc[0]
            other = best["second_attr"] if best["first_attr"] == attr else best["first_attr"]
            pair = (attr, other)
            # avoid duplicates like (a,b) and (b,a)
            if pair not in marginals and (other, attr) not in marginals:
                marginals.append(pair)
        else:
            marginals.append((attr,))

    return marginals
