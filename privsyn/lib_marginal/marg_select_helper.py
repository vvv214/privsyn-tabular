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
import itertools
import os

from privsyn.lib_marginal.marg import Marginal
from tqdm import tqdm
import privsyn.config as config


def calculate_indif(logger, dataset, dataset_name, rho):
    logger.info("calculating pair indif")

    indif_df = pd.DataFrame(
        columns=["first_attr", "second_attr", "num_cells", "error"])
    workloads = list(itertools.combinations(dataset.domain, 2))

    for first_attr, second_attr in tqdm(workloads, desc='Calculating Indif'):
        two_way = dataset.project((first_attr, second_attr)).datavector(flatten=False)
        indep_two_way = np.outer(
            dataset.project((first_attr, )).datavector(flatten=False),
            dataset.project((second_attr, )).datavector(flatten=False)
        )/two_way.sum()

        error = np.sum(np.abs(two_way - indep_two_way))

        num_cells = dataset.domain.project(first_attr).shape[0] * dataset.domain.project(second_attr).shape[0]
        indif_df.loc[len(indif_df)] = [first_attr, second_attr, num_cells, error]

    # add noise
    if rho != 0.0:
        indif_df.error += np.random.normal(
            scale=np.sqrt(8 * indif_df.shape[0]/rho), size=indif_df.shape[0])

    # publish indif (ensure directory exists)
    out_path = config.DEPENDENCY_PATH + dataset_name
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle.dump(indif_df, open(out_path, "wb"))

    logger.info("calculated pair indif")
    return indif_df



def handle_isolated_attrs(dataset_domain, selected_attrs, indif_df, marginals, method="isolate", sort=False):
    # find attrs that does not appear in any of the pairwise marginals
    missing_attrs = set(dataset_domain.attrs) - selected_attrs

    if sort:
        # self.dependency_df["error"] /= np.sqrt(self.dependency_df["num_cells"].astype("float"))
        indif_df.sort_values(
            by="error", ascending=False, inplace=True)
        indif_df.reset_index(drop=True, inplace=True)

    for attr in missing_attrs:
        if method == "isolate":
            marginals.append((attr,))

        elif method == "connect":
            match_missing_df = indif_df.loc[
                (indif_df["first_attr"] == attr) | (indif_df["second_attr"] == attr)]
            match_df = match_missing_df.loc[(match_missing_df["first_attr"].isin(selected_attrs)) | (
                match_missing_df["second_attr"].isin(selected_attrs))]
            match_df.reset_index(drop=True, inplace=True)
            if match_df.empty:
                marginals.append((attr,))
            else:
                marginals.append(
                    (match_df.loc[0, "first_attr"], match_df.loc[0, "second_attr"]))

    return marginals
