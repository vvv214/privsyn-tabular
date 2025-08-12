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

from privsyn.lib_marginal.marg import Marginal
from tqdm import tqdm
import privsyn.config as config


def transform_records_distinct_value(logger, df, dataset_domain):
    '''
    Replaces the attribute value with the index of the unique value on that attribute.
    Fix the domain size at the same time.
    
    In our preprocessing step, this has already been done by ordinal encoding. 
    Therefore, you do not need to apply this function before opeartion
    '''
    logger.info("transforming records")

    distinct_shape = []
    for attr_index, attr in enumerate(dataset_domain.attrs):
        record = np.copy(df.loc[:, attr])
        unique_value = np.unique(record)
        distinct_shape.append(unique_value.size)

        for index, value in enumerate(unique_value):
            indices = np.where(record == value)[0]
            # self.df.loc[indices, attr] = index
            # df.value[indices, attr_index] = index
            df.iloc[indices, attr_index] = index
    dataset_domain.shape = tuple(distinct_shape)
    dataset_domain.config = dict(
        zip(dataset_domain.attrs, distinct_shape))
    logger.info("transformed records")

    return df


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

    # publish indif
    pickle.dump(indif_df, open(
        config.DEPENDENCY_PATH + dataset_name, "wb"))

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
            marginals.append(
                (match_df.loc[0, "first_attr"], match_df.loc[0, "second_attr"]))

    return marginals
