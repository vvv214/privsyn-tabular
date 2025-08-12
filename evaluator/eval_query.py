import numpy as np
import os 
import tempfile
import shutil
import random
import pandas as pd
import json
# from TabDDPM.scripts.sample import sample
from copy import deepcopy
from pathlib import Path
from evaluator.data.dataset import read_pure_data
from evaluator.data.data_utils import *
# from DP_MERF.sample import merf_heterogeneous_sample

def query_succeed(x: np.array, query_attr, query, query_type):
    query_res = np.full(len(x), True)
    for i in range(len(query)):
        if query_type[i] == 'num':
            query_res = query_res & (x[:, query_attr[i]].astype(float) >= query[i][0]) & (x[:, query_attr[i]].astype(float) <= query[i][1])
        elif query_type[i] == 'cat':
            query_res = query_res & (x[:, query_attr[i]] == query[i][0])
    return sum(query_res)

def make_query(
    synthetic_data_path,
    data_path,
    task_type,
    query_times,
    attr_num,
    seeds = 0
):
    random.seed(seeds)
    
    print('Starting query error evaluation')

    X_num_real, X_cat_real = read_pure_data(data_path, split = 'test')
    if X_num_real is None and X_cat_real is None:
        raise ValueError("Both numerical and categorical data are None for real data.")
    elif X_num_real is None: 
        X_cat_real = X_cat_real.astype(str)
        real_data = X_cat_real
    elif X_cat_real is None: 
        X_num_real = X_num_real.astype(float)
        real_data = X_num_real
    else:
        X_num_real = X_num_real.astype(float)
        X_cat_real = X_cat_real.astype(str)
        real_data = np.concatenate((X_num_real, X_cat_real), axis=1)

    info = json.load(open(os.path.join(data_path, 'info.json')))
    df_fake = pd.read_csv(synthetic_data_path)
    
    num_cols = info.get('num_columns', [])
    cat_cols = info.get('cat_columns', [])

    X_num_fake = df_fake[num_cols].to_numpy() if num_cols else None
    X_cat_fake = df_fake[cat_cols].to_numpy() if cat_cols else None

    if X_num_fake is None and X_cat_fake is not None:
        X_cat_fake = X_cat_fake.astype(str)
        fake_data = X_cat_fake
    elif X_cat_fake is None and X_num_fake is not None: 
        X_num_fake = X_num_fake.astype(float)
        fake_data = X_num_fake
    elif X_num_fake is not None and X_cat_fake is not None:
        X_num_fake = X_num_fake.astype(float)
        X_cat_fake = X_cat_fake.astype(str)
        fake_data = np.concatenate((X_num_fake, X_cat_fake), axis=1)
    else: # both are None
        raise ValueError("Both numerical and categorical data are None for fake data.")

    # obtain the domain of variables
    num_attr = 0
    cat_attr = 0
    num_range = None
    cat_range = None
    if X_num_real is not None: 
        num_attr = X_num_real.shape[1]
        num_range = get_numerical_range(X_num_real)
    if X_cat_real is not None: 
        cat_attr = X_cat_real.shape[1]
        cat_range = get_category_range(X_cat_real)

    error = []
    for i in range(query_times):
        # in each query time, choose attr respectively
        query_attr = np.random.choice(np.arange(0, num_attr + cat_attr), size = attr_num, replace = False)
        # real_query_data = real_data[:, query_attr]
        # fake_query_data = fake_data[:, query_attr]
        query = []
        query_type = []
        for x in query_attr:
            if x < num_attr:
                query.append(sorted([random.uniform(num_range[x][0], num_range[x][1]), random.uniform(num_range[x][0], num_range[x][1])]))
                query_type.append('num')
            elif x >= num_attr and x < (num_attr + cat_attr): 
                query.append(np.random.choice(cat_range[x - num_attr], 1, replace=False))
                query_type.append('cat')
            
        error.append(abs(
            query_succeed(real_data, query_attr, query, query_type)/len(real_data) - 
            query_succeed(fake_data, query_attr, query, query_type)/len(fake_data)
        ))
    
    
    return np.mean(error)

def main(eval_args, original_data_dir, synthesized_csv_path):
    # Extract relevant args from eval_args or use defaults
    # For now, let's use some reasonable defaults or infer from info.json if available
    task_type = "binclass" # Default, or infer from info.json
    query_times = 100 # Default
    attr_num = 2 # Default
    seeds = 0 # Default

    # Call the make_query function
    query_error = make_query(
        synthetic_data_path=synthesized_csv_path,
        data_path=original_data_dir,
        task_type=task_type,
        query_times=query_times,
        attr_num=attr_num,
        seeds=seeds
    )
    print(f"Query Error: {query_error}")