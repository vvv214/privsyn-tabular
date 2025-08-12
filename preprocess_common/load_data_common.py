import numpy as np 
import pandas as pd
import os
import json
import math
import sklearn.preprocessing
from preprocess_common.preprocess import * 
from util.rho_cdp import cdp_rho

class data_preporcesser_common():
    def __init__(self, args):
        self.num_encoder = None
        self.cat_encoder = None
        self.num_col = 0
        self.cat_col = 0
        self.args = args
        self.cat_output_type = 'one_hot' if self.args.method == 'merf' else 'ordinal'

    def load_data(self, path, rho, user_domain_data: dict = None, user_info_data: dict = None):
        # preprocesser and column info will be saved in this class
        # dataframe, domain domain, portion of rho will be returned

        num_prep = self.args.num_preprocess
        rare_threshold = self.args.rare_threshold
        print(f'Numerical discretizer is {num_prep}')

        X_num = None 
        X_cat = None 
        domain_for_clipping = {} # Initialize domain_for_clipping 

        # Load domain data (either user-provided or from file)
        if user_domain_data is not None:
            domain = user_domain_data
        else:
            domain = json.load(open(os.path.join(path, 'domain.json'))) 

        if os.path.exists(os.path.join(path, 'X_num_train.npy')):
            X_num = np.load(os.path.join(path, 'X_num_train.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'X_cat_train.npy')):
            X_cat_raw = np.load(os.path.join(path, 'X_cat_train.npy'), allow_pickle=True)
            # Convert to pandas DataFrame to ensure proper handling of mixed types and then to object array
            X_cat = pd.DataFrame(X_cat_raw).astype(str).to_numpy()
        
        print(f"X_num shape: {X_num.shape if X_num is not None else None}")
        print(f"X_cat shape: {X_cat.shape if X_cat is not None else None}")
        
        num_divide, cat_divide = calculate_rho_allocate(X_num, X_cat, num_prep)

        if (rho == 0) and (num_divide + cat_divide > 0):
            if self.args.epsilon > 0:
                rho = cdp_rho(0.1*(num_divide+cat_divide)*self.args.epsilon, 0.1*(num_divide+cat_divide)*self.args.delta)
            else:
                rho = cdp_rho(0.1*(num_divide+cat_divide)*1.0, 0.1*(num_divide+cat_divide)*1e-5) # this is the default value setting
    
        if X_num is not None:
            # Apply clipping based on user_domain_data for numerical features
            for key, value in domain_for_clipping.items(): # Use domain_for_clipping
                if key.startswith('num_attr_') and isinstance(value, dict) and 'min' in value and 'max' in value:
                    try:
                        # Extract the index from 'num_attr_X' (e.g., 'num_attr_1' -> 0)
                        col_index = int(key.split('_')[-1]) - 1
                        if 0 <= col_index < X_num.shape[1]:
                            X_num[:, col_index] = np.clip(X_num[:, col_index], value['min'], value['max'])
                    except (ValueError, IndexError):
                        # Handle cases where key format is unexpected or index is out of bounds
                        pass
            
            print("Defining domain_for_clipping")
        # The 'domain' variable for clipping will be user_domain_data if provided, else domain
        domain_for_clipping = user_domain_data if user_domain_data is not None else domain

        

        if X_num is not None:
            print("Entering clipping loop")
            # Apply clipping based on user_domain_data for numerical features
            for key, value in domain_for_clipping.items(): # Use domain_for_clipping
                if key.startswith('num_attr_') and isinstance(value, dict) and 'min' in value and 'max' in value:
                    try:
                        # Extract the index from 'num_attr_X' (e.g., 'num_attr_1' -> 0)
                        col_index = int(key.split('_')[-1]) - 1
                        if 0 <= col_index < X_num.shape[1]:
                            X_num[:, col_index] = np.clip(X_num[:, col_index], value['min'], value['max'])
                    except (ValueError, IndexError):
                        # Handle cases where key format is unexpected or index is out of bounds
                        pass
            
            if num_prep != 'none':
                ord = False if self.args.method in ['rap', 'gsd'] else True
                self.num_encoder = discretizer(num_prep, num_divide * 0.1 * rho, ord=ord) # return an ordinal encoded discrete data
            else:
                self.num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
            X_num = self.num_encoder.fit_transform(X_num).astype(int)
            self.num_col = X_num.shape[1]
        if X_cat is not None:
            self.cat_encoder = rare_merger(cat_divide * 0.1 * rho, rare_threshold=rare_threshold, output_type=self.cat_output_type) #by default return ordinal encoded data
            X_cat = self.cat_encoder.fit_transform(X_cat).astype(int)
            self.cat_col = X_cat.shape[1]

        

        print("Defining domain_for_clipping")

        if self.args.method in ('merf', 'ddpm'):
            df = {
                "X_num": X_num,
                "X_cat": X_cat,
            }
            domain = {
                "X_num": [len(set(X_num[:, i])) for i in range(self.num_col)] if X_num is not None else [],
                "X_cat": [len(cat) for cat in self.cat_encoder.ordinal_encoder.categories_] if X_cat is not None else [],
            }
        else:
            col_name = [f'num_attr_{i}' for i in range(1, self.num_col + 1)] + [f'cat_attr_{i}' for i in range(1, self.cat_col + 1)]

            if X_num is None:
                df = pd.DataFrame(X_cat, columns=col_name)
            elif X_cat is None:
                df = pd.DataFrame(X_num, columns=col_name)
            else:
                df = pd.DataFrame(np.concatenate((X_num, X_cat), axis=1), columns=col_name) 
    
        return df, domain, 0.1*(num_divide + cat_divide)


    def reverse_data(self, df, path=None):
        if isinstance(df, pd.DataFrame):
            df = df.to_numpy()
        
        x_num = None 
        x_cat = None 
        if self.num_col > 0:
            x_num = df[:, 0:self.num_col]
            if x_num.ndim == 1:
                x_num = x_num.reshape(-1,1)
            
            if self.num_encoder is not None:
                x_num = self.num_encoder.inverse_transform(x_num).astype(float)
            if path is not None:
                np.save(os.path.join(path, 'X_num_train.npy'), x_num)

        if self.cat_col > 0:
            x_cat = df[:, self.num_col:self.num_col+self.cat_col]
            if x_cat.ndim == 1:
                x_cat = x_cat.reshape(-1,1)
            
            if self.cat_encoder is not None:
                x_cat = self.cat_encoder.inverse_transform(x_cat).astype(str)
            if path is not None:
                np.save(os.path.join(path, 'X_cat_train.npy'), x_cat) 
        
        return x_num, x_cat