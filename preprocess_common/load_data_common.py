import numpy as np 
import pandas as pd
import logging
import os
import json
import math
import sklearn.preprocessing
from preprocess_common.preprocess import * 
from util.rho_cdp import cdp_rho

class data_preporcesser_common():
    def __init__(self, args):
        self.logger = logging.getLogger("preprocess_common.load_data_common")
        self.num_encoder = None
        self.cat_encoder = None
        self.num_col = 0
        self.cat_col = 0
        self.args = args
        self.cat_output_type = 'one_hot' if self.args.method == 'merf' else 'ordinal'

    def load_data(self, X_num_raw: np.ndarray, X_cat_raw: np.ndarray, rho, user_domain_data: dict = None, user_info_data: dict = None):
        # preprocesser and column info will be saved in this class
        # dataframe, domain domain, portion of rho will be returned

        num_prep = self.args.num_preprocess
        rare_threshold = self.args.rare_threshold
        self.logger.info(f'Numerical discretizer is {num_prep}')

        X_num = X_num_raw
        X_cat = X_cat_raw
        # Normalize empty arrays to None to avoid shape errors later
        if isinstance(X_num, np.ndarray):
            if X_num.size == 0:
                X_num = None
            elif X_num.ndim == 1 and X_num.shape[0] == 0:
                X_num = None
        if isinstance(X_cat, np.ndarray):
            if X_cat.size == 0:
                X_cat = None
            elif X_cat.ndim == 1 and X_cat.shape[0] == 0:
                X_cat = None
        domain_for_clipping = {} # Initialize domain_for_clipping 

        # Load domain data (either user-provided or from file)
        if user_domain_data is not None:
            domain = user_domain_data
        else:
            # This path should ideally not be taken if X_num_raw and X_cat_raw are provided
            # If it is, it means domain.json is expected from disk, which is inconsistent with in-memory data
            raise ValueError("Domain data must be provided via user_domain_data when X_num_raw and X_cat_raw are used.")
        
        self.logger.debug(f"X_num shape: {X_num.shape if X_num is not None else None}")
        self.logger.debug(f"X_cat shape: {X_cat.shape if X_cat is not None else None}")
        
        num_divide, cat_divide = calculate_rho_allocate(X_num, X_cat, num_prep)

        if (rho == 0) and (num_divide + cat_divide > 0):
            if self.args.epsilon > 0:
                rho = cdp_rho(0.1*(num_divide+cat_divide)*self.args.epsilon, 0.1*(num_divide+cat_divide)*self.args.delta)
            else:
                rho = cdp_rho(0.1*(num_divide+cat_divide)*1.0, 0.1*(num_divide+cat_divide)*1e-5) # this is the default value setting
    
        # The 'domain' variable for clipping will be user_domain_data if provided, else domain
        domain_for_clipping = user_domain_data if user_domain_data is not None else domain

        if X_num is not None:
            self.logger.debug("Entering clipping loop")
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
            self.num_col = X_num.shape[1] if X_num.ndim > 1 and X_num.shape[1] > 0 else 0
        if X_cat is not None:
            self.cat_encoder = rare_merger(cat_divide * 0.1 * rho, rare_threshold=rare_threshold, output_type=self.cat_output_type) #by default return ordinal encoded data
            X_cat = self.cat_encoder.fit_transform(X_cat).astype(int)
            self.cat_col = X_cat.shape[1] if X_cat.ndim > 1 and X_cat.shape[1] > 0 else 0

        self.logger.debug("Defining domain_for_clipping")

        # Get actual column names from user_info_data
        num_col_names_actual = user_info_data.get("num_columns", [])
        cat_col_names_actual = user_info_data.get("cat_columns", [])

        # Construct df using actual column names
        if X_num is None:
            df = pd.DataFrame(X_cat, columns=cat_col_names_actual) # Use actual cat names
        elif X_cat is None:
            df = pd.DataFrame(X_num, columns=num_col_names_actual) # Use actual num names
        else:
            # Concatenate with actual column names
            df = pd.DataFrame(np.concatenate((X_num, X_cat), axis=1), 
                              columns=num_col_names_actual + cat_col_names_actual) 

        if self.args.method not in ('merf', 'ddpm'): # Only if we are using our user_domain_data
            # Construct domain_list from the processed df columns and the domain dictionary
            # The order of columns in df is important here
            domain_list = []
            for col_name in df.columns: # Iterate through the columns of the processed dataframe
                if col_name in domain and 'size' in domain[col_name]:
                    domain_list.append(domain[col_name]['size'])
                else:
                    # This should ideally not happen if domain_data is well-formed
                    # Log an error or raise an exception if a column is missing from domain_data
                    self.logger.warning(f"Column '{col_name}' not found or 'size' missing in domain data. Skipping.")
            domain = domain_list # Update the 'domain' variable to be returned
    
        if df.empty or df.shape[1] == 0:
            raise ValueError("Processed dataframe has no columns. Please check input data and preprocessing steps.")

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
                try:
                    # Only apply clipping if it's a KBinsDiscretizer (i.e., num_prep was 'uniform_kbins')
                    if isinstance(self.num_encoder, discretizer) and hasattr(self.num_encoder, 'encoder') and hasattr(self.num_encoder.encoder, 'n_bins_'):
                        x_num_clipped = x_num.copy() # Make a copy before clipping
                        self.logger.debug(f"x_num before clip - min: {x_num_clipped.min()}, max: {x_num_clipped.max()}")
                        max_bin_index = len(self.num_encoder.encoder.categories_[0]) - 2 # Correct max ordinal index
                        self.logger.debug(f"max_bin_index: {max_bin_index}")
                        x_num_clipped = np.clip(x_num_clipped, 0, max_bin_index)
                        self.logger.debug(f"x_num after clip - min: {x_num_clipped.min()}, max: {x_num_clipped.max()}")
                        x_num = x_num_clipped # Use the clipped copy
                    
                    # Ensure x_num is a contiguous integer array before inverse_transform
                    x_num = np.ascontiguousarray(x_num).astype(int)
                    x_num = self.num_encoder.inverse_transform(x_num).astype(float)
                except IndexError as e:
                    self.logger.warning(f"IndexError during numerical inverse_transform. Returning None for numerical data. Error: {e}")
                    x_num = None # Return None for numerical data if inverse_transform fails
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
