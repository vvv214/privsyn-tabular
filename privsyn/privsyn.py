#####################################################################
#                                                                   #
#                The main procedure class of dpsyn,                 #
#               which implements algorithm 3 in paper               #
#                                                                   #
#####################################################################

import sys
target_path="./"
sys.path.append(target_path)

import numpy as np
import pandas as pd
import logging
import math
import copy

from privsyn.parameter_parser import parameter_parser
from privsyn.lib_synthesize.converge_imp import sep_graph, clip_graph, append_attrs
from privsyn.lib_synthesize.update_config import UpdateConfig
from privsyn.lib_dataset.dataset import Dataset
from privsyn.lib_marginal.marg import Marginal
from privsyn.lib_marginal.consistent import Consistenter
from privsyn.lib_composition.advanced_composition import AdvancedComposition
from privsyn.lib_marginal.marg_determine import marginal_selection, marginal_combine
from privsyn.lib_marginal.filter import Filter
from privsyn.lib_dataset.data_store import DataStore
from privsyn.lib_dataset.domain import Domain
from privsyn.lib_synthesize.GUM import GUM_Mechanism

from functools import reduce

class PrivSyn():
    def __init__(self, args, df, domain, rho):
        '''
        Initialize the PrivSyn class, requiring:
            1. args: a dict of hyper-parameters
            2. df: a dataframe
            3. domain: a dict of attributes domain
            4. rho: total rho for PrivSyn. 
               The rho of each modules will be allocated based on the total rho, which is achieved by `privacy_budget_allocation`.
        '''

        self.logger = logging.getLogger('PrivSyn')        
        self.args = args
        self.total_rho = rho
        
        self.dataset_name = args['dataset']
        self.data_store = DataStore(self.args)
        self.load_data_from_df(df, domain)

        self.num_records = self.original_dataset.df.shape[0]
        self.num_attributes = self.original_dataset.df.shape[1]

        self.one_way_marg_dict  = {} # save one-way marginals
        self.combined_marg_dict  = {} # save multi-way marginals
        
        self.privacy_budget_allocation() # allocate rho to each steps

        

    def marginal_selection(self):
        '''
        This method consists of two steps:
            1. select two-way marginals and combine them

            2. construct all marginals (one-way and multi-way marginals) into `Marginal` class
               measure and introduce DP noise to all measured marginals
        '''
        self.sel_marg_name = self.select_and_combine_marginals(self.original_dataset)
        self.one_way_marg_dict = self.construct_margs(mode = 'one_way') 
        self.combined_marg_dict = self.construct_margs(mode = 'combined')

    

    def syn(self, n_sample, preprocesser, parent_dir, **kwargs):
        '''
        This method is the main synthesis process of PrivSyn:
            1. initialize a GUM class, requiring: 
                dataset (in Dataset class)
                dictionary of multi-way marginals {(attr1, attr2): Marginal, (attr2, attr3): Marginal, ...} 
                dictionary of one-way marginals {(attr1,): Marginal, (attr2,): Marginal, ...}

            2. run GUM, achieved by two steps: consistency + updation 

            3. postprocess the synthesized dataset and save it in parent_dir
        '''
        self.model = GUM_Mechanism(self.args, self.original_dataset, self.combined_marg_dict, self.one_way_marg_dict)
        self.synthesized_df = self.model.run(n_sample)
        self.postprocessing(preprocesser, parent_dir)




    ########################################### helper function ###########################################
    
    def load_data_from_df(self, df, domain):
        self.logger.info("loading dataset %s" % (self.dataset_name,))
        domain_list = [v for v in domain.values()]
        domain = Domain(df.columns, domain_list)
        self.original_dataset = Dataset(df, domain)
        

    def privacy_budget_allocation(self):
        self.indif_rho = self.total_rho * 0.1
        self.one_way_marginal_rho = self.total_rho * 0.1
        self.combined_marginal_rho = self.total_rho * 0.8
    

    def select_and_combine_marginals(self, dataset):
        '''
        implements algorithm 1(marginal selection) and algorithm 2(marginal combine) in paper
        '''
        if self.args['is_cal_marginals']:
            self.logger.info("selecting marginals")
    
            select_args = copy.deepcopy(self.args)
            select_args['indif_rho'] = self.indif_rho
            select_args['combined_marginal_rho'] = self.combined_marginal_rho
            select_args['threshold'] = 5000
            
            marginals = marginal_selection(dataset, select_args) #alg1

            if select_args['is_combine']:
                marginals = marginal_combine(dataset, select_args, marginals) #alg2

            self.data_store.save_marginal(marginals)
        else:
            marginals = self.data_store.load_marginal()
        
        return marginals

    
    def construct_marg(self, dataset, marginal):
        marg = Marginal(dataset.domain.project(marginal), dataset.domain)
        marg.count_records(dataset.df.values)
        
        return marg
    
    def anonymize_marg(self, marg, rho=0.0):
        sigma = math.sqrt(self.args['marg_add_sensitivity'] ** 2 / (2.0 * rho))
        noise = np.random.normal(scale=sigma, size=marg.num_key)
        marg.count += noise

        return marg

    def construct_margs(self, mode):
        if mode == 'one_way':
            self.logger.info("constructing one-way marginals")

            one_way_marg_dict = {}
            rho = self.one_way_marginal_rho / len(self.original_dataset.domain.attrs)
            self.gauss_sigma_4_one_way = math.sqrt(self.args['marg_add_sensitivity'] ** 2 / (2.0 * rho))

            for attr in self.original_dataset.domain.attrs:
                marg = self.construct_marg(self.original_dataset, (attr,))
                self.anonymize_marg(marg, rho)
                one_way_marg_dict[(attr,)] = marg
            
            self.logger.info("constructed one-way marginals")
            return one_way_marg_dict
        
        elif mode == 'combined':
            self.logger.info("constructing combined marginals")

            divider = 0.0
            combined_marg_dict = {}
            rho = self.combined_marginal_rho

            for i, marginal in enumerate(self.sel_marg_name):
                self.logger.debug('%s th marginal' % (i,))
                combined_marg_dict[marginal] = self.construct_marg(self.original_dataset, marginal)

            for key, marg in combined_marg_dict.items():
                divider += math.pow(marg.num_key, 2.0 / 3.0)
            for key, marg in combined_marg_dict.items():
                marg.rho = rho * math.pow(marg.num_key, 2.0 / 3.0) / divider
                self.anonymize_marg(marg, rho=marg.rho)
            
            self.logger.info("constructed combined marginals")
            return combined_marg_dict
            

    def postprocessing(self, preprocesser, save_path = None):
        print(self.synthesized_df['y_attr'].value_counts())
        preprocesser.reverse_data(self.synthesized_df, save_path)



    ######################################## Static Method ##############################################
    # This function extract the process of two-way marginal selection, which can be used for other synthesize methods

    @staticmethod
    def two_way_marginal_selection(df, domain, rho_indif, rho_measure):
        args = {}
        args['indif_rho'] = rho_indif
        args['combined_marginal_rho'] = rho_measure # don't used in this phase, just as a penalty term
        args['dataset_name'] = 'temp_data'
        args['is_cal_depend'] = True
        args['marg_sel_threshold'] = 20000

        domain_list = [v for v in domain.values()]
        domain = Domain(df.columns, domain_list)
        dataset = Dataset(df, domain)
        
        marginals = marginal_selection(dataset, args)

        return marginals




def config_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def add_default_params(args):
    args.dataset_name = args.dataset
    args.is_cal_marginals = True 
    args.is_cal_depend = True
    args.is_combine = True 
    args.marg_add_sensitivity = 1.0
    args.marg_sel_threshold = 20000
    args.non_negativity = "N3"
    args.consist_iterations = 501
    args.initialize_method = "singleton"
    args.update_method = "S5"
    args.append = True 
    args.sep_syn = False 
    args.update_rate_method = "U4"
    args.update_rate_initial = 1.0
    args.update_iterations = 50

    return args

def privsyn_main(args, df, domain, rho, **kwargs):
    config_logger()

    args = vars(add_default_params(args))
    privsyn_generator = PrivSyn(args, df, domain, rho) 
    privsyn_generator.marginal_selection()

    return {"privsyn_generator": privsyn_generator}



if __name__ == "__main__":
    args = parameter_parser()
    
    privsyn_main(args)

