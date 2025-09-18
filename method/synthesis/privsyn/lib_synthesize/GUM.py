import numpy as np 
import pandas as pd
import logging
import math
from method.synthesis.privsyn.lib_synthesize.converge_imp import sep_graph, clip_graph, append_attrs
from method.synthesis.privsyn.lib_synthesize.update_config import UpdateConfig
from method.synthesis.privsyn.lib_dataset.dataset import Dataset
from method.synthesis.privsyn.lib_marginal.consistent import Consistenter



class GUM_Mechanism():
    def __init__(self, args, dataset, combined_marg_dict, one_way_marg_dict, progress_report=None):
        self.args = args
        self.check_args()

        self.original_dataset = dataset 
        self.synthesized_df = None

        self.sel_marg_name = list(combined_marg_dict.keys())
        self.one_way_marg_dict = one_way_marg_dict
        self.combined_marg_dict = combined_marg_dict
        self.marg_dict = {**self.one_way_marg_dict, **self.combined_marg_dict}

        self.logger = logging.getLogger('GUM')
        self.logger.setLevel(logging.WARNING)
        self.progress_report = progress_report


    def run(self, n_sample):
        self.improving_convergence()
        self.consist_marginals(self.original_dataset.domain, self.marg_dict)
        self.synthesize_records(n_sample)
        append_attrs(self.logger, self.original_dataset.domain, self.clip_layers, self.synthesized_df, self.marg_dict)
        return self.synthesized_df



    ########################################### helper functions #########################################
    def check_args(self):
        self.args.setdefault('consist_iterations', 501)
        self.args.setdefault('non_negativity', 'N3')
        self.args.setdefault('append', True)
        self.args.setdefault('sep_syn', False)
        self.args.setdefault('initialize_method', 'singleton')
        self.args.setdefault('update_method', 'S5')
        self.args.setdefault('update_rate_method', 'U4')
        self.args.setdefault('update_rate_initial', 1.0)
        self.args.setdefault('update_iterations', 50)


    def update_selected_marginals(self, combined_marg_dict, one_way_marg_dict=None):
        self.sel_marg_name = list(combined_marg_dict.keys())

        if not one_way_marg_dict:
            self.one_way_marg_dict = one_way_marg_dict
        self.combined_marg_dict = combined_marg_dict

        self.marg_dict = {**self.one_way_marg_dict, **self.combined_marg_dict}


    def consist_marginals(self, recode_domain, marg_dict):
        self.logger.info("consisting margs")
        
        consist_parameters = {
            "consist_iterations": self.args['consist_iterations'],
            "non_negativity": self.args['non_negativity'],
        }
        
        if self.progress_report:
            self.progress_report({"status": "running", "stage": "consistency", "overall_step": 3, "overall_total": 5, "message": "Consist marginals"})
        consistenter = Consistenter(marg_dict, recode_domain, consist_parameters)
        consistenter.consist_marginals()
        
        self.logger.info("consisted margs")


    def improving_convergence(self): 
        # This is for seperate attributes appending

        logger = logging.getLogger("GUM.improving_convergence")
        iterate_marginals, self.clip_layers = clip_graph(logger, self.original_dataset.domain, self.sel_marg_name, enable=self.args['append'])

        self.logger.info("iterate_marginals after clip_graph is %s" % (iterate_marginals,))
        self.iterate_keys = sep_graph(logger,self.original_dataset.domain, self.sel_marg_name, iterate_marginals, enable=self.args['sep_syn'])
    

    def synthesize_records(self, n_sample):
        self.args['num_synthesize_records'] = n_sample
        self.logger.info("updating data")

        temp_synthesized_df = pd.DataFrame(data=np.zeros([self.args['num_synthesize_records'], self.original_dataset.df.shape[1]], dtype=np.uint32),
                                        columns=self.original_dataset.domain.attrs)
        self.error_tracker = pd.DataFrame()
        
        # main procedure for synthesizing records
        for key, marginal_keys in self.iterate_keys.items():
            synthesizer = self._update_records(marginal_keys)
            normalized_key = self._normalize_key(key)
            cols = list(normalized_key)
            available = [c for c in cols if c in temp_synthesized_df.columns and c in synthesizer.update.df.columns]
            if available:
                temp_synthesized_df.loc[:, available] = synthesizer.update.df.loc[:, available]
            if hasattr(synthesizer, 'update') and hasattr(synthesizer.update, 'error_tracker'):
                self.error_tracker = pd.concat([self.error_tracker, synthesizer.update.error_tracker])

        self.logger.info("updated data")
        self.synthesized_df = temp_synthesized_df.copy(deep=True)


    def _update_records(self, margs_iterate_key):
        update_config = {
            "alpha": self.args['update_rate_initial'],
            "alpha_update_method": self.args['update_rate_method'],
            "update_method": self.args['update_method'],
            "threshold": 0.0
        }
        # update records in each sep graph

        singletons = {singleton: self.one_way_marg_dict[(singleton,)] for singleton in self.original_dataset.domain.attrs}

        synthesizer = UpdateConfig(self.original_dataset.domain, self.args['num_synthesize_records'], update_config, init_df = self.synthesized_df)
        synthesizer.update.initialize_records(margs_iterate_key, method=self.args['initialize_method'], singletons=singletons)

        for update_iteration in range(self.args['update_iterations']):
            synthesizer.update_alpha(update_iteration)
            margs_iterate_key = synthesizer.update_order(update_iteration, self.marg_dict, margs_iterate_key)

            for index, key in enumerate(margs_iterate_key):
                normalized_key = self._normalize_key(key)
                if normalized_key not in self.marg_dict:
                    raise KeyError(f"Missing marginal entry for key {normalized_key}")
                marg = self.marg_dict[normalized_key]
                synthesizer.update_records(marg, normalized_key, update_iteration)

            if self.progress_report:
                self.progress_report({
                    "status": "running",
                    "stage": "synthesize",
                    "overall_step": 4,
                    "overall_total": 5,
                    "inner_step": update_iteration + 1,
                    "inner_total": self.args['update_iterations'],
                    "message": f"Synthesize iteration {update_iteration + 1}/{self.args['update_iterations']}"
                })

        return synthesizer

    @staticmethod
    def _normalize_key(key):
        if isinstance(key, tuple):
            return key
        if isinstance(key, list):
            return tuple(key)
        if isinstance(key, str):
            if "::" in key:
                parts = tuple(part for part in key.split("::") if part)
                return parts if parts else (key,)
            return (key,)
        return (key,)


    def project(self, cols):
        data = self.synthesized_df.iloc[:, cols]
        domain = self.original_dataset.domain.project(cols)
        return Dataset(data, domain)
