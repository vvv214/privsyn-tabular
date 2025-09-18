import pickle
import os

import method.synthesis.privsyn.config as config


class DataStore:
    def __init__(self, args):
        self.args = args
        exp_root = getattr(config, "EXPERIMENT_BASE_PATH", os.path.join("privsyn", "temp_data", "exp"))
        self.parent_dir = os.path.join(
            exp_root,
            str(self.args['dataset']),
            str(self.args['method']),
            f"{self.args['epsilon']}_{self.args['num_preprocess']}_{self.args['rare_threshold']}",
        )
        self.determine_data_path()
        self.generate_folder()
    
    def determine_data_path(self):
        synthesized_records_name = '_'.join((self.args['dataset_name'], 'records'))
        marginal_name = '_'.join((self.args['dataset_name'], 'marginal'))
        
        self.synthesized_records_file = os.path.join(self.parent_dir, synthesized_records_name)
        self.marginal_file = os.path.join(self.parent_dir, marginal_name)
        
    def generate_folder(self):
        # Ensure core temp/data paths exist
        for path in config.ALL_PATH:
            os.makedirs(path, exist_ok=True)
        # Ensure experiment parent dir exists for saving marginals/records
        os.makedirs(self.parent_dir, exist_ok=True)
    
    def load_processed_data(self):
        processed_file = os.path.join(config.PROCESSED_DATA_PATH, self.args['dataset_name'])
        with open(processed_file, 'rb') as fh:
            return pickle.load(fh)
    
    def save_synthesized_records(self, records, save_path = None):
        if save_path is None:
            with open(self.synthesized_records_file, 'wb') as fh:
                pickle.dump(records, fh)
        else:
            target = os.path.join(save_path, '_'.join((self.args['dataset_name'], str(self.args['epsilon']))))
            with open(target, 'wb') as fh:
                pickle.dump(records, fh)
        
    def save_marginal(self, marginals):
        with open(self.marginal_file, 'wb') as fh:
            pickle.dump(marginals, fh)
    
    def load_marginal(self):
        with open(self.marginal_file, 'rb') as fh:
            return pickle.load(fh)
