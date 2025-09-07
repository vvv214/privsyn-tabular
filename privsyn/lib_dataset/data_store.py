import pickle
import os

import privsyn.config as config


import os

class DataStore:
    def __init__(self, args):
        self.args = args
        # Allow override via args.output_dir or PRIVSYN_OUTPUT_DIR
        base_dir = getattr(self.args, 'output_dir', None) if hasattr(self.args, '__dict__') else self.args.get('output_dir') if isinstance(self.args, dict) else None
        base_dir = base_dir or os.getenv('PRIVSYN_OUTPUT_DIR')
        if base_dir:
            self.parent_dir = os.path.join(base_dir, str(self.args['dataset']))
        else:
            self.parent_dir = 'exp/'+ str(self.args['dataset'])+ '/' + str(self.args['method'])+'/'+str(self.args['epsilon']) + '_' + str(self.args['num_preprocess']) + '_' + str(self.args['rare_threshold'])
        self.determine_data_path()
        self.generate_folder() 
    
    def determine_data_path(self):
        synthesized_records_name = '_'.join((self.args['dataset_name'], 'records'))
        marginal_name = '_'.join((self.args['dataset_name'], 'marginal'))
        
        self.synthesized_records_file = os.path.join(self.parent_dir, synthesized_records_name)
        self.marginal_file = os.path.join(self.parent_dir, marginal_name)
        
    def generate_folder(self):
        for path in config.ALL_PATH:
            if not os.path.exists(path):
                os.makedirs(path)
        
    def save_marginal(self, marginals):
        os.makedirs(os.path.dirname(self.marginal_file), exist_ok=True)
        pickle.dump(marginals, open(self.marginal_file, 'wb'))
    
    def load_marginal(self):
        return pickle.load(open(self.marginal_file, 'rb'))
