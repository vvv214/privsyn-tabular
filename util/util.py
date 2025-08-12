import os 
import json

def make_exp_dir(args):
    if args.test:
        parent_dir = f'exp/{args.dataset}/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}_test'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)
    elif args.syn_test:
        parent_dir = f'exp/{args.dataset}/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}_syn'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)
    else:
        parent_dir = f'exp/{args.dataset}/{args.method}/{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}'
        data_path = f'data/{args.dataset}/'
        os.makedirs(parent_dir, exist_ok=True)

    return parent_dir, data_path


def prepare_eval_config(args, parent_dir):
    with open(f'data/{args.dataset}/info.json', 'r') as file:
        data_info = json.load(file)

    config = {
        'parent_dir': parent_dir,
        'real_data_path': f'data/{args.dataset}/',
        'model_params':{'num_classes': data_info['n_classes']},
        'sample': {'seed': 0, 'sample_num': data_info['train_size']}
    }

    with open(os.path.join(parent_dir, 'eval_config.json'), 'w') as file:
        json.dump(config, file, indent=4)
    return config

def algo_method(args):
    if args.method == 'privsyn':
        from privsyn.privsyn import privsyn_main
        algo = privsyn_main 
    else:
        raise 'Invalid Method Name'
    return algo