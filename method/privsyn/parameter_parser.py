import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()
    
    ####################################### general parameters ###################################
    # parameters for single run
    parser.add_argument('--dataset_name', type=str, default="colorado",
                        help='options: colorado')
    parser.add_argument('--device', type=str, default="cuda:0")
    
    # parameters for workflow control
    parser.add_argument('--is_cal_marginals', type=str2bool, default=True)
    parser.add_argument('--is_cal_depend', type=str2bool, default=True)
    parser.add_argument('--is_combine', type=str2bool, default=True)
    
    # parameters for privacy
    parser.add_argument('-e', '--epsilon', type=float, default=2.0,
                        help="when run main(), specify epsilon here")
    parser.add_argument('--marg_add_sensitivity', type=float, default=1.0)
    
    # parameters for marginal selection
    parser.add_argument('--marg_sel_threshold', type=float, default=20000)
    ############################################# specific parameters ############################################
    # parameters for marg consist and non-negativity
    parser.add_argument('--non_negativity', type=str, default="N3",
                        help='N1 -> norm_cut; N2 -> norm_sub; N3 -> norm_sub + norm_cut')
    parser.add_argument('--consist_iterations', type=int, default=501)
    
    # parameters for synthesizing
    parser.add_argument('--initialize_method', type=str, default="singleton")
    parser.add_argument('--update_method', type=str, default="S5",
                        help='S1 -> all replace; S2 -> all duplicate; S3 -> all half-half;'
                             'S4 -> replace+duplicate; S5 -> half-half+duplicate; S6 -> half-half+replace.'
                             'The optimal one is S5')
    parser.add_argument('--append', type=str2bool, default=True)
    parser.add_argument('--sep_syn', type=str2bool, default=False)
    
    parser.add_argument('--update_rate_method', type=str, default="U4",
                        help='U4 -> step decay; U5 -> exponential decay; U6 -> linear decay; U7 -> square root decay.'
                             'The optimal one is U4')
    parser.add_argument('--update_rate_initial', type=float, default=1.0)
    parser.add_argument('--num_synthesize_records', type=int, default=None)
    parser.add_argument('--update_iterations', type=int, default=50)
    parser.add_argument('--num_prep', dtype=str, default='privtree')
    parser.add_argument('--rare_threshold', dtype=float, default=0.005)

    return vars(parser.parse_args())

