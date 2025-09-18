import argparse
import pytest

from method.synthesis.privsyn.parameter_parser import str2bool, parameter_parser


def test_str2bool_true():
    assert str2bool('yes') == True
    assert str2bool('true') == True
    assert str2bool('t') == True
    assert str2bool('y') == True
    assert str2bool('1') == True
    assert str2bool(True) == True


def test_str2bool_false():
    assert str2bool('no') == False
    assert str2bool('false') == False
    assert str2bool('f') == False
    assert str2bool('n') == False
    assert str2bool('0') == False
    assert str2bool(False) == False


def test_str2bool_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool('invalid')


def test_parameter_parser_defaults():
    # Mock sys.argv to test default arguments
    import sys
    old_argv = sys.argv
    sys.argv = ['test_script.py']

    args = parameter_parser()

    sys.argv = old_argv # Restore original sys.argv

    assert isinstance(args, dict)
    assert args['dataset_name'] == "colorado"
    assert args['device'] == "cuda:0"
    assert args['is_cal_marginals'] == True
    assert args['epsilon'] == 2.0
    assert args['marg_sel_threshold'] == 20000


def test_parameter_parser_custom_args():
    # Mock sys.argv to test custom arguments
    import sys
    old_argv = sys.argv
    sys.argv = ['test_script.py', '--dataset_name', 'adult', '--epsilon', '1.0', '--is_cal_marginals', 'false']

    args = parameter_parser()

    sys.argv = old_argv # Restore original sys.argv

    assert isinstance(args, dict)
    assert args['dataset_name'] == "adult"
    assert args['epsilon'] == 1.0
    assert args['is_cal_marginals'] == False
