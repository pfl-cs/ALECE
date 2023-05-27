import numpy as np
import os
import sys
sys.path.append("../")
from src.utils import arg_parser_utils, eval_utils
from src.arg_parser import arg_parser
import math


def process_error_val(val):
    if val < 100:
        s = str(round(val, 2))
        terms = s.split('.')
        if len(terms) == 1:
            s += '.00'
        elif len(terms[1]) == 1:
            s += '0'

        return f'${s}$'

    if val >= 1e10:
        return '>$10^{10}$'
    if val < 1e5:
        int_val = int(val)
        s = format(int_val, ',d')
        return f'${s}$'
    exponent = int(math.log10(val))
    x = math.pow(10, exponent)
    a = val / x

    a_str = str(round(a,1))
    terms = a_str.split('.')
    if len(terms) == 1:
        a_str += '.0'
    return f'${a_str}$$\\cdot$$10^{exponent}$'


def calc_p_error(args, test_wl_type=None, model=None):
    if test_wl_type is None:
        test_wl_type = args.test_wl_type
    if model is None:
        model = args.model
    p_error_dir, _ = arg_parser_utils.get_p_q_error_dir(args,test_wl_type)
    fname = f'{model}.npy'
    model_errors_path = os.path.join(p_error_dir, fname)
    if os.path.exists(model_errors_path) == False:
        print(f'{model_errors_path} does not exist')
        return None

    optimal_errors_path = os.path.join(p_error_dir, 'optimal.npy')
    assert os.path.exists(optimal_errors_path)

    model_errors = np.load(model_errors_path)
    optimal_errors = np.load(optimal_errors_path)

    min_e = np.min(optimal_errors)
    assert min_e > 0
    model_errors = np.clip(model_errors, a_min=min_e, a_max=1e25)
    p_error_test = eval_utils.generic_calc_q_error(model_errors, optimal_errors)
    p_error_test = np.sort(p_error_test)
    n = p_error_test.shape[0]
    # print(f'p_error.shape = {p_error_test.shape}')
    ratios = [0.5, 0.9, 0.95, 0.99]

    error_vals = []
    for ratio in ratios:
        idx = int(n * ratio)
        # print(f'idx = {idx}')
        error_vals.append(p_error_test[idx])

    # print(error_vals)
    results = []
    for val in error_vals:
        results.append(process_error_val(val))
    result_str = ' & '.join(results)
    print(f'{args.data}-{args.wl_type}-{args.model}: {result_str}')
    return error_vals


if __name__ == '__main__':
    args = arg_parser.get_arg_parser()
    error_vals = calc_p_error(args)
    print(error_vals)


# python benchmark/calc_q_error.py --data STATS --wl_type ins_heavy --model ALECE