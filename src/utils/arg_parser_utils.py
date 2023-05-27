import os

def get_workload_dir(args, wl_type=None):
    if wl_type is None:
        wl_type = args.wl_type

    workload_dir = os.path.join(args.workload_base_dir, wl_type)
    return workload_dir


def get_query_paths_in_static_wl_dir(args, wl_data_type=None):
    if wl_data_type is None:
        wl_data_type = args.wl_data_type

    static_workload_dir = get_workload_dir(args, wl_type='static')

    query_path = os.path.join(static_workload_dir, wl_data_type + '_queries.sql')
    sub_queries_path = os.path.join(static_workload_dir, wl_data_type + '_sub_queries.sql')
    single_tbls_path = os.path.join(static_workload_dir, wl_data_type + '_single_tbls.sql')

    return static_workload_dir, query_path, sub_queries_path, single_tbls_path


def get_p_q_error_dir(args, test_wl_type=None):
    if test_wl_type is None:
        assert args.test_wl_type is not None
        test_wl_type = args.test_wl_type

    if test_wl_type != 'static':
        q_error_dir = os.path.join(args.experiments_dir, f'q_error/{test_wl_type}')

        p_error_dir = os.path.join(args.experiments_dir, f'p_error/{test_wl_type}')
    else:
        q_error_dir = os.path.join(args.experiments_dir, 'q_error/static')
        p_error_dir = os.path.join(args.experiments_dir, 'p_error/static')

    return p_error_dir, q_error_dir

def get_wl_type_pre_and_pg_cards_paths(args, wl_type=None, test_wl_type=None):
    if wl_type is None:
        wl_type = args.wl_type
    if test_wl_type is None:
        test_wl_type = args.test_wl_type
    terms = wl_type.split('_')
    train_wl_type_pre = terms[0]
    terms = test_wl_type.split('_')
    test_wl_type_pre = terms[0]
    pg_cards_path = f'pg_{args.data}_{test_wl_type_pre}.txt'
    return train_wl_type_pre, test_wl_type_pre, pg_cards_path


#--------------------For parameter studies---------------
def get_feature_data_dir(args, wl_type=None):
    workload_dir = get_workload_dir(args, wl_type)
    feature_data_dir = os.path.join(workload_dir, args.feature_data_dirname)
    histogram_ckpt_dir = os.path.join(workload_dir, 'histogram_ckpt')

    return workload_dir, feature_data_dir, histogram_ckpt_dir

def get_ckpt_dir(args, wl_type=None):
    if wl_type is None:
        wl_type = args.wl_type

    ckpt_dir = os.path.join(args.experiments_dir, args.ckpt_dirname.format(args.model, wl_type))
    return ckpt_dir