import os
import argparse
import pathlib

def get_stats_arg_parser():
    parser = argparse.ArgumentParser(description='ALECE')
    parser.add_argument('--data', type=str, default='STATS',
                        help='')

    # ----------------------------------- Data Path Params -----------------------------------
    parser.add_argument('--base_dir', type=str, default='../data/STATS/',
                        help='')
    parser.add_argument('--absolute_base_dir', type=str,
                        default='$WORKSPACE_DIR$/data/STATS',
                        help='')
    parser.add_argument('--data_dir', type=str, default='../data/STATS/data',
                        help='data dir')
    parser.add_argument('--workload_base_dir', type=str, default='../data/STATS/workload/',
                        help='')
    parser.add_argument('--data_dirname', type=str, default='data',
                        help='data dirname')
    parser.add_argument('--int_data_dirname', type=str, default='int',
                        help='')

    parser.add_argument('--experiments_dir', type=str, default='../exp/STATS/',
                        help='Directory to put the experimental results.')
    parser.add_argument('--feature_data_dirname', type=str, default='features',
                        help='its path is os.path.join(args.data_dir/args.dynamic_workload_dirname, args.feature_data_dirname)')
    parser.add_argument('--workload_fname', type=str, default='workload.sql',
                        help='')
    parser.add_argument('--train_queries_fname', type=str, default='train_queries.sql',
                        help='its path is os.path.join(args.data_dir/static_workload_dirname, args.train_queries_file)')
    parser.add_argument('--train_sub_queries_fname', type=str, default='train_sub_queries.sql',
                        help='its path is os.path.join(args.data_dir/static_workload_dirname, args.train_sub_queries_file)')
    parser.add_argument('--train_single_tbls_fname', type=str, default='train_single_tbls.sql',
                        help='its path is os.path.join(args.data_dir/static_workload_dirname, args.train_single_tbls_fname)')

    parser.add_argument('--test_queries_fname', type=str, default='test_queries.sql',
                        help='its path is os.path.join(args.data_dir/static_workload_dirname, args.test_queries_file)')
    parser.add_argument('--test_sub_queries_fname', type=str, default='test_sub_queries.sql',
                        help='its path is os.path.join(args.data_dir/static_workload_dirname, args.test_sub_queries_file)')
    parser.add_argument('--test_single_tbls_fname', type=str, default='test_single_tbls.sql',
                        help='its path is os.path.join(args.data_dir/static_workload_dirname, args.test_single_tbls_fname)')

    parser.add_argument('--base_queries_fname', type=str, default='base_queries.sql',
                        help='its path is os.path.join(args.data_dir, args.base_queries_fname)')
    parser.add_argument('--tables_info_fname', type=str, default='tables_info.txt',
                        help='its path is os.path.join(args.data_dir, args.tables_info_file)')

    # ----------------------------------- DB Params -----------------------------------
    parser.add_argument('--db_data_dir', type=str, default='$PG_DATADIR$', help='')
    parser.add_argument('--db_name', type=str, default='', help='')
    parser.add_argument('--db_subqueries_fname', type=str, default='sub_queries.txt', help='')
    parser.add_argument('--db_single_tbls_fname', type=str, default='single_tbl_queries.txt', help='')

    # ----------------------------------- Model Params -----------------------------------
    parser.add_argument('--model', type=str, default='ALECE', help='')
    parser.add_argument('--input_dim', type=int, default=97, help='')
    parser.add_argument('--use_float64', type=int, default=0, help='')
    parser.add_argument('--latent_dim', type=int, default=256, help='dimension of latent variables.')
    parser.add_argument('--mlp_num_layers', type=int, default=6, help='number of hidden layers in a mlp')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512,
                        help='number of neurons in a mlp layer.')
    parser.add_argument('--use_positional_embedding', type=int, default=0, help='')
    parser.add_argument('--use_dropout', type=int, default=0, help='')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='')
    parser.add_argument('--num_attn_heads', type=int, default=8, help='')
    parser.add_argument('--attn_head_key_dim', type=int, default=511, help='')
    parser.add_argument('--feed_forward_dim', type=int, default=2048, help='')
    parser.add_argument('--num_self_attn_layers', type=int, default=6, help='')
    parser.add_argument('--num_external_attn_layers', type=int, default=6, help='')

    # ----------------------------------- Featurization Params -----------------------------------
    parser.add_argument('--num_tables', type=int, default=8,
                        help='')

    # Histogram Feature Params
    parser.add_argument('--n_bins', type=int, default=40,
                        help='')
    parser.add_argument('--histogram_feature_dim', type=int, default=430, help='')
    parser.add_argument('--num_attrs', type=int, default=43, help='')

    # Query Part Feature Params
    parser.add_argument('--query_part_feature_dim', type=int, default=96, help='')
    parser.add_argument('--join_pattern_dim', type=int, default=11, help='')

    # ----------------------------------- Training Params -----------------------------------
    parser.add_argument('--gpu', type=int, default=1, help='')
    parser.add_argument('--buffer_size', type=int, default=32, help='')
    parser.add_argument('--use_loss_weights', type=int, default=1, help='')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--shuffle_buffer_size', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Adam optimizer.')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--min_n_epochs', type=int, default=3, help='Minimum number of epochs.')
    parser.add_argument('--card_log_scale', type=int, default=1, help='take logarithm of the card')
    parser.add_argument('--scaling_ratio', type=float, default=20., help='log(card)/scaling_ratio')

    # ----------------------------------- workload Params -----------------------------------
    parser.add_argument('--wl_data_type', type=str, default='init', help='train or test')
    parser.add_argument('--wl_type', type=str, default='ins_heavy', help='ins_heavy or upd_heavy or dist_shift')
    parser.add_argument('--test_wl_type', type=str, default=None, help='ins_heavy or upd_heavy or dist_shift')

    # ----------------------------------- e2e Params -----------------------------------
    parser.add_argument('--db_task', type=str, default='query_exec',
                        help='select one from \{query_exec, pg_card_access\}')
    parser.add_argument('--e2e_dirname', type=str, default='e2e',
                        help='')
    parser.add_argument('--e2e_print_sub_queries', type=int, default=0,
                        help='')
    parser.add_argument('--e2e_write_pg_join_cards', type=int, default=0,
                        help='')
    parser.add_argument('--ignore_single_cards', type=int, default=1,
                        help='')

    # ----------------------------------- ckpt Params -----------------------------------
    parser.add_argument('--ckpt_dirname', type=str, default='ckpt/{0:s}_{1:s}',
                        help='')
    parser.add_argument('--keep_train', type=int, default=0,
                        help='')

    # ----------------------------------- P-error Params -----------------------------------
    parser.add_argument('--costs_dirname', type=str, default='costs', help='')
    parser.add_argument('--hints_dirname', type=str, default='hints', help='')

    # ----------------------------------- calc Params -----------------------------------
    parser.add_argument('--calc_task', type=str, default='q_error',
                        help='')

    args = parser.parse_args()
    return args

def get_arg_parser():
    args = get_stats_arg_parser()
    workspace_dir = str(pathlib.Path().resolve().parent.absolute())
    args.absolute_base_dir = args.absolute_base_dir.replace('$WORKSPACE_DIR$', workspace_dir)

    if args.test_wl_type is None:
        args.test_wl_type = args.wl_type

    if args.test_wl_type == 'static':
        args.db_name = args.data.lower()
    else:
        terms = args.wl_type.split('_')
        wl_type_pre = terms[0]

        terms = args.test_wl_type.split('_')
        test_wl_type_pre = terms[0]
        args.db_name = f'{args.data}_{args.model}_{wl_type_pre}_{test_wl_type_pre}'.lower()

    return args

if __name__ == '__main__':
    args = get_arg_parser()
