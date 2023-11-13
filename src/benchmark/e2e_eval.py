import shutil

import psycopg2
import time
import os
import sys
import numpy as np
sys.path.append("../")
from src.utils import string_utils, FileViewer, file_utils, pg_utils, arg_parser_utils
from src.arg_parser import arg_parser
from decimal import Decimal


def get_test_sqls_and_copy_sqls(workload_lines):
    init_copy_sqls = []
    lines = workload_lines
    assert lines[0].lower().startswith('copy ')
    start_i = -1
    n_lines = len(lines)
    for i in range(n_lines):
        line = lines[i]
        if line.startswith('--'):
            start_i = i
            break
        else:
            init_copy_sqls.append(line.strip())

    test_sqls = []
    n_lines = len(lines)
    prefix_len = len('test_query: ')

    for i in range(start_i + 1, n_lines):
        line = lines[i]
        if line.startswith('--'):
            assert line.find('evaluation-part') >= 0
            start_i = i
            break

    num_sub_queries = 0
    num_single_tbls = 0
    num_sub_queries_list = []
    num_single_tbls_list = []
    original_query_nos = []
    is_first_test_query = True

    for i in range(start_i + 1, n_lines):
        line = lines[i]
        assert not (line.startswith('COPY') or line.startswith('copy'))

        if line.startswith('test_query'):
            terms = line.split('||')
            sql = terms[0][prefix_len:]
            query_no = terms[1]
            test_sqls.append(sql + '||' + query_no)

            if not is_first_test_query:
                num_sub_queries_list.append(num_sub_queries)
                num_single_tbls_list.append(num_single_tbls)
            else:
                is_first_test_query = False

            num_sub_queries = 0
            num_single_tbls = 0
            original_query_nos.append(query_no)
        elif line.startswith('test_sub'):
            num_sub_queries += 1
        elif line.startswith('test_single'):
            num_single_tbls += 1
        elif line.startswith('delete') or line.startswith('update'):
            terms = line.split('##')
            test_sqls.append(terms[0])
        elif line.startswith('insert'):
            test_sqls.append(line.strip())

    num_sub_queries_list.append(num_sub_queries)
    num_single_tbls_list.append(num_single_tbls)

    copy_sqls = []
    for copy_sql in init_copy_sqls:
        new_copy_sql = copy_sql.replace('/init/', '/after_train/')
        copy_sqls.append(new_copy_sql)

    return copy_sqls, test_sqls, num_sub_queries_list, num_single_tbls_list


def prepare_before_running_workload(args):
    if_static_workload = args.test_wl_type == 'static'
    workload_dir = arg_parser_utils.get_workload_dir(args, args.test_wl_type)
    num_sub_queries_list = []
    num_single_tbls_list = []
    if if_static_workload:
        test_queries_path = os.path.join(workload_dir, args.test_queries_fname)
        lines = file_utils.read_all_lines(test_queries_path)
        sqls = []
        for i, line in enumerate(lines):
            assert line.startswith('select')
            sqls.append(f'{line.strip()}||{i}')

        pg_utils.database_init(args, static_workload=True)
        conn = pg_utils.get_db_conn(args.db_name)
        cur = conn.cursor()

    else:
        workload_path = os.path.join(workload_dir, args.workload_fname)
        lines = file_utils.read_all_lines(workload_path)
        copy_sqls, sqls, num_sub_queries_list, num_single_tbls_list = get_test_sqls_and_copy_sqls(lines)

        conn = pg_utils.get_db_conn(args.db_name)
        cur = conn.cursor()

        assert len(copy_sqls) == args.num_tables
        table_cards_map = {}
        for copy_sql in copy_sqls:
            terms = copy_sql.split(' ')
            table_name = terms[1]
            path = terms[3][1:-1]
            assert os.path.exists(path)
            table_lines = file_utils.read_all_lines(path)
            table_cards_map[table_name] = len(table_lines) - 1

        for i, sql in enumerate(copy_sqls):
            print(f'Exec {sql}')
            cur.execute(sql)
            conn.commit()

        # print('x =', x)
        for table_name in table_cards_map:
            count_sql = 'select count(*) from {0:s}'.format(table_name)
            cur.execute(count_sql)
            res = cur.fetchall()
            card = res[0][0]
            # print('table = {0:s}, card = {1:d}'.format(table_name, card))
            assert card == table_cards_map[table_name]

        # print('All csv files are imported.')


    return if_static_workload, conn, cur, sqls, num_sub_queries_list, num_single_tbls_list


def get_single_cards_read_sqls(args, single_est_no=0, model_name=None):
    train_wl_type_pre, test_wl_type_pre, pg_cards_path = arg_parser_utils.get_wl_type_pre_and_pg_cards_paths(args)
    if model_name is None:
        model_name = args.model
    setting_sqls = []
    setting_sqls.append('SET single_read_flag=1;')
    setting_sqls.append('SET read_single_cards=true;')
    if args.test_wl_type != 'static':
        method_cardest_fname = f'{model_name}_single_tbls_{args.data}_{test_wl_type_pre}.txt'
    else:
        method_cardest_fname = f'{model_name}_single_tbls_{args.data}_static.txt'
    method_cardest_path = os.path.join(args.db_data_dir, method_cardest_fname)
    assert os.path.exists(method_cardest_path)
    set_method_cardest_fname = 'SET single_cards_fname=\'{0:s}\';'.format(method_cardest_fname)
    setting_sqls.append(set_method_cardest_fname)
    setting_sqls.append(f'SET single_est_no={single_est_no};')
    return setting_sqls


def get_setting_sqls(args, join_est_no, single_est_no, ignore_single_cards=True, db_task=None):
    train_wl_type_pre, test_wl_type_pre, pg_cards_path = arg_parser_utils.get_wl_type_pre_and_pg_cards_paths(args)
    if db_task is None:
        db_task = args.db_task
    setting_sqls = []
    if (db_task == 'query_exec' or db_task == 'p_error' or db_task == 'hints_gen') and args.model != 'pg':
        setting_sqls.append('SET read_join_cards=true;')
        setting_sqls.append('SET join_read_flag=1;')

        if args.test_wl_type != 'static':
            method_joinest_fname = f'{args.model}_{args.data}_{train_wl_type_pre}_{test_wl_type_pre}.txt'
        else:
            method_joinest_fname = f'{args.model}_{args.data}_static.txt'

        method_joinest_path = os.path.join(args.db_data_dir, method_joinest_fname)
        # print('method_joinest_path =', method_joinest_path)
        assert os.path.exists(method_joinest_path)
        set_method_joinest_fname = 'SET join_cards_fname=\'{0:s}\';'.format(method_joinest_fname)

        setting_sqls.append(set_method_joinest_fname)
        setting_sqls.append(f'SET join_est_no={join_est_no};')

        if ignore_single_cards == False:
            model_name = None
            if db_task == 'p_error':
                model_name = 'optimal'
            single_cards_read_sqls = get_single_cards_read_sqls(args, single_est_no, model_name=model_name)
            setting_sqls.extend(single_cards_read_sqls)
    elif args.db_task == 'pg_card_access':
        setting_sqls.append('SET write_pg_card_estimates=true;')
        setting_sqls.append(f'SET pg_join_cards_fname=\'{pg_cards_path}\';')

    timeout = 1000 * 3600
    set_timeout = f'SET statement_timeout = {timeout};'
    setting_sqls.append(set_timeout)

    return setting_sqls

def judge_if_ignore_single_cards(args):
    ignore_single_cards = (args.ignore_single_cards == 1)
    if args.model == 'optimal':
        ignore_single_cards = False
    elif args.model == 'pg':
        ignore_single_cards = True
    return ignore_single_cards

def run_workload(args):
    if_static_workload, conn, cur, sqls, _, num_single_tbls_list = prepare_before_running_workload(args)
    test_queries = []
    id_raw_id_map = {}
    results = []

    cancel_setting_sqls = [
        'SET read_single_cards=false;',
        'SET single_read_flag=0;'
        # 'SET read_join_cards=false;',
        # 'SET join_read_flag=0;'
    ]

    test_query_count = 0
    sub_query_no = 0
    single_tbl_no = 0

    total_time = 0

    ignore_single_cards = judge_if_ignore_single_cards(args)
    try:
        setting_sqls = get_setting_sqls(args, 0, 0, ignore_single_cards)
        for setting_sql in setting_sqls:
            cur.execute(setting_sql)
            conn.commit()

        if args.db_task == 'query_exec':
            for i, sql in enumerate(sqls):
                if sql.startswith('insert') or sql.startswith('delete') or sql.startswith('update'):
                    try:
                        cur.execute(sql)
                        conn.commit()
                    except:
                        print(f'line-{i+1}, sql = {sql}')
                        raise Exception()

                else:
                    assert sql.startswith('select')

                    terms = sql.split('||')
                    query = terms[0]
                    test_query_no = int(terms[1])

                    print('{0:d}, {1:d}, {2:d}, query = {3:s}'.format(test_query_count, i, test_query_no, sql))
                    if (not ignore_single_cards) and (not if_static_workload):
                        single_cards_read_sqls = get_single_cards_read_sqls(args, single_tbl_no)
                        for setting_sql in single_cards_read_sqls:
                            cur.execute(setting_sql)
                            conn.commit()

                    start_i = time.time()
                    try:
                        cur.execute(query)
                        res = cur.fetchall()
                        card = res[0][0]
                    except:
                        card = -1
                    stop_i = time.time()
                    t_i = stop_i - start_i
                    total_time += t_i
                    results.append((card, t_i, test_query_no))
                    if (not ignore_single_cards) and (not if_static_workload):
                        for cancel_setting_sql in cancel_setting_sqls:
                            cur.execute(cancel_setting_sql)
                            conn.commit()

                    if args.wl_type != 'static':
                        single_tbl_no += num_single_tbls_list[test_query_count]

                    test_query_count += 1

        elif args.db_task == 'pg_card_access':
            id = 0
            for i, sql in enumerate(sqls):
                if sql.startswith('insert') or sql.startswith('delete') or sql.startswith('update'):
                    cur.execute(sql)
                    conn.commit()
                else:
                    assert sql.startswith('select')

                    print('{0:d}, {1:d}, query = {2:s}'.format(id, i, sql))
                    terms = sql.split('||')
                    query = terms[0]
                    raw_id = int(terms[1])
                    cur.execute("SET mainquery_no={0:d}".format(raw_id))
                    cur.execute("EXPLAIN (FORMAT JSON) " + query)
                    res = cur.fetchall()
                    id_raw_id_map[id] = raw_id
                    # cur.execute("SET subquery_no=0")
                    test_queries.append(query.strip())
                    id += 1

    finally:
        cur.close()
        conn.close()
    return total_time, results, test_queries, id_raw_id_map



if __name__ == '__main__':
    args = arg_parser.get_arg_parser()

    logs_dir = '../logs'
    FileViewer.detect_and_create_dir(logs_dir)
    if_static_workload = (args.wl_type == 'static')
    train_wl_type_pre, test_wl_type_pre, pg_cards_fname = arg_parser_utils.get_wl_type_pre_and_pg_cards_paths(args)

    pg_utils.database_init(args, static_workload=if_static_workload)

    if args.db_task == 'pg_card_access':
        pg_cards_path = os.path.join(args.db_data_dir, pg_cards_fname)
        if os.path.exists(pg_cards_path):
            os.remove(pg_cards_path)
        assert os.path.exists(pg_cards_path) == False

        run_workload(args)
        if not os.path.exists(pg_cards_path):
            print(f'pg_cards_path = {pg_cards_path}')
        assert os.path.exists(pg_cards_path) == True
        lines = file_utils.read_all_lines(pg_cards_path)
        pg_cards = [float(s.strip()) for s in lines]
        pg_cards = np.array(pg_cards, dtype=np.float64)
        _, q_error_dir = arg_parser_utils.get_p_q_error_dir(args)
        FileViewer.detect_and_create_dir(q_error_dir)
        pg_cards_npy_path = os.path.join(q_error_dir, 'pg.npy')
        np.save(pg_cards_npy_path, pg_cards)
        # print('pg_cards.shape =', pg_cards.shape)

    else:
        if if_static_workload:
            model_spec = f'{args.model}_{args.data}_static'
            results_path = os.path.join(logs_dir, f'{args.db_task}_{model_spec}.txt')
        else:
            model_spec = f'{args.model}_{args.data}_{train_wl_type_pre}_{test_wl_type_pre}'
            results_path = os.path.join(logs_dir, f'{args.db_task}_{model_spec}.txt')

        t, results, test_queries, _ = run_workload(args)

        if t is not None and results is not None:
            line = f'method = {args.model}, t = {int(t)}'
            print(line)
            lines = [line + '\n']

            lines.extend([f'{s[0]}, {Decimal(s[1]).quantize(Decimal("0.00"))}, {s[2]}\n' for s in results])
            file_utils.write_all_lines(results_path, lines)

# python benchmark/e2e_eval.py --db_task query_exec --data STATS --wl_type ins_heavy --model optimal
