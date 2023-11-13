import os
import sys
sys.path.append("..")
from src.arg_parser import arg_parser
from src.utils import file_utils, FileViewer, pg_utils, arg_parser_utils
import numpy as np
import e2e_eval



def hints_gen_exec(args, costs_path, hints_path):
    if_static_workload = (args.test_wl_type == 'static')
    pg_utils.database_init(args, static_workload=if_static_workload)
    if_static_workload, conn, cur, sqls, _, num_single_tbls_list = e2e_eval.prepare_before_running_workload(args)
    ignore_single_cards = e2e_eval.judge_if_ignore_single_cards(args)
    setting_sqls = e2e_eval.get_setting_sqls(args, 0, 0, db_task='p_error', ignore_single_cards=ignore_single_cards)

    cancel_setting_sqls = [
        'SET read_single_cards=false;',
        'SET single_read_flag=0;'
        # 'SET read_join_cards=false;',
        # 'SET join_read_flag=0;'
    ]

    hint_str_list = []
    costs = []
    single_tbl_no = 0
    test_query_count = 0

    try:
        for setting_sql in setting_sqls:
            # print('setting_sql =', setting_sql)
            cur.execute(setting_sql)
            conn.commit()
        for i, sql in enumerate(sqls):
            if sql.startswith('insert') or sql.startswith('delete') or sql.startswith('update'):
                cur.execute(sql)
                conn.commit()
            else:
                assert sql.startswith('select')

                # print(f'{i}, query = {sql}')
                terms = sql.split('||')
                query = terms[0]
                if (not ignore_single_cards) and (not if_static_workload):
                    single_cards_read_sqls = e2e_eval.get_single_cards_read_sqls(args, single_tbl_no,
                                                                                 model_name='optimal')
                    for setting_sql in single_cards_read_sqls:
                        cur.execute(setting_sql)
                        conn.commit()

                # cur.execute("SET mainquery_no={0:d}".format(raw_id))
                cur.execute("EXPLAIN (FORMAT JSON) " + query)
                res = cur.fetchall()
                plan_obj = res[0][0][0]['Plan']
                hint_str = pg_utils.get_hints(plan_obj)
                hint_str = hint_str.replace('\n', ' ').strip()
                hint_str_list.append(hint_str + '\n')
                cost = float(plan_obj['Total Cost'])
                costs.append(cost)

                if (not ignore_single_cards) and (not if_static_workload):
                    for cancel_setting_sql in cancel_setting_sqls:
                        cur.execute(cancel_setting_sql)
                        conn.commit()

                if args.wl_type != 'static':
                    single_tbl_no += num_single_tbls_list[test_query_count]

                test_query_count += 1
        if args.model == 'optimal':
            costs = np.array(costs, dtype=np.float64)
            np.save(costs_path, costs)
        file_utils.write_all_lines(hints_path, hint_str_list)
    finally:
        cur.close()
        conn.close()


def p_error_exec(args, costs_path, hints_path):
    if_static_workload = (args.test_wl_type == 'static')
    pg_utils.database_init(args, static_workload=if_static_workload)
    if_static_workload, conn, cur, sqls, _, num_single_tbls_list = e2e_eval.prepare_before_running_workload(args)

    args_model = args.model
    args.model = 'optimal'
    ignore_single_cards = e2e_eval.judge_if_ignore_single_cards(args)

    setting_sqls = e2e_eval.get_setting_sqls(args, 0, 0, db_task='p_error', ignore_single_cards=ignore_single_cards)
    cancel_setting_sqls = [
        'SET read_single_cards=false;',
        'SET single_read_flag=0;'
        # 'SET read_join_cards=false;',
        # 'SET join_read_flag=0;'
    ]
    single_tbl_no = 0
    test_query_count = 0


    costs = []
    try:
        for setting_sql in setting_sqls:
            # print('setting_sql =', setting_sql)
            cur.execute(setting_sql)
            conn.commit()

        assert os.path.exists(hints_path)
        hint_str_list = file_utils.read_all_lines(hints_path)

        cur.execute('LOAD \'pg_hint_plan\';')
        query_count = 0
        for i, sql in enumerate(sqls):
            if sql.startswith('insert') or sql.startswith('delete') or sql.startswith('update'):
                cur.execute(sql)
                conn.commit()
            else:
                assert sql.startswith('select')

                # print(f'{query_count}, query = {sql}')
                terms = sql.split('||')
                query = terms[0]
                if (not ignore_single_cards) and (not if_static_workload):
                    single_cards_read_sqls = e2e_eval.get_single_cards_read_sqls(args, single_tbl_no)
                    for setting_sql in single_cards_read_sqls:
                        cur.execute(setting_sql)
                        conn.commit()

                # cur.execute("SET mainquery_no={0:d}".format(raw_id))
                hint_str = hint_str_list[query_count]
                query_count += 1
                new_sql = hint_str.strip() + ' ' + "EXPLAIN (FORMAT JSON) " + query
                cur.execute(new_sql)
                res = cur.fetchall()
                plan_obj = res[0][0][0]['Plan']
                cost = float(plan_obj['Total Cost'])
                costs.append(cost)
                # hint_str = pg_utils.get_hints(plan_obj)
                # hint_str = hint_str.replace('\n', ' ')

                if (not ignore_single_cards) and (not if_static_workload):
                    for cancel_setting_sql in cancel_setting_sqls:
                        cur.execute(cancel_setting_sql)
                        conn.commit()

                if args.wl_type != 'static':
                    single_tbl_no += num_single_tbls_list[test_query_count]

                test_query_count += 1

        if args_model != 'optimal':
            costs = np.array(costs, dtype=np.float64)
            np.save(costs_path, costs)

    finally:
        cur.close()
        conn.close()
    args.model = args_model



def calc_p_error(args):
    p_error_dir, _ = arg_parser_utils.get_p_q_error_dir(args)
    hints_dir = os.path.join(p_error_dir, args.hints_dirname)
    FileViewer.detect_and_create_dir(hints_dir)

    costs_fname = f'{args.model}.npy'
    hints_fname = f'{args.model}.txt'

    costs_path = os.path.join(p_error_dir, costs_fname)
    hints_path = os.path.join(hints_dir, hints_fname)

    require_hints_gen = True
    if args.model == 'optimal':
        if os.path.exists(costs_path):
            optimal_costs = np.load(costs_path)
            require_hints_gen = False
    else:
        # assert (os.path.exists(hints_path))
        if os.path.exists(hints_path):
            hints_str_list = file_utils.read_all_lines(hints_path)
            require_hints_gen = False

    if require_hints_gen:
        hints_gen_exec(args, costs_path, hints_path)
    p_error_exec(args, costs_path, hints_path)



if __name__ == '__main__':
    args = arg_parser.get_arg_parser()

    calc_p_error(args)
    # calc_p_error(args, db_task='p_error')

# python benchmark/p_error_cmp.py --data STATS --wl_type ins_heavy --model attn --ignore_single_cards 0
