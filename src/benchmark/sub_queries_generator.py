import psycopg2
import time
import os
import sys
import shutil
sys.path.append("../")
from src.utils import pg_utils, sql_utils, arg_parser_utils
from src.arg_parser import arg_parser


def generate_sub_queries_with_pg(args):
    db_subqueries_path = os.path.join(args.db_data_dir, args.db_subqueries_fname)
    db_single_tbls_path = os.path.join(args.db_data_dir, args.db_single_tbls_fname)
    if os.path.exists(db_subqueries_path):
        os.remove(db_subqueries_path)

    static_workload_dir, query_path, sub_queries_path, single_tbls_path = \
        arg_parser_utils.get_query_paths_in_static_wl_dir(args)

    # db_name = 'stats'
    db_name = args.db_name
    conn = pg_utils.get_db_conn(db_name)
    cur = conn.cursor()

    queries = sql_utils.load_queries(query_path)


    init_setting_sqls = ['SET print_sub_queries=true;',
                         'SET single_est_no=0;',
                         'SET join_est_no=0;'
                         ]

    start = time.time()
    try:
        for init_setting_sql in init_setting_sqls:
            print('init_setting_sql =', init_setting_sql)
            cur.execute(init_setting_sql)
            conn.commit()

        for id, query in enumerate(queries):
            assert query.startswith('select')
            print('{0:d}, query = {1:s}'.format(id, query))
            cur.execute("SET mainquery_no={0:d}".format(id))
            cur.execute("EXPLAIN (FORMAT JSON) " + query)
            res = cur.fetchall()

    except:
        cur.close()
        conn.close()
        raise Exception()
    stop = time.time()

    assert (os.path.exists(db_subqueries_path))
    shutil.move(db_subqueries_path, sub_queries_path)
    assert (os.path.exists(db_single_tbls_path))
    shutil.move(db_single_tbls_path, single_tbls_path)
    print('time =', stop - start)

if __name__ == '__main__':
    args = arg_parser.get_arg_parser()
    pg_utils.database_init(args, static_workload=True)
    generate_sub_queries_with_pg(args)

# python benchmark/sub_queries_generator.py --data STATS --wl_data_type train
# python benchmark/sub_queries_generator.py --data STATS --wl_data_type test
