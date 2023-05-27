import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import os
sys.path.append("../")
from . import file_utils, FileViewer, arg_parser_utils
# import file_utils, FileViewer

PG_USER = '$PG_USER$'

def db_conn_str(db_name, port=4321):
    return f'dbname={db_name} user={PG_USER} host=localhost port={port}'

def get_db_conn(db_name, port=4321):
    conn_str = db_conn_str(db_name, port)
    conn = psycopg2.connect(conn_str)
    conn.set_client_encoding('UTF8')
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn


def drop_dbs(db_name_prefix):
    port = 4321
    conn = get_db_conn("postgres", port)
    cur = conn.cursor()
    cur.execute("SELECT datname FROM pg_database;")
    list_dbs = cur.fetchall()
    list_dbs = [db[0] for db in list_dbs]
    # print(list_dbs)
    for db_name in list_dbs:
        if db_name.startswith(db_name_prefix):
            sql = 'drop database {0:s}'.format(db_name)
            cur.execute(sql)
            conn.commit()
    cur.close()
    conn.close()

def drop_db(db_name):
    assert (db_name != 'postgres')
    port = 4321
    conn = get_db_conn("postgres", port)
    cur = conn.cursor()
    cur.execute("SELECT datname FROM pg_database;")
    list_dbs = cur.fetchall()
    list_dbs = [db[0] for db in list_dbs]
    # print(list_dbs)
    for _db_name in list_dbs:
        if _db_name == db_name:
            sql = 'drop database {0:s}'.format(db_name)
            cur.execute(sql)
            conn.commit()
    cur.close()
    conn.close()

def detect_db_exists(db_name):
    assert (db_name != 'postgres')
    port = 4321
    conn = get_db_conn("postgres", port)
    cur = conn.cursor()
    cur.execute("SELECT datname FROM pg_database;")
    list_dbs = cur.fetchall()
    list_dbs = [db[0] for db in list_dbs]
    # print(list_dbs)
    exists = False
    for _db_name in list_dbs:
        if _db_name == db_name:
            exists = True
            break
    cur.close()
    conn.close()
    return exists

def db_create_multi_thread(n_machines, n_threads, machine_id, thread_id, db_name_prefix):
    db_name = '{0:s}{1:d}_{2:d}_{3:d}_{4:d}'.format(db_name_prefix, n_machines, n_threads, machine_id, thread_id)
    return db_create(db_name)

def db_create(db_name):
    assert (db_name != 'postgres')
    port = 4321
    conn = get_db_conn("postgres", port)
    cur = conn.cursor()
    cur.execute("SELECT datname FROM pg_database;")
    list_dbs = cur.fetchall()
    list_dbs = [db[0] for db in list_dbs]
    # print(list_dbs)
    if db_name in list_dbs:
        sql = 'drop database {0:s}'.format(db_name)
        cur.execute(sql)
        conn.commit()
    sql = 'create database {0:s}'.format(db_name)
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
    return db_name


def create_tables(db_name, create_tables_path):
    port = 4321
    conn = get_db_conn(db_name, port)
    cur = conn.cursor()
    sqls = file_utils.read_all_lines(create_tables_path)

    for sql in sqls:
        if len(sql) <= 1:
            continue
        elif sql.startswith('--'):
            continue

        # print('create_sql =', sql)
        cur.execute(sql.strip())
        conn.commit()
    cur.close()
    conn.close()


def database_init(args, static_workload=False):
    static_workload_dir = arg_parser_utils.get_workload_dir(args, wl_type='static')
    create_tables_path = os.path.join(static_workload_dir, 'create_tables.sql')

    not_init = False
    if static_workload:
        not_init = detect_db_exists(args.db_name)

    print('not_init =', not_init, 'static_workload =', static_workload)
    if not not_init:
        drop_db(args.db_name)
        db_name = db_create(args.db_name)
        create_tables(db_name, create_tables_path)

        if static_workload:
            data_dir = os.path.join(args.absolute_base_dir, args.data_dirname)
            table_paths = FileViewer.list_files(data_dir, suffix='csv', isdepth=False)
            copy_sqls = []
            for path in table_paths:
                table_fname = os.path.basename(path)
                table_name = table_fname[0:-4]
                sql = 'COPY {0:s} FROM \'{1:s}\' CSV header;'.format(table_name, path)
                copy_sqls.append(sql)

            conn = get_db_conn(args.db_name)
            cur = conn.cursor()

            # conn_str = pg_utils.db_conn_str(db_name, 4321)
            # conn = psycopg2.connect(conn_str)
            # conn.set_client_encoding('UTF8')
            # conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            for i, sql in enumerate(copy_sqls):
                print('sql =', sql)
                cur.execute(sql)
                conn.commit()
            cur.close()
            conn.close()


def plan_to_pg_hint(plan_obj, scan_hint_list, join_hint_list):
    FEATURE_LIST = ['Node Type', 'Startup Cost',
                    'Total Cost', 'Plan Rows', 'Plan Width']
    LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

    UNKNOWN_OP_TYPE = "Unknown"
    SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
    JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
    OTHER_TYPES = ['Bitmap Index Scan']
    OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
               + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES

    node_type = plan_obj['Node Type']
    tables = None
    if "Plans" in plan_obj:
        children = plan_obj["Plans"]

        left_tables = []
        right_tables = []
        if len(children) == 2:
            left_tables = plan_to_pg_hint(children[0], scan_hint_list, join_hint_list)
            right_tables = plan_to_pg_hint(children[1], scan_hint_list, join_hint_list)
            tables = (left_tables, right_tables)
        else:
            assert len(children) == 1
            left_tables = plan_to_pg_hint(children[0], scan_hint_list, join_hint_list)
            tables = left_tables

        if node_type in JOIN_TYPES:
            join_hint_list.append(node_type.replace(" ", "").replace("Nested", "Nest") + "(" + \
                                  str(tables).replace("'", "").replace(",", " ").replace("(", "").replace(")",
                                                                                                          "") + ")")

        if node_type == "Bitmap Heap Scan":
            assert 'Alias' in plan_obj
            assert len(left_tables) == 0 and len(right_tables) == 0
            assert len(children) == 1
            bitmap_idx_scan = children[0]
            index_name = None
            if "Index Name" in bitmap_idx_scan:
                index_name = bitmap_idx_scan["Index Name"]

            table_name = plan_obj['Alias']
            tables = table_name
            scan_hint_list.append("BitmapScan(" + table_name + ("" if index_name is None else " " + index_name) + ")")


    else:
        if node_type == "Bitmap Index Scan":
            return []

        assert node_type in SCAN_TYPES
        assert 'Alias' in plan_obj
        index_name = None
        if "Index Name" in plan_obj:
            index_name = plan_obj["Index Name"]
        table_name = plan_obj['Alias']
        tables = table_name
        scan_hint_list.append(
            node_type.replace(" ", "") + "(" + table_name + ("" if index_name is None else " " + index_name) + ")")

    return tables


def get_hints(plan_obj):
    scan_hint_list, join_hint_list = [], []
    tables = plan_to_pg_hint(plan_obj, scan_hint_list, join_hint_list)
    hint_str = "/*+\n"
    hint_str += "\n".join(scan_hint_list) + "\n"
    hint_str += "\n".join(join_hint_list) + "\n"
    hint_str += "Leading(" + str(tables).replace("'", "").replace(",", " ") + ")\n"
    hint_str += "*/\n"
    return hint_str
