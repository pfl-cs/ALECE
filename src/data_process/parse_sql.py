# select A.a1,..., A.an, B.b1,...,B.bm,...,X.x1,...,X.xt from A join B on A.ax = B.bx,..., where A.a1 in [],...,
# join: one-hot encoding. We organize all pairs of PK-FK attributes from A, B,..,X in triangular order.

import numpy as np
import os
import time
# import pandas as pd
import sys
sys.path.append("../..")
from src.utils import FileViewer, string_utils, sql_utils, file_utils, data_utils, arg_parser_utils
# from utils import FileViewer, string_utils, sql_utils
import random
import shutil
import copy

epsilon_for_float = 1e-6

def isnumber(s):
    return s.lstrip('-').replace('.', '', 1).isdigit()

def get_table_no_and_attr_no(table_attr, table_no_map, attr_no_map_list):
    table_name, attr_name = sql_utils.table_and_attr(table_attr)
    table_no = table_no_map[table_name]
    attr_no_map = attr_no_map_list[table_no]
    assert attr_name in attr_no_map
    attr_no = attr_no_map[attr_name]
    return table_no, attr_no


def parse_predicate_lhs(lhs, table_no_map, attr_no_map_list):
    lhs = lhs.lower()
    idx = lhs.find("::timestamp")
    template = "2000-01-01 00:00:00"
    if idx >= 0:
        assert idx >= len(template) + 2
        lhs = lhs[0:idx - 1].strip()
        lhs = lhs[1:]
        timestamp = data_utils.time_to_int(lhs)
        return (-1, timestamp)
    terms = lhs.split(".")
    if len(terms) == 2:
        table_name = terms[0]
        attr = terms[1]
        if table_name in table_no_map:
            table_no = table_no_map[table_name]
            attr_no_map = attr_no_map_list[table_no]
            assert attr in attr_no_map
            attr_no = attr_no_map[attr]
            return (table_no, attr_no)
        else:
            assert isnumber(lhs)
            return (-1, float(lhs))
    else:
        if not isnumber(lhs):
            print('lhs =', lhs)
            raise Exception()
        assert isnumber(lhs)
        return (-1, float(lhs))



def parse_sql(sql_and_other_info,
              table_no_map,
              attr_no_map_list,
              attr_no_types_list,
              attr_ranges_list,
              min_card_threshold,
              delim):

    """
    :param sql_and_other_info: string, looks like select count * from A, B,.., X where A.a1 = B.b2 and A.a2 = X.x1 and A.a3 >=1 and A.a3 < 10 and B.b2 = 1 and X.x1 <= 100
    :param table_no_map: table_name-table_id dict
    :param attr_no_map_list: list of dicts, each element is a attr_name-attr_no dict
    :param attr_no_type_map_list: list of dicts, each element is a attr_no-data_type dict
    :param attr_ranges_list: list of numpy ndarray. Each element r is a M x 2 matrix. r[i,0] and r[i,1] denotes the lower and upper bound of the ith attr
    :param min_card_threshold:
    :return:
    """
    sql_and_other_info = sql_and_other_info.lower().strip().replace(';;', ';')
    idx = sql_and_other_info.find('select')
    assert (idx >= 0)
    sql_and_other_info = sql_and_other_info[idx:]
    # terms = sql_and_other_info.split(" from ")
    # sql_and_other_info = terms[1]
    terms = sql_and_other_info.split(delim) # sql || join_type || selection_type || sql_no (used in postgres) || true card || cartesian card

    join_type = None
    selection_type = None
    sql_no = -1
    true_card = -1
    cartesian_join_card = -1
    natural_join_card = -1

    if len(terms) >= 5:
        join_type = terms[1]
        true_card = int(terms[4])
        if len(terms) > 5:
            cartesian_join_card = int(terms[5])
        if len(terms) > 6:
            natural_join_card = int(terms[6])
    elif len(terms) == 3:
        true_card = int(terms[2])
    elif len(terms) == 2:
        true_card = int(terms[1])

    if true_card >= 0:
        if true_card < min_card_threshold:
            return None, None, None, None, None, None, None, None

    short_full_table_name_map, join_predicates, filter_predicates = sql_utils.simple_parse(terms[0])
    relevant_tables = []
    for short_name in short_full_table_name_map:
        full_name = short_full_table_name_map[short_name]
        table_no = table_no_map[full_name]
        table_no_map[short_name] = table_no
        relevant_tables.append(table_no)
    relevant_tables.sort()

    equi_classes = sql_utils.get_equi_classes(join_predicates)


    join_strs = []
    for equi_class in equi_classes.subsets():
        table_attr_list = list(equi_class)
        # table_attr_list.sort()
        for i, l_table_attr in enumerate(table_attr_list):
            for j, r_table_attr in enumerate(table_attr_list):
                if i != j:
                    join_strs.append(l_table_attr + ' = ' + r_table_attr)

    join_strs.sort()
    join_conds = []
    for join_str in join_strs:
        terms = join_str.split(' = ')
        l_table_attr = terms[0].strip()
        r_table_attr = terms[1].strip()
        l_table_no, l_attr_no = get_table_no_and_attr_no(l_table_attr, table_no_map, attr_no_map_list)
        r_table_no, r_attr_no = get_table_no_and_attr_no(r_table_attr, table_no_map, attr_no_map_list)
        join_conds.append([l_table_no, l_attr_no, r_table_no, r_attr_no])

    if len(join_conds) == 0:
        join_conds = None
    else:
        join_conds = np.array(join_conds, dtype=np.int64)

    attr_range_conds = [x.copy() for x in attr_ranges_list]
    filter_conds = []
    for filter_predicate in filter_predicates:
        (lhs, op, rhs) = filter_predicate
        (lhs_table_no, lhs_attr_no) = parse_predicate_lhs(lhs, table_no_map, attr_no_map_list)
        rhs_res = None
        try:
            rhs_res = parse_predicate_lhs(rhs, table_no_map, attr_no_map_list)
        except:
            print('filter_predicate =', filter_predicate)
            print('sql =', sql_and_other_info)
            raise Exception()


        assert (rhs_res[0] < 0)
        # Note I skip border check here. I will add it in the future
        attr_type = attr_no_types_list[lhs_table_no][lhs_attr_no]  # 0 for int, 1 for float
        rhs_val = rhs_res[1]
        single_attr_range_cond = attr_range_conds[lhs_table_no]
        if attr_type == 0:
            epsilon = 0.5
        else:
            epsilon = epsilon_for_float
        if op == '=':
            single_attr_range_cond[lhs_attr_no][0] = rhs_val - epsilon
            single_attr_range_cond[lhs_attr_no][1] = rhs_val + epsilon
        elif op == '<=':
            single_attr_range_cond[lhs_attr_no][1] = rhs_val + epsilon
        elif op == '<':
            single_attr_range_cond[lhs_attr_no][1] = rhs_val - epsilon
        elif op == '>=':
            single_attr_range_cond[lhs_attr_no][0] = rhs_val - epsilon
        else:  # '>'
            single_attr_range_cond[lhs_attr_no][0] = rhs_val + epsilon

    for table_no in relevant_tables:
        filter_conds.append(attr_range_conds[table_no])


    assert len(relevant_tables) > 0
    for x in filter_conds:
        assert len(x.shape) == 2
        assert x.shape[0] > 0
    filter_conds = np.concatenate(filter_conds, axis=0)

    attr_range_conds = np.concatenate(attr_range_conds, axis=0, dtype=np.float64)
    attr_range_conds = np.reshape(attr_range_conds, [-1])
    assert join_conds is None or (join_conds.shape[0] > 0)

    return join_conds, equi_classes, attr_range_conds, true_card, cartesian_join_card, natural_join_card, join_type, relevant_tables, filter_conds



def get_table_info(table_path, attr_type_list):
    attr_no_map = {}
    attr_no_types = []
    lines = file_utils.read_all_lines(table_path)

    attr_names_str = lines[0].strip()
    attr_names = attr_names_str.split(",")
    for i, attr_name in enumerate(attr_names):
        attr_no_map[attr_name.lower()] = i

    assert attr_type_list is not None
    assert len(attr_type_list) == len(attr_names)

    epsilons = []
    for attr_type in attr_type_list:
        assert attr_type == 0 or attr_type == 1
        if attr_type == 0:
            epsilons.append(0.5)
        else:
            epsilons.append(epsilon_for_float)
        attr_no_types.append(0)  # all attrs' types are int

    attr_no_types = np.array(attr_type_list, dtype=np.int64)

    lines = lines[1:]
    card = len(lines)

    data = []
    for i in range(len(attr_names)):
        data.append([])
    for line in lines:
        items = line.strip().split(",")
        try:
            for i, x in enumerate(items):
                if len(x) > 0:
                    data[i].append(float(x))
        except:
            print('-' * 20 + line)
            for i, x in enumerate(items):
                print('i =', i, 'x =', x, 'len(x) =', len(x), )
                if len(x) > 0:
                    data[i].append(float(x))
            raise

    table_attr_ranges = []
    data_0 = np.array(data[0], dtype=np.float64)
    data_1 = np.array(data[1], dtype=np.float64)
    diff = np.abs(data_1 - data_0)
    print('table =', os.path.basename(table_path)[0:-4], ', diff.max =', np.max(diff), 'diff.shape =', diff.shape, '-' * 10)
    for i, data_i in enumerate(data):
        data_i_np = np.array(data_i, dtype=np.float64)
        minv, maxv = np.min(data_i_np) - epsilons[i], np.max(data_i_np) + epsilons[i]
        # tmp1 = np.min(data_i_np)
        # tmp2 = np.max(data_i_np)
        # print('table =', os.path.basename(table_path)[0:-4], ', attr =', attr_names[i], ', minv =', minv, ', maxv =',
        #       maxv)
        # print('tmp1 =', tmp1, 'tmp2 =', tmp2)
        table_attr_ranges.append([minv, maxv])
    table_attr_ranges = np.array(table_attr_ranges, dtype=np.float64)

    return attr_no_map, attr_no_types, table_attr_ranges, card


# skip check
def parse_str_int_int_map(line, item_sep="|", key_value_sep=","):
    terms = line.strip().split(item_sep)
    m1 = {}
    m2 = {}
    for term in terms:
        items = term.strip().split(key_value_sep)
        no = int(items[1])
        m1[items[0].strip()] = no
        m2[no] = int(items[2])
    return m1, m2

# skip check
def parse_str_int_map(line, item_sep="|", key_value_sep=","):
    terms = line.strip().split(item_sep)
    m = {}
    for term in terms:
        items = term.strip().split(key_value_sep)
        m[items[0].strip()] = int(items[1])
    return m


def load_tables_info_from_file(table_names, tables_info_path):
    attr_no_map_list = []
    attr_no_types_list = []
    attr_ranges_list = []
    lines = file_utils.read_all_lines(tables_info_path)

    table_no_map, table_card_map = parse_str_int_int_map(lines[0])
    n_tables = len(table_no_map)
    table_card_list = []
    for i in range(n_tables):
        table_card_list.append(table_card_map[i])

    lines = lines[3:]
    for i in range(n_tables):
        line = lines[i]
        terms = line.split(":")
        line = terms[1]
        attr_no_map = parse_str_int_map(line)
        attr_no_map_list.append(attr_no_map)

    lines = lines[n_tables + 1:]
    for i in range(n_tables):
        line = lines[i]
        terms = line.split(":")
        line = terms[1].strip()
        terms = line.split(",")
        attr_no_types = [int(x) for x in terms]
        attr_no_types = np.array(attr_no_types, dtype=np.int64)
        attr_no_types_list.append(attr_no_types)

    lines = lines[n_tables + 1:]
    for i in range(n_tables):
        line = lines[i]
        terms = line.split(":")
        line = terms[1].strip()
        terms = line.split(",")
        attr_ranges = [float(x) for x in terms]
        attr_ranges = np.array(attr_ranges, dtype=np.float64)
        attr_ranges = np.reshape(attr_ranges, [-1, 2])
        attr_ranges_list.append(attr_ranges)

    no_table_map = {}
    for table_name in table_names:
        no = table_no_map[table_name.lower()]
        no_table_map[no] = table_name
    return (table_no_map, no_table_map, table_card_list, attr_no_map_list, attr_no_types_list, attr_ranges_list)



def get_tables_info(args):
    static_workload_dir = arg_parser_utils.get_workload_dir(args, wl_type='static')
    create_tables_path = os.path.join(static_workload_dir, 'create_tables.sql')
    _, table_attr_types_map = sql_utils.get_all_table_attr_infos(create_tables_path)

    table_names = table_attr_types_map.keys()
    table_names = list(table_names)
    table_names.sort()
    table_card_list = []

    tables_info_path = os.path.join(args.data_dir, args.tables_info_fname)

    attr_no_map_list = []
    attr_no_types_list = []
    attr_ranges_list = []
    if os.path.exists(tables_info_path) == False:
        table_no_map = {}
        no_keep_cap_letter_table_name_map = {}
        for i, table_name in enumerate(table_names):
            table_no_map[table_name.lower()] = i
            no_keep_cap_letter_table_name_map[i] = table_name

        for table_name in table_names:
            attr_type_list = table_attr_types_map[table_name]
            table_file_path = os.path.join(args.data_dir, table_name + ".csv")
            attr_no_map, attr_no_types, table_attr_ranges, table_card = get_table_info(table_file_path, attr_type_list)
            attr_no_map_list.append(attr_no_map)
            attr_no_types_list.append(attr_no_types)
            attr_ranges_list.append(table_attr_ranges)
            table_card_list.append(table_card)

        with open(tables_info_path, "w") as writer:
            lines = []
            terms = []
            terms2 = []

            # table_no_map
            for table_name, table_no in table_no_map.items():
                terms.append(table_name + "," + str(table_no) + "," + str(table_card_list[table_no]))
                terms2.append(
                    table_name + "," + str(table_no) + "," + no_keep_cap_letter_table_name_map[table_no] + "," + str(
                        table_card_list[table_no]))
            lines.append('|'.join(terms) + "\n")
            lines.append('|'.join(terms2) + "\n")
            lines.append("\n")

            # attr_no_map_list
            for i, attr_no_map in enumerate(attr_no_map_list):
                terms = []

                for attr_name, attr_no in attr_no_map.items():
                    terms.append(attr_name + "," + str(attr_no))
                line = table_names[i] + ' attr nos: ' + '|'.join(terms) + "\n"
                lines.append(line)
            lines.append("\n")

            # attr_no_type_map_list
            for i, attr_no_types in enumerate(attr_no_types_list):
                terms = attr_no_types.tolist()
                terms = [str(x) for x in terms]

                line = table_names[i] + ' attr types: ' + ','.join(terms) + "\n"

                lines.append(line)
            lines.append("\n")

            # table_attr_ranges_list
            for i, attr_ranges in enumerate(attr_ranges_list):
                print("+++++i =", i, attr_ranges)
                tmp = np.reshape(attr_ranges, [-1])
                terms = tmp.tolist()
                terms = [str(x) for x in terms]
                line = table_names[i] + ' attr ranges: ' + ','.join(terms) + "\n"
                lines.append(line)

            writer.writelines(lines)

    return load_tables_info_from_file(table_names, tables_info_path)


def parse_queries(lines, table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list, min_card_threshold, delim, baseline_results_delim):
    join_conds_list = []
    equi_classes_list = []
    attr_range_conds_list = []
    true_card_list = []
    cartesian_join_card_list = []
    natural_join_card_list = []

    join_type_list = []
    relevant_tables_list = []
    filter_conds_list = []

    baseline_results = []
    for line in lines:
        first_part = line
        if baseline_results_delim is not None:
            two_parts = line.strip().split(baseline_results_delim)
            if len(two_parts) == 2:
                first_part = two_parts[0]
                second_part = two_parts[1]
                terms = second_part.split(delim)
                card_preds = [float(x) for x in terms]
                baseline_results.append(card_preds)

        join_conds, equi_classes, attr_range_conds, true_card, cartesian_join_card, natural_join_card \
            , join_type, relevant_tables, filter_conds = parse_sql(
            first_part,
            table_no_map,
            attr_no_map_list,
            attr_no_types_list,
            attr_ranges_list,
            min_card_threshold,
            delim
        )
        # join_equi_classes = sql_utils.get_equi_classes(join_conds)

        if true_card is None:
            continue
        equi_classes_list.append(equi_classes)
        join_conds_list.append(join_conds)
        attr_range_conds_list.append(attr_range_conds)
        true_card_list.append(true_card)
        cartesian_join_card_list.append(cartesian_join_card)
        natural_join_card_list.append(natural_join_card)
        # print("attr_range_conds.shape =", attr_range_conds.shape)
        join_type_list.append(join_type)
        relevant_tables_list.append(relevant_tables)
        filter_conds_list.append(filter_conds)

    possible_join_strs = []
    for equi_classes in equi_classes_list:
        for equi_class in equi_classes.subsets():
            table_attr_list = list(equi_class)
            table_attr_list.sort()
            for i, l_table_attr in enumerate(table_attr_list):
                for j in range(i + 1, len(table_attr_list)):
                    r_table_attr = table_attr_list[j]
                    possible_join_strs.append(l_table_attr + ' = ' + r_table_attr)

    possible_join_strs = set(possible_join_strs)
    possible_join_strs = list(possible_join_strs)
    possible_join_strs.sort()

    possible_join_attrs = []
    for join_str in possible_join_strs:
        terms = join_str.split(' = ')
        l_table_attr = terms[0].strip()
        r_table_attr = terms[1].strip()
        l_table_no, l_attr_no = get_table_no_and_attr_no(l_table_attr, table_no_map, attr_no_map_list)
        r_table_no, r_attr_no = get_table_no_and_attr_no(r_table_attr, table_no_map, attr_no_map_list)

        possible_join_attrs.append([l_table_no, l_attr_no, r_table_no, r_attr_no])

    # possible_join_attrs = np.concatenate(join_conds_list, axis=0)
    possible_join_attrs = np.array(possible_join_attrs, dtype=np.int64)
    # print("possible_join_conds.shape =", possible_join_attrs.shape)
    if len(baseline_results) == 0:
        baseline_results = None
    else:
        baseline_results = np.array(baseline_results, dtype=np.float64)

    return (possible_join_attrs, join_conds_list, attr_range_conds_list, true_card_list, cartesian_join_card_list,
            natural_join_card_list, join_type_list, relevant_tables_list, filter_conds_list, baseline_results)

def parse_queries_from_file(query_path, table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list, min_card_threshold, delim, baseline_results_delim):
    lines = file_utils.read_all_lines(query_path)

    return parse_queries(lines, table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list,
                         min_card_threshold, delim, baseline_results_delim)


def predicate_attrs_encode(predicate_attrs_sorted):
    str_list = []
    for x in predicate_attrs_sorted:
        l = [str(s) for s in x]
        str_list.append(",".join(l))
    return "/".join(str_list)



def access_database_info(args):
    table_no_map, no_table_map, table_card_list, attr_no_map_list \
        , attr_no_types_list, attr_ranges_list = get_tables_info(args)
    tables_info = (table_no_map, no_table_map, table_card_list, attr_no_map_list, attr_no_types_list, attr_ranges_list)

    query_path = os.path.join(args.data_dir, args.base_query_file)

    queries_info = parse_queries_from_file(
        query_path,
        table_no_map,
        attr_no_map_list,
        attr_no_types_list,
        attr_ranges_list,
        min_card_threshold=10,
        delim="||",
        baseline_results_delim="|*|"
    )

    return (tables_info, queries_info)
