import sys
import time
import re
from scipy.cluster.hierarchy import DisjointSet
from . import file_utils, data_utils
# import file_utils, data_utils

def time_to_int_in_sql(time_str):#, with_apostrophe=True):
    idx = time_str.find("::timestamp")
    if idx >= 0:
        template = "2000-01-01 00:00:00"
        assert idx >= len(template) + 2
        s = time_str[0:idx - 1].strip()
        # if with_apostrophe:
        #     s = s[1:]
        s = s[1:]

        timestamp = data_utils.time_to_int(s)
        return timestamp
    else:
        return None

def time_to_int_in_data(time_str):#, with_apostrophe=True):
    idx = time_str.find(":")
    if idx >= 0:
        s = time_str.strip()
        timestamp = data_utils.time_to_int(s)
        return timestamp
    else:
        return None

def remove_extra_blanks(sql):
    sql = sql.replace('\t', ' ').lower()
    while True:
        l = len(sql)
        sql = sql.replace('  ', ' ')
        if l == len(sql):
            break
    s = 'select count(*) from'
    # sql[0:len(s)] = s
    sql = s + sql[len(s):]
    sql = sql.replace(' :: timestamp', '::timestamp')
    return sql

def convert_time_to_int_in_predicate(rhs):
    rhs = rhs.lower()
    idx = rhs.find("::timestamp")
    template = "2000-01-01 00:00:00"
    if idx >= 0:
        assert idx >= len(template) + 2
        rhs = rhs[0:idx - 1].strip()
        rhs = rhs[1:]
        timestamp = data_utils.time_to_int(rhs)
        return (1, timestamp)
    else:
        return (0, rhs)

def timestamp_to_int_in_stats_sql(sql):
    sql = remove_extra_blanks(sql)
    short_full_table_name_map, join_conds, filter_conds = simple_parse(sql)
    new_fileter_conds = []
    for filter_cond in filter_conds:
        (lhs, op, rhs) = filter_cond
        # print(lhs, op, rhs)
        (flag, new_rhs) = convert_time_to_int_in_predicate(rhs)
        new_fileter_conds.append((lhs, op, str(new_rhs)))

    new_sql = merge_elements_into_query(short_full_table_name_map, join_conds, new_fileter_conds)
    return new_sql

def is_join_predicate(rhs, short_full_table_name_map):
    for short_name in short_full_table_name_map.keys():
        s = short_name + "."
        if rhs.startswith(s):
            return True
    return False

def predicates_parse(predicate):
    delim_pattern = '(?:(?:<|>)?=)|<|>'
    patt = re.compile(delim_pattern)
    items = re.split(delim_pattern, predicate)
    search_obj = patt.search(predicate)
    if search_obj is None:
        return None
    op = search_obj.group()

    lhs = items[0].strip()
    rhs = items[1].strip()
    return (lhs, op, rhs)

def simple_parse(_query):
    query = _query.lower().strip()
    try:
        terms = query.split(' from ')
        sql = terms[1].strip()
        sql = sql[0:-1]
    except:
        print('query =', query)
        raise Exception()

    sql_parts = sql.split(" where ")

    short_names = sql_parts[0].strip()
    terms = short_names.split(",")
    short_full_table_name_map = {}
    for term in terms:
        items = term.strip().split(" as ")

        if len(items) != 2:
            items = term.strip().split(" ")

        # assert (len(items) == 2)
        full_name = items[0].strip()
        short_name = items[1].strip()
        short_full_table_name_map[short_name] = full_name

    if len(sql_parts) < 2:
        return short_full_table_name_map, [], []

    predicates_str = sql_parts[1].strip()
    delim_pattern = '(?:(?:<|>)?=)|<|>'
    patt = re.compile(delim_pattern)

    predicates = predicates_str.split(" and ")

    join_conds = []
    filter_conds = []
    for predicate in predicates:
        items = re.split(delim_pattern, predicate)
        # if len(items) != 2:
        #     print(len(items), items)
        assert (len(items) == 2)
        search_obj = patt.search(predicate)
        op = search_obj.group()

        lhs = items[0].strip()
        rhs = items[1].strip()

        if is_join_predicate(rhs, short_full_table_name_map):
            join_conds.append((lhs, op, rhs))
        else:
            filter_conds.append((lhs, op, rhs))

    return short_full_table_name_map, join_conds, filter_conds

def merge_elements_into_query(short_full_table_name_map, join_conds, filter_conds):
    query = 'select count(*) from '

    sub_clauses = []
    for short_name in short_full_table_name_map:
        full_name = short_full_table_name_map[short_name]
        sub_clauses.append(full_name + ' as ' + short_name)
    sub_clauses.sort()
    query += ', '.join(sub_clauses)
    if len(join_conds) <= 0 and len(filter_conds) <= 0:
        return query + ';'
    # assert (len(join_conds) > 0 or len(filter_conds) > 0)
    query += ' where '

    if len(join_conds) > 0:
        sub_clauses = []
        equi_join_conds = []
        for join_cond in join_conds:
            (lhs, op, rhs) = join_cond
            if op == '=':
                equi_join_conds.append(join_cond)
            else:
                sub_clauses.append(lhs + ' ' + op + ' ' + rhs)
        equi_classes = get_equi_classes(equi_join_conds)
        sub_clauses.extend(equi_classes_to_join_predicates(equi_classes))

        sub_clauses.sort()
        query += ' and '.join(sub_clauses)
        if len(filter_conds) > 0:
            query += ' and '

    if len(filter_conds) > 0:
        sub_clauses = []
        for filter_cond in filter_conds:
            (lhs, op, rhs) = filter_cond
            sub_clauses.append(lhs + ' ' + op + ' ' + rhs)
        sub_clauses.sort()
        query += ' and '.join(sub_clauses)
    query += ';'

    return query

def get_join_query(query):
    short_full_table_name_map, join_conds, filter_conds = simple_parse(query)
    if join_conds is None:
        return None
    else:
        return merge_elements_into_query(short_full_table_name_map, join_conds, [])

def get_attr_ranges(table_attr, short_full_table_name_map, tables_info):
    terms = table_attr.strip().split('.')
    table = terms[0]
    attr = terms[1]
    table_no_map, no_table_map, table_card_list, attr_no_map_list \
        , attr_no_types_list, attr_ranges_list = tables_info

    full_name = short_full_table_name_map[table]
    table_no = table_no_map[full_name]
    attr_no = attr_no_map_list[table_no][attr]
    return attr_ranges_list[table_no][attr_no]

def get_equi_classes(join_conds):
    equi_classes = DisjointSet()
    for join_cond in join_conds:
        (lhs, op, rhs) = join_cond
        equi_classes.add(lhs)
        equi_classes.add(rhs)
        equi_classes.merge(lhs, rhs)
    return equi_classes

def split_query(query, tables_info):
    short_full_table_name_map, join_conds, filter_conds = simple_parse(query)

    if len(join_conds) == 0:
        return None

    # full_short_table_name_map = {}
    # for (short_name, full_name) in short_full_table_name_map:
    #     full_short_table_name_map[full_name] = short_name
    equi_classes = DisjointSet()

    table_attrs = set()
    for join_cond in join_conds:
        (lhs, op, rhs) = join_cond
        equi_classes.add(lhs)
        equi_classes.add(rhs)
        equi_classes.merge(lhs, rhs)
        table_attrs.add(lhs)
        table_attrs.add(rhs)

    # for filter_cond in filter_conds:
    #     (lhs, op, rhs) = filter_cond
    #     if lhs in table_attrs:
    #         table_attrs.remove(lhs)


    min_diff = 1e100
    x_table_attr = None
    x_lbd = None
    x_ubd = None
    for table_attr in table_attrs:
        attr_ranges = get_attr_ranges(table_attr, short_full_table_name_map, tables_info)
        diff = attr_ranges[1] - attr_ranges[0]
        assert (diff >= 0)
        if diff < min_diff:
            min_diff = diff

            x_table_attr = table_attr
            x_lbd = int(attr_ranges[0])
            x_ubd = int(attr_ranges[1])

    query_template = query.strip()[0:-1]
    subset = equi_classes.subset(x_table_attr)
    for s in subset:
        query_template += " and " + s + " = {0:d}"
    query_template += ";"
    new_queries = []
    for i in range(x_lbd, x_ubd + 1):
        new_query = query_template.format(i)
        new_queries.append(new_query)

    return new_queries

def table_and_attr(s):
    terms = s.split(".")
    return terms[0], terms[1]

def equi_classes_to_join_predicates(equi_classes):
    sub_clauses = []
    for sub_equi_class in equi_classes.subsets():
        table_attrs = list(sub_equi_class)
        table_attrs.sort()
        for j in range(1, len(table_attrs)):
            sub_clause = table_attrs[j - 1] + ' = ' + table_attrs[j]
            sub_clauses.append(sub_clause)
    return sub_clauses

def load_queries(path):
    lines = file_utils.read_all_lines(path)
    queries = []
    for _line in lines:
        terms = _line.strip().split('||')
        query = terms[0].lower()
        queries.append(query)
    return queries

# create table movie_companies (movie_id integer not null, company_id integer not null, company_type_id integer not null);
def parse_create_sql(create_sql):
    terms = create_sql.strip().lower().split('create table')
    info = terms[1].strip()
    lidx = info.find('(')
    table_name = info[0:lidx].strip()
    ridx = info.rfind(')')
    attr_infos_str = info[lidx + 1: ridx].strip()
    attr_infos = attr_infos_str.split(',')

    attr_descs = []
    attr_extra_infos = []
    for attr_info in attr_infos:
        terms = attr_info.strip().split(' ')
        attr_name = terms[0].strip()
        data_type = terms[1].strip()
        extra_info = None
        if len(terms) > 2:
            extra_info = ' '.join(terms[2:])
        attr_extra_infos.append(extra_info)
        assert data_type in {'bigint', 'integer', 'character', 'double', 'smallint', 'timestamp', 'serial'}
        if data_type == 'character':
            varying_str = terms[2].strip()
            assert (varying_str.startswith('varying'))
        # assert (data_type == 'integer')
        attr_descs.append((attr_name, data_type))
    return table_name, attr_descs, attr_extra_infos

def get_attr_infos_from_create_sql(create_sql):
    table_name, attr_descs, attr_extra_infos = parse_create_sql(create_sql)
    attr_type_list = []
    attr_names = []
    for attr_desc in attr_descs:
        attr_name = attr_desc[0]
        attr_type = attr_desc[1]
        attr_names.append(attr_name)
        if attr_type in {'bigint', 'integer', 'smallint', 'serial'}:
            attr_type_list.append(0)
        elif attr_type == 'double':
            attr_type_list.append(1)
        else:
            attr_type_list.append(-1)

    return table_name, attr_names, attr_type_list

def get_all_table_attr_infos(create_tables_path):
    lines = file_utils.read_all_lines(create_tables_path)
    table_attr_infos_list = []
    table_attr_types_map = {}
    for _line in lines:
        line = _line.lower()
        if line.startswith('create table'):
            table_name, attr_names, attr_type_list = get_attr_infos_from_create_sql(line.strip())
            table_attr_infos_list.append((table_name, attr_names, attr_type_list))
            table_attr_types_map[table_name] = attr_type_list
    return table_attr_infos_list, table_attr_types_map
