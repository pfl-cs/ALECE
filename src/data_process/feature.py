import os
import numpy as np
import sys
sys.path.append("../")
from src.utils import file_utils, arg_parser_utils
from src.arg_parser import arg_parser
from src.data_process import histogram
from .parse_sql import *

# now, a query is assumed to contain range selection and equi-join condition
class queryFeature(object):
    def __init__(self, table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list, possible_join_attrs):
        """
        :param attr_ranges_list: list with each element is a n x 2 numpy float matrix.
        denoting the ith table has n attrs. These n attrs' range are represented by this matrix.
        :param possible_join_attrs: N * 4 numpy int matrx with each row looks like [table_no1, table_no1.attr_no, table_no2, table_no2.attr_no]
        """

        self.table_no_map, self.attr_no_map_list, self.attr_no_types_list, self.attr_ranges_list \
            = table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list

        self.attr_ranges_all = np.concatenate(attr_ranges_list, axis=0, dtype=np.float64)
        self.attr_types_all = np.concatenate(attr_no_types_list, dtype=np.int64)

        # for feature calc only
        self.attr_lbds = np.zeros_like(self.attr_ranges_all, dtype=self.attr_ranges_all.dtype)
        self.attr_range_measures = np.zeros_like(self.attr_ranges_all, dtype=self.attr_ranges_all.dtype)

        self.attr_lbds[:, 0] = self.attr_ranges_all[:, 0]
        self.attr_lbds[:, 1] = self.attr_ranges_all[:, 0]

        tmp = self.attr_ranges_all[:, 1] - self.attr_ranges_all[:, 0]

        self.attr_range_measures[:, 0] = tmp
        self.attr_range_measures[:, 1] = tmp

        self.attr_lbds = np.reshape(self.attr_lbds, [-1])
        self.attr_range_measures = np.reshape(self.attr_range_measures, [-1])
        # print("attr_range_measures_all.shape =", self.attr_range_measures_all.shape)

        self.n_tables = len(attr_ranges_list)
        self.n_attrs_total = self.attr_types_all.shape[0]

        self.maxn_attrs_single_table = self.attr_ranges_list[0].shape[0]
        for i in range(1, self.n_tables):
            table_i = self.attr_ranges_list[i]
            n_attrs = table_i.shape[0]
            if n_attrs > self.maxn_attrs_single_table:
                self.maxn_attrs_single_table = n_attrs

        # print("n_attrs_total =", self.n_attrs_total)
        join_attrs_trans = possible_join_attrs.transpose()

        t1, t1_attr, t2, t2_attr =  join_attrs_trans[0], join_attrs_trans[1], join_attrs_trans[2], join_attrs_trans[3]
        m1 = t1 * self.maxn_attrs_single_table + t1_attr
        m2 = t2 * self.maxn_attrs_single_table + t2_attr
        M = self.maxn_attrs_single_table * self.n_tables

        join_ids = set()
        equi_relations = {}
        for i in range(m1.shape[0]):
            id_1 = m1[i]
            id_2 = m2[i]
            if id_1 <= id_2:
                join_id = id_1 * M + id_2
                symm_join_id = id_2 * M + id_1
            else:
                join_id = id_2 * M + id_1
                symm_join_id = id_1 * M + id_2
            equi_relations[join_id] = symm_join_id
            join_ids.add(join_id)
        join_ids = list(join_ids)
        join_ids.sort()

        self.join_id_no_map = {}
        for i, join_id in enumerate(join_ids):
            self.join_id_no_map[join_id] = i
            symm_join_id = equi_relations[join_id]
            self.join_id_no_map[symm_join_id] = i

        self.n_possible_joins = len(self.join_id_no_map)


    # join_id: a number from [0, M * M), where M = self.maxn_attrs_single_table * self.n_tables
    def calc_join_ids(self, join_conds):
        join_conds_trans = np.transpose(join_conds)
        m1 = join_conds_trans[0] * self.maxn_attrs_single_table + join_conds_trans[1]
        m2 = join_conds_trans[2] * self.maxn_attrs_single_table + join_conds_trans[3]
        M = self.maxn_attrs_single_table * self.n_tables
        return M * m1 + m2

    # join_no: a number from [0, self.n_possible_joins)
    def calc_join_nos(self, join_conds):
        join_idxes = self.calc_join_ids(join_conds)
        for i in range(join_idxes.shape[0]):
            join_idxes[i] = self.join_id_no_map[join_idxes[i]]
        return join_idxes

    def encode(self, sql_join_conds, sql_attr_ranges_conds, relevant_tables):
        """
        :param sql_join_conds: shape=[m, self.n_possible_joins]
        :param sql_attr_ranges_conds: shape=[self.n_attrs_total * 2]
        :return:
        """
        feature = np.zeros(self.n_tables + self.n_possible_joins + self.n_attrs_total * 2, dtype=np.float64)

        # encode tables appeared in query
        feature[relevant_tables] = 1

        # encode conjunctive join conds
        if sql_join_conds is not None:
            join_id_idxes = self.calc_join_ids(sql_join_conds)
            join_id_idxes += self.n_tables
            feature[join_id_idxes] = 1

        cursor = self.n_tables + self.n_possible_joins

        # encode conjunctive filter conds
        feature[cursor:cursor + self.n_attrs_total * 2] = ((sql_attr_ranges_conds - self.attr_lbds) / self.attr_range_measures) * 2.0 - 1
        return feature

    #encode_batch_w_appeared_tables(
    def encode_batch(self, sql_join_conds_batch, sql_attr_ranges_conds_batch, relevant_tables_list):
        """
        :param sql_join_conds_batch: list of sql_join_conds
        :param sql_attr_ranges_conds_batch: shape=[batch_size, self.n_attrs_total * 2]
        :return:
        """
        batch_size = len(sql_join_conds_batch)
        features = np.zeros(shape=[batch_size,self.n_tables + self.n_possible_joins + self.n_attrs_total * 2], dtype=np.float64)
        for i in range(batch_size):
            relevant_tables = relevant_tables_list[i]
            # encode tables appeared in query
            features[i][relevant_tables] = 1
            sql_join_conds = sql_join_conds_batch[i]
            if sql_join_conds is not None:
                # encode conjunctive join conds
                join_id_idxes = self.calc_join_nos(sql_join_conds)
                join_id_idxes += self.n_tables
                # print(join_id_idxes)
                features[i][join_id_idxes] = 1
        cursor = self.n_tables + self.n_possible_joins

        # encode conjunctive filter conds
        features[:, cursor:cursor + self.n_attrs_total * 2] = ((sql_attr_ranges_conds_batch - self.attr_lbds) / self.attr_range_measures) * 2.0 - 1
        return features



def query_part_features_gen(qF, join_conds_list, attr_range_conds_list, true_card_list, cartesian_join_card_list, natural_join_card_list, relevant_tables_list):
    true_cards = np.array(true_card_list, dtype=np.float64)
    attr_range_conds_batch = np.array(attr_range_conds_list, dtype=np.float64)
    features = qF.encode_batch(join_conds_list, attr_range_conds_batch, relevant_tables_list)
    n_possible_joins = qF.n_possible_joins
    cartesian_join_cards = None
    natural_join_cards = None
    if cartesian_join_card_list is not None:
        cartesian_join_cards = np.array(cartesian_join_card_list, dtype=np.float64)
    if natural_join_card_list is not None:
        natural_join_cards = np.array(natural_join_card_list, dtype=np.float64)

    return (features, true_cards, cartesian_join_cards, natural_join_cards, n_possible_joins)


def normalize_query_features(train_query_features, test_query_features_list, n_possible_joins=None):
    train_std = train_query_features.std(axis=0)
    nonzero_idxes = np.where(train_std > 0)[0]
    train_query_features = train_query_features[:, nonzero_idxes]
    train_mean = train_query_features.mean(axis=0)
    train_std = train_std[nonzero_idxes]
    train_query_features = (train_query_features - train_mean) / train_std

    num = len(test_query_features_list)
    for i in range(num):
        test_query_features = test_query_features_list[i]
        test_query_features = test_query_features[:, nonzero_idxes]
        test_query_features = (test_query_features - train_mean) / train_std
        test_query_features_list[i] = test_query_features

    if n_possible_joins is not None:
        join_pattern_train_std = train_std[0:n_possible_joins]
        join_pattern_nonzero_idxes = np.where(join_pattern_train_std > 0)[0]
        join_pattern_dim = join_pattern_nonzero_idxes.shape[0]

        return train_query_features, test_query_features_list, join_pattern_dim
    else:
        return train_query_features, test_query_features_list


def _load_data_from_workload(args, wl_type=None):
    workload_dir, feature_data_dir, histogram_ckpt_dir = arg_parser_utils.get_feature_data_dir(args, wl_type)
    workload_path = os.path.join(workload_dir, args.workload_fname)

    FileViewer.detect_and_create_dir(feature_data_dir)
    all_features_path = os.path.join(feature_data_dir, 'all_features.npy')
    all_cards_path = os.path.join(feature_data_dir, 'all_cards.npy')
    all_num_inserts_path = os.path.join(feature_data_dir, 'all_num_inserts.npy')

    train_idxes_path = os.path.join(feature_data_dir, 'train_idxes.npy')
    train_sub_idxes_path = os.path.join(feature_data_dir, 'train_sub_idxes.npy')
    test_idxes_path = os.path.join(feature_data_dir, 'test_idxes.npy')
    test_sub_idxes_path = os.path.join(feature_data_dir, 'test_sub_idxes.npy')
    test_single_idxes_path = os.path.join(feature_data_dir, 'test_single_idxes.npy')

    meta_infos_path = os.path.join(feature_data_dir, 'feature_meta_infos.npy')

    required_paths = [all_features_path, all_cards_path, all_num_inserts_path, train_idxes_path, train_sub_idxes_path,
                      test_idxes_path, test_sub_idxes_path, test_single_idxes_path, meta_infos_path]
    all_files_exist = True
    for path in required_paths:
        if os.path.exists(path) == False:
            all_files_exist = False
            break

    if all_files_exist:
        all_features = np.load(all_features_path)
        all_cards = np.load(all_cards_path)
        all_num_inserts = np.load(all_num_inserts_path)

        train_idxes = np.load(train_idxes_path)
        train_sub_idxes = np.load(train_sub_idxes_path)

        test_idxes = np.load(test_idxes_path)
        test_sub_idxes = np.load(test_sub_idxes_path)
        test_single_idxes = np.load(test_single_idxes_path)

        meta_infos = np.load(meta_infos_path).tolist()
        # [histogram_feature_dim, num_attrs, n_possible_joins] = meta_infos

        return (all_features, all_cards, all_num_inserts, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes, meta_infos)

    tables_info = get_tables_info(args)
    table_no_map, no_table_map, table_card_list, attr_no_map_list \
        , attr_no_types_list, attr_ranges_list = tables_info

    num_attrs = 0
    for attr_no_map in attr_no_map_list:
        num_attrs += len(attr_no_map)

    data_dir = args.data_dir
    base_query_path = os.path.join(data_dir, args.base_queries_fname)
    print('\tParsing statements in the workload...')
    base_queries_info = parse_queries_from_file(
        base_query_path,
        table_no_map,
        attr_no_map_list,
        attr_no_types_list,
        attr_ranges_list,
        min_card_threshold=0,
        delim="||",
        baseline_results_delim="|*|"
    )
    base_possible_join_attrs = base_queries_info[0]

    qF = queryFeature(table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list, base_possible_join_attrs)

    DH = histogram.databaseHistogram(tables_info, workload_path, args.n_bins, histogram_ckpt_dir)
    DH.build_histogram_features(workload_path)
    query_info_strs, query_ids, split_idxes, histogram_features, num_inserts_before_queries, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes = DH.current_data()

    queries_info = parse_queries(
        query_info_strs,
        table_no_map,
        attr_no_map_list,
        attr_no_types_list,
        attr_ranges_list,
        min_card_threshold=0,
        delim="||",
        baseline_results_delim=None
    )

    possible_join_attrs, join_conds_list, attr_range_conds_list, true_card_list, cartesian_join_card_list, natural_join_card_list \
        , join_type_list, relevant_tables_list, selection_conds_list, baseline_results = queries_info

    if cartesian_join_card_list[0] is None:
        cartesian_join_card_list = None

    if natural_join_card_list[0] is None:
        natural_join_card_list = None

    print('\tBuilding query featurizations...')
    query_part_data = query_part_features_gen(
        qF,
        join_conds_list,
        attr_range_conds_list,
        true_card_list,
        cartesian_join_card_list,
        natural_join_card_list,
        relevant_tables_list
    )
    (query_part_features, true_cards, cartesian_join_cards, natural_join_cards, n_possible_joins) = query_part_data

    all_features = np.concatenate([histogram_features, query_part_features], axis=1, dtype=histogram_features.dtype)

    histogram_feature_dim = histogram_features.shape[1]

    meta_infos = [histogram_feature_dim, num_attrs, n_possible_joins]
    meta_infos = np.array(meta_infos, dtype=np.int64)
    np.save(all_features_path, all_features)
    all_cards = true_cards
    np.save(all_cards_path, all_cards)
    all_num_inserts = num_inserts_before_queries
    np.save(all_num_inserts_path, all_num_inserts)

    np.save(train_idxes_path, train_idxes)
    np.save(train_sub_idxes_path, train_sub_idxes)

    np.save(test_idxes_path, test_idxes)
    np.save(test_sub_idxes_path, test_sub_idxes)
    np.save(test_single_idxes_path, test_single_idxes)

    np.save(meta_infos_path, meta_infos)


    return (all_features, all_cards, all_num_inserts, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes, meta_infos.tolist())

def normalize_data(train_features, test_features_list, histogram_feature_dim, n_possible_joins):
    train_query_features = train_features[:, histogram_feature_dim:]
    test_query_features_list = []
    for test_features in test_features_list:
        test_query_features = test_features[:, histogram_feature_dim:]
        test_query_features_list.append(test_query_features)
    train_query_features, test_query_features_list, join_pattern_dim = normalize_query_features(train_query_features, test_query_features_list, n_possible_joins)
    query_feature_dim = train_query_features.shape[1]
    train_features[:, histogram_feature_dim:histogram_feature_dim+query_feature_dim] = train_query_features
    train_features = train_features[:, 0:histogram_feature_dim+query_feature_dim]
    for i in range(len(test_query_features_list)):
        test_features = test_features_list[i]
        test_query_features = test_query_features_list[i]
        test_features[:, histogram_feature_dim:histogram_feature_dim + query_feature_dim] = test_query_features
        test_features = test_features[:, 0:histogram_feature_dim + query_feature_dim]
        test_features_list[i] = test_features
    return train_features, test_features_list, join_pattern_dim


def load_workload_data(args):
    if args.test_wl_type == args.wl_type:
        (all_features_1, all_cards_1, _, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes,
         test_single_idxes, meta_infos) = _load_data_from_workload(args)

        all_features_2 = all_features_1
        all_cards_2 = all_cards_1
    else:
        train_wl_type = args.wl_type
        test_wl_type = args.test_wl_type
        (all_features_1, all_cards_1, _, train_idxes, train_sub_idxes, _, _,
         _, meta_infos) = _load_data_from_workload(args, train_wl_type)

        (all_features_2, all_cards_2, _, _, _, test_idxes, test_sub_idxes,
         test_single_idxes, meta_infos_2) = _load_data_from_workload(args, test_wl_type)

        for i in range(meta_infos.shape[0]):
            assert meta_infos_2[i] == meta_infos[i]

    [histogram_feature_dim, num_attrs, n_possible_joins] = meta_infos

    train_features = all_features_1[train_idxes]
    train_sub_features = all_features_1[train_sub_idxes]
    all_train_features = np.concatenate([train_features, train_sub_features], axis=0, dtype=train_features.dtype)

    train_cards = all_cards_1[train_idxes]
    train_sub_cards = all_cards_1[train_sub_idxes]
    all_train_cards = np.concatenate([train_cards, train_sub_cards], dtype=train_cards.dtype)
    all_train_cards = np.reshape(all_train_cards, [-1])
    valid_idxes = np.where(all_train_cards >= 0)[0]

    all_train_features = all_train_features[valid_idxes]
    all_train_cards = all_train_cards[valid_idxes]

    test_sub_features = all_features_2[test_sub_idxes]
    test_sub_cards = all_cards_2[test_sub_idxes]

    valid_idxes = np.where(test_sub_cards >= 0)[0]

    assert valid_idxes.shape[0] == test_sub_cards.shape[0]

    # test_single_tbls_cards = all_cards_2[test_single_idxes]
    # valid_idxes = np.where(test_single_tbls_cards >= 0)[0]
    # print('test_single_tbls_cards.shape =', test_single_tbls_cards.shape)
    # print('valid_idxes.shape =', valid_idxes.shape)
    # assert valid_idxes.shape[0] == test_single_tbls_cards.shape[0]


    all_train_features, test_sub_features, join_pattern_dim = normalize_data(all_train_features, [test_sub_features], histogram_feature_dim, n_possible_joins)
    test_sub_features = test_sub_features[0]
    feature_dim = all_train_features.shape[1]

    query_part_feature_dim = all_train_features.shape[1] - histogram_feature_dim
    meta_infos = (histogram_feature_dim, query_part_feature_dim, join_pattern_dim, num_attrs)

    return all_train_features, all_train_cards, test_sub_features, test_sub_cards, meta_infos


if __name__ == '__main__':
    args = arg_parser.get_arg_parser()
    load_workload_data(args)

# python data_process/feature.py --wl_type ins_heavy