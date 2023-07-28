import os
import numpy as np
import sys
import copy
sys.path.append("../")
from src.utils import file_utils, FileViewer, sql_utils
from src.arg_parser import arg_parser

class tableHistogram(object):
    def __init__(self, initial_data_path, n_bins, min_vals, max_vals, attr_no_map, table_size_threshold):
        self.n_bins = n_bins
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.bin_sizes = (max_vals - min_vals) / self.n_bins
        self.attr_no_map = attr_no_map
        self.n_attrs = len(attr_no_map)
        # self.attr_type_list = attr_type_list
        arange_idxes = np.reshape(np.arange(0, self.n_attrs, dtype=np.int64), [1, -1])
        self.idxes = np.concatenate([arange_idxes, np.zeros(shape=arange_idxes.shape, dtype=arange_idxes.dtype)],
                                    axis=0)
        self.table_size_threshold = table_size_threshold

        if initial_data_path is not None:
            with open(initial_data_path, "r") as reader:
                lines = reader.readlines()
            lines = lines[1:]
            # self.table_size = len(lines)
            # self.histogram = np.zeros(shape=[self.n_attrs, self.n_bins], dtype=np.int64)
            initial_data = []
            for k in range(self.n_attrs):
                initial_data.append([])
            for line in lines:
                terms = line.split(",")
                for i, _term in enumerate(terms):
                    term = _term.strip()
                    if len(term) > 0:
                        # print('term =', term)
                        val_int = sql_utils.time_to_int_in_data(term)
                        if val_int is not None:
                            val = val_int
                        else:
                            # val = int(term)
                            val = float(term)
                        initial_data[i].append(val)
            initial_data = [np.sort(np.array(data_i, dtype=np.float64)) for data_i in initial_data]
            boundary_points = []
            for k in range(self.n_attrs):
                arange_idxes = np.reshape(np.arange(1, self.n_bins + 1, dtype=np.float64), [1, -1])
                boundary_points.append(arange_idxes)
            boundary_points = np.concatenate(boundary_points, axis=0)
            boundary_points *= np.reshape(self.bin_sizes, [-1, 1])
            assert (boundary_points.shape[0] == min_vals.shape[0])
            for i in range(min_vals.shape[0]):
                boundary_points[i] += min_vals[i]
            boundary_points[:, -1] += 1
            print(f'boundary_points.shape = {boundary_points.shape}, len(lines) = {len(lines)}')

            self.histogram = []
            for i in range(self.n_attrs):
                sorted_idxes = np.searchsorted(initial_data[i], boundary_points[i], side='left')
                print(
                    f'table = {os.path.basename(initial_data_path)[0:-4]}, i = {i}, sorted_idxes[0:10] = {sorted_idxes[0:10]}, sorted_idxes[-10:] = {sorted_idxes[-10:]}')
                assert sorted_idxes[-1] == initial_data[i].shape[0]
                sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
                sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
                sorted_idxes_left_translation[0] = 0
                diff = sorted_idxes - sorted_idxes_left_translation
                self.histogram.append(np.reshape(diff, [1, -1]))
            self.histogram = np.concatenate(self.histogram, axis=0, dtype=np.float64)
        else:
            self.histogram = np.zeros(shape=[self.n_attrs, self.n_bins], dtype=np.float64)

        # print('histogram.shape =', self.histogram.shape)

    def insert(self, sql):
        # self.table_size += 1
        terms = sql.split(" values")
        _values = (terms[1].strip()[1:-2]).split(",")
        # values = copy.deepcopy(_values)
        assert len(_values) == self.n_attrs

        values = np.array([float(x) for x in _values], dtype=np.float64)
        self.insert_one_row(values)


    def delete(self, sql):
        terms = sql.split("##")
        _values = (terms[1].strip()).split(",")
        assert len(_values) == self.n_attrs
        values = np.array([float(x) for x in _values], dtype=np.float64)
        self.delete_one_row(values)

    def update(self, sql):
        terms = sql.split("##")
        second_part = terms[1].strip()
        terms = second_part.split('#')
        assert len(terms) == 2
        _values_0 = terms[0].split(",")
        assert len(_values_0) == self.n_attrs
        values_0 = np.array([float(x) for x in _values_0], dtype=np.float64)
        _values_1 = terms[1].split(",")
        assert len(_values_1) == self.n_attrs
        values_1 = np.array([float(x) for x in _values_1], dtype=np.float64)
        self.delete_one_row(values_0)
        self.insert_one_row(values_1)


    def insert_one_row(self, values):
        # assert values.shape[0] == self.n_attrs
        self.idxes[1] = (values - self.min_vals) // self.bin_sizes
        self.histogram[self.idxes[0], self.idxes[1]] += 1

    def delete_one_row(self, values):
        self.idxes[1] = (values - self.min_vals) // self.bin_sizes
        self.histogram[self.idxes[0], self.idxes[1]] -= 1

    def histogram_feature(self):
        feature = np.reshape(self.histogram / self.table_size_threshold, [-1])
        return feature


class databaseHistogram(object):
    def __init__(self, tables_info, workload_path, n_bins, checkpoint_dir):
        table_no_map, no_table_map, table_card_list, attr_no_map_list \
            , attr_no_types_list, attr_ranges_list = tables_info
        self.table_no_map = table_no_map

        self.no_table_map = no_table_map
        self.n_tables = len(attr_ranges_list)
        self.table_historgrams = []

        self.existing_histogram_features = None
        self.existing_num_inserts_before_queries = None
        self.existing_train_idxes = None
        self.existing_train_sub_idxes = None
        self.existing_test_idxes = None
        self.existing_test_sub_idxes = None
        self.existing_test_single_idxes = None
        self.existing_line_no = None

        self.histogram_features = None
        self.num_inserts_before_queries = None
        self.train_idxes = None
        self.train_sub_idxes = None
        self.test_idxes = None
        self.test_sub_idxes = None
        self.split_idxes = None

        self.query_and_results = None
        self.query_ids = None
        self.checkpoint_dir = checkpoint_dir
        FileViewer.detect_and_create_dir(self.checkpoint_dir)

        print('\tInitialzing DB states...')
        for i in range(self.n_tables):
            th = tableHistogram(None,
                                n_bins=n_bins,
                                min_vals=attr_ranges_list[i][:, 0],
                                max_vals=attr_ranges_list[i][:, 1],
                                attr_no_map=attr_no_map_list[i],
                                table_size_threshold=table_card_list[i])
            self.table_historgrams.append(th)

        lines = file_utils.read_all_lines(workload_path)
        table_path_map = {}
        for line in lines:
            if line.startswith('--'):
                break
            terms = line.split(' ')
            path = terms[3][1:-1]
            fname = os.path.basename(path)
            table_info = fname[0:-4]
            terms = table_info.split('-')
            table_name = terms[0]
            table_path_map[table_name] = path

        if len(table_path_map) != self.n_tables:
            print('table_path_map =', table_path_map)
            print('self.n_tables =', self.n_tables)
        assert len(table_path_map) == self.n_tables
        for table_name in table_path_map:
            assert table_name in table_no_map

        flag = self.restore()
        if flag == False:
            self.table_historgrams.clear()
            for i in range(self.n_tables):
                table_name = self.no_table_map[i]
                assert table_name in table_path_map
                initial_data_path = table_path_map[table_name]
                th = tableHistogram(initial_data_path,
                                    n_bins=n_bins,
                                    min_vals=attr_ranges_list[i][:, 0],
                                    max_vals=attr_ranges_list[i][:, 1],
                                    attr_no_map=attr_no_map_list[i],
                                    table_size_threshold=table_card_list[i])
                self.table_historgrams.append(th)


    def process_sql(self, _sql, print_feature=False):
        try:
            sql = _sql.lower().strip()
            if sql.startswith('insert'):
                terms = sql.split(' ')
                table = terms[2].strip()
                table_no = self.table_no_map[table]
                # print('table =', table)
                self.table_historgrams[table_no].insert(sql)
            elif sql.startswith('delete'):
                terms = sql.split(' ')
                table = terms[2].strip()
                table_no = self.table_no_map[table]
                self.table_historgrams[table_no].delete(sql)
            elif sql.startswith('update'):
                terms = sql.split(' ')
                table = terms[1].strip()
                table_no = self.table_no_map[table]
                self.table_historgrams[table_no].update(sql)
        except:
            print('sql =', _sql)
            raise Exception()

    def batch_process(self, sqls):
        for sql in sqls:
            self.process_sql(sql)

    def current_histogram_feature(self):
        feature_list = [self.table_historgrams[i].histogram_feature() for i in range(self.n_tables)]
        feature = np.concatenate(feature_list)
        return feature

    def build_histogram_features(self, workload_results_path):
        with open(workload_results_path, "r") as reader:
            lines = reader.readlines()

        start_line_no = 0
        for i, line in enumerate(lines):
            if line.startswith('--'):
                start_line_no = i + 1
                break

        # print('start_line_no =', start_line_no)
        if self.query_and_results is not None:
            assert self.existing_line_no >= start_line_no
            j = self.existing_histogram_features.shape[0]
            num_inserts = self.existing_num_inserts_before_queries[-1]
            k = self.existing_line_no
            assert (k >= start_line_no)
        else:
            self.query_and_results = []
            self.query_ids = []
            self.split_idxes = []
            j = 0
            num_inserts = 0
            k = start_line_no + 1

        batch_sqls = []

        self.histogram_features = []
        self.num_inserts_before_queries = []
        self.train_idxes = []
        self.train_sub_idxes = []
        self.test_idxes = []
        self.test_sub_idxes = []
        self.test_single_idxes = []

        i = 0
        if k >= len(lines):
            return
        nlines = len(lines)
        for i in range(nlines):
            if i < k:
                continue
            line = lines[i].strip().lower()
            if line.startswith("insert") or line.startswith("delete") or line.startswith("update"):
                batch_sqls.append(line)
                num_inserts += 1
                # DH.add(line)
            elif line.startswith("t"): # train or train_sub or test or test_sub
                if i >= start_line_no:
                    terms = line.split("||")
                    card_str = "-1"
                    if line.startswith("train"):
                        if line.startswith("train_sub"):
                            if len(terms) >= 5:
                                card_str = terms[-2]
                            self.train_sub_idxes.append(j)
                            query_id = "-".join(["0", terms[1], terms[2]])
                        else:
                            assert line.startswith("train_query")
                            if len(terms) >= 4:
                                card_str = terms[-2]
                            self.train_idxes.append(j)
                            query_id = "-".join(["1", terms[1]])
                    else: # line.startswith("train"):
                        if line.startswith("test_sub"):
                            if len(terms) >= 5:
                                card_str = terms[-2]
                            self.test_sub_idxes.append(j)
                            query_id = "-".join(["3", terms[1], terms[2]])
                        elif line.startswith("test_single"):
                            if len(terms) >= 5:
                                card_str = terms[-2]
                            self.test_single_idxes.append(j)
                            query_id = "-".join(["4", terms[1], terms[2]])
                        else:
                            if len(terms) >= 4:
                                card_str = terms[-2]
                            self.test_idxes.append(j)
                            query_id = "-".join(["5", terms[1]])
                    j += 1
                    query_str = terms[0]
                    idx = query_str.find(':')
                    query_str = query_str[idx + 2:]
                    self.query_and_results.append(query_str + "||" + card_str + "\n")
                    self.query_ids.append(query_id + "\n")
                    self.batch_process(batch_sqls)
                    curr_feature = self.current_histogram_feature()
                    self.histogram_features.append(curr_feature)
                    self.num_inserts_before_queries.append(num_inserts)
                    batch_sqls = []
            elif line.startswith('--'):
                self.split_idxes.append(j)

            if i % 10000 == 0:
                print(f'\tBuilding DB states: {(i * 100) // nlines}%', end='\r')
        self.split_idxes.append(j)
        self.save(i + 1)
        print()

    def save(self, line_no):
        FileViewer.detect_and_create_dir(self.checkpoint_dir)
        all_exist = (self.existing_histogram_features is not None) and (self.existing_num_inserts_before_queries is not None) \
                    and (self.existing_train_idxes is not None) and (self.existing_train_sub_idxes is not None) \
                    and (self.existing_test_idxes is not None) and (self.existing_test_sub_idxes is not None) \
                    and (self.existing_test_single_idxes is not None)


        all_not_exist = (self.existing_histogram_features is None) and (self.existing_num_inserts_before_queries is None) \
                    and (self.existing_train_idxes is None) and (self.existing_train_sub_idxes is None) \
                    and (self.existing_test_idxes is None) and (self.existing_test_sub_idxes is None) \
                        and (self.existing_test_single_idxes is None)

        # print('all_exist =', all_exist)

        assert all_exist or all_not_exist
        if all_exist and len(self.histogram_features) == 0:
            return
        for table_no in range(self.n_tables):
            th = self.table_historgrams[table_no]
            histogram_i_path = os.path.join(self.checkpoint_dir, 'histogram_{0:d}_{1:d}.npy'.format(table_no, line_no))
            np.save(histogram_i_path, th.histogram)

        if self.query_and_results is not None:
            query_results_path = os.path.join(self.checkpoint_dir, 'query_results_{0:d}.txt'.format(line_no))
            with open(query_results_path, 'w') as writer:
                writer.writelines(self.query_and_results)

        if self.query_ids is not None:
            query_ids_path = os.path.join(self.checkpoint_dir, 'query_ids_{0:d}.txt'.format(line_no))
            with open(query_ids_path, 'w') as writer:
                writer.writelines(self.query_ids)

        if self.split_idxes is not None:
            split_idxes_path = os.path.join(self.checkpoint_dir, 'split_idxes_{0:d}.npy'.format(line_no))
            split_idxes = np.array(self.split_idxes, dtype=np.int64)
            np.save(split_idxes_path, split_idxes)

        histogram_features_path = os.path.join(self.checkpoint_dir, 'histogram_features_{0:d}.npy'.format(line_no))
        num_inserts_before_queries_path = os.path.join(self.checkpoint_dir,
                                                       'num_inserts_before_queris_{0:d}.npy'.format(line_no))
        train_idxes_path = os.path.join(self.checkpoint_dir, 'train_idxes_{0:d}.npy'.format(line_no))
        train_sub_idxes_path = os.path.join(self.checkpoint_dir, 'train_sub_idxes_{0:d}.npy'.format(line_no))
        test_idxes_path = os.path.join(self.checkpoint_dir, 'test_idxes_{0:d}.npy'.format(line_no))
        test_sub_idxes_path = os.path.join(self.checkpoint_dir, 'test_sub_idxes_{0:d}.npy'.format(line_no))
        test_single_idxes_path = os.path.join(self.checkpoint_dir, 'test_single_idxes_{0:d}.npy'.format(line_no))

        histogram_features, num_inserts_before_queries, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes =\
            self.current_features_and_train_test_indexes(all_exist)

        if histogram_features is not None:
            np.save(histogram_features_path, histogram_features)
            # print('histogram_features.shape =', histogram_features.shape)
        if num_inserts_before_queries is not None:
            np.save(num_inserts_before_queries_path, num_inserts_before_queries)
            # print('num_inserts_before_queries.shape =', num_inserts_before_queries.shape)
        if train_idxes is not None:
            np.save(train_idxes_path, train_idxes)
            # print('n_train =', train_idxes.shape[0])
        if train_sub_idxes is not None:
            np.save(train_sub_idxes_path, train_sub_idxes)
            # print('n_train_sub =', train_sub_idxes.shape[0])
        if test_idxes is not None:
            np.save(test_idxes_path, test_idxes)
            # print('n_test =', test_idxes.shape[0])
        if test_sub_idxes is not None:
            np.save(test_sub_idxes_path, test_sub_idxes)
            # print('n_test_sub =', test_sub_idxes.shape[0])
        if test_single_idxes is not None:
            np.save(test_single_idxes_path, test_single_idxes)
            # print('n_test_single =', test_single_idxes.shape[0])


    def restore(self):
        line_no = -1
        histogram_i_files = FileViewer.list_filenames(self.checkpoint_dir, suffix='npy',
                                                             prefix='histogram')
        if len(histogram_i_files) == 0:
            return False

        for file in histogram_i_files:
            name = file[0:-4]
            terms = name.split('_')
            x = int(terms[-1])
            if x > line_no:
                line_no = x
        self.existing_line_no = line_no

        files = []
        for table_no in range(self.n_tables):
            # th = self.table_historgrams[table_no]
            histogram_i_path = os.path.join(self.checkpoint_dir, 'histogram_{0:d}_{1:d}.npy'.format(table_no, line_no))
            files.append(histogram_i_path)

        histogram_features_path = os.path.join(self.checkpoint_dir, 'histogram_features_{0:d}.npy'.format(line_no))
        num_inserts_before_queries_path = os.path.join(self.checkpoint_dir,
                                                       'num_inserts_before_queris_{0:d}.npy'.format(line_no))
        train_idxes_path = os.path.join(self.checkpoint_dir, 'train_idxes_{0:d}.npy'.format(line_no))
        train_sub_idxes_path = os.path.join(self.checkpoint_dir, 'train_sub_idxes_{0:d}.npy'.format(line_no))
        test_idxes_path = os.path.join(self.checkpoint_dir, 'test_idxes_{0:d}.npy'.format(line_no))
        test_sub_idxes_path = os.path.join(self.checkpoint_dir, 'test_sub_idxes_{0:d}.npy'.format(line_no))
        test_single_idxes_path = os.path.join(self.checkpoint_dir, 'test_single_idxes_{0:d}.npy'.format(line_no))
        query_results_path = os.path.join(self.checkpoint_dir, 'query_results_{0:d}.txt'.format(line_no))
        query_ids_path = os.path.join(self.checkpoint_dir, 'query_ids_{0:d}.txt'.format(line_no))
        split_idxes_path = os.path.join(self.checkpoint_dir, 'split_idxes_{0:d}.npy'.format(line_no))

        files.append(histogram_features_path)
        files.append(num_inserts_before_queries_path)
        files.append(train_idxes_path)
        files.append(train_sub_idxes_path)
        files.append(test_idxes_path)
        files.append(test_sub_idxes_path)
        files.append(test_single_idxes_path)
        files.append(query_results_path)
        files.append(query_ids_path)
        files.append(split_idxes_path)

        for file in files:
            if os.path.exists(file) == False:
                return False


        for table_no in range(self.n_tables):
            th = self.table_historgrams[table_no]
            histogram_i_path = os.path.join(self.checkpoint_dir, 'histogram_{0:d}_{1:d}.npy'.format(table_no, line_no))
            # assert os.path.exists(histogram_i_path)
            th.histogram = np.load(histogram_i_path)

        self.existing_histogram_features = np.load(histogram_features_path)
        self.existing_num_inserts_before_queries = np.load(num_inserts_before_queries_path)
        self.existing_train_idxes = np.load(train_idxes_path)
        self.existing_train_sub_idxes = np.load(train_sub_idxes_path)
        self.existing_test_idxes = np.load(test_idxes_path)
        self.existing_test_sub_idxes = np.load(test_sub_idxes_path)
        self.existing_test_single_idxes = np.load(test_single_idxes_path)

        with open(query_results_path, 'r') as reader:
            self.query_and_results = reader.readlines()

        with open(query_ids_path, 'r') as reader:
            self.query_ids = reader.readlines()

        split_idxes = np.load(split_idxes_path)
        self.split_idxes = split_idxes.tolist()

        return True

    def current_features_and_train_test_indexes(self, all_exist):
        if self.histogram_features is None or self.num_inserts_before_queries is None or self.train_idxes is None or \
                self.train_sub_idxes is None or self.test_idxes is None or self.test_sub_idxes is None or self.test_single_idxes is None:
            return None, None, None, None, None, None, None

        histogram_features = np.array(self.histogram_features, dtype=np.float64)
        num_inserts_before_queries = np.array(self.num_inserts_before_queries, dtype=np.int64)
        train_idxes = np.array(self.train_idxes, dtype=np.int64)
        train_sub_idxes = np.array(self.train_sub_idxes, dtype=np.int64)
        test_idxes = np.array(self.test_idxes, dtype=np.int64)
        test_sub_idxes = np.array(self.test_sub_idxes, dtype=np.int64)
        test_single_idxes = np.array(self.test_single_idxes, dtype=np.int64)

        if len(self.histogram_features) == 0:
            assert all_exist
            return self.existing_histogram_features, self.existing_num_inserts_before_queries, self.existing_train_idxes, self.existing_train_sub_idxes, self.existing_test_idxes, self.existing_test_sub_idxes, self.existing_test_single_idxes
        elif all_exist and len(self.histogram_features) > 0:
            histogram_features = np.concatenate([self.existing_histogram_features, histogram_features], axis=0)
            num_inserts_before_queries = np.concatenate([self.existing_num_inserts_before_queries, num_inserts_before_queries])
            train_idxes = np.concatenate([self.existing_train_idxes, train_idxes])
            train_sub_idxes = np.concatenate([self.existing_train_sub_idxes, train_sub_idxes])
            test_idxes = np.concatenate([self.existing_test_idxes, test_idxes])
            test_sub_idxes = np.concatenate([self.existing_test_sub_idxes, test_sub_idxes])
            test_single_idxes = np.concatenate([self.existing_test_single_idxes, test_single_idxes])

        return histogram_features, num_inserts_before_queries, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes

    def current_data(self):
        all_exist = (self.existing_histogram_features is not None) and (self.existing_num_inserts_before_queries is not None) \
                    and (self.existing_train_idxes is not None) and (self.existing_train_sub_idxes is not None) \
                    and (self.existing_test_idxes is not None) and (self.existing_test_sub_idxes is not None) \
                    and (self.existing_test_single_idxes is not None)



        histogram_features, num_inserts_before_queries, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes = self.current_features_and_train_test_indexes(all_exist)
        if not (self.histogram_features is None or self.train_idxes is None or self.train_sub_idxes is None or self.test_idxes is None or self.test_sub_idxes is None or self.test_single_idxes is None):
            return (self.query_and_results, [s.strip() for s in self.query_ids], self.split_idxes, histogram_features,
                    num_inserts_before_queries, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes)

        else:
            return (None, None, None, None, None, None, None, None, None, None)

