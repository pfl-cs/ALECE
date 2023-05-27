import numpy as np
import tensorflow as tf
import os
from ALECE import ALECE
import math
from arg_parser import arg_parser

from utils import FileViewer, file_utils, eval_utils, arg_parser_utils
from data_process import feature


def train_with_batch(model, args, train_data, validation_data, curr_ckpt_step):
    _train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    _validation_dataset = tf.data.Dataset.from_tensor_slices(validation_data)

    train_labels = train_data[-1]
    validation_labels = validation_data[-1]

    n_batches = math.ceil(train_labels.shape[0] / args.batch_size)
    validation_n_batches = math.ceil(validation_labels.shape[0] / args.batch_size)

    if model.require_init_train_step == True:
        for epoch in range(args.n_epochs):
            train_dataset = _train_dataset.shuffle(args.shuffle_buffer_size).batch(args.batch_size)
            for batch in train_dataset.take(n_batches):
                model.init_train_step(batch)

    best_loss = 1e100
    loss = 0
    if curr_ckpt_step >= 0:
        validation_dataset = _validation_dataset.batch(args.batch_size)
        for batch in validation_dataset.take(validation_n_batches):
            batch_loss = model.eval_validation(batch)
            loss += batch_loss.numpy()
        best_loss = loss

    if best_loss > 1e50:
        print('best_loss = inf')
    else:
        print(f'best_loss = {best_loss}')
    for epoch in range(1, args.n_epochs + 1):
        train_dataset = _train_dataset.shuffle(args.shuffle_buffer_size).batch(args.batch_size)
        batch_no = 0
        for batch in train_dataset.take(n_batches):
            model.train_step(batch)
            batch_no += 1

        loss = 0
        if curr_ckpt_step >= 0 or epoch >= args.min_n_epochs:
            validation_dataset = _validation_dataset.batch(args.batch_size)
            for batch in validation_dataset.take(validation_n_batches):
                batch_loss = model.eval_validation(batch)
                loss += batch_loss.numpy()

            if loss < best_loss:
                ckpt_step = curr_ckpt_step + epoch
                model.save(ckpt_step)
                best_loss = loss
            print(f'Epoch-{epoch}, loss = {loss}, best_loss = {best_loss}')
        else:
            print(f'Epoch-{epoch}')

def eval(model, args, test_data, q_error_dir):
    batch_preds_list = []

    test_labels = test_data[-1]
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(args.batch_size)
    n_batches = math.ceil(test_labels.shape[0] / args.batch_size)

    for batch in test_dataset.take(n_batches):
        batch_preds = model.eval_test(batch)
        batch_preds_list.append(batch_preds.numpy())

    preds = np.concatenate(batch_preds_list, axis=0)
    preds = label_preds_to_card_preds(preds, args)

    labels = test_data[-1].numpy()
    labels = label_preds_to_card_preds(np.reshape(labels, [-1]), args)
    q_error = eval_utils.generic_calc_q_error(preds, labels)
    idexes = np.where(q_error < 10)[0]
    n = idexes.shape[0]
    print('ratio =', n / q_error.shape[0])

    if q_error_dir is not None:
        fname = model.model_name
        result_path = os.path.join(q_error_dir, fname + ".npy")
        np.save(result_path, preds)

    return preds


def _normalize(data, X_mean, X_std, nonzero_idxes):
    norm_data = (data - X_mean)
    norm_data[:, nonzero_idxes] /= X_std[nonzero_idxes]
    return norm_data


def normalizations(datas):
    X = datas[0]
    X_std = X.std(axis=0)
    nonzero_idxes = np.where(X_std > 0)[0]
    X_mean = X.mean(axis=0)
    norm_data = tuple(_normalize(data, X_mean, X_std, nonzero_idxes) for data in datas)
    return norm_data


def valid_datasets(datas):
    cards = datas[-1]
    valid_idxes = np.where(cards >= 0)[0]
    valid_datas = tuple(data[valid_idxes] for data in datas)
    return valid_datas


def organize_data(raw_data_i, args):
    features = raw_data_i[0]
    db_states = features[:, 0:args.histogram_feature_dim]
    query_part_features = features[:, args.histogram_feature_dim:]

    data = [db_states, query_part_features]
    data.extend(raw_data_i[1:])
    data = tuple(data)
    return data


def cards_to_labels(cards, args):
    card_min = np.min(cards)
    assert card_min >= 0
    dtype = np.float32
    if args.use_float64 == 1:
        dtype = np.float64

    cards += 1
    cards = cards.astype(dtype)
    if args.card_log_scale == 1:
        labels = np.log(cards) / args.scaling_ratio
    else:
        labels = cards
    labels = np.reshape(labels, [-1, 1])
    return labels.astype(dtype)


def label_preds_to_card_preds(preds, args):
    preds = np.reshape(preds, [-1])
    if args.card_log_scale:
        preds *= args.scaling_ratio
        preds = np.clip(preds, a_max=25, a_min=0)
        preds = np.exp(preds) - 1
    return preds


def data_compile(data, args):
    dtype = tf.float32
    if args.use_float64 == 1:
        dtype = tf.float64

    tf_data = tuple(tf.convert_to_tensor(x, dtype=dtype) for x in data)
    return tf_data


def load_data(args):
    print('Loading data...')

    workload_data = feature.load_workload_data(args)
    ckpt_dir = arg_parser_utils.get_ckpt_dir(args)
    _, q_error_dir = arg_parser_utils.get_p_q_error_dir(args)

    FileViewer.detect_and_create_dir(ckpt_dir)
    FileViewer.detect_and_create_dir(q_error_dir)

    (train_features, train_cards, test_features, test_cards, meta_infos) = workload_data
    (histogram_feature_dim, query_part_feature_dim, join_pattern_dim, num_attrs) = meta_infos

    args.histogram_feature_dim = histogram_feature_dim
    args.query_part_feature_dim = query_part_feature_dim
    args.join_pattern_dim = join_pattern_dim
    args.num_attrs = num_attrs

    print('Processing data...')

    # print('before processing, train_features.shape =', train_features.shape)
    (train_features, train_cards) = valid_datasets((train_features, train_cards))
    # print('after processing, train_features.shape =', train_features.shape)

    train_labels = cards_to_labels(train_cards, args)

    _n_test = test_features.shape[0]
    # print('before processing, test_features.shape =', test_features.shape)
    (test_features, test_cards) = valid_datasets((test_features, test_cards))
    # print('test_features.shape =', test_features.shape)
    assert test_features.shape[0] == _n_test

    test_labels = cards_to_labels(test_cards, args)

    (train_features, test_features) = normalizations(
        (train_features, test_features)
    )

    #randomly select 10% of train data as validation data
    N_train = train_features.shape[0]
    shuffle_idxes = np.arange(0, N_train, dtype=np.int64)
    np.random.shuffle(shuffle_idxes)

    train_features = train_features[shuffle_idxes]
    train_labels = train_labels[shuffle_idxes]

    validation_features = train_features[N_train:]
    validation_labels = train_labels[N_train:]

    train_features = train_features[0: N_train]
    train_labels = train_labels[0: N_train]

    label_mean = np.mean(train_labels)
    if args.use_loss_weights == 1:
        weights = train_labels / label_mean
        weights = np.reshape(weights, [-1])
        weights = np.clip(weights, a_min=1e-3, a_max=np.max(weights))
    else:
        weights = np.ones(shape=[train_labels.shape[0]], dtype=train_labels.dtype)

    train_data = (train_features, weights, train_labels)
    validation_data = (validation_features, validation_labels)
    test_data = (test_features, test_labels)

    train_data = organize_data(train_data, args)
    validation_data = organize_data(validation_data, args)
    test_data = organize_data(test_data, args)

    train_data = data_compile(train_data, args)
    validation_data = data_compile(validation_data, args)
    test_data = data_compile(test_data, args)

    return (train_data, validation_data, test_data), q_error_dir, ckpt_dir


if __name__ == '__main__':
    args = arg_parser.get_arg_parser()
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu:args.gpu + 1], 'GPU')

    test_wl_type = args.test_wl_type
    FileViewer.detect_and_create_dir(args.experiments_dir)

    datasets, q_error_dir, ckpt_dir = load_data(args)
    (train_data, validation_data, test_data) = datasets

    model_name = f'{args.model}_{args.wl_type}'
    # ====================================================
    model = ALECE.ALECE(args)
    model.set_model_name(model_name)
    model.ckpt_init(ckpt_dir)
    ckpt_files = FileViewer.list_files(ckpt_dir)
    if len(ckpt_files) > 0:
        ckpt_step = model.restore().numpy()
    else:
        ckpt_step = -1
    print('ckpt_step =', ckpt_step)

    training = ckpt_step < 0 or args.keep_train == 1
    if training:
        train_with_batch(model, args, train_data, validation_data, ckpt_step)
        model.compile(train_data)

    ckpt_step = model.restore().numpy()
    assert ckpt_step >= 0
    preds = eval(model, args, test_data, q_error_dir)

    # ====================================================
    preds = preds.tolist()
    workload_dir = arg_parser_utils.get_workload_dir(args, test_wl_type)
    e2e_dir = os.path.join(args.experiments_dir, args.e2e_dirname)
    FileViewer.detect_and_create_dir(e2e_dir)
    train_wl_type_pre, test_wl_type_pre, pg_cards_path = arg_parser_utils.get_wl_type_pre_and_pg_cards_paths(args)

    if test_wl_type == 'static':
        path = os.path.join(e2e_dir, f'{args.model}_{args.data}_static.txt')
    else:
        path = os.path.join(e2e_dir, f'{args.model}_{args.data}_{train_wl_type_pre}_{test_wl_type_pre}.txt')

    lines = [(str(x) + '\n') for x in preds]
    file_utils.write_all_lines(path, lines)

# python train.py --model ALECE --batch_size 128 --keep_train 0 --gpu 0 --data STATS --wl_type ins_heavy
