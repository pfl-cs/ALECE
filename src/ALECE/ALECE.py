import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append("../")
from src.utils import tf_utils
from src import arg_parser


class attention(tf.keras.layers.Layer):
    def __init__(self,
                 args,
                 if_self_attn=True
                 ):
        super(attention, self).__init__()
        self.attn_head_key_dim = args.attn_head_key_dim
        self.num_attn_heads = args.num_attn_heads
        self.if_self_attn = if_self_attn
        self.attr_states_dim = args.n_bins
        self.query_part_feature_dim = args.query_part_feature_dim
        self.dropout_rate = 0.0
        if args.use_dropout == 1:
            self.dropout_rate = args.dropout_rate
        self.feed_forward_dim = args.feed_forward_dim
        self._dtype = tf.float32
        if args.use_float64 == 1:
            self._dtype = tf.float64

        # Multi-head self-attention.
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_attn_heads,
            key_dim=self.attn_head_key_dim,  # Size of each attention head for query Q and key K.
            dropout=self.dropout_rate,
        )

        # Point-wise feed-forward network.
        if self.if_self_attn:
            self.ffn, self.regularization = tf_utils.point_wise_feed_forward_network(self.attr_states_dim, args.feed_forward_dim, self._dtype)
        else:
            self.ffn, self.regularization = tf_utils.point_wise_feed_forward_network(self.query_part_feature_dim, args.feed_forward_dim, self._dtype)


        # Layer normalization.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)



    def call(self, x, training):
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=x,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attention_mask=None,  # A boolean mask that prevents attention to certain positions.
            training=training,  # A boolean indicating whether the layer should behave in training mode.
        )
        # tf.print('attn_output.shape =', attn_output.shape)

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(x + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`
        # tf.print('out1.shape =', out1.shape)

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # tf.print('ffn_output.shape =', ffn_output.shape)

        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return out2

    def self_attn(self, x, training):
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=x,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attention_mask=None,  # A boolean mask that prevents attention to certain positions.
            training=training,  # A boolean indicating whether the layer should behave in training mode.
        )
        # tf.print('attn_output.shape =', attn_output.shape)

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(x + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`
        # tf.print('out1.shape =', out1.shape)

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # tf.print('ffn_output.shape =', ffn_output.shape)

        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return out2

    def attn(self, x, q, training):
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=q,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attention_mask=None,  # A boolean mask that prevents attention to certain positions.
            training=training  # A boolean indicating whether the layer should behave in training mode.
        )
        # tf.print('attn_output.shape =', attn_output.shape)

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(q + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`
        # tf.print('out1.shape =', out1.shape)

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # tf.print('ffn_output.shape =', ffn_output.shape)

        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return out2


class attnModel(tf.keras.Model):
    def __init__(self,
                 args
                 ):
        super(attnModel, self).__init__()

        self.num_self_attn_layers = args.num_self_attn_layers
        self.num_cross_attn_layers = args.num_cross_attn_layers
        self.num_attrs = args.num_attrs
        self.attr_states_dim = args.n_bins
        self._dtype = tf.float32
        if args.use_float64 == 1:
            self._dtype = tf.float64

        self.use_positional_embedding = (args.use_positional_embedding != 0)
        self.use_dropout = (args.use_dropout != 0)

        # tf.print('self.pos_encoding.shape =', self.pos_encoding.shape)
        self.regularization = tf.constant(0, dtype=self._dtype)

        # Data-encoder
        self.self_attn_layers = [
            attention(
                args,
                if_self_attn=True
            )
            for _ in range(self.num_self_attn_layers)]

        # Query-analyzer
        self.cross_attn_layers = [
            attention(
                args,
                if_self_attn=False
            )
            for _ in range(self.num_cross_attn_layers)]

        for layer in self.self_attn_layers:
            self.regularization += layer.regularization

        for layer in self.cross_attn_layers:
            self.regularization += layer.regularization

        mlp_layers = []
        for _ in range(args.mlp_num_layers - 1):
            layer = tf.keras.layers.Dense(args.mlp_hidden_dim,
                                          activation='elu',
                                          dtype=self._dtype,
                                          kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                                          bias_regularizer=tf.keras.regularizers.L2(1e-4))
            mlp_layers.append(layer)
            self.regularization += tf.math.reduce_sum(layer.losses)

        final_layer = tf.keras.layers.Dense(1,
                                            dtype=self._dtype,
                                            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                                            bias_regularizer=tf.keras.regularizers.L2(1e-4))

        self.regularization += tf.math.reduce_sum(final_layer.losses)
        mlp_layers.append(final_layer)
        self.reg = tf.keras.Sequential(mlp_layers)

        # Dropout.
        self.dropout_rate = 0.0
        if args.use_dropout:
            self.dropout_rate = args.dropout_rate
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # self.x_normalize = tf.keras.layers.Normalization()
        # self.q_normalize = tf.keras.layers.Normalization()



    def call(self, x, q, training):
        """
        :param x: Shape `(batch_size, num_attrs, attr_states_dim)
        :param q: Shape `(batch_size, 1, query_part_features_dim)
        :param training: tf tensor
        :return:
        """
        # with self.strategy.scope():
        seq_len = tf.shape(x)[1]

        if self.use_positional_embedding:
            x *= tf.math.sqrt(tf.cast(self.attr_states_dim, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
        # Add dropout.
        if self.use_dropout:
            x = self.dropout(x, training=training)

        # Data-encoder forward
        for i in range(self.num_self_attn_layers):
            x = self.self_attn_layers[i].attn(x, x, training)

        # Query-analyzer forward
        for i in range(self.num_cross_attn_layers):
            q = self.cross_attn_layers[i].attn(x, q, training)

        preds = self.reg(q, training=training)

        return preds


attn_train_step_signature = [
    (tf.TensorSpec(shape=(None, None), dtype=tf.float32),
     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
     tf.TensorSpec(shape=(None), dtype=tf.float32),
     tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
     ),
]

class ALECE(object):
    def __init__(self, args):
        self.model_name = 'ALECE'
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.list_logical_devices('GPU')
        self.strategy = tf.distribute.MirroredStrategy(gpus)
        with self.strategy.scope():
            self.model_name = args.model
            self.num_attrs = args.num_attrs
            self.attr_states_dim = args.n_bins
            self.query_part_feature_dim = args.query_part_feature_dim
            self.attn_model = attnModel(args)
            self.mse_loss_object = tf.keras.losses.MeanSquaredError()

            self.require_init_train_step = False
            self.lr = args.lr


    def ckpt_init(self, ckpt_dir):
        with self.strategy.scope():
            self.ckpt_step = tf.Variable(-1, trainable=False)
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
            self.ckpt_dir = ckpt_dir

            self.ckpt = tf.train.Checkpoint(
                step=self.ckpt_step,
                model=self.attn_model,
                # x_normalize=self.x_normalize,
                # q_normalize=self.q_normalize,
                optimizer=self.optimizer
            )
            self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=3)

    def ckpt_reinit(self, ckpt_dir):
        with self.strategy.scope():
            self.ckpt_step.assign(-1)
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
            self.ckpt_dir = ckpt_dir

            self.ckpt = tf.train.Checkpoint(
                step=self.ckpt_step,
                model=self.attn_model,
                # x_normalize=self.x_normalize,
                # q_normalize=self.q_normalize,
                optimizer=self.optimizer
            )
            self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=3)


    def set_model_name(self, mname):
        self.model_name = mname

    def save(self, ckpt_step):
        self.ckpt_step.assign(ckpt_step)
        self.manager.save()

    def restore(self):
        if os.path.exists(self.ckpt_dir):
            # vars = tf.train.list_variables(self.ckpt_dir)
            self.ckpt.restore(self.manager.latest_checkpoint)
        return self.ckpt_step

    def compile(self, train_data):
        # train_X, train_Q, train_labels = train_data
        pass

    def forward(self, X, Q, training):
        with self.strategy.scope():

            X = tf.reshape(X, [-1, self.num_attrs, self.attr_states_dim])
            Q = tf.reshape(Q, [-1, 1, self.query_part_feature_dim])
            # Q = tf.reshape(Q, [-1, 1, Q.shape[1]])
            preds = self.attn_model(
                x=X,
                q=Q,
                training=training
            )
            return preds


    def build_loss(self, train_X, train_Q, train_weights, train_labels):
        preds = self.forward(train_X, train_Q, True)
        loss = self.mse_loss_object(train_labels, preds, sample_weight=train_weights) + self.attn_model.regularization
        return loss

    @tf.function(input_signature=attn_train_step_signature)
    def train_step(self, train_data):
        (train_X, train_Q, train_weights, train_labels) = train_data
        with tf.GradientTape() as tape:
            loss = self.build_loss(train_X, train_Q, train_weights, train_labels)
        gradients = tape.gradient(loss, self.attn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.attn_model.trainable_variables))


    def eval_validation(self, valid_data):
        (valid_X, valid_Q, valid_labels) = valid_data
        preds = self.forward(valid_X, valid_Q, training=False)
        loss = self.mse_loss_object(valid_labels, preds)
        return loss

    def eval_test(self, test_data):
        (test_X, test_Q, _) = test_data
        preds = self.forward(test_X, test_Q, training=False)
        return preds
