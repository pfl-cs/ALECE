import tensorflow as tf

# Inner-layer dimensionality.
def point_wise_feed_forward_network(
        d_model, # Input/output dimensionality.
        dff, # Inner-layer dimensionality.
        _dtype):

    layers = [
        tf.keras.layers.Dense(
            dff,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_regularizer=tf.keras.regularizers.L2(1e-4)
        ),  # Shape `(batch_size, seq_len, dff)`.
        tf.keras.layers.Dense(
            d_model,
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_regularizer=tf.keras.regularizers.L2(1e-4)
        )  # Shape `(batch_size, seq_len, d_model)`.
    ]
    reg_regularization = tf.constant(0, dtype=_dtype)
    for layer in layers:
        reg_regularization += tf.math.reduce_sum(layer.losses)
    seq = tf.keras.Sequential(layers)

    return seq, reg_regularization
