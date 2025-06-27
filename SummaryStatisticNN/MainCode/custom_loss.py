import tensorflow as tf

def gated_loss(y_true, y_pred):
    # need to unpack the values from the call in the NN
    p = tf.shape(y_true)[-1]
    theta_hat = y_pred[:, :p]
    g         = y_pred[:, p:]

    # prediction error
    mse = tf.reduce_mean(tf.square(y_true - theta_hat))

    # gate should match the true indicator 1{θ≠0}
    γ_true = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(γ_true, g))

    # optional l1 penalty on g to encourage fewer non-zeros
    l1 = 1e-3 * tf.reduce_mean(g)

    return mse + bce + l1
