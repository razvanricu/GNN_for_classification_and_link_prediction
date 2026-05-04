import tensorflow as tf
from tensorflow import keras


def focal_loss_custom(gamma=2.0):
    """Focal Loss returning a per-node vector.
    Keras multiplies each element by sample_weight before averaging."""

    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        y_true_casted = tf.cast(y_true, tf.int32)
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(batch_size), y_true_casted], axis=1)
        p_t = tf.gather_nd(y_pred, indices)

        modulating_factor = tf.pow(1.0 - p_t, gamma)

        return modulating_factor * cross_entropy

    loss_fn.__name__ = "focal_loss_custom"
    return loss_fn
