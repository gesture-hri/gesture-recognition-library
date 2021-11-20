import numpy as np
import tensorflow as tf


def _normalize_array(array: np.ndarray):
    return (array - tf.math.reduce_mean(array, axis=0)) / tf.math.reduce_std(
        array, axis=0
    )


def simple_preprocessing(array: np.ndarray):
    return _normalize_array(array)


def default_preprocessing(array: np.ndarray):
    return tf.reshape(_normalize_array(array), (-1,))


def euclidean_preprocessing(array: np.ndarray):
    distances = tf.norm((array[:, None] - array), axis=2)
    return tf.reshape(_normalize_array(distances), (-1,))
