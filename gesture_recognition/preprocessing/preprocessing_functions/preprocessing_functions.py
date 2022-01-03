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


def l1_preprocessing(array: np.ndarray):
    distances = tf.norm((array[:, None] - array), axis=2, ord=1)
    return tf.reshape(_normalize_array(distances), (-1,))


def cosine_preprocessing(array: np.ndarray):
    normalized = array / tf.reshape(tf.norm(array, axis=1), (-1, 1))
    cosine_similarity = tf.linalg.matmul(normalized, normalized, transpose_b=True)
    normalized_cosine_similarity = _normalize_array(cosine_similarity)
    return tf.subtract(
        tf.ones(shape=cosine_similarity.shape), normalized_cosine_similarity
    )
