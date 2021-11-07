import logging
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle

from gesture_recognition.classification import *
from gesture_recognition.gesture_recognizer import GestureRecognizer
from gesture_recognition.mediapipe_cache import PickleCache
from gesture_recognition.preprocessing import *

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """
    This script is intended to serve two purposes:
        1) For library developers themselves to test its efficiency on different datasets
        2) For library users to be an example of how to train and use recognizer in their projects.
    It expects the following arguments:
        :param dataset_name: dataset string identifier
        :param dataset_path: path (absolute or relative) to dataset that MUST have all image files grouped
        into directories based on gesture they represent. Directory for each gesture should be named by the gesture it
        represents
        :param cache_path: path (absolut or relative) to directory where mediapipe output on dataset will be cached
        :param classifier_binary_path: path (absolute or relative) under which serialized classifier instance
        will be stored
    Library developers are encouraged to experiment with classifier and preprocessor variables to find optimal
    combination, specific to dataset they are working with.
    """

    try:
        (
            _script,
            dataset_name,
            dataset_path,
            cache_path,
            classifier_binary_path,
        ) = sys.argv
    except IndexError:
        raise ValueError("Invalid number of arguments")

    logger = logging.getLogger(f"{dataset_name} dataset POC")

    paths = [
        [
            file
            for file in os.scandir(gesture.path)
            if any([file.path.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]])
        ]
        for gesture in os.scandir(dataset_path)
    ]
    random_state = 42

    samples = (
        np.asarray(Image.open(file.path))
        for gesture in paths
        for file in shuffle(gesture, random_state=random_state)
    )
    labels = (
        label
        for label, gesture in enumerate(paths)
        for _file in shuffle(gesture, random_state=random_state)
    )

    keras_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(paths), activation="softmax"),
        ]
    )
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["acc"],
        loss="sparse_categorical_crossentropy",
    )
    classifier = TFLiteClassifier(
        keras_model=keras_model,
        test_size=0.2,
        random_state=random_state,
        keras_training_params={
            "epochs": 10,
            "verbose": 0,
        },
    )
    preprocessor = TFLitePreprocessor.from_function(default_preprocessing)
    cache = PickleCache(cache_path)
    categories = [gesture.name for gesture in os.scandir(dataset_path)]

    gesture_recognizer = GestureRecognizer(classifier, preprocessor, cache, categories)
    score = gesture_recognizer.train_end_evaluate(
        dataset_name,
        samples,
        labels,
    )
    classifier.save_classifier(classifier_binary_path)

    logger.info(f"Keras classifier scored {score} on {dataset_name} dataset")
