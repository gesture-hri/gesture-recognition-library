import logging
import os
import sys

import numpy as np
import tensorflow
from PIL import Image
from sklearn.utils import shuffle

from gesture_recognition.classifiers import *
from gesture_recognition.gesture_recognizer import GestureRecognizer
from gesture_recognition.mediapipe_cache import PickleCache
from gesture_recognition.preprocessors import *

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
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

    keras_model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Dense(128, activation='relu'),
        tensorflow.keras.layers.Dense(len(paths), activation='softmax'),
    ])
    keras_model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['acc'],
        loss='sparse_categorical_crossentropy',
    )
    classifier = TFLiteClassifier(
        keras_model=keras_model,
        test_size=0.2,
        random_state=random_state,
        keras_training_params={
            'epochs': 10,
            'verbose': 0,
        },
    )
    preprocessor = DefaultPreprocessor()
    cache = PickleCache(cache_path)
    categories = [gesture.name for gesture in os.scandir(dataset_path)]

    gesture_recognizer = GestureRecognizer(classifier, preprocessor, cache)
    score = gesture_recognizer.train_end_evaluate(
        dataset_name, samples, labels, categories
    )
    classifier.save_classifier(classifier_binary_path)

    logger.info(f"Keras classifier scored {score} on {dataset_name} dataset")