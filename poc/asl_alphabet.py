import logging
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle

from gesture_recognition.classifiers import SklearnClassifier
from gesture_recognition.gesture_recognizer import GestureRecognizer
from gesture_recognition.mediapipe_cache import PickleCache
from gesture_recognition.preprocessors import DefaultPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asl alphabet pipeline")

if __name__ == "__main__":
    paths = [
        [file for file in os.scandir(letter.path) if file.path.endswith(".npy")]
        for letter in os.scandir("../data/asl_alphabet_train")
    ]

    random_state = 42

    samples = (
        np.load(file.path)
        for letter in paths
        for file in shuffle(letter, random_state=random_state)
    )
    labels = (
        label
        for label, letter in enumerate(paths)
        for _file in shuffle(letter, random_state=random_state)
    )

    classifier = SklearnClassifier(
        ExtraTreesClassifier(), random_state=random_state, test_size=0.2
    )
    preprocessor = DefaultPreprocessor()
    cache = PickleCache("../pickle_cache_dir")
    gesture_recognizer = GestureRecognizer(classifier, preprocessor, cache, hands=True)
    score = gesture_recognizer.train_end_evaluate(
        identifier="asl_alphabet",
        samples=samples,
        labels=labels,
    )

    logger.info(f"Extra trees classifier scored {score} on ASL alphabet dataset")
