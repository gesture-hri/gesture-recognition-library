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
logger = logging.getLogger("rock paper scissors pipeline")

if __name__ == "__main__":
    paths = [
        [file for file in os.scandir(gesture.path) if file.path.endswith(".npy")]
        for gesture in os.scandir("../data/rock_paper_scissors")
    ]
    random_state = 42

    samples = (
        np.load(file.path)
        for gesture in paths
        for file in shuffle(gesture, random_state=random_state)
    )
    labels = (
        label
        for label, gesture in enumerate(paths)
        for _file in shuffle(gesture, random_state=random_state)
    )

    classifier = SklearnClassifier(
        ExtraTreesClassifier(), random_state=random_state, test_size=0.2
    )
    preprocessor = DefaultPreprocessor()
    cache = PickleCache("../pickle_cache_dir")
    categories = ["rock", "paper", "scissors"]
    gesture_recognizer = GestureRecognizer(classifier, preprocessor, cache)
    score = gesture_recognizer.train_end_evaluate(
        "rock_paper_scissors", samples, labels, categories
    )
    gesture_recognizer.to_pickle_binary(
        "../pretrained_recognizers/rock_paper_scissors.pkl"
    )
    logger.info(f"Extra trees classifier scored {score} on rock-paper-scissors dataset")
