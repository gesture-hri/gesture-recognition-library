import logging
import os

import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle

from gesture_recognition.classifiers import SklearnClassifier
from gesture_recognition.gesture_recognizer import GestureRecognizer
from gesture_recognition.mediapipe_cache import PickleCache
from gesture_recognition.preprocessors import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("senz3d dataset pipeline")

if __name__ == "__main__":
    paths = [
        [file for file in os.scandir(gesture.path) if file.path.endswith(".png")]
        for gesture in os.scandir("../data/senz3d")
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

    classifier = SklearnClassifier(
        ExtraTreesClassifier(), random_state=random_state, test_size=0.2
    )
    preprocessor = DistancePreprocessor(DistancePreprocessor.Metrics.L2)
    cache = PickleCache("../pickle_cache_dir")
    categories = sorted(
        [
            "five",
            "thumb",
            "small-finger",
            "two",
            "three",
            "palm",
            "fist",
            "hard-rock",
            "pointer",
            "four",
            "ok",
        ]
    )
    gesture_recognizer = GestureRecognizer(classifier, preprocessor, cache)
    score = gesture_recognizer.train_end_evaluate("senz3d", samples, labels, categories)
    gesture_recognizer.to_pickle_binary("../pretrained_recognizers/senz3d.pkl")
    logger.info(f"Extra trees classifier scored {score} on senz3d dataset")
