import logging
import os
import sys

import numpy as np
from PIL import Image
from sklearn.utils import shuffle

from gesture_recognition.gesture_recognizer.gesture_recognizer_builder import (
    GestureRecognizerBuilder,
)
from gesture_recognition.mediapipe_cache import PickleCache

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
        :param recognizer_save_dir_path: path (absolute or relative) under which recognizer instance
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
            recognizer_save_dir_path,
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

    cache = PickleCache(cache_path)
    categories = [gesture.name for gesture in os.scandir(dataset_path)]
    gesture_recognizer = (
        GestureRecognizerBuilder(mode="hand", num_classes=len(paths))
        .set_cache(cache)
        .set_categories(categories)
        .build_recognizer()
    )

    score = gesture_recognizer.train_end_evaluate(
        dataset_name,
        samples,
        labels,
    )
    gesture_recognizer.save_recognizer(recognizer_save_dir_path)

    logger.info(f"Recognizer scored {score} on {dataset_name} dataset")
