import logging
from typing import List, Iterable

import mediapipe
import numpy as np

from gesture_recognition.classification import TrainableClassifier
from gesture_recognition.mediapipe_cache import MediapipeCache
from gesture_recognition.preprocessors import Preprocessor

logger = logging.getLogger("gesture recognizer")


# TODO: Add .from_config file method.
class GestureRecognizer:
    def __init__(
        self,
        classifier: TrainableClassifier,
        preprocessor: Preprocessor,
        cache: MediapipeCache = None,
        categories: List[any] = None,
        hands=True,
    ):
        """
        :param categories: List of objects associated with gesture numeric label.
        :param classifier: Classifier that will be used on top of mediapipe output to classify gestures
        :param preprocessor: Object responsible for additional semantic preprocessing of mediapipe output before
        passing it to classifier.
        :param hands: Whether gestures to be recognized are hands gestures.
        :param cache: Indicates whether to cache output of mediapipe application on training dataset.
        """
        self.hands = hands
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.cache = cache
        self.categories = categories

        if self.hands:
            # TODO: what about two-handed gestures in the same dataset with single-handed?
            self.mediapipe_handle = mediapipe.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
            )
        else:
            self.mediapipe_handle = mediapipe.solutions.pose.Pose(
                static_image_mode=True
            )

        if self.cache is not None:
            self.cache.initialize()

    def _image_flow(self, image: np.ndarray):
        """
        Performs normalization and mediapipe processing on raw image before it is fed into classifier.
        :param image: Image to perform operations on.
        :return: Data format that can be accepted by preprocessor.
        """
        normalized = self.preprocessor.normalize(image)

        if self.hands:
            try:
                return self.mediapipe_handle.process(normalized).multi_hand_landmarks
            except TypeError:
                raise TypeError("Mediapipe expects 3D np.ndarray of np.unit8")

        pose_mediapipe_output = self.mediapipe_handle.process(normalized).pose_landmarks
        if pose_mediapipe_output is not None:
            # In case of Pose solution only Single NormalizedLandmarkList object is returned.
            # self.preprocessor expects list of that
            return [pose_mediapipe_output]

    def train_end_evaluate(
        self,
        identifier: str = None,
        samples: Iterable[np.ndarray] = None,
        labels: Iterable[np.int] = None,
    ):
        """
        Trains and evaluates top classifier.
        :param samples: Raw images in numpy array format.
        :param labels: Categories corresponding to images from samples.
        :param identifier: Dataset name. Essential for cache framework.
        :return: Score achieved by classifier on provided dataset. Score format is defined by classifier class.
        """
        if identifier is None and self.cache is None:
            raise Exception("Unable to use dataset cache without identifier specified.")

        try:
            x, y = self.cache.retrieve_mediapipe_output(identifier)
            logger.info(f"Dataset identified as {identifier} found in cache.")
        except (MediapipeCache.Error, AttributeError):
            x = [self._image_flow(sample) for sample in samples]
            y = [label for idx, label in enumerate(labels) if x[idx] is not None]
            x = [sample for sample in x if sample is not None]

            if self.cache is not None:
                self.cache.store_mediapipe_output(x, y, identifier)

        x = [self.preprocessor.preprocess(sample) for sample in x]

        logger.info(f"Using {len(x)} samples for training and evaluation")
        return self.classifier.train(x, y)

    def recognize(self, image: np.ndarray):
        """
        Recognizes gesture present on image.
        :param image: Image with gesture to be recognized.
        :return: Detected gesture label index or corresponding object from categories (if is not None)
        """
        mediapipe_output = self._image_flow(image)
        if mediapipe_output is None:
            return mediapipe_output

        preprocessed = [self.preprocessor.preprocess(mediapipe_output)]
        classification = self.classifier.infer(preprocessed)

        if self.categories:
            return [self.categories[label] for label in classification]
        return classification
