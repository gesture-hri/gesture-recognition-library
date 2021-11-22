import json
import logging
import os
from typing import List, Iterable, Union

import mediapipe
import numpy as np

from gesture_recognition.classification import TrainableClassifier, TFLiteClassifier
from gesture_recognition.mediapipe_cache import MediapipeCache
from gesture_recognition.preprocessing import Preprocessor, TFLitePreprocessor

logger = logging.getLogger("gesture recognizer")


class GestureRecognizer:
    _HAND = "hand"
    _POSE = "pose"
    _SUPPORTED_MODES = [_HAND, _POSE]

    class LandmarkShapes:
        """
        Mediapipe results shape holder
        """

        HAND_LANDMARK_SHAPE = (21, 3)
        POSE_LANDMARK_SHAPE = (33, 3)

    def __init__(
        self,
        classifier: TrainableClassifier,
        preprocessor: Preprocessor,
        cache: MediapipeCache = None,
        categories: List[any] = None,
        mode: str = _HAND,
    ):
        """
        :param categories: List of objects associated with gesture numeric label.
        :param classifier: Classifier that will be used on top of mediapipe output to classify gestures
        :param preprocessor: Object responsible for additional semantic preprocessing of mediapipe output before
        passing it to classifier.
        :param mode: Specifies which body parts landmarks will be estimated by mediapipe. This attribute controls
        which particular mediapipe solution will be used and which attributes of its output will be
        passed to preprocessor. Must be one of: "hands", "pose"
        :param cache: Indicates whether to cache output of mediapipe application on training dataset.
        """
        if mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Mode: {mode} is not supported. Supported modes are: {', '.join(self._SUPPORTED_MODES)}."
            )

        self.mode = mode
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.cache = cache
        self.categories = categories

        if self.mode == self._HAND:
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

    def _image_flow(self, image: np.ndarray) -> Union[List[np.ndarray], None]:
        """
        Performs normalization and mediapipe processing on raw image before it is fed into classifier.
        :param image: Image to perform operations on.
        :return: List of numpy arrays representing extracted landmarks specific to particular mediapipe solution,
        or None in case of unsuccessful mediapipe inference.
        """

        try:
            if self.mode == self._HAND:
                mediapipe_output = self.mediapipe_handle.process(
                    image
                ).multi_hand_landmarks
            else:
                mediapipe_output = self.mediapipe_handle.process(image).pose_landmarks
        except TypeError:
            raise TypeError("Mediapipe expects 3D np.ndarray of np.unit8")

        if mediapipe_output is not None:
            # In case of Pose solution only Single NormalizedLandmarkList object is returned.
            # self.preprocessor expects list of that
            if self.mode == self._HAND:
                landmarks = [
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z]
                            for landmarks in mediapipe_output
                            for landmark in landmarks.landmark
                        ]
                    ).astype(np.float32)
                ]
            else:
                landmarks = [
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in mediapipe_output.landmark
                        ]
                    ).astype(np.float32)
                ]
            return landmarks
        return mediapipe_output

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
        landmarks = self._image_flow(image)
        if landmarks is None:
            return landmarks

        preprocessed = self.preprocessor.preprocess(landmarks)
        classification = self.classifier.infer(preprocessed)

        if self.categories:
            return [self.categories[label] for label in classification]
        return classification

    def save_recognizer(self, path):
        if not isinstance(self.classifier, TFLiteClassifier):
            raise AttributeError(
                "Attribute `classifier` must be instance of TFLiteClassifier class to save recognizer using "
                "this method. Override it if you are working with custom TrainableClassifier derived class."
            )

        if not isinstance(self.preprocessor, TFLitePreprocessor):
            raise AttributeError(
                "Attribute `preprocessor` must be instance of TFLitePreprocessor class to save recognizer using "
                "this method. Override it if you are working with custom Preprocessor derived class."
            )

        os.makedirs(path, os.O_RDWR, exist_ok=True)

        classifier_path = os.path.join(path, "classifier.tflite")
        preprocessor_path = os.path.join(path, "preprocessor.tflite")
        config_path = os.path.join(path, "config.json")

        self.classifier.save_classifier(classifier_path)
        self.preprocessor.save_preprocessor(preprocessor_path)
        with open(config_path, "w+") as config_fd:
            config = {
                "mode": self.mode,
            }
            json.dump(config, config_fd)

    @classmethod
    def from_recognizer_dir(cls, path):
        classifier_path = os.path.join(path, "classifier.tflite")
        preprocessor_path = os.path.join(path, "preprocessor.tflite")
        config_path = os.path.join(path, "config.json")

        classifier = TFLiteClassifier.from_file(classifier_path)
        preprocessor = TFLitePreprocessor.from_file(preprocessor_path)

        with open(config_path, "rb") as config_fd:
            config = json.load(config_fd)

        return cls(
            classifier=classifier, preprocessor=preprocessor, mode=config["mode"]
        )
