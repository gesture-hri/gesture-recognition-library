import json
import logging
import os
from typing import List, Iterable, Union

import mediapipe
import numpy as np

from gesture_recognition.classification import TrainableClassifier, TFLiteClassifier
from gesture_recognition.mediapipe_cache import MediaPipeCache
from gesture_recognition.preprocessing import Preprocessor, TFLitePreprocessor

logger = logging.getLogger("gesture recognizer")


class GestureRecognizer:
    _HAND = "hand"
    _POSE = "pose"
    _SUPPORTED_MODES = [_HAND, _POSE]

    _LOW_COMPLEXITY = "low"
    _MEDIUM_COMPLEXITY = "medium"
    _HIGH_COMPLEXITY = "high"

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
        cache: MediaPipeCache = None,
        categories: List[any] = None,
        mode: str = _HAND,
        complexity: str = _LOW_COMPLEXITY,
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
        :param complexity: Specifies mediapipe solution complexity to be used for landmarks estimation. The higher the
        complexity, the higher landmarks estimation accuracy, but also the higher estimation latency. Must be on of:
        "high", "low" if mode is "hand" and one of: "high", "medium", "low" if mode is "pose".
        """

        if mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Mode: {mode} is not supported. Supported modes are: {', '.join(self._SUPPORTED_MODES)}."
            )

        self.mode = mode
        self.complexity = complexity
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.cache = cache
        self.categories = categories

        if self.mode == self._HAND:
            if self.complexity not in [self._HIGH_COMPLEXITY, self._LOW_COMPLEXITY]:
                raise ValueError(
                    "Complexity: {} for mode: hands not supported. Supported complexities are: {}".format(
                        self.complexity,
                        ", ".join([self._HIGH_COMPLEXITY, self._LOW_COMPLEXITY]),
                    )
                )

            self.mediapipe_handle = mediapipe.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=0 if self.complexity == self._LOW_COMPLEXITY else 1,
            )
        else:
            if self.complexity not in [self._HIGH_COMPLEXITY, self._LOW_COMPLEXITY]:
                raise ValueError(
                    "Complexity: {} for mode: pose not supported. Supported complexities are: {}".format(
                        self.complexity,
                        ", ".join(
                            [
                                self._HIGH_COMPLEXITY,
                                self._LOW_COMPLEXITY,
                                self._MEDIUM_COMPLEXITY,
                            ]
                        ),
                    )
                )

            self.mediapipe_handle = mediapipe.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity={
                    self._LOW_COMPLEXITY: 0,
                    self._MEDIUM_COMPLEXITY: 1,
                    self._HIGH_COMPLEXITY: 2,
                }[self.complexity],
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
    ) -> Union[np.float, List[np.float]]:
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
        except (MediaPipeCache.Error, AttributeError):
            x = [self._image_flow(sample) for sample in samples]
            y = [label for idx, label in enumerate(labels) if x[idx] is not None]
            x = [sample for sample in x if sample is not None]

            if self.cache is not None:
                self.cache.store_mediapipe_output(x, y, identifier)

        x = [self.preprocessor.preprocess(sample) for sample in x]

        logger.info(f"Using {len(x)} samples for training and evaluation")
        return self.classifier.train(x, y)

    def recognize(self, image: np.ndarray) -> Union[List[int], List[any], None]:
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

    def save_recognizer(self, path: str):
        """
        Serializes recognizer content and stores in directory pointed by path argument.
        :param path: Path under which serialized recognizer content will be stored.
        """
        os.makedirs(path, exist_ok=True)

        config_path = os.path.join(path, "config.json")

        self.classifier.save_classifier(path)
        self.preprocessor.save_preprocessor(path)
        with open(config_path, "w+") as config_fd:
            config = {
                "mode": self.mode,
                "complexity": self.complexity,
            }
            json.dump(config, config_fd)

    @classmethod
    def from_recognizer_dir(
        cls,
        path: str,
        classifier_module=TFLiteClassifier,
        preprocessor_module=TFLitePreprocessor,
    ):
        """
        Deserializes recognizer content from directory pointed by path argument and uses it to instantiate recognizer.
        :param path: Path from which recognizer content will be restored.
        :param classifier_module: TrainableClassifier derived class with 'from_file' method implementation.
        :param preprocessor_module: Preprocessor derived class with 'from_file' method implementation.
        :return: Restored GestureRecognizer instance that has been previously saved.
        """
        config_path = os.path.join(path, "config.json")

        classifier = classifier_module.from_file(path)
        preprocessor = preprocessor_module.from_file(path)

        with open(config_path, "rb") as config_fd:
            config = json.load(config_fd)

        return cls(
            classifier=classifier,
            preprocessor=preprocessor,
            mode=config["mode"],
            complexity=config["complexity"],
        )
