import logging
import pickle
from typing import List, Iterable

import mediapipe
import numpy as np

from gesture_recognition.classifiers import TrainableClassifier
from gesture_recognition.mediapipe_cache import MediapipeCache
from gesture_recognition.preprocessors import Preprocessor

logger = logging.getLogger('gesture recognizer')


# TODO: Add .from_config file method.
#  Override pickling to ensure classifier is correctly pickled.
class GestureRecognizer:
    categories: List[any]

    def __init__(
            self,
            classifier: TrainableClassifier,
            preprocessor: Preprocessor,
            cache: MediapipeCache = None,
            hands=True,
    ):
        """
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

        if self.hands:
            # TODO: what about two-handed gestures in the same dataset with single-handed?
            self.mediapipe_handle = mediapipe.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
            )
        else:
            self.mediapipe_handle = mediapipe.solutions.holistic.Holistic(static_image_mode=True)

        if self.cache is not None:
            self.cache.initialize()

    def _image_flow(self, image: np.ndarray):
        """
        Performs all operations that single raw image needs to be fed into classifier.
        :param image: Image to perform operations on.
        :return: Data format that can be accepted by classifier.
        """
        normalized = self.preprocessor.normalize(image)

        if self.hands:
            mediapipe_output = self.mediapipe_handle.process(normalized).multi_hand_landmarks
        else:
            mediapipe_output = self.mediapipe_handle.process(normalized).pose_landmarks

        return mediapipe_output

    def train_end_evaluate(
            self,
            identifier: str = None,
            samples: Iterable[np.ndarray] = None,
            labels: Iterable[np.int] = None,
            categories: List[any] = None,
    ):
        """
        Trains and evaluates top classifier.
        :param categories: List of objects associated with gesture numeric label.
        :param samples:Raw images in numpy array format.
        :param labels: Categories corresponding to images from samples.
        :param identifier: Dataset name. Essential for cache framework.
        :return: Accuracy score achieved by classifier on provided dataset.
        """
        if identifier is None and self.cache is None:
            raise Exception('Unable to use dataset cache without identifier specified.')

        self.categories = categories

        try:
            x, y = self.cache.retrieve_mediapipe_output(identifier)
            logger.info(f'Dataset identified as {identifier} found in cache.')
        except (MediapipeCache.Error, AttributeError):
            x = [self._image_flow(sample) for sample in samples]
            y = [label for idx, label in enumerate(labels) if x[idx] is not None]
            x = [sample for sample in x if sample is not None]

            if self.cache is not None:
                self.cache.store_mediapipe_output(x, y, identifier)

        x = [self.preprocessor.preprocess(sample) for sample in x]

        logger.info(f'Using {len(x)} samples for training and evaluation')
        return self.classifier.train(x, y)

    def recognize(self, image: np.ndarray):
        """
        Recognizes gesture present on image.
        :param image: Image with gesture to be recognized.
        :return: Detected gesture label index or corresponding object from categories (if is not None)
        """
        preprocessed = self._image_flow(image)
        classification = self.classifier.infer(preprocessed)

        if self.categories:
            return [self.categories[label] for label in classification]
        return classification

    def to_pickle_binary(self, path):
        """
        Serializes GestureRecognizer to pickle binary format and saves under provided path.
        Custom serialization and deserialization is necessary as mediapipe instance cannot be picled.
        :param path: File path under which GestureRecognizer instance is expected to be stored.
        """
        with open(path, 'w+b') as f:
            pickle.dump(
                obj=(self.classifier, self.preprocessor, self.cache, self.hands),
                file=f,
            )

    @classmethod
    def from_pickle_binary(cls, path):
        """
        Deserializes and creates GestureRecognizer instance from pickle binary stored under provided path.
        :param path: File path where GestureRecognizer binary is stored.
        :return: GestureRecognizer instance.
        """
        with open(path, 'rb') as f:
            classifier, preprocessor, cache, hands = pickle.load(f)
            return cls(classifier, preprocessor, cache, hands)
