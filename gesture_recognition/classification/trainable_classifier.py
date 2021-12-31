from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class TrainableClassifier(ABC):
    """
    All classifiers used on top of mediapipe must override this class by implementing abstract methods below
    """

    class Error(Exception):
        """
        Base class for related classifier errors and exceptions.
        """

        pass

    @abstractmethod
    def train(
        self, samples: List[np.ndarray], labels: List[np.ndarray]
    ) -> Union[np.float, List[np.float]]:
        """
        This method will be called by GestureRecognizer to train and evaluate its top classifier
        to semantic preprocessing output.
        :param samples: Raw images in numpy array format.
        :param labels: Categories corresponding to images from samples.
        :return: Keras model evaluation result on samples fraction used for testing.
        """
        pass

    @abstractmethod
    def evaluate(
        self, samples: List[np.ndarray], labels: List[np.ndarray]
    ) -> Union[np.float, List[np.float]]:
        """
        This method will be called by GestureRecognizer to evaluate its top classifier on semantic preprocessing output.
        :param samples: Raw images in numpy array format.
        :param labels: Categories corresponding to images from samples.
        :return: Keras model evaluation result on samples.
        """
        pass

    @abstractmethod
    def infer(self, samples: List[np.ndarray]) -> List[int]:
        """
        This method will be called by GestureRecognizer to classify incoming images.
        :param samples: Raw images in numpy array format.
        :return: List of predicted class labels for each image in samples.
        """
        pass

    @abstractmethod
    def infer_probabilities(self, samples: List[np.ndarray]) -> List[np.ndarray]:
        """
        This method can be used for evaluation of more advanced metrics than simple accuracy. Note that some
        classifiers like SVM are incapable of providing probability distribution over classes. Appropriate exception
        should be thrown.
        :param samples: Raw images in numpy array format.
        :return: List of numpy arrays of per-class probabilities for each image in samples.
        """
        pass

    @abstractmethod
    def save_classifier(self, gesture_recognizer_path: str):
        """
        This method will be called by GestureRecognizer during saving process.
        :param gesture_recognizer_path: Path to the directory under which GestureRecognizer content will be stored.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, gesture_recognizer_path: str):
        """
        This class method will be called during GestureRecognizer deserialization from directory, which requires class
        module to be provided.
        :param gesture_recognizer_path: Path to the directory from which GestureRecognizer content will be restored.
        :return: Instantiated classifier ready to run inference.
        """
        pass
