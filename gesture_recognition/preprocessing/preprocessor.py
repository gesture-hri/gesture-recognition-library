from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Preprocessor(ABC):
    """
    All preprocessors used by GestureRecognizer must override this class by implementing methods below
    """

    @abstractmethod
    def preprocess(self, landmark_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        This method will be called on numpy array that represent mediapipe output
        to perform additional semantic preprocessing.
        :param landmark_vectors: List of numpy arrays with 3D body joint positions estimated by MediaPipe
        :return: List of numpy arrays being feature matrices.
        """
        pass

    @abstractmethod
    def save_preprocessor(self, gesture_recognizer_path: str):
        """
        This method will be called by GestureRecognizer during saving process.
        :param gesture_recognizer_path:  Path to the directory under which GestureRecognizer content will be stored.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, gesture_recognizer_path: str):
        """
        This class method will be called during GestureRecognizer deserialization from directory, which requires class
        module to be provided.
        :param gesture_recognizer_path: Path to the directory from which GestureRecognizer content will be restored.
        :return: Instantiated preprocessor ready to run preprocessing.
        """
        pass
