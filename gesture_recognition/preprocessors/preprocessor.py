from abc import ABC, abstractmethod

import numpy as np


class Preprocessor(ABC):
    """
    All preprocessors used by GestureRecognizer must override this class by implementing methods below
    """

    def normalize(self, image: np.ndarray, *_args, **_kwargs):
        """
        This method will be called on mediapipe input to perform standard normalization.
        :param image: Image in numpy array format.
        :return: Normalized image, ready to be processed by mediapipe
        """
        return image

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        This method will be called on numpy array that represent mediapipe output
        to perform additional semantic preprocessing.
        """
        pass
