from abc import ABC, abstractmethod

import cv2
import numpy as np


class Preprocessor(ABC):
    """
    All preprocessors used by GestureRecognizer must override this class by implementing methods below
    """

    def normalize(self, image: np.ndarray, video_mode=False, *_args, **_kwargs):
        """
        This method will be called on mediapipe input to perform standard normalization.
        :param image: Image in numpy array format.
        :param video_mode: Specifies whether image comes from photo file or video stream.
        :return: Normalized image, ready to be processed by mediapipe
        """
        if video_mode:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        This method will be called on mediapipe output to perform additional semantic preprocessing.
        """
        pass
