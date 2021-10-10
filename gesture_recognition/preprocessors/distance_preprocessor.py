from enum import Enum

import numpy as np
from sklearn.preprocessing import StandardScaler

from gesture_recognition.preprocessors.preprocessor import Preprocessor


class DistancePreprocessor(Preprocessor):
    """
    Distance preprocessor extracts 3D coordinates from joint landmarks, calculates norm of euclidean distance
    between each pair of landmarks and flattens the output.
    """
    class Metrics(Enum):
        """
        Class to hold available metrics as enumeration type.
        """
        L1 = 1
        L2 = 2

    def __init__(self, metrics):
        """
        :param metrics: Metrics used to calculate vector norm in preprocess method
        """
        if not isinstance(metrics, DistancePreprocessor.Metrics):
            raise ValueError("Currently only metrics defined in DistancePreprocessor class are supported")

        self.metrics = metrics

    def preprocess(self, mediapipe_output, *args, **kwargs):
        landmarks = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmarks in mediapipe_output
                for landmark in landmarks.landmark
            ],
        )

        return (
            StandardScaler()
            .fit_transform(
                np.linalg.norm((landmarks[:, np.newaxis] - landmarks), axis=2, ord=self.metrics.value),
            )
            .reshape((-1,))
        )
