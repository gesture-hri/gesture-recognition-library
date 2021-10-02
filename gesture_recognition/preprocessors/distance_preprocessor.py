import numpy as np
from sklearn.preprocessing import StandardScaler

from gesture_recognition.preprocessors.preprocessor import Preprocessor


class DistancePreprocessor(Preprocessor):
    """
    Distance preprocessor extracts 3D coordinates from joint landmarks, calculates norm of euclidean distance
    between each pair of landmarks and flattens the output.
    """
    def normalize(self, image, *args, **kwargs):
        return image

    def preprocess(self, mediapipe_output, *args, **kwargs):
        landmarks = np.array(
            [[landmark.x, landmark.y, landmark.z]
             for landmarks in mediapipe_output for landmark in landmarks.landmark],
        )

        return StandardScaler().fit_transform(
            np.linalg.norm((landmarks[:, np.newaxis] - landmarks), axis=1),
        ).reshape((-1,))
