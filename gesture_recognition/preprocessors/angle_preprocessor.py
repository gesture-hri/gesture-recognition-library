import numpy as np
from sklearn.preprocessing import StandardScaler

from gesture_recognition.preprocessors.preprocessor import Preprocessor


class AnglePreprocessor(Preprocessor):
    """
    Angle preprocessor extracts 3D coordinates from joint landmarks, calculates cosine distance
    between each pair of landmarks and flattens the output.
    """

    def preprocess(self, landmarks, *args, **kwargs):
        landmarks = landmarks / np.linalg.norm(landmarks, axis=1).reshape(-1, 1)
        return StandardScaler().fit_transform((landmarks @ landmarks.T)).reshape((-1,))
