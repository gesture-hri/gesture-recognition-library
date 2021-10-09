import numpy as np
from sklearn.preprocessing import StandardScaler

from gesture_recognition.preprocessors.preprocessor import Preprocessor


class DefaultPreprocessor(Preprocessor):
    """
    Default preprocessor functionality scope is quite restricted. It simply extracts 3D coordinates of joint landmarks
    and flattens resulting array.
    """

    def normalize(self, image, *args, **kwargs):
        return image

    def preprocess(self, mediapipe_output, *args, **kwargs):
        landmarks = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmarks in mediapipe_output
                for landmark in landmarks.landmark
            ],
        )

        return StandardScaler().fit_transform(landmarks).reshape((-1,))
