import numpy as np
from sklearn.preprocessing import StandardScaler

from gesture_recognition.preprocessors.preprocessor import Preprocessor


class SimplePreprocessor(Preprocessor):
    def preprocess(self, mediapipe_output, *args, **kwargs):
        landmarks = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmarks in mediapipe_output
                for landmark in landmarks.landmark
            ],
        )

        return StandardScaler().fit_transform(landmarks)
