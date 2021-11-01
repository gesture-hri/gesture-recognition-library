from sklearn.preprocessing import StandardScaler

from gesture_recognition.preprocessors.preprocessor import Preprocessor


class SimplePreprocessor(Preprocessor):
    def preprocess(self, landmarks, *args, **kwargs):
        return StandardScaler().fit_transform(landmarks)
