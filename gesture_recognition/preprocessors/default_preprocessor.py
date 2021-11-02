from sklearn.preprocessing import StandardScaler

from gesture_recognition.preprocessors.preprocessor import Preprocessor


class DefaultPreprocessor(Preprocessor):
    """
    Default preprocessor functionality scope is quite restricted. It simply extracts 3D coordinates of joint landmarks
    and flattens resulting array.
    """

    def preprocess(self, landmarks, *args, **kwargs):
        return StandardScaler().fit_transform(landmarks).reshape((-1,))
