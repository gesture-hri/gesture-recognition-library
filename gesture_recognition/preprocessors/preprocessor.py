from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """
    All preprocessors used by GestureRecognizer must override this class by implementing methods below
    """
    @abstractmethod
    def normalize(self, *args, **kwargs):
        """
        This method will be called on mediapipe input to perform standard normalization.
        """
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        This method will be called on mediapipe output to perform additional semantic preprocessing.
        """
        pass
