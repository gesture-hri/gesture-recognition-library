from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """
    All preprocessors used by GestureRecognizer must override this class by implementing methods below
    """

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        This method will be called on numpy array that represent mediapipe output
        to perform additional semantic preprocessing.
        """
        pass
