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

    @abstractmethod
    def save_preprocessor(self, *args, **kwargs):
        """
        This method will be called by GestureRecognizer during saving process.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, *args, **kwargs):
        """
        This class method will be called during GestureRecognizer deserialization from directory, which requires class
        module to be provided.
        """
        pass
