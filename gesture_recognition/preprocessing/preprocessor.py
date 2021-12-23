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
    def save_preprocessor(self, path):
        """
        This method will be called by GestureRecognizer during saving process.
        :param path: Path to the directory in which preprocessor will be saved. Providing appropriate suffix to this
        path is left for overriding.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, path):
        """
        :param path: Path to directory from which preprocessor will be restored. Providing appropriate suffix to this
        path is left for overriding.
        :return: Preprocessor instance restored from file.
        """
        pass
