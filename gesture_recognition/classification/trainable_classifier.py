from abc import ABC, abstractmethod


class TrainableClassifier(ABC):
    """
    All classifiers used on top of mediapipe must override this class by implementing abstract methods below
    """

    class Error(Exception):
        """
        Base class for related classifier errors and exceptions.
        """

        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        This method will be called by GestureRecognizer to train and evaluate its top classifier
        to semantic preprocessing output.
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        This method will be called by GestureRecognizer to evaluate its top classifier on semantic preprocessing output.
        """
        pass

    @abstractmethod
    def infer(self, *args, **kwargs):
        """
        This method will be called by GestureRecognizer to classify incoming images.
        """
        pass

    @abstractmethod
    def infer_probabilities(self, *args, **kwargs):
        """
        This method can be used for evaluation of more advanced metrics than simple accuracy. Note that some
        classifiers like SVM are incapable of providing probability distribution over classes. Appropriate exception
        should be thrown.
        """
        pass

    @abstractmethod
    def save_classifier(self, path):
        """
        This method will be called by GestureRecognizer during saving process.
        :param path: Path to the directory in which classifier will be saved. Providing appropriate suffix to this path
        is left for overriding.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, path):
        """
        :param path: Path to directory from which classifier will be restored. Providing appropriate suffix to this
        path is left for overriding.
        :return: TrainableClassifier instance restored from file.
        """
        pass
