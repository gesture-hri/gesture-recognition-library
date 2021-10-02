from abc import ABC, abstractmethod


# TODO: Add save()/pickle() method to enable custom saving/pickling in GestureRecognizer()
class TrainableClassifier(ABC):
    """
    All classifiers used on top of mediapipe must override this class by implementing abstract methods below
    """
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
        This method can be used for evaluation of more advanced metrics than simple accuracy.
        """
        pass
