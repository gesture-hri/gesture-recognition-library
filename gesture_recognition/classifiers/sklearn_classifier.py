from sklearn.model_selection import train_test_split

from gesture_recognition.classifiers.trainable_classifier import TrainableClassifier


class SklearnClassifier(TrainableClassifier):
    def __init__(self, sklearn_model, test_size=None, random_state=None):
        """
        :param sklearn_model: Instantiated sklearn model (LogisticRegression, IsolationForest etc.)
        :param test_size: Parameter controlling train test ration while training.
        :param random_state: Globally saved and used random state might be useful for repeatable results.
        """
        self.sklearn_model = sklearn_model
        self.test_size = test_size
        self.random_state = random_state

    def train(self, samples, labels, *args, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(
            samples,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.sklearn_model.fit(x_train, y_train, *args, **kwargs)
        return self.sklearn_model.score(x_test, y_test)

    def evaluate(self, samples, labels, *args, **kwargs):
        return self.sklearn_model.score(samples, labels, *args, **kwargs)

    def infer(self, samples, *args, **kwargs):
        return self.sklearn_model.predict(samples, *args, **kwargs)

    def infer_probabilities(self, samples, *args, **kwargs):
        try:
            return self.sklearn_model.predict_proba(samples)
        except AttributeError:
            raise TrainableClassifier.Error(
                "Provided classifier incapable of inferring classes probabilities.",
            )
