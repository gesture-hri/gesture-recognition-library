import os.path
import sys
from typing import List

import numpy as np

from gesture_recognition.classification.trainable_classifier import TrainableClassifier


class TFLiteClassifier(TrainableClassifier):
    def __init__(
        self,
        keras_model=None,
        keras_training_params=None,
        test_size=None,
        random_state=None,
    ):
        """
        :param keras_model: Compiled instance of  keras model.
        :param test_size: Parameter controlling train test ration while training.
        :param keras_training_params: Dictionary of parameters (learning rate, batch size etc.)
        needed to call keras model fit method.
        :param random_state: Globally saved and used random state might be useful for repeatable results.
        """

        self.keras_model = keras_model
        self.test_size = test_size
        self.random_state = random_state
        self.training_params = keras_training_params

        self.tf_lite_model = None
        self.tf_lite_interpreter = None
        self.input_meta = None
        self.output_meta = None

    def _setup_interpreter_meta(self):
        self.input_meta = [
            (meta["index"], meta["shape"], meta["dtype"])
            for meta in self.tf_lite_interpreter.get_input_details()
        ]
        self.output_meta = [
            (meta["index"], meta["shape"], meta["dtype"])
            for meta in self.tf_lite_interpreter.get_output_details()
        ]

    @classmethod
    def from_file(cls, gesture_recognizer_path: str):
        tf_lite_classifier_path = os.path.join(
            gesture_recognizer_path, "classifier.tflite"
        )
        if "tensorflow" in sys.modules:
            import tensorflow

            interpreter = tensorflow.lite.Interpreter(
                model_path=tf_lite_classifier_path
            )
        else:
            from tflite_runtime.interpreter import Interpreter

            interpreter = Interpreter(model_path=tf_lite_classifier_path)

        instance = cls()
        instance.tf_lite_interpreter = interpreter
        instance._setup_interpreter_meta()
        return instance

    @staticmethod
    def _adapt_input_shape(x, y):
        if not x:
            return np.array(x), np.array(y)

        if len(x[0]) == 1:
            return np.array([sample[0] for sample in x]), np.array(y)

        x_reshaped = [[] for _ in x[0]]
        for sample in enumerate(x):
            for idx, input_elem in enumerate(sample):
                x_reshaped[idx].append(sample)

        x_reshaped = [np.array(input_samples) for input_samples in x_reshaped]
        return x_reshaped, np.array(y)

    def train(self, samples: List[np.ndarray], labels: List[np.int]):
        if self.keras_model is None or self.training_params is None:
            raise AttributeError(
                "TFLiteClassifier instances that were not instantiated via constructor cannot be trained."
            )
        import tensorflow
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(
            samples,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        x_train, y_train = self._adapt_input_shape(x_train, y_train)
        self.keras_model.fit(x_train, y_train, **self.training_params)

        x_test, y_test = self._adapt_input_shape(x_test, y_test)
        _, score = self.keras_model.evaluate(x_test, y_test, verbose=0)

        self.tf_lite_model = tensorflow.lite.TFLiteConverter.from_keras_model(
            self.keras_model
        ).convert()

        self.tf_lite_interpreter = tensorflow.lite.Interpreter(
            model_content=self.tf_lite_model
        )

        self._setup_interpreter_meta()
        return score

    def evaluate(self, samples: List[np.ndarray], labels: List[np.int]):
        _, score = self.keras_model.evaluate(samples, labels)
        return score

    def _invoke_inference(self, samples):
        if len(self.input_meta) != len(samples):
            raise ValueError(
                f"Invalid number of input tensors. {len(self.input_meta)} expected, but {len(samples)} given"
            )

        self.tf_lite_interpreter.allocate_tensors()
        for meta, tensor in zip(self.input_meta, samples):
            if np.any(meta[1][1:] != tensor.shape):
                raise ValueError(
                    f"Shape mismatch for input {meta[0]}. {meta[1][1:]} expected but {tensor.shape} given."
                )

            if meta[2] != tensor.dtype:
                try:
                    tensor = tensor.astype(meta[2])
                except Exception:
                    raise ValueError(
                        f"Couldn't convert type {tensor.dtype} to {meta[2]} which is required by classifier."
                    )

            self.tf_lite_interpreter.set_tensor(meta[0], [tensor])

        self.tf_lite_interpreter.invoke()

    def infer(self, samples: List[np.ndarray]):
        self._invoke_inference(samples)
        return [
            np.argmax(self.tf_lite_interpreter.get_tensor(meta[0]), axis=1)[0]
            for meta in self.output_meta
        ]

    def infer_probabilities(self, samples: List[np.ndarray]):
        self._invoke_inference(samples)
        return [
            self.tf_lite_interpreter.get_tensor(meta[0])[0] for meta in self.output_meta
        ]

    def save_classifier(self, gesture_recognizer_path: str):
        classifier_path = os.path.join(gesture_recognizer_path, "classifier.tflite")
        with open(classifier_path, "w+b") as classifier_binary:
            classifier_binary.write(self.tf_lite_model)
