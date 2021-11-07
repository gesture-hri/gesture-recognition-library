import numpy as np
from sklearn.model_selection import train_test_split

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

    def setup_interpreter_meta(self):
        self.input_meta = [
            (meta["index"], meta["shape"], meta["dtype"])
            for meta in self.tf_lite_interpreter.get_input_details()
        ]
        self.output_meta = [
            (meta["index"], meta["shape"], meta["dtype"])
            for meta in self.tf_lite_interpreter.get_output_details()
        ]

    @classmethod
    def from_file(cls, tf_lite_model_path):
        try:
            from tflite_runtime.interpreter import Interpreter

            interpreter = Interpreter(model_path=tf_lite_model_path)
        except ImportError:
            import tensorflow

            interpreter = tensorflow.lite.Interpreter(model_path=tf_lite_model_path)

        instance = cls()
        instance.tf_lite_interpreter = interpreter
        instance.setup_interpreter_meta()
        return instance

    def train(self, samples, labels, *args, **kwargs):
        if self.keras_model is None or self.training_params is None:
            raise AttributeError(
                "TFLiteClassifier instances that were not instantiated via constructor cannot be trained."
            )
        import tensorflow

        x_train, x_test, y_train, y_test = train_test_split(
            samples,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.keras_model.fit(
            np.array(x_train).astype(np.float32),
            np.array(y_train),
            **self.training_params,
        )
        _, score = self.keras_model.evaluate(
            np.array(x_test).astype(np.float32), np.array(y_test), verbose=0
        )

        self.tf_lite_model = tensorflow.lite.TFLiteConverter.from_keras_model(
            self.keras_model
        ).convert()

        self.tf_lite_interpreter = tensorflow.lite.Interpreter(
            model_content=self.tf_lite_model
        )

        self.setup_interpreter_meta()
        return score

    def evaluate(self, samples, labels, *args, **kwargs):
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

    def infer(self, samples, *args, **kwargs):
        self._invoke_inference(samples)
        return [
            np.argmax(self.tf_lite_interpreter.get_tensor(meta[0]), axis=1)[0]
            for meta in self.output_meta
        ]

    def infer_probabilities(self, samples, *args, **kwargs):
        self._invoke_inference(samples)
        return [
            self.tf_lite_interpreter.get_tensor(meta[0])[0] for meta in self.output_meta
        ]

    def save_classifier(self, path: str):
        with open(path, "w+b") as classifier_binary:
            classifier_binary.write(self.tf_lite_model)
