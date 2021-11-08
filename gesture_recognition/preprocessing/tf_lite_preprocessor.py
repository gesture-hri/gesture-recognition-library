import sys
from typing import List, Tuple, Callable

import numpy as np

from gesture_recognition.preprocessing.preprocessor import Preprocessor


class TFLitePreprocessor(Preprocessor):
    def __init__(self, tf_lite_interpreter, function_as_tf_lite_model=None):
        """
        :param tf_lite_interpreter: TensorflowLite runtime interpreter created from appropriate preprocessing function
        :param function_as_tf_lite_model: This argument will be specified only when creating preprocessor from function.
        It will be used to create binary file when serializing mode. Instances created from binary file will not have
        this attribute set, thus .save_preprocessor() cannot be called on them.
        """
        self.tf_lite_interpreter = tf_lite_interpreter
        self.function_as_tf_lite_model = function_as_tf_lite_model

        self.input_meta = [
            (meta["index"], meta["shape"], meta["dtype"])
            for meta in self.tf_lite_interpreter.get_input_details()
        ]
        self.output_index = self.tf_lite_interpreter.get_output_details()[0]["index"]

    def preprocess(
        self, landmark_vectors: List[np.ndarray], *args, **kwargs
    ) -> np.ndarray:
        """
        Executes behavior of preprocessing function that was used to create this TFLitePreprocessor instance.
        :param landmark_vectors: List of numpy arrays that represent content and order of arguments of function
        used to create TFLitePreprocessor instance.
        :return: Result that would be obtained from preprocessing_function(*landmark_vectors)
        """

        self.tf_lite_interpreter.allocate_tensors()
        for landmark_vector, meta in zip(landmark_vectors, self.input_meta):
            if (
                np.any(landmark_vector.shape != meta[1])
                or landmark_vector.dtype != meta[2]
            ):
                raise ValueError(
                    "Preprocessor expects input vector of shape {} and type {}, but {} and {} was provided".format(
                        meta[1],
                        meta[2],
                        landmark_vector.shape,
                        landmark_vector.dtype,
                    )
                )
            self.tf_lite_interpreter.set_tensor(meta[0], landmark_vector)
        self.tf_lite_interpreter.invoke()
        return self.tf_lite_interpreter.get_tensor(self.output_index)

    def save_preprocessor(self, path):
        """
        :param path: Path under which serialized TFLitePreprocessor binary will be stored.
        """
        if self.function_as_tf_lite_model is None:
            raise AttributeError("Trying to serialize already serialized preprocessor")

        with open(path, "w+b") as preprocessor_binary:
            preprocessor_binary.write(self.function_as_tf_lite_model)

    @classmethod
    def from_function(
        cls, function: Callable[..., np.ndarray], input_shapes: List[Tuple[int, ...]]
    ):
        """
        :param function: Preprocessing function that takes arbitrary number of numpy array as arguments and returns
        single numpy array as a result. Note that this function should only use tensorflow libraries for
        matrix manipulation.
        :param input_shapes: List of shapes of function arguments.
        :return: TFLitePreprocessor instance, that can be serialized into file and used for inference.
        """
        import tensorflow as tf

        tf_function = tf.function(
            function,
            input_signature=[
                tf.TensorSpec(shape=shape, dtype=tf.float32) for shape in input_shapes
            ],
        ).get_concrete_function()
        function_as_tf_lite_model = tf.lite.TFLiteConverter.from_concrete_functions(
            [tf_function]
        ).convert()
        interpreter = tf.lite.Interpreter(model_content=function_as_tf_lite_model)
        return cls(interpreter, function_as_tf_lite_model)

    @classmethod
    def from_file(cls, tf_lite_preprocessor_path):
        """
        :param tf_lite_preprocessor_path: Path to binary file containing serialized TFLitePreprocessor instance.
        :return: TFLitePreprocessor instance, that can used for inference but not re-serialized again.
        """
        if "tensorflow" in sys.modules:
            import tensorflow

            interpreter = tensorflow.lite.Interpreter(
                model_path=tf_lite_preprocessor_path
            )
        else:
            from tflite_runtime.interpreter import Interpreter

            interpreter = Interpreter(model_path=tf_lite_preprocessor_path)
        return cls(interpreter)
