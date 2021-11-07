import sys
from typing import Callable

import numpy as np

from gesture_recognition.preprocessing.preprocessor import Preprocessor


class TFLitePreprocessor(Preprocessor):
    HAND_LANDMARKS_SHAPE = [21, 3]
    POSE_LANDMARK_SHAPE = [33, 3]

    def __init__(self, tf_lite_interpreter, function_as_tf_lite_model=None):
        """
        :param tf_lite_interpreter: TensorflowLite runtime interpreter created from appropriate preprocessing function
        :param function_as_tf_lite_model: This argument will be specified only when creating preprocessor from function.
        It will be used to create binary file when serializing mode. Instances created from binary file will not have
        this attribute set, thus .save_preprocessor() cannot be called on them.
        """
        self.tf_lite_interpreter = tf_lite_interpreter
        self.function_as_tf_lite_model = function_as_tf_lite_model

        input_details = self.tf_lite_interpreter.get_input_details()[0]
        output_details = self.tf_lite_interpreter.get_output_details()[0]

        self.input_meta = (
            input_details["index"],
            input_details["shape"],
            input_details["dtype"],
        )
        self.output_meta = (
            output_details["index"],
            output_details["shape"],
            input_details["dtype"],
        )

    def preprocess(self, landmark_vector: np.ndarray, *args, **kwargs):
        landmark_vector = landmark_vector.astype(self.input_meta[2])
        self.tf_lite_interpreter.allocate_tensors()
        if np.any(landmark_vector.shape != self.input_meta[1]):
            raise ValueError(
                "Preprocessor expects input vector of shape {}. Shape {} was provided".format(
                    self.input_meta[1], landmark_vector.shape
                )
            )
        self.tf_lite_interpreter.set_tensor(self.input_meta[0], landmark_vector)
        self.tf_lite_interpreter.invoke()
        return self.tf_lite_interpreter.get_tensor(self.output_meta[0])

    def save_preprocessor(self, path):
        if self.function_as_tf_lite_model is None:
            raise AttributeError("Trying to serialize already serialized preprocessor")

        with open(path, "w+b") as preprocessor_binary:
            preprocessor_binary.write(self.function_as_tf_lite_model)

    @classmethod
    def from_function(
        cls,
        function: Callable[[np.ndarray], np.ndarray],
        input_shape=None,
    ):
        if input_shape is None:
            input_shape = TFLitePreprocessor.HAND_LANDMARKS_SHAPE
        import tensorflow as tf

        tf_function = tf.function(
            function,
            input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)],
        ).get_concrete_function()
        function_as_tf_lite_model = tf.lite.TFLiteConverter.from_concrete_functions(
            [tf_function]
        ).convert()
        interpreter = tf.lite.Interpreter(model_content=function_as_tf_lite_model)
        return cls(interpreter, function_as_tf_lite_model)

    @classmethod
    def from_file(cls, tf_lite_preprocessor_path):
        if "tensorflow" in sys.modules:
            import tensorflow

            interpreter = tensorflow.lite.Interpreter(
                model_path=tf_lite_preprocessor_path
            )
        else:
            from tflite_runtime.interpreter import Interpreter

            interpreter = Interpreter(model_path=tf_lite_preprocessor_path)
        return cls(interpreter)
