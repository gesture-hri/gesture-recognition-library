from typing import Callable, Dict, List, Union

import numpy as np
import tensorflow as tf

from gesture_recognition.classification import TFLiteClassifier
from gesture_recognition.gesture_recognizer.gesture_recognizer import GestureRecognizer
from gesture_recognition.mediapipe_cache import MediaPipeCache
from gesture_recognition.preprocessing import TFLitePreprocessor
from gesture_recognition.preprocessing.preprocessing_functions import (
    default_preprocessing,
)


class GestureRecognizerBuilder:
    """
    Automates process of GestureRecognizer instances creation with TFLitePreprocessor and TFLiteClassifier instances as
    preprocessor and classifier attributes respectively. Developers familiar with machine learning who want to
    experiment with advanced keras architectures or use their own derived class as preprocessor or classifier should use
    GestureRecognizer constructor API directly. This class automatically builds predefined keras architecture so that
    developer does not need neither to use nor understand keras/tensorflow API. Keras architectures provided by this
    class are simple but should be absolutely sufficient for majority of cases as they are built to work
    with low-dimensional data produced by Mediapipe which pose-estimation capabilities are of high quality.
    """

    _MODE_SHAPES_MAPPER = {
        "hand": [GestureRecognizer.LandmarkShapes.HAND_LANDMARK_SHAPE],
        "pose": [GestureRecognizer.LandmarkShapes.POSE_LANDMARK_SHAPE],
    }

    _MODE_NEURON_COUNT_MAPPER = {
        "hand": [128],
        "pose": [128],
    }

    def __init__(self, num_classes: int, mode: str):
        """
        :param num_classes: Number of classes that created GestureRecognizer instance is expected to
        distinguish between.
        :param mode: This attribute has the same meaning as 'mode' attribute of GestureRecognizer. It will be
        directly passed to GestureRecognizer constructor.
        """

        if mode not in self._MODE_SHAPES_MAPPER:
            raise ValueError(
                f"Mode: {mode} is not supported. Supported modes are: {', '.join(self._MODE_SHAPES_MAPPER.keys())}."
            )
        self.mode = mode
        self.num_classes = num_classes
        self.preprocessor = TFLitePreprocessor.from_function(
            default_preprocessing, self._MODE_SHAPES_MAPPER[self.mode]
        )

        self.compilation_config = {
            "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["acc"],
        }

        self.training_config = {
            "verbose": 0,
            "epochs": 100,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor="acc", patience=5, min_delta=0.005
                )
            ],
        }

        self.test_size = 0.2
        self.random_state = 42

        self.categories = None
        self.cache = None
        self.complexity = "low"

    def set_complexity(self, complexity: str):
        """
        :param complexity: This attribute has the same meaning as 'complexity' attribute of GestureRecognizer. It will
        be directly passed to GestureRecognizer constructor.
        """
        self.complexity = complexity
        return self

    def set_preprocessing(
        self, preprocessing: Callable[..., Union[np.ndarray, List[np.ndarray]]]
    ):
        """
        :param preprocessing: Preprocessing function that takes arbitrary number of numpy array as arguments, and
        returns single numpy array or tuple of arbitrary number of numpy arrays as a result.
        Note that this function should only use tensorflow libraries for matrix manipulation or it will fail to
        serialize into TensorFlow Lite format.
        """
        self.preprocessor = TFLitePreprocessor.from_function(
            preprocessing, self._MODE_SHAPES_MAPPER[self.mode]
        )
        return self

    def set_keras_compilation_params(self, keras_compilation_params: Dict[str, any]):
        """
        :param keras_compilation_params: Parameters used to compile keras model of classifier attribute of returned
        GestureRecognizer instance.
        """
        for key, val in keras_compilation_params.items():
            self.compilation_config[key] = val
        return self

    def set_keras_training_params(self, keras_training_params: Dict[str, any]):
        """
        :param keras_training_params: Parameters used to train keras model of classifier attribute of returned
        GestureRecognizer instance.
        """
        for key, val in keras_training_params.items():
            self.training_config[key] = val
        return self

    def set_train_test_split(self, test_size, random_state):
        """
        :param test_size: Fraction of training data used for classifier evaluation after training.
        :param random_state: Random seed used during train - test split of training data.
        """
        self.test_size = test_size
        self.random_state = random_state
        return self

    def set_categories(self, categories: List[any]):
        """
        :param categories: List of objects associated with gesture numeric label.
        """
        self.categories = categories
        return self

    def set_cache(self, cache: MediaPipeCache):
        """
        :param cache: MediapipeCache instance responsible for storage of mediapipe application
        result on training dataset.
        """
        self.cache = cache
        return self

    def build_recognizer(self):
        """
        :return: GestureRecognizer instance created according to set parameters.
        """
        keras_model = self._create_keras_model()
        classifier = TFLiteClassifier(
            keras_model=keras_model,
            keras_training_params=self.training_config,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        return GestureRecognizer(
            classifier=classifier,
            preprocessor=self.preprocessor,
            cache=self.cache,
            categories=self.categories,
            mode=self.mode,
            complexity=self.complexity,
        )

    def _create_keras_model(self):
        input_layers = [
            tf.keras.Input(shape=shape)
            for _index, shape in self.preprocessor.output_meta
        ]
        analyzer_layers = [
            tf.keras.layers.Dense(neuron_count, activation="relu")(input_layer)
            for neuron_count, input_layer in zip(
                self._MODE_NEURON_COUNT_MAPPER[self.mode], input_layers
            )
        ]
        concatenation_layer = tf.keras.layers.Concatenate(axis=1)(analyzer_layers)
        flatten_layer = tf.keras.layers.Flatten()(concatenation_layer)
        classification_layer = tf.keras.layers.Dense(
            self.num_classes, activation="softmax"
        )(flatten_layer)

        model = tf.keras.models.Model(inputs=input_layers, outputs=classification_layer)
        model.compile(**self.compilation_config)
        return model
