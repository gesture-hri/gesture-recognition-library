# Gesture Recognition Library
Python library that provides rich yet highly customizable interface for training and evaluation of 
gesture image classifiers on any dataset.

### Development setup
Following steps describe setting up development environment on your local computer.

#### Runtime setup
The recommended way to install language/runtimes is with the `asdf` version manager tool. Follow the instructions [here](https://asdf-vm.com/#/core-manage-asdf-vm) to install it.

Once `asdf` is installed you can install the needed plugins with: 
```shell
asdf plugin-add python
```
Then you can install the required version of Python with: 
```shell
asdf install
```
as the project has a `.tool-version` file that asdf looks at to set the above language runtimes to specific versions when inside this project directory.

For python dependency management, we use `poetry`. Please install it following instructions [here](https://python-poetry.org/docs/#installation). 
After installing `poetry` you can run 
```shell
poetry install
```
to install project dependencies defined in `pyproject.toml` to your setup. 

### Build and installation
Library can be build with two options - with both recognizer training and inference and with inference functionality only.
The former is provided to enable inference in environments where tensorflow cannot be installed. Using the second option
requires previously trained classifier deserialization. To build library with full functionality run:
```shell
python setup.py sdist bdist_wheel
```
To build inference-only version run:
```shell
python setup_lite.py sdist bdist_wheel 
````
Once build is finished find .whl file in dist directory and run:
```shell
pip install path/to/.whl/file 
```

### Code formatting
To format code automatically, run: 
```shell
poetry run black .
```

### Basic Usage
The core functionality of the library is provided by GestureRecognizer class, from gesture_recognizer module.
To instantiate GestureRecognizer three other objects are needed: 

 - `TrainableClassifier` instance - an adapter for an external classifier (sklearn, keras etc.) 
that will be trained classify gestures from mediapipe pose estimation.
 - `Preprocessor` instance - object responsible for additional preprocessing of mediapipe pose estimation before passing
it to classifier.
 - `MediapipeCache` instance - this parameter is optional, however recommended. Currently mediapipe does not provide
batch inference which can be harmful for repeated trainings on large dataset. With MediapipeCache instance pose
estimations are stored and retrieved on repeated trainings.

However `Preprocessor` and `TrainableClassifier` are generic base classes that are easy to override, library is 
designed and optimized to work with it own derived classes `TFLitePreprocessor` and `TFLiteClassifier` respectively.
Therefore, it is highly recommended not to override abstract classes and work the ones provided that take advantage
of highly performant and functional TensorFlow Lite framework.

Example usage might look like this:
```shell
from gesture_recognition.classification import *
from gesture_recognition.gesture_recognizer import GestureRecognizer
from gesture_recognition.mediapipe_cache import PickleCache
from gesture_recognition.preprocessing import TFLitePreprocessor
from gesture_recognition.preprocessing.preprocessing_functions import (
    default_preprocessing,
)


classifier = TFLiteClassifier(...)
preprocessor = TFLitePreprocessor.from_function(
    default_preprocessing, [GestureRecognizer.LandmarkShapes.HAND_LANDMARK_SHAPE]
)
cache = PickleCache('path/to/mediapipe/output')

gesture_recognizer = GestureRecognizer(classifier, preprocessor, cache, hands=True)
gesture_recognizer.train_and_evaluate(dataset_identifier: str, samples: Iterable[np.ndarray], labels: Iterable[np.int])
```

Example above resembles `train_recognizer.py` script from `poc` package. The script can be used train, evaluate and save 
GestureRecognizer instance on any dataset. Dataset should consist of folders named after each gesture to be recognized
containing image files (.jpg, .jpeg, .png formats) that belong to the same gesture label. At this point is has to be stated
that mediapipe expects RGB images. Violation to this assumption might result in poor classification accuracy,
so make sure that source of your images follows RGB format.
Example of usage:
```shell
python train_recognizer.py <dataset_name> path/to/dataset path/to/cache/directory path/to/recognizer/save/destination
```

For more detailed information on available classes, their methods and attributes refer to appropriate class docstring.

### Test Trained Gesture Recognizer With Video Inference
Although the main objective ot this library is to simplify and automate gesture classifiers training, 
`video_debugging` package contains script, `continuous_video_debugger.py` that allows to test 
trained GestureRecognizer instance in real life like scenario. The script can be used on both video (.mp4) file:
```shell
python continuous_video_debugger.py path/to/.mp4/file path/to/trained/recognizer/save/source
```
and live webcam feed:
```shell
python continuous_video_debugger.py path/to/trained/recognizer/save/source
```
In both cases frame timestamp, classification result and inference are logged for each frame mediapipe successfully 
detects human posture. Conversion from BGR format used by opencv to RGB expected by mediapipe is handled automatically
if running inference on live webcam feed.