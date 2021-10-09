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
To build library simply run:
```shell
poetry build
```
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

 - TrainableClassifier instance - an adapter for an external classifier (sklearn, keras etc.) 
that will be trained classify gestures from mediapipe pose estimation.
 - Preprocessor instance - object responsible for additional preprocessing of mediapipe pose estimation before passing
it to classifier.
 - MediapipeCache instance - this parameter is optional, however recommended. Currently mediapipe does not provide
batch inference which can be harmful for repeated trainings on large dataset. With MediapipeCache instance pose
estimations are stored and retrieved on repeated trainings.

Example usage might look like this:
```shell
from sklearn.ensemble import RandomForestClassifier

from gesture_recognition.classifiers import SklearnClassifier
from gesture_recognition.gesture_recognizer import GestureRecognizer
from gesture_recognition.mediapipe_cache import PickleCache
from gesture_recognition.preprocessors import DefaultPreprocessor


classifier = SklearnClassifier(RandomForestClassifier(), test_size=0.2)
preprocessor = DefaultPreprocessor()
cache = PickleCache('path/to/mediapipe/output')

gesture_recognizer = GestureRecognizer(classifier, preprocessor, cache, hands=True)
gesture_recognizer.train_and_evaluate(dataset_identifier: str, samples: Iterable[np.ndarray], labels: Iterable[np.int])
```
Two examples similar to the one above are provided in poc package, which is a part of this repository, but not a library
itself. Examples include GestureRecognizer usage on rock-paper-scissors and ASL alphabet datasets. 

For more detailed information on available classes, their methods and attributes refer to appropriate class docstring.