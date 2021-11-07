import logging
import os
import sys
import time

import cv2

from gesture_recognition.classification import TFLiteClassifier
from gesture_recognition.gesture_recognizer import GestureRecognizer
from gesture_recognition.preprocessing import *
from video_debugging import FrameSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video file debugger")

if __name__ == "__main__":
    """
    This script is intended to serve two purposes:
        1) For library developers themselves to test its efficiency in real-time video inference conditions.
        2) For library users to be an example of how to use recognizer for real-time video inference.
    It expects the following arguments:
        :param video_path: Optional. If provided inference will run on .mpr4 file stored under the path. If not
        :param model_binary_path: Path to model binary that can be deserialized into classifier instance.
        :param dataset_path: Path to dataset that was used to train classifier. Its purpose is to provide named
        gesture labels for that recognizer will use for logging during inference.
        specified inference will run on live webcam feed.
    """
    try:
        if len(sys.argv) == 4:
            (_script, video_path, model_binary_path, dataset_path) = sys.argv
        else:
            (_script, model_binary_path, dataset_path) = sys.argv
            video_path = 0
    except IndexError:
        raise ValueError("Invalid number of arguments")

    classifier = TFLiteClassifier.from_file(model_binary_path)
    preprocessor = TFLitePreprocessor.from_function(default_preprocessing)
    categories = [path.name for path in os.scandir(dataset_path)]

    pretrained_recognizer = GestureRecognizer(
        classifier=classifier,
        preprocessor=preprocessor,
        categories=categories,
    )
    source = FrameSource(video_path)

    for counter, fps, frame in source:
        start_inference = time.time()
        classification = pretrained_recognizer.recognize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        inference_time = time.time() - start_inference

        seconds_passed = counter / fps
        video_time_stamp = time.strftime("%H:%M:%S", time.gmtime(seconds_passed))

        if video_path != 0:
            # there is not point to show frames during live video debugging.
            cv2.imshow("capture", frame)
            cv2.waitKey(0)

        if classification is not None:
            logger.info(
                f"Recognized {classification[0]} at time {video_time_stamp}. Inference took {inference_time} seconds."
            )

    cv2.destroyAllWindows()
