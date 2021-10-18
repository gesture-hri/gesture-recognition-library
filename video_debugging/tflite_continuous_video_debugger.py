import logging
import sys
import time

import cv2

from gesture_recognition.classifiers import TFLiteClassifier
from gesture_recognition.gesture_recognizer import GestureRecognizer
from gesture_recognition.preprocessors import *
from video_debugging import FrameSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video file debugger")

if __name__ == "__main__":
    """
    This script serves experimental purposes only, until common serialization for all TrainableClassifier derived classes
    is established
    """
    video_path = sys.argv[1] if len(sys.argv) == 3 else 0
    classifier = TFLiteClassifier.from_tf_lite_model_path(sys.argv[-1])
    pretrained_recognizer = GestureRecognizer(
        classifier=classifier,
        preprocessor=DefaultPreprocessor(),
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
