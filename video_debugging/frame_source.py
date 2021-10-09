from threading import Thread
from typing import Iterator, Union, Tuple

import cv2
import numpy as np


class FrameSource(Iterator):
    """
    This class is and adapter for reading video file/webcam input and providing it to gesture recognizer
    during video analysis and debugging.
    """
    def __init__(self, video_path: Union[str, int], flush=False):
        """
        :param video_path: Path to .mp4 file or system webcam index
        :param flush: Setting this parameter launches daemonic thread constantly flushing webcam buffer
        so that every __next__() call results in the latest frame available. Use it only if __next__ calls
        happen irregularly or frame processing is slow.
        """
        self.video_path = video_path
        self.capture: cv2.VideoCapture = None
        self.flusher: Thread = None
        self.flush = flush
        self.fps = 0.0
        self.counter = 0

        if self.flush:
            self.flusher = Thread(target=self._flush, daemon=True)

    def __iter__(self):
        self.capture = cv2.VideoCapture(self.video_path)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.flush:
            self.flusher.start()
        return self

    def __next__(self) -> Tuple[int, float, np.ndarray]:
        available, frame = self.capture.read()
        if not available:
            self.flush = False
            self.capture.release()
            raise StopIteration
        self.counter += 1
        return self.counter, self.fps, frame

    def _flush(self):
        while self.flush:
            self.capture.grab()
