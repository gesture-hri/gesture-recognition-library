import queue
from queue import Queue
from threading import Thread
from typing import Iterator, Union, Tuple

import cv2
import numpy as np


class FrameSource(Iterator):
    """
    This class is and adapter for reading video file/webcam input and providing it to gesture recognizer
    during video analysis and debugging.
    """
    def __init__(self, video_path: Union[str, int] = 0, flush=False):
        """
        :param video_path: Path to .mp4 file or system webcam index
        :param flush: Setting this parameter allows frame buffer reader thread to empty the buffer once it is full
        so that every __next__() call results in the latest frame available. Use it only if __next__ calls
        happen irregularly or frame processing is slow.
        """
        self.capture = cv2.VideoCapture(video_path)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        self.block_buffer_reader = not flush
        self.counter = 0

        self.buffer = Queue(1)
        self.buffer_reader = Thread(target=self._read_buffer, daemon=True)

    def __iter__(self):
        self.buffer_reader.start()
        return self

    def __next__(self) -> Tuple[int, float, np.ndarray]:
        """
        :return: Tuple of number of frames already read, video source FPS property, frame in numpy array format
        """
        available, counter, frame = self.buffer.get()
        if not available:
            self.capture.release()
            raise StopIteration
        return self.counter, self.fps, frame

    def _read_buffer(self):
        while True:
            available, frame = self.capture.read()
            self.counter += 1
            try:
                self.buffer.put(block=self.block_buffer_reader, item=(available, self.counter, frame))
            except queue.Full:
                self.buffer.get()
                self.buffer.put(item=(available, self.counter, frame))
            if not available:
                break
