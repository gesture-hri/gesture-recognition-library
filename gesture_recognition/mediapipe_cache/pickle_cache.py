import os
import pickle
from typing import List

import numpy as np

from gesture_recognition.mediapipe_cache.mediapipe_cache import MediapipeCache


class PickleCache(MediapipeCache):
    """
    Derived class of MediapipeCache. Implements simplest type of storage based on pickle library and filesystem
    """

    def __init__(self, root_path: str):
        """
        :param root_path: Path to directory you want to store pickled mediapipe outputs in.
        """
        self.root_path = root_path

    def initialize(self):
        """
        Creates directory specified by root_path attribute if not exists.
        """
        os.makedirs(self.root_path, os.O_RDWR, exist_ok=True)

    def store_mediapipe_output(
        self, samples: List[np.ndarray], labels: List[np.ndarray], identifier: str
    ):
        with open(os.path.join(self.root_path, identifier) + ".pkl", "w+b") as file:
            pickle.dump(
                obj=(samples, labels),
                file=file,
            )

    def retrieve_mediapipe_output(self, identifier: str):
        try:
            with open(os.path.join(self.root_path, identifier) + ".pkl", "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise MediapipeCache.Error(
                f"Dataset identified as {identifier} not found in cache"
            )

    def remove_mediapipe_output(self, identifier: str):
        try:
            os.remove(os.path.join(self.root_path, identifier))
        except FileNotFoundError:
            raise MediapipeCache.Error(
                f"Dataset identified as {identifier} not found in cache"
            )
