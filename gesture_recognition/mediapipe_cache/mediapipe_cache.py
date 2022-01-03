from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class MediaPipeCache(ABC):
    """
    All cache frameworks used by GestureRecognizer to store mediapipe output between repeated experiments
    should derive from this class.
    """

    class Error(Exception):
        """
        Base class for related mediapipe cache output errors and exceptions
        """

        pass

    @abstractmethod
    def initialize(self):
        """
        This method should run all necessary cache setup like directory creation, database connection etc.
        """
        pass

    @abstractmethod
    def store_mediapipe_output(
        self,
        samples: List[Union[np.ndarray, List[np.ndarray]]],
        labels: List[np.int],
        identifier: str,
    ):
        """
        This method should persist mediapipe output in storage that is wrapped by this class.
        :param samples:
        :param labels:
        :param identifier:
        """
        pass

    @abstractmethod
    def retrieve_mediapipe_output(
        self, identifier: str
    ) -> Tuple[List[Union[np.ndarray, List[np.ndarray]]], List[np.int]]:
        """
        This method should retrieve mediapipe output from storage that is wrapped by this class.
        :param identifier:
        :return:
        """
        pass

    @abstractmethod
    def remove_mediapipe_output(self, identifier: str):
        """
        This method should remove mediapipe output from storage that is wrapped by this class.
        :param identifier:
        """
        pass
