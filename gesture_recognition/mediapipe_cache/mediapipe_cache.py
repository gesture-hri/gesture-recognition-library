from abc import ABC, abstractmethod


class MediapipeCache(ABC):
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
    def initialize(self, *args, **kwargs):
        """
        This method should run all necessary cache setup like directory creation, database connection etc.
        """
        pass

    @abstractmethod
    def store_mediapipe_output(self, *args, **kwargs):
        """
        This method should persist mediapipe output in storage that is wrapped by this class.
        """
        pass

    @abstractmethod
    def retrieve_mediapipe_output(self, *args, **kwargs):
        """
        This method should retrieve mediapipe output from storage that is wrapped by this class.
        """
        pass

    @abstractmethod
    def remove_mediapipe_output(self, *args, **kwargs):
        """
        This method should remove mediapipe output from storage that is wrapped by this class.
        """
        pass
