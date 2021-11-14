from setuptools import setup, find_packages

setup(
    name="gesture-recognition-lite",
    packages=find_packages(include=["gesture_recognition.*"]),
    version="0.0.1",
    install_requires=[
        "mediapipe==0.8.8.1",
        "numpy==1.19.5",
        "tflite_runtime==2.5.0",
    ],
)
