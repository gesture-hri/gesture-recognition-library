from setuptools import setup, find_packages

setup(
    name='gesture-recognition',
    packages=find_packages(include=['gesture_recognition']),
    version='0.1.0',
    install_requires=[
        'tensorflow==2.6.0',
        'mediapipe==0.8.7.3',
        'opencv-python==4.5.3.56',
        'sklearn==0.0',
        'keras==2.6.0',
        'numpy==1.19.5',
    ],
)
