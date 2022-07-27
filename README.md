# AR mirror for sportive training application

## Introduction

In this repository I show a little project done in the Computer Vision course. The project is carried out also with Alessandro Castellaz. The project consists in studying the MediaPipe library in two gym exercises.

## Basic libraryes used

In order to run the code of this project two main libraries are needed, Opencv and numpy. OpenCV is an open source computer vision and machine learning software library while NumPy is a Python library used for working with arrays and for working in domain of linear algebra, fourier transform, and matrices.
The following commands are run in the command prompt to install Opencv and numpy libraries.

```
pip install opencv-python
pip install numpy
```

## GUI library used

The main file of the project is called GUI.py and in order to run it it's important to have installed the library Tkinter which is a standard library in Python used for GUI application. 
This library can be installed using pip. 
The following command is run in the command prompt to install Tkinter library.

```
pip install tk
```

This command will start downloading and installing packages related to the Tkinter library. Once done, the message of successful installation will be displayed.

## Code execution

Firstly it's important to put all the files in the same folder and from terminal it's necessary to enter in the folder of the project.
So, the user is asked to run the main file called "GUI.py" with the command:

```
python3 GUI.py
```

If the version of python installed is python 2 the program has to be run with the command:

```
python GUI.py
```
and the script GUI.py has to be modified where there is the call of the subroutine _Squat.py_ and _Biceps.py_ changing _python3_ with _python_.

So, after running it, a graphic user interface is shown presenting the two possible exercises the user has to complete. By clicking on one of them, another Python script will be run in order to access the logic of the chosen exercise, then, the user has to follow the instructions that appears on the screen. In practice, what we have exploited is the Python library subproces that will allow to associate each Python script for the exercises, i.e. "Biceps.py" and "Squat.py", to two different subprocesses. Each subprocess is than called each time the user presses the corresponding button on the screen and the corresponding Python script will be run.

## Documentation

All the relevant documentation used to develop this project can be found at the following link:  https://google.github.io/mediapipe/solutions/pose.html
