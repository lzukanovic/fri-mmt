# Drowsiness Detection System
This project offers the following features:
- At startup calibration of the eyes is performed to determine the threshold for the eyes to be considered closed.
- The program is able to detect the eyes of the driver and determine if they are closed or open.
- If the eyes are closed for more than cca. 2 seconds, the driver will be alerted.
- When calibration is performed, the program displays a loading bar on the camera live feed and also displays a loading bar on the attached Sense HAT LED matrix.
- When drowsiness is detected, the program displays a warning message on the camera live feed and also displays a warning message on the attached Sense HAT LED matrix.
- The program displays a FPS counter on the camera live feed.
- The program auto-rotates the LED matrix display according to the Raspberry PI orientation/accelerometer data.
- The program also listens to the Sense HAT joystick events:
    - when joystick is pressed up or down, the LED matrix is rotated accordingly, 
    - when joystick is pressed left or right, the LED matrix is flipped vertically or horizontally,
    - when joystick is pressed, the program recalibrates the persons eye just like in the initial calibration step.
    
    > The joystick events will only be processed if the device is laying flat, otherwise they will be overridden by the auto-rotation feature.
- The program sends the following data to the defined MQTT broker (which is then read by Node-RED and displayed in a dashboard):
    - The calculated eye aspect ratio (EAR) value for all detected faces.
    - The status of drowsiness (EAR over or under the threshold) for all detected faces.

## Environment variables
The environment variables include configuration for the MQTT broker connection.

## Models
The `models` folder contains the pre-trained models for the face detector and the facial landmark detector.
- `shape_predictor_68_face_landmarks.dat` - the facial landmark detector model for dlib. [link](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
- `haarcascade_frontalface_default.xml` - the face detector model for OpenCV Viola-Jones. [link](https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml)
- `lbfmodel.yaml` - the facial landmark detector model for OpenCV. [link](https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml)

## Comparisons
The `comparisons` folder contains simplified versions of the main program, which contain only the logic to detect faces and facial landmarks and to calculate different metrics (EAR, FPS, drowsiness status). These programs are used to compare the performance of different face detection and facial landmark detection algorithms:
- `drowsiness_detector_dlib_dlib.py` - uses the dlib face detector and the dlib facial landmark detector.
- `drowsiness_detector_dlib_opencv.py` - uses the dlib face detector and the OpenCV facial landmark detector.
- `drowsiness_detector_vj_dlib.py` - uses the OpenCV Viola-Jones face detector and the dlib facial landmark detector.
- `drowsiness_detector_vj_opencv.py` - uses the OpenCV Viola-Jones face detector and the OpenCV facial landmark detector.

## Node-RED Flow
The `node_red_flow.json` file contains the Node-RED flow for the dashboard. The flow is used to display the data sent by the program to the MQTT broker.
The dashboard represents the operator's view of all the drivers and their drowsiness status.