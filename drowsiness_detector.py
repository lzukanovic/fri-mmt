import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import time
from sense_hat import SenseHat, ACTION_RELEASED
import paho.mqtt.client as mqtt
import threading
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sense HAT setup
sense = SenseHat()
sense.low_light = True

# MQTT setup
mqtt_client_name = os.getenv("MQTT_CLIENT_NAME")
mqtt_broker = os.getenv("MQTT_BROKER")
mqttc = mqtt.Client(mqtt_client_name)
mqttc.username_pw_set(os.getenv("MQTT_USER"), os.getenv("MQTT_PASSWORD"))

# Global values
connected_to_mqtt = False
recalibrate = False
face_data = [] # Array of faces and their EAR values and drowsiness status

# Function to handle MQTT connection
def on_connect(client, userdata, flags, rc):
    global connected_to_mqtt
    if rc == 0:
        print("Connected to MQTT Broker!")
        connected_to_mqtt = True
    else:
        print("Failed to connect to MQTT Broker, return code %d\n", rc)
        connected_to_mqtt = False

# MQTT connection setup
mqttc.on_connect = on_connect
mqttc.connect(mqtt_broker)
mqttc.loop_start()

# MQTT Publishing Thread
def mqtt_publish_thread():
    global connected_to_mqtt, face_data
    while True:
        if connected_to_mqtt:
            face_data_json = json.dumps(face_data)
            mqttc.publish(mqtt_client_name + "FACE/", face_data_json)
        time.sleep(1)

# Configuration Parameters
CAMERA_INDEX = 0
LANDMARK_MODEL_PATH = "./models/shape_predictor_68_face_landmarks.dat"
EAR_THRESHOLD = 0.25
CALIBRATION_FRAMES = 30  # Number of frames for calibration
EYE_CLOSED_FRAMES_LIMIT = 4  # Number of consecutive frames to consider for drowsiness (e.g., 4 frames at 1.6fps is 2.5 seconds)

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to draw eye landmarks
def draw_eye(frame, eye, bgr_color):
    for i in range(len(eye)):
        next_point = (i + 1) % len(eye)  # loop to the start at the end
        cv2.line(frame, eye[i], eye[next_point], bgr_color, 1)

# Function to draw progress bar
def draw_progress_bar(frame, progress, total):
    width = frame.shape[1]
    progress_width = int((progress / total) * width)
    cv2.rectangle(frame, (0, frame.shape[0] - 20), (progress_width, frame.shape[0]), (255, 255, 255), -1)
    cv2.putText(frame, f"Calibrating: {int((progress / total) * 100)}%",
                (10, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

# Function to display FPS
def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 80, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

# Function to display a spinner for calibration
def display_calibration_spinner(progress, total):
    X = [255, 255, 255]  # White color for progress
    O = [0, 0, 0]  # Off

    # Initialize all LEDs to off
    progress_indicator = [O for _ in range(64)]

    # Calculate the number of LEDs to light up based on progress
    num_leds_lit = int((progress / total) * 64)

    # Update the LEDs that should be lit
    for i in range(num_leds_lit):
        progress_indicator[i] = X

    sense.set_pixels(progress_indicator)

# Function to display a red flashing exclamation mark
def display_exclamation_mark():
    R = [255, 0, 0]  # Red color
    O = [0, 0, 0]  # Off

    exclamation_mark = [
    O, O, O, R, R, O, O, O,
    O, O, O, R, R, O, O, O,
    O, O, O, R, R, O, O, O,
    O, O, O, R, R, O, O, O,
    O, O, O, R, R, O, O, O,
    O, O, O, R, R, O, O, O,
    O, O, O, O, O, O, O, O,
    O, O, O, R, R, O, O, O
    ]

    sense.set_pixels(exclamation_mark)
    time.sleep(0.3)
    sense.clear()
    time.sleep(0.3)
 
# Rotate the LED matrix based on accelerometer axis
def adjust_display_based_on_accelerometer():
    acceleration = sense.get_accelerometer_raw()
    x = acceleration['x']
    y = acceleration['y']
    z = acceleration['z']

    if abs(x) > abs(y) and abs(x) > abs(z):
        # Sense HAT is tilted towards the x-axis
        if x > 0:
            sense.set_rotation(270)
        else:
            sense.set_rotation(90)
    elif abs(y) > abs(x) and abs(y) > abs(z):
        # Sense HAT is tilted towards the y-axis
        if y > 0:
            sense.set_rotation(0)
        else:
            sense.set_rotation(180)
    # Else the Sense HAT is facing up or down, do nothing
    # When it is flat we want to control the rotation using the joystick
    # The default rotation is 0

def pushed_up(event):
    if event.action == ACTION_RELEASED:
        print("up: rotate +90")
        sense.rotation = (sense.rotation + 90) % 360

def pushed_down(event):
    if event.action == ACTION_RELEASED:
        print("down: rotate -90")
        sense.rotation = (sense.rotation - 90) % 360

def pushed_left(event):
    if event.action == ACTION_RELEASED:
        print("left: flip horizontally")
        sense.flip_h()

def pushed_right(event):
    if event.action == ACTION_RELEASED:
        print("right: flip vertically")
        sense.flip_v()

def pushed_middle(event):
    global recalibrate
    if event.action == ACTION_RELEASED:
        print("middle: recalibrate")
        recalibrate = True

def draw_text_center_top(frame, text, font_scale, font_thickness, color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 20  # 20 pixels from the top
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, font_scale, color, font_thickness)

def draw_text_center_bottom(frame, text, font_scale, font_thickness, color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 20  # 20 pixels from the bottom
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, font_scale, color, font_thickness)

# Calibration for EAR threshold
def calibrate(face_detector, dlib_facelandmark, cap):
    print("Calibrating... Please keep your eyes open normally.")
    sense.clear()
    ear_list = []
    for i in range(CALIBRATION_FRAMES):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break

        # Fix orientation
        adjust_display_based_on_accelerometer()
        
        # Display calibration message and spinner
        draw_progress_bar(frame, i + 1, CALIBRATION_FRAMES)
        display_calibration_spinner(i + 1, CALIBRATION_FRAMES)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        for face in faces:
            landmarks = dlib_facelandmark(gray, face)
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
            ear_list.append((calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0)
        
        cv2.imshow("Drowsiness Detector", frame)
        cv2.waitKey(1)

    sense.clear() 
    if len(ear_list) > 0:
        calculated_threshold = np.mean(ear_list) * 0.8  # 80% of the average EAR
    else:
        print("No faces detected during calibration, using default threshold...")
        calculated_threshold = EAR_THRESHOLD

    print(f"Calibration completed. EAR threshold set to {calculated_threshold:.2f}")
    return calculated_threshold

def main():
    global recalibrate
    global connected_to_mqtt
    global face_data

    # Joystick listeners
    sense.stick.direction_up = pushed_up
    sense.stick.direction_down = pushed_down
    sense.stick.direction_left = pushed_left
    sense.stick.direction_right = pushed_right
    sense.stick.direction_middle = pushed_middle

    # Initialize the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Camera with index {CAMERA_INDEX} could not be opened.")
        return
    
    # Start MQTT Publishing Thread
    publish_thread = threading.Thread(target=mqtt_publish_thread)
    publish_thread.daemon = True  # This makes the thread exit when the main program exits
    publish_thread.start()

    # Load dlib's face detector and landmark predictor
    try:
        face_detector = dlib.get_frontal_face_detector()
        dlib_facelandmark = dlib.shape_predictor(LANDMARK_MODEL_PATH)
    except Exception as e:
        print(f"Error loading dlib models: {e}")
        return
    
    # Initial Calibration
    calculated_threshold = calibrate(face_detector, dlib_facelandmark, cap)

    # Main loop
    prev_frame_time = time.time()
    closed_eye_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break

        # Recalibrate
        if recalibrate:
            recalibrate = False
            calculated_threshold = calibrate(face_detector, dlib_facelandmark, cap)

        # Fix orientation
        adjust_display_based_on_accelerometer()
        
        # FPS
        current_frame_time = time.time()
        fps = 1 / (current_frame_time - prev_frame_time)
        prev_frame_time = current_frame_time

        # Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        face_data = [] # Reset face data
        for face in faces:
            landmarks = dlib_facelandmark(gray, face)

            # Extracting points for left and right eye
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

            # Drawing the eyes
            draw_eye(frame, left_eye, (0, 255, 0))
            draw_eye(frame, right_eye, (255, 255, 0))

            # Calculating EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below the threshold
            drowsy = 0
            if ear < calculated_threshold:
                closed_eye_frame_count += 1
                if closed_eye_frame_count >= EYE_CLOSED_FRAMES_LIMIT:
                    # Draw drowsiness alert
                    draw_text_center_top(frame, "DROWSINESS DETECTED", 2, 2, (0, 0, 255))
                    draw_text_center_bottom(frame, "ALERT! WAKE UP!", 2, 2, (0, 0, 255))
                    # Display exclamation mark on LED matrix
                    display_exclamation_mark()
                    # Set drowsiness status to True
                    drowsy = 1
            else:
                closed_eye_frame_count = 0  # Reset counter if eyes are not closed
                # Set drowsiness status to False
                drowsy = 0
            
            # Add face data to array for publishing
            face_data.append((ear, drowsy))
                
        draw_fps(frame, fps)
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()