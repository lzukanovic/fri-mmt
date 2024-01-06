import cv2
import dlib
from scipy.spatial import distance
import time
import numpy as np

# Configuration Parameters
CAMERA_INDEX = 0
LANDMARK_MODEL_PATH = "../models/lbfmodel.yaml"
EAR_THRESHOLD = 0.25
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

# Function to display FPS
def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 80, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

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

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Camera with index {CAMERA_INDEX} could not be opened.")
        return

    # Load dlib's face detector and landmark predictor
    try:
        face_detector = dlib.get_frontal_face_detector()
        landmark_detector  = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(LANDMARK_MODEL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Main loop
    prev_frame_time = time.time()
    closed_eye_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break
        
        # FPS
        current_frame_time = time.time()
        fps = 1 / (current_frame_time - prev_frame_time)
        prev_frame_time = current_frame_time

        # Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        if len(faces) > 0:
            # Convert dlib rectangles to OpenCV format
            faces_formatted = []
            for dlib_face in faces:
                x = dlib_face.left()
                y = dlib_face.top()
                w = dlib_face.right() - x
                h = dlib_face.bottom() - y
                faces_formatted.append((x, y, w, h))
            faces_formatted = np.array(faces_formatted, dtype=np.int32)

            # Landmark detection
            _, landmarks = landmark_detector.fit(gray, faces_formatted)

            for (x, y, w, h) in faces_formatted:
                # Draw face bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            for landmark in landmarks:
                # Assuming the landmark indices for eyes are the same as dlib (36-41 for left eye, 42-47 for right eye)
                left_eye = [(int(point[0]), int(point[1])) for point in landmark[0][36:42]]
                right_eye = [(int(point[0]), int(point[1])) for point in landmark[0][42:48]]

                # Drawing the eyes
                draw_eye(frame, left_eye, (0, 255, 0))
                draw_eye(frame, right_eye, (255, 255, 0))

                # Calculating EAR for both eyes
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Check if EAR is below the threshold
                if ear < EAR_THRESHOLD:
                    closed_eye_frame_count += 1
                    if closed_eye_frame_count >= EYE_CLOSED_FRAMES_LIMIT:
                        # Draw drowsiness alert
                        draw_text_center_top(frame, "DROWSINESS DETECTED", 2, 2, (0, 0, 255))
                        draw_text_center_bottom(frame, "ALERT! WAKE UP!", 2, 2, (0, 0, 255))
                else:
                    closed_eye_frame_count = 0  # Reset counter if eyes are not closed
                
        # Display camera feed
        draw_fps(frame, fps)
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()