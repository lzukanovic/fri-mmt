import cv2
import dlib
from scipy.spatial import distance
import time

# Configuration Parameters
CAMERA_INDEX = 0
FACE_MODEL_PATH = "../models/haarcascade_frontalface_default.xml"
LANDMARK_MODEL_PATH = "../models/shape_predictor_68_face_landmarks.dat"
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
        faceCascade = cv2.CascadeClassifier(FACE_MODEL_PATH)
        dlib_facelandmark = dlib.shape_predictor(LANDMARK_MODEL_PATH)
    except Exception as e:
        print(f"Error loading dlib models: {e}")
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
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )

        for (x, y, w, h) in faces:
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = dlib_facelandmark(gray, dlib_rect)

            # Extracting points for left and right eye
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

            # Drawing the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

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
                
        draw_fps(frame, fps)
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()