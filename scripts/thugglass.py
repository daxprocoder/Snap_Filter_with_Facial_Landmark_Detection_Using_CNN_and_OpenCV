import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

import sys
import os

# Load the facial landmark model
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
# Load the pre-trained facial landmark model
model = load_model('models/facial_landmark_model.h5')

# Load the glasses PNG with alpha channel
glasses = cv2.imread('filters/glasses/thugglass.png', cv2.IMREAD_UNCHANGED)
if glasses is None:
    print("Error: Glasses PNG not found. Check the file path.")
    exit()

# Function to apply the glasses filter
def apply_thugglasses_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        try:
            # Crop and preprocess the detected face
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (96, 96))
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_resized = face_resized / 255.0
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)

            # Predict facial landmarks
            prediction = model.predict(face_resized, verbose=0)
            landmarks = prediction[0]

            # Calculate eye positions
            right_eye = (int(landmarks[0] * w) + x, int(landmarks[1] * h) + y)
            left_eye = (int(landmarks[2] * w) + x, int(landmarks[3] * h) + y)

            # Calculate glasses width and height based on distance between eyes
            glasses_width = int(2.5 * abs(right_eye[0] - left_eye[0]))
            glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

            # Resize the glasses to the calculated dimensions
            glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

            # Calculate the overlay position (align glasses over the eyes)
            x_offset = min(left_eye[0], right_eye[0]) - int(0.3 * glasses_width)
            y_offset = min(left_eye[1], right_eye[1]) - int(0.5 * glasses_height)

            # Ensure the overlay stays within frame boundaries
            y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + glasses_height)
            x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + glasses_width)
            glasses_y1, glasses_y2 = max(0, -y_offset), min(glasses_height, frame.shape[0] - y_offset)
            glasses_x1, glasses_x2 = max(0, -x_offset), min(glasses_width, frame.shape[1] - x_offset)

            # Overlay the glasses using alpha blending
            for c in range(0, 3):  # Iterate over color channels (B, G, R)
                alpha = glasses_resized[glasses_y1:glasses_y2, glasses_x1:glasses_x2, 3] / 255.0
                frame[y1:y2, x1:x2, c] = (frame[y1:y2, x1:x2, c] * (1 - alpha) +
                                          glasses_resized[glasses_y1:glasses_y2, glasses_x1:glasses_x2, c] * alpha)

        except Exception as e:
            print(f"Error processing face: {e}")

    return frame
