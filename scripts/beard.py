import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

# Load the facial landmark model
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
model = load_model('models/facial_landmark_model.h5')

# Load the beard filter image
beard = cv2.imread('filters/beard/white beard.png', cv2.IMREAD_UNCHANGED)
if beard is None:
    raise FileNotFoundError("Error: Beard image not found. Check the file path.")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Function to overlay the beard filter on the frame
def apply_beard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        try:
            beard_width = int(w)
            beard_height = int(beard_width * beard.shape[0] / beard.shape[1])
            beard_resized = cv2.resize(beard, (beard_width, beard_height))

            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (96, 96))
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_resized = face_resized / 255.0
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)

            prediction = model.predict(face_resized, verbose=0)
            landmarks = prediction[0]

            if len(landmarks) > 25:
                lip_top = (int(landmarks[20] * w) + x, int(landmarks[21] * h) + y)
                beard_offset = lip_top[1] - 27

                x_offset = x
                y_offset = beard_offset

                y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + beard_height)
                x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + beard_width)
                beard_y1, beard_y2 = max(0, -y_offset), min(beard_height, frame.shape[0] - y_offset)
                beard_x1, beard_x2 = max(0, -x_offset), min(beard_width, frame.shape[1] - x_offset)

                for c in range(0, 3):
                    alpha = beard_resized[beard_y1:beard_y2, beard_x1:beard_x2, 3] / 255.0
                    frame[y1:y2, x1:x2, c] = (
                        frame[y1:y2, x1:x2, c] * (1 - alpha) +
                        beard_resized[beard_y1:beard_y2, beard_x1:beard_x2, c] * alpha
                    )
        except Exception as e:
            print(f"Error applying beard: {e}")
    return frame
