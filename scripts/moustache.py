import cv2
import numpy as np
from tensorflow.keras.models import load_model

import sys
import os

# Load the facial landmark model
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
# Load the facial landmark model
model = load_model('models/facial_landmark_model.h5')

# Load the moustache filter image
moustache = cv2.imread('filters/moustache/light.png', cv2.IMREAD_UNCHANGED)
if moustache is None:
    print("Error: Moustache image not found. Check the file path.")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Function to overlay the moustache filter on the frame
def apply_moustache(frame):
    face_coords = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in face_coords:
        try:
            # Resize the moustache image to match the face size
            moustache_width = int(w)
            moustache_height = int(moustache_width * moustache.shape[0] / moustache.shape[1])
            moustache_resized = cv2.resize(moustache, (moustache_width, moustache_height))

            # Predict facial landmarks
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (96, 96))  # Resize face region to match model input size
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            face_resized = face_resized / 255.0  # Normalize the image
            face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
            face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

            # Predict landmarks from the model
            prediction = model.predict(face_resized, verbose=0)
            landmarks = prediction[0]

            # Print landmarks shape and values for debugging
            # print(f"Landmarks shape: {landmarks.shape}")
            # print(f"Landmarks values: {landmarks}")

            # Ensure there are enough landmarks
            if len(landmarks) > 25:
                # Get the coordinates of the upper lip (top of the lips)
                lip_top = (int(landmarks[20] * w) + x, int(landmarks[21] * h) + y)  # Lip top
                lip_bottom = (int(landmarks[24] * w) + x, int(landmarks[25] * h) + y)  # Lip bottom

                # Adjust the moustache position above the lips
                moustache_offset = lip_top[1] - 74  # Move the moustache above the top lip

                # Calculate the X and Y position for the moustache
                x_offset = x
                y_offset = moustache_offset

                # Ensure the moustache stays within frame boundaries
                y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + moustache_height)
                x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + moustache_width)
                moustache_y1, moustache_y2 = max(0, -y_offset), min(moustache_height, frame.shape[0] - y_offset)
                moustache_x1, moustache_x2 = max(0, -x_offset), min(moustache_width, frame.shape[1] - x_offset)

                # Overlay the moustache using alpha blending
                for c in range(0, 3):
                    alpha = moustache_resized[moustache_y1:moustache_y2, moustache_x1:moustache_x2, 3] / 255.0
                    frame[y1:y2, x1:x2, c] = (
                        frame[y1:y2, x1:x2, c] * (1 - alpha) +
                        moustache_resized[moustache_y1:moustache_y2, moustache_x1:moustache_x2, c] * alpha
                    )
            else:
                print("Insufficient landmarks for moustache application.")
        except Exception as e:
            print(f"Error applying moustache: {e}")
    return frame
