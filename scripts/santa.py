import cv2
import numpy as np
from tensorflow.keras.models import load_model

import sys
import os

# Load the facial landmark model
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
# Load the facial landmark model
model = load_model('models/facial_landmark_model.h5')

# Load the Santa hat image
santa_hat = cv2.imread('filters/santa/santahat.png', cv2.IMREAD_UNCHANGED)
if santa_hat is None:
    print("Error: Santa hat image not found. Check the file path.")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Function to overlay the Santa hat on the frame
def apply_santa_hat(frame):
    face_coords = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in face_coords:
        try:
            # Increase the size of the Santa hat (scale by 1.2)
            scale_factor = 1.2  # Increase the size by 20%
            hat_width = int(w * scale_factor)  # Scale width
            hat_height = int(hat_width * santa_hat.shape[0] / santa_hat.shape[1])  # Scale height
            santa_hat_resized = cv2.resize(santa_hat, (hat_width, hat_height))

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

            # Ensure there are enough landmarks
            if len(landmarks) > 25:
                # Get the coordinates of the top of the head (above the eyes)
                head_top = (int(landmarks[10] * w) + x, int(landmarks[11] * h) + y)  # Example coordinates for head top

                # Adjust the Santa hat position above the head
                hat_offset = head_top[1] - hat_height - 10  # Move the hat above the top of the head

                # Calculate the X and Y position for the Santa hat
                x_offset = x
                y_offset = hat_offset

                # Ensure the hat stays within frame boundaries
                y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + hat_height)
                x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + hat_width)
                hat_y1, hat_y2 = max(0, -y_offset), min(hat_height, frame.shape[0] - y_offset)
                hat_x1, hat_x2 = max(0, -x_offset), min(hat_width, frame.shape[1] - x_offset)

                # Overlay the Santa hat using alpha blending
                for c in range(0, 3):
                    alpha = santa_hat_resized[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255.0
                    frame[y1:y2, x1:x2, c] = (
                        frame[y1:y2, x1:x2, c] * (1 - alpha) +
                        santa_hat_resized[hat_y1:hat_y2, hat_x1:hat_x2, c] * alpha
                    )
            else:
                print("Insufficient landmarks for Santa hat application.")
        except Exception as e:
            print(f"Error applying Santa hat: {e}")
    return frame