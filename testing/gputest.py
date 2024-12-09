import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Check if GPU is available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is being used.")
else:
    print("GPU not available, using CPU.")

face_cascade = cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')

model = load_model('../models/facial_landmark_model.h5')

glasses = cv2.imread('../filters/thugglass.png', cv2.IMREAD_UNCHANGED)
if glasses is None:
    print("Error: Glasses PNG not found. Check the file path.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        try:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (96, 96))
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_resized = face_resized / 255.0
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)

            # Predict facial landmarks (runs on GPU)
            prediction = model.predict(face_resized, verbose=0)
            landmarks = prediction[0]

            # Calculate eye positions
            left_eye = (int(landmarks[0] * w) + x, int(landmarks[1] * h) + y)  # Left eye
            right_eye = (int(landmarks[2] * w) + x, int(landmarks[3] * h) + y)  # Right eye

            # Calculate glasses size and position based on eye positions
            glasses_width = int(1.5 * abs(right_eye[0] - left_eye[0]))
            glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])
            glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

            # Determine the top-left corner for overlay (based on eye positions)
            x_offset = left_eye[0] - glasses_width // 2
            y_offset = left_eye[1] - glasses_height // 2

            # Ensure the overlay stays within frame boundaries
            y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + glasses_height)
            x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + glasses_width)
            glasses_y1, glasses_y2 = max(0, -y_offset), min(glasses_height, frame.shape[0] - y_offset)
            glasses_x1, glasses_x2 = max(0, -x_offset), min(glasses_width, frame.shape[1] - x_offset)

            # Overlay the glasses using alpha blending
            for c in range(0, 3):  
                alpha = glasses_resized[glasses_y1:glasses_y2, glasses_x1:glasses_x2, 3] / 255.0
                frame[y1:y2, x1:x2, c] = (frame[y1:y2, x1:x2, c] * (1 - alpha) +
                                          glasses_resized[glasses_y1:glasses_y2, glasses_x1:glasses_x2, c] * alpha)

        except Exception as e:
            print(f"Error processing face: {e}")

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with glasses and FPS
    cv2.imshow('Facial Landmarks with Glasses', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
