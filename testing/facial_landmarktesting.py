import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')


# Load the pre-trained model for facial landmarks (if available)
model = load_model('../models/facial_landmark_model.h5')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Resize the face to 96x96 and convert it to grayscale
        face_resized = cv2.resize(face, (96, 96))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Normalize the face image to [0, 1]
        face_resized = face_resized / 255.0
        
        # Expand dimensions to match the input shape of the model
        face_resized = np.expand_dims(face_resized, axis=-1)
        face_resized = np.expand_dims(face_resized, axis=0)
        
        # Make prediction (facial landmarks or other attributes)
        prediction = model.predict(face_resized)
        landmarks = prediction[0]  # Get the prediction for the current face
        
        # Scale the predicted values to match the size of the detected face
        for i in range(0, len(landmarks), 2):
            # Get x and y coordinates
            x_point = int(landmarks[i] * w)  
            y_point = int(landmarks[i+1] * h)  
            cv2.circle(frame, (x + x_point, y + y_point), 2, (0, 255, 0), -1)
        
    # Show the frame with landmarks applied
    cv2.imshow('Facial Landmarks', frame)
    
    # Press 'q' to exit the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
