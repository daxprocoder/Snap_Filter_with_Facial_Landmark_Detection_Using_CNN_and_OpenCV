import cv2
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Load the cowboy hat filter
hat_path = 'filters/hats/realhat.png'
cowboy_hat = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
if cowboy_hat is None:
    print("Error: Cowboy hat image not found. Check the file path.")
    exit()

# Function to apply cowboy hat filter
def apply_cowboy_hat(frame, face_coords):
    for (x, y, w, h) in face_coords:
        try:
            # Resize the cowboy hat image to match face width
            hat_width = int(1.5 * w)
            hat_height = int(hat_width * cowboy_hat.shape[0] / cowboy_hat.shape[1])
            hat_resized = cv2.resize(cowboy_hat, (hat_width, hat_height))

            # Calculate position to overlay the hat
            x_offset = x - hat_width // 4
            
            # Adjust the height based on the hat file path
            y_offset = y - hat_height // 1 + 100  # Lower position for realhat.png

            # Ensure the hat stays within frame boundaries
            y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + hat_height)
            x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + hat_width)
            hat_y1, hat_y2 = max(0, -y_offset), min(hat_height, frame.shape[0] - y_offset)
            hat_x1, hat_x2 = max(0, -x_offset), min(hat_width, frame.shape[1] - x_offset)

            # Overlay the hat using alpha blending
            for c in range(0, 3):
                alpha = hat_resized[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255.0
                frame[y1:y2, x1:x2, c] = (
                    frame[y1:y2, x1:x2, c] * (1 - alpha) +
                    hat_resized[hat_y1:hat_y2, hat_x1:hat_x2, c] * alpha
                )
        except Exception as e:
            print(f"Error applying cowboy hat: {e}")
    return frame

# Function to detect faces in the frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces