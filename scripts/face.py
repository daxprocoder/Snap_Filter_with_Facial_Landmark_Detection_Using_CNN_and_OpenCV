import cv2
import numpy as np


# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Load the overlay image (ensure this is a transparent PNG)
overlay_img = cv2.imread('filters/face/mrbean.png', -1)  # -1 loads the image with alpha channel

# Check if the overlay image is loaded correctly
if overlay_img is None:
    print("Error: Overlay image not loaded. Check the file path.")
    exit()

# Function to apply the face overlay filter to the frame
def apply_face_overlay(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Parameters for scaling and positioning
    scaling_factor_w = 2.0  # Scaling factor for width (1.5 means 50% larger)
    scaling_factor_h = 2.0  # Scaling factor for height (2.0 means 100% larger)
    x_offset = -10  # Move the overlay slightly to the left (negative value moves it left)
    y_offset = -20  # Move the overlay slightly above the face (negative value moves it up)

    for (x, y, w, h) in faces:
        # Calculate new width and height with scaling factors
        new_w = int(w * scaling_factor_w)
        new_h = int(h * scaling_factor_h)

        # Ensure the new width and height fit within the frame (avoid out of bounds)
        new_w = min(new_w, frame.shape[1] - x)  # Ensure overlay doesn't go outside the frame width
        new_h = min(new_h, frame.shape[0] - y)  # Ensure overlay doesn't go outside the frame height

        # Resize the overlay image to the calculated size
        overlay_resized = cv2.resize(overlay_img, (new_w, new_h))

        # Extract the alpha channel (transparency) from the overlay image
        alpha_overlay = overlay_resized[:, :, 3] / 255.0  # Normalize alpha channel to 0-1
        alpha_frame = 1.0 - alpha_overlay

        # Adjust the x, y position of the overlay
        overlay_x = x + (w - new_w) // 2 + x_offset  # Move to the left of the center
        overlay_y = y - new_h + y_offset  # Move above the face

        # Ensure that the overlay fits within the frame
        if overlay_y + new_h <= frame.shape[0] and overlay_x + new_w <= frame.shape[1]:
            # Extract the region of interest (roi) from the frame where the overlay will be placed
            roi = frame[overlay_y : overlay_y + new_h, overlay_x : overlay_x + new_w]

            # Check if roi is valid (i.e., non-empty)
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue  # Skip this face if the roi is invalid (out of bounds)

            # Blend the overlay onto the frame using the alpha channel
            for c in range(0, 3):  # Iterate over color channels (R, G, B)
                roi[:, :, c] = (
                    alpha_overlay * overlay_resized[:, :, c] +
                    alpha_frame * roi[:, :, c]
                )

            # Place the blended region back in the frame
            frame[overlay_y : overlay_y + new_h, overlay_x : overlay_x + new_w] = roi

    return frame
