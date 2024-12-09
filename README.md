# Snap_Filter_with_Facial_Landmark_Detection_Using_CNN_and_OpenCV

![Alt Text](https://cdn.discordapp.com/attachments/1315702512837197862/1315704210666553434/ui_ux.jpg?ex=67586088&is=67570f08&hm=0a28b9a4437d6c28faeb7b0942836d02366f25cae63e2f00c844a0419019dfe3&)


Real-time facial landmark detection with glasses overlay.

Overview
This project implements a real-time system for detecting facial landmarks using a Convolutional Neural Network (CNN) and OpenCV. The system identifies 30 facial landmarks, such as the eyes, nose, and mouth, and can overlay effects like glasses. By leveraging GPU acceleration, the system achieves high frames-per-second (FPS) performance, making it suitable for real-time applications like augmented reality, face recognition, and human-computer interaction.

Features
Real-Time Detection: Efficient facial landmark detection with webcam integration.
Facial Landmarks: Predicts 30 key landmarks for feature mapping.
Face Detection: Uses OpenCV's Haar Cascade for face detection.
GPU Acceleration: Optimized for high-speed predictions using TensorFlow's GPU capabilities.
Visual Overlay: Demonstrates overlays like glasses applied to detected faces.

Requirements

To run this project, install the following Python libraries:

pip install tensorflow opencv-python numpy matplotlib customtkinter

Alternatively, use the requirements.txt file:

pip install -r requirements.txt

How It Works
Face Detection:
The Haar Cascade (haarcascade_frontalface_default.xml) detects face regions in a video stream or image.
These regions are cropped and resized to 
96×96 pixels.
Landmark Prediction:
The CNN model predicts 30 facial landmarks for the detected face.
Coordinates are normalized and then mapped back to the original image.
Overlay Integration:
Demonstrates practical usage by overlaying a glasses image (thugglass.png) based on predicted landmarks.

Results
Accuracy and Loss
The model was trained for 20 epochs with the following results:

![Alt Text](https://media.discordapp.net/attachments/1315702512837197862/1315702838525038653/Screenshot_2024-12-09_103329.png?ex=67585f41&is=67570dc1&hm=5765991c9d176e348e0dfbc142ee57fec8d4c52c7f08ecd47a6cbb3b1c86f122&=&format=webp&quality=lossless&width=688&height=662)

Training Accuracy: 94%
Validation Accuracy: 91%
Loss Graph: 

![Alt Text](https://media.discordapp.net/attachments/1315702512837197862/1315702839200186429/Screenshot_2024-12-09_103407.png?ex=67585f41&is=67570dc1&hm=fb7f0f75ade2ec79cc8956594d5e93a96d213d31d74d41161e66430b39e13ff7&=&format=webp&quality=lossless&width=655&height=662)

Performance (FPS)
The system achieves the following performance:

![Alt Text](https://media.discordapp.net/attachments/1315702512837197862/1315702840186110002/Screenshot_2024-12-09_133950.png?ex=67585f41&is=67570dc1&hm=8581c98caa706b3e94c76937f593db1371b374e326ebbd2a0e0c8eddf5272102&=&format=webp&quality=lossless&width=502&height=70)

CPU: 8 FPS
GPU: 25 FPS

Real-Time Detection
The system can successfully detect facial landmarks and overlay glasses in real-time.
Example output:

![Alt Text](https://media.discordapp.net/attachments/1315702512837197862/1315702840479453204/Screenshot_2024-12-09_134921.png?ex=67585f41&is=67570dc1&hm=220d98355697dea1501bd162350dd60d8482107bd9c5a7963ccd8805f3bda4d0&=&format=webp&quality=lossless&width=768&height=662)

Model Architecture
The CNN model is structured as follows:

Input: 
96×96×1 grayscale images.
Convolutional Layers: Extract spatial features with 
3×3 filters.
MaxPooling Layers: Reduce spatial dimensions.
Dense Layers: Fully connected layers for regression.
Output: 30 coordinates representing facial landmarks.
Model Summary:

Layer (Type)             Output Shape        Param    
======================================================
Conv2D                   (96, 96, 32)        320       
MaxPooling2D             (48, 48, 32)        0         
Conv2D                   (48, 48, 64)        18496     
MaxPooling2D             (24, 24, 64)        0         
Flatten                  (36864)             0         
Dense                    (128)               4718720   
Dense                    (30)                3870      
======================================================
Total params: 4,737,406
Trainable params: 4,737,406


Result:
![Alt Text](https://media.discordapp.net/attachments/1315702512837197862/1315702838113861762/Screenshot_2024-12-09_103154.png?ex=67585f41&is=67570dc1&hm=af1012749c51721a11174b87d73a27a048bd9e0047688311caf473757a2c8f72&=&format=webp&quality=lossless&width=628&height=662)


Getting Started
1. Clone the Repository
git clone https://github.com/daxprocoder/Snap_Filter_with_Facial_Landmark_Detection_Using_CNN_and_OpenCV.git
cd facial-landmark-detection

2. Download the Dataset
Download the Kaggle dataset: [Kaggle](https://www.kaggle.com/datasets/nagasai524/facial-keypoint-detection/code) or [Drive](https://drive.google.com/drive/folders/18H1Gtdn7Wvx3T2t9iv60VrsLXB1WLuvB?usp=sharing).

3. Run the Scripts
Face Detection:
python testing/facedetect.py

4. Load the model.
create a folder name and paste the facial_landmark_model.h5 mode in that folder.
Model download link [Model link](https://drive.google.com/drive/folders/18H1Gtdn7Wvx3T2t9iv60VrsLXB1WLuvB?usp=sharing)

6. Run the Scripts
Face keypoint
python testing/facial_landmarktesting.py

7. Run the main file
Main Ui
python main.py

Future Work
Extend to detect multiple faces simultaneously.
Add support for non-frontal face detection.
Implement Transformer-based architectures for enhanced accuracy.

References
TensorFlow Documentation
OpenCV Documentation
Kaggle: Facial Keypoint Detection Dataset

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.



