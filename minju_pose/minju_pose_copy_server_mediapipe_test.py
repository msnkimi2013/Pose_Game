import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt










# Initialize the Pose Detection Model

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose

# Setting up the pose function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation
mp_drawing = mp.solutions.drawing_utils










# Read an ImageRead an Image

# Read an image from the specified path
sample_img = cv2.imread('./sample.jpg')

# Specify a size of the figure
plt.figure(figsize = [10, 10])

# Display the sample image, also convert BGR to RGB for display
# plt.title("Sample image")
# plt.axis('off')
# plt.imshow(sample_img[:,:,::-1])
# plt.show()










# Perform Pose Detection

# Perform pose detection after converting the image into RGB format
results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

# Check if any landmarks are found
if results.pose_landmarks:
    
    # Iterate two times as we only want to display first two landmarks
    for i in range(2):
        
        #Display the found normalized landmarks:
        print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')










# Retrieve the height and width of the sample image
image_height, image_width, _ = sample_img.shape

# Check if any landmarks are found:
if results.pose_landmarks:
    
    # Iterate two times as we only want to  display first two landmark
    for i in range(2):
        
        # Display  the found landmarks after converting then into their original scate
        print(f'{mp_pose.PoseLandmark(i).name}:')
        print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
        print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')
        print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
        print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')










# Draw the detected land marks on the Sample image

# Create a copy of the sample image to draw landmark on
img_copy = sample_img.copy()

# Check any landmarks are found
if results.pose_landmarks:
    
    # Draw pose landmarks on the sample image
    mp_drawing.draw_landmarks(image = img_copy, landmark_list = results.pose_landmarks, connections = mp_pose.POSE_CONNECTIONS)
    
    # Specify a size of the figure
    fig = plt.figure(figsize = [10, 10])
    
    #Display the output image with the landmarks drawn, also convert BGR to RGB for display
    # plt.title("Output")
    # plt.axis('off')
    # plt.imshow(img_copy[:,:,::-1])
    # plt.show()

    cv2.imwrite('./sample_img_landmarks.png',img_copy)



