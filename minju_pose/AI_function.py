import cv2
import numpy as np
import math
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt




# Initialize the Pose Detection Model

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose

# Setting up the pose function
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation
mp_drawing = mp.solutions.drawing_utils







# Create a Pose Detection Function

def detectPose(image, pose, display = True):
    """
    This function performs pose detection on an image
    Args:
        image: The in put image with a person whose pose landmarks need to bbe detected
        pose: The pose setup function required  to perform the pose detection
        display: A boolean value that is if set to true the function displays the original input image. 
    
    Return:
        ouutput_image : input image with detected pose
        landmarks: A list of detected landmarks converted into scale
    """
    
    # Create a copy of the input image 
    output_image = image.copy()
    
    # Convert the image from BGR to RGB
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the pose detection
    results = pose.process(imageRGB)
    
    height, width, _ = image.shape
    
    # Innitialize a list to store the detected landmarks
    landmarks = []
    
    # Check if any landmarks were detected
    if results.pose_landmarks:
        
        # Draw  pose lanndmarks  on the putput image
        mp_drawing.draw_landmarks(image = output_image, landmark_list = results.pose_landmarks, connections = mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
            
    # Check if the original input image and the resultant image are specified to be displayed
    if display:
        
        # Display the original input image and the resultant image
        plt.figure(figsize = [22,22])
        plt.subplot(121)
        plt.imshow(image[:,:,::-1])
        plt.title("Original image")
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(output_image[:,:,::-1])
        plt.title("Output image")
        plt.axis('off')
        
        # Also plot the pose lanndmarks in 3D
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    else:
        
        # Return the output image and the found landmark
        return output_image, landmarks










# Pose Classification with Angle

def calculateAngle(landmark1, landmark2, landmark3):
    
    '''
    calculate angle between three points
    args:
        
    Returns:
        angle
    '''
    
    # Get the required landmarks coordinate
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    
    # Claculate the angle between 3 points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than 0
    if angle < 0:
        
        angle += 360
        
    return angle










# Create a Function to Perform Pose Classification

def classifyPose(landmarks, ouput_image, display=False):
    '''
    Yoga pose classification suing angle of body jopints
    Args:
        landmarks: list of detected pose landmarks
        output_image: image with detected pose
        display: boolean 
    Returns:
        output_image: with detected pose landmarks
        label: classified pose
    '''
    
    # Initialize the label of the pose
    label = 'Unknown Pose'
    
    # Specify the color of label
    color = (0,0,255)
    
    # calculate the required angles
    # =============================================================================================
    
    # Get angle: left shoulder, elbow, wrist
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get angle: right shoulder, elbow, wrist
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    
    # Get angle: left elbow, wrist, hip
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    
    # Get angle: right hip, shoulder, elbow
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    
    # Get the angle: left hip, knee, ankle
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    # Get the angle: right hip, knee, ankle
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #===================================================================================================================
    # Check warrior 2 pose or T pose
    # arms be straight and shoulder should be at the specific angle
    #==================================================================================================================
    
    # Check if the both arms straight 
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        
        # Check if shoulders are at the required angle
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110 :
        
    # Check if it is the warrior 2 pose
    #=========================================================================================================================
    
            # Check if one leg is straight
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle >165 and right_knee_angle < 195 :
            
                # Check if the other leg is bended at the required angle
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120 :
                
                    # Specify the label warrior 2
                    label = 'Warrior 2 pose'
                    
    #============================================================================================================================
    # Check if T pose
    #===========================================================================================================================
            # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195 :
                
                #Specify the label 
                label = 'T pose'
                
    #===============================================================================================================================
    # Check Tree pose
    #===============================================================================================================================
    
    #Check if one leg is straigh
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195 :
        
        # Check the other leg at the required angle
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45 :
            
            label = 'Tree pose'
            
    #===========================================================================================================================
    
    # Check if the pose label success
    if label != 'Unknown Pose':
        
        # Update the color to green
        color = (0, 255, 0)
        
    # wirte the label on the output image 
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to display
    if display:
        
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1])
        plt.title("Output Image")
        plt.axis('off')
    
    else:
        
        # Return the output image and the pose label
        return output_image, label



def handle(video):

    
    

    # Jin code here ***



    # Pose Classification on Realtime Webcam

    # Initialize a resizable window
    # cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    # video.set(3,1280)
    # video.set(4,960)


    # Iterate untill the webcam success
    while video.isOpened():

        ok, frame = video.read()

        # check if frame is not ok
        if not ok:
    
            continue
            # break
    
        # Flip the frame horizontally
        # frame = cv2.flip(frame,1)

        frame_height, frame_width, _ = frame.shape

        # resize and keep ratio
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        # pose landmark detection
        frame, landmarks = detectPose(frame, pose, display=False)

        if landmarks:
    
            # pose classification
            frame, label_minju = classifyPose(landmarks, frame, display=False)
    
        # cv2.imshow('Pose Classification', frame)

        # how to  break
        if label_minju != 'Unknown Pose':
            break
        

    # Release the video capture object  
    # camera_video.release()
    # cv2.destroyAllWindows()





    # output label text ***
    # answer = ai_model(video)   by minju



    return label_minju
