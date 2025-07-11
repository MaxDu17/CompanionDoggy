from unitree_sdk2py.go2.video.video_client import VideoClient


from Insta360Camera.CalibratedInsta360 import Insta360Calibrated 
from Insta360Camera.Insta360_x4_client import Insta360SharedMem 

from read_aruco_360 import estimate_aruco_pose 

import cv2 
import numpy as np
import sys
import imageio 

import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
import math
import numpy as np 

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from datetime import datetime
now = datetime.now() # current date and time
date_time = now.strftime("%m_%d_%Y__%H_%M_%S")

from collections import deque

from safety_stack import RemoteHandler 

from scipy.spatial.transform import Rotation

import json 

if len(sys.argv)>1:
    ChannelFactoryInitialize(0, sys.argv[1])
else:
    ChannelFactoryInitialize(0)

sport_client = SportClient()  
sport_client.SetTimeout(10.0)
sport_client.Init()

remoteControl = RemoteHandler()
remoteControl.Init()

color_output = imageio.get_writer(f"videos/{date_time}.mp4", fps = 10)


# position_queue = deque(maxlen = 3)

HEADLESS_MODE = False
DEVELOP_MODE = True
DETECT_GESTURES = True

last_tag = None 
last_error = 0 
integral_error = 0
vanish_counter = 0 

current_name = ""

# BaseOptions = mp.tasks.BaseOptions
# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
# GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
# VisionRunningMode = mp.tasks.vision.RunningMode
#
# last_behavior = time.time()
# BEHAVIOR_COOLDOWN = 1 # seconds between behavior exercution
# executing = False
# def process_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
#     # print(result.hand_landmarks)
#     global current_name, last_behavior, BEHAVIOR_COOLDOWN, executing
#     if len(result.hand_landmarks) > 0:
#         landmark_x = [x.x for x in result.hand_landmarks[0]]
#         x_range = max(landmark_x) - min(landmark_x)
#         # print(x_range)
#     gestures = result.gestures
#     if len(gestures) > 0:
#         print(gestures[0][0].category_name)
#         current_name = gestures[0][0].category_name
#     else:
#         current_name = ""
#
# def execute_behavior(name):
#     if name == "Open_Palm":
#         print("OPEN PALM EXECUTE")
#         sport_client.Hello()
#         return True
#     if name == "Victory":
#         print("VICTORY EXECUTE")
#         sport_client.Stretch()
#         return True
#     if name == "ILoveYou":
#         print("I LOVE YOU EXECUTE")
#         sport_client.Dance1()
#         return True
#     return False
#
#
# options = GestureRecognizerOptions(
#     base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=process_result)
# recognizer = vision.GestureRecognizer.create_from_options(options)


camera = Insta360SharedMem() # ('127.0.0.1', 8080)

frame_rate = 10 
frame_rate_detector = 4

with open("../Insta360Camera/camera_calibration/fisheye_calibration.json", "r") as f:
    calibration = json.load(f)
camera_matrix = np.array(calibration["K"])
dist_coeffs = np.array(calibration["D"])

start = time.time() 
last_detection_frame = time.time()



# Create a window for the trackbars
cv2.namedWindow("Trackbars")

def nothing():
    pass
# Create trackbars for lower and upper HSV
cv2.createTrackbar("Lower H", "Trackbars", 90, 179, nothing)
cv2.createTrackbar("Lower S", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 110, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)






while True: # MAIN EXECUTION LOOP 
    # safety  
    if remoteControl.getEstopState() == 1: 
        sport_client.Damp() 
        last_tag = None 

    # read = cam.get_camera_frame()
    # if read is None:
    #     continue 
    # front, back = read.front_rgb, read.back_rgb 
    # # Define camera intrinsic parameters (example values, replace with actual calibration data)
    # camera_matrix = cam.K
    # dist_coeffs = cam.D

    front = camera.receive_image(crop = "back").copy()
    # read = cam.get_camera_frame()
    if front is None:
        continue

        # Convert the frame to HSV
    hsv = cv2.cvtColor(front, cv2.COLOR_BGR2HSV)

    # Get current positions of all trackbars
    l_h = cv2.getTrackbarPos("Lower H", "Trackbars")
    l_s = cv2.getTrackbarPos("Lower S", "Trackbars")
    l_v = cv2.getTrackbarPos("Lower V", "Trackbars")
    u_h = cv2.getTrackbarPos("Upper H", "Trackbars")
    u_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    u_v = cv2.getTrackbarPos("Upper V", "Trackbars")

    # Set the HSV range from trackbars
    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])

    # Create a mask using the HSV range
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours found
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 500:
            cv2.drawContours(front, [largest_contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(front, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(front, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(front, f"Line Center: ({cx}, {cy})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.imshow("Green Line Tracker", front)
            cv2.imshow("Mask", mask)
            cv2.waitKey(1)

            continue # don't move if you're not confieent
    # print(cx, front.shape)
    if abs(cx - front.shape[0] / 2) > 100:
        cv2.putText(front, f"LINE TOO FAR AWAY", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Green Line Tracker", front)
        cv2.waitKey(1)

        continue 
    # Show windows
    cv2.imshow("Green Line Tracker", front)
    cv2.imshow("Mask", mask)
    

    position_error = cx - front.shape[0] // 2
    position_error *= 0.01
    # pd_error = position_error  # + 1 * velocity_error # + 0.05 * integral_error
    # # scaled_position_error = -np.clip(pd_error, -1, 1)
    # pd_error = position_error # + 1 * velocity_error # + 0.05 * integral_error
    # scaled_position_error = -np.clip(pd_error, -1, 1)
    scaled_position_error = -np.clip(position_error, -0.5, 0.5)
    print(scaled_position_error)

    # sport_client.Move(scaled_distance_error, scaled_position_error, yaw_error)
    sport_client.Move(0.7, 0, scaled_position_error)
    # forwards, sideways, rotation


    # rvecs, tvecs, image = estimate_aruco_pose(front.copy(), camera_matrix, dist_coeffs)
    # tag_location = None
    # if rvecs is not None:
    #     for i in range(len(rvecs)):
    #         pass
    #         # print(f"Marker {i}: Rotation Vector: {rvecs[i].flatten()} Translation Vector: {tvecs[i].flatten()}")
    #         # print(f"Marker {i}: Translation Vector: {tvecs[i].flatten()}")
    #
    #     tag_location = tvecs[0] # currently only tracking one tag
    #
    # if tag_location is not None:
    #     # TODO: what happens to PD control when the tag isn't detected for a bit?
    #     tag_location = tag_location[:, 0] # removethe exgtradimension
    #     position_error = 0.002 * tag_location[0]
    #     velocity_error = position_error - last_error
    #     integral_error += position_error
    #     # turn position into velocity
    #     pd_error = position_error # + 1 * velocity_error # + 0.05 * integral_error
    #     # scaled_position_error = -np.clip(pd_error, -1, 1)
    #     scaled_position_error = -np.clip(pd_error, -0.5, 0.5)
    #
    #     # print(pd_error)
    #     # print(velocity_error)
    #     # print(integral_error)
    #
    #     distance_error = 0.0015 * (tag_location[2] - 750)
    #     # scaled_distance_error = np.clip(distance_error, -1.5, 2)
    #     scaled_distance_error = np.clip(distance_error, -0.5, 1)
    #
    #
    #     last_error = position_error
    #     # print(distance_error)
    #
    #
    #     print(scaled_distance_error, scaled_position_error)
    #     # sport_client.Move(scaled_distance_error, scaled_position_error, yaw_error)
    #     sport_client.Move(scaled_distance_error, 0, scaled_position_error)
    #     # forwards, sideways, rotation
    
    # color_output.append_data(cv2.cvtColor(color_img,cv2.COLOR_BGR2RGB))
    # color_output.append_data(image)
    # don't record video for deployment

    # GESTURE DETECTION CODE
    # if time.time() - last_detection_frame > (1 / frame_rate_detector) and DETECT_GESTURES:
    #     imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     current_timestamp_ms = int(time.time() * 1000)
    #     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    #     recognizer.recognize_async(mp_image, current_timestamp_ms) #, time.time() - start)
    #     cv2.putText(image, current_name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)
    #     last_detection_frame = time.time()
    # if time.time()  - last_behavior > BEHAVIOR_COOLDOWN:
    #     executed = execute_behavior(current_name)
    #     if executed:
    #         last_behavior = time.time()
        

    if not HEADLESS_MODE: 
        # cv2.imshow("Output",image)
        if cv2.waitKey(1) == 27:
            break

    time_elapsed = time.time() - start 
    # print(time_elapsed)
    time.sleep(max((1/frame_rate) - time_elapsed, 0))
    start = time.time() 
    cv2.waitKey(1)

color_output.close()