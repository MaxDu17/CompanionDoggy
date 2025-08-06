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

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

last_behavior = time.time()
BEHAVIOR_COOLDOWN = 1 # seconds between behavior exercution 
executing = False 
def process_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # print(result.hand_landmarks)
    global current_name, last_behavior, BEHAVIOR_COOLDOWN, executing 
    if len(result.hand_landmarks) > 0:
        landmark_x = [x.x for x in result.hand_landmarks[0]]
        x_range = max(landmark_x) - min(landmark_x)
        # print(x_range)
    gestures = result.gestures
    if len(gestures) > 0:
        print(gestures[0][0].category_name)
        current_name = gestures[0][0].category_name
        if current_name == "None":
            current_name = "" 
    else:
        current_name = ""


# 1 - Closed fist, label: Closed_Fist
# 2 - Open palm, label: Open_Palm
# 3 - Pointing up, label: Pointing_Up
# 4 - Thumbs down, label: Thumb_Down
# 5 - Thumbs up, label: Thumb_Up
# 6 - Victory, label: Victory
# 7 - Love, label: ILoveYou

import random 
def execute_behavior(name):
    if name == "Open_Palm":
        print("OPEN PALM EXECUTE -> should only print once")
        sport_client.Hello()
        return True 
    if name == "Victory":
        print("VICTORY EXECUTE -> should only print once")
        sport_client.Stretch()
        return True 
    if name == "ILoveYou":
        print("I LOVE YOU EXECUTE -> should only print once")
        if random.random() > 0.2: 
            sport_client.Dance1()
        else: 
            sport_client.Dance2() # randomly select a dance to do. There's a rare dance that is quite long 
        return True 
    if name == "Thumb_Up":
        print("STANDING UP AND ACTIVE")
        sport_client.StandUp()
        sport_client.BalanceStand()
        return True 

    if name == "Thumb_Down": 
        print("SITTING DOWN AND IDLE")
        sport_client.StandDown()
        return True 

    
    if name == "Closed_Fist":
        pass # BARK BARK  
        return True 

    
    if name == "EXCEPTIONAL_POUNCE":
        # ANNOUNCE  
        sport_client.FrontPounce()
        return True 

    if name == "EXCEPTIONAL_JUMP":
        sport_client.FrontJump()
        return True 


    return False 
    

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result)


recognizer = vision.GestureRecognizer.create_from_options(options)


camera = Insta360SharedMem() # ('127.0.0.1', 8080)

frame_rate = 10 
frame_rate_detector = 4

with open("Insta360Camera/camera_calibration/fisheye_calibration.json", "r") as f:
    calibration = json.load(f)


camera_matrix = np.array(calibration["K"])
dist_coeffs = np.array(calibration["D"])

start = time.time() 
last_detection_frame = time.time()

map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), camera_matrix, (720, 720), cv2.CV_32FC1)


active_control = False # master kill switch 
ispressing = False 
active_tracking = False # this is if we are tracking the large code  
do_stop = False 

behavior_dict = {
    1 : "Thumb_Up",
    2 : "Thumb_Down",
    3 : "ILoveYou",
    4 : "Open_Palm",
    5 : "EXCEPTIONAL_POUNCE",
    6 : "EXCEPTIONAL_JUMP"
}

while True: # MAIN EXECUTION LOOP 
    # safety  
    if remoteControl.getEstopState() == 1: 
        sport_client.Damp() 
        last_tag = None 
    if remoteControl.getDisableState() == 1 and not ispressing:
        active_control = not active_control 
        ispressing = True 
    if remoteControl.getDisableState() == 0:
        ispressing = False 
    
    while True: #this needs to go up top to prevent program from freezinggi
        key = cv2.waitKey(1)
        if key == -1:  # no keycode reported
            break
        if key == ord('q'):
            do_stop = True
    if do_stop:
        break 


    front = camera.receive_image(crop = "back")
    undistorted_front = cv2.remap(front, map1, map2, interpolation=cv2.INTER_LINEAR)

    if front is None:
        print("failed to retrieve frame")
        continue 
    rvecs, tvecs, image, ids = estimate_aruco_pose(front.copy(), camera_matrix, dist_coeffs)

    # TODO: only include one ID to prevent glitch 
    tag_location = None
    if rvecs is not None and ids[0] == 0:
        tag_location = tvecs[0] # currently only tracking one tag 
    elif rvecs is not None and ids[0][0] in behavior_dict: # make sure we are not in tracking mode
        current_name = behavior_dict[ids[0][0]]

    active_tracking = (tag_location is not None) 

    if tag_location is not None: 
        tag_location = tag_location[:, 0] # removethe exgtradimension 
        position_error = 0.002 * tag_location[0]
        velocity_error = position_error - last_error 
        # integral_error += position_error 
        # turn position into velocity 
        pd_error = position_error # + 1 * velocity_error # + 0.05 * integral_error 
        # scaled_position_error = -np.clip(pd_error, -1, 1)
        scaled_position_error = -np.clip(pd_error, -0.5, 0.5)
       
        distance_error = 0.0015 * (tag_location[2] - 750)
        # distance_error = 0.0015 * (tag_location[2] - 2000)

        # scaled_distance_error = np.clip(distance_error, -1.5, 2)
        scaled_distance_error = np.clip(distance_error, -0.5, 1)

    
        last_error = position_error 
        # print(scaled_distance_error, scaled_position_error)
        # sport_client.Move(scaled_distance_error, scaled_position_error, yaw_error)
        sport_client.Move(scaled_distance_error, 0, scaled_position_error)
        # forwards, sideways, rotation 
    

    h, w = front.shape[:2]
    x1 = w // 4
    y1 = h // 4
    x2 = 3 * w // 4
    y2 = 3 * h // 4
    if time.time() - last_detection_frame > (1 / frame_rate_detector) and DETECT_GESTURES:
        cropped_img = front[y1:y2, x1:x2]
        imgRGB = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        # imgRGB = cv2.cvtColor(undistorted_front, cv2.COLOR_BGR2RGB)

        current_timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        recognizer.recognize_async(mp_image, current_timestamp_ms) #, time.time() - start)

        last_detection_frame = time.time()
    if time.time()  - last_behavior > BEHAVIOR_COOLDOWN and not active_tracking and active_control:
        executed = execute_behavior(current_name)
        current_name = "" # clear just in case 
        if executed:
            last_behavior = time.time()
        

    if not HEADLESS_MODE: 
        cv2.rectangle(undistorted_front, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        cv2.putText(undistorted_front, "Detected behavior: " + current_name, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 0), 2, cv2.LINE_AA)

        cv2.putText(undistorted_front, "Control Status: " + ("active" if active_control else "disabled"), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1, cv2.LINE_AA)
        cv2.putText(undistorted_front, "Tracking status: " + str(active_tracking), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1, cv2.LINE_AA)
        cv2.imshow("Undistorted", undistorted_front)
        cv2.imshow("Raw Feed",image)
        if cv2.waitKey(1) == 27:
            break

    time_elapsed = time.time() - start 
    # print(time_elapsed)
    time.sleep(max((1/frame_rate) - time_elapsed, 0))
    start = time.time() 
    cv2.waitKey(1)

color_output.close()