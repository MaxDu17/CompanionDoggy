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

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

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

from dog_line import LineDetector, create_color_sliders    


sport_client = SportClient()  
sport_client.SetTimeout(10.0)
sport_client.Init()

remoteControl = RemoteHandler()
remoteControl.Init()

color_output = imageio.get_writer(f"videos/{date_time}.mp4", fps = 10)


# position_queue = deque(maxlen = 3)

DEVELOP_MODE = True
DETECT_GESTURES = True

last_tag = None 
last_error = 0 
integral_error = 0
vanish_counter = 0 

current_name = ""

camera = Insta360SharedMem() # ('127.0.0.1', 8080)

frame_rate = 10 
frame_rate_detector = 4

with open("Insta360Camera/camera_calibration/fisheye_calibration.json", "r") as f:
    calibration = json.load(f)
camera_matrix = np.array(calibration["K"])
dist_coeffs = np.array(calibration["D"])

start = time.time() 
last_detection_frame = time.time()


line_detector = LineDetector(
    width = 720, height = 720,
    K = camera_matrix,
    D = dist_coeffs
)

create_color_sliders("white") #line_detector.preload_colors)

ispressing = False 
active_control = False # needs an orange button press to start and stop 
do_stop = False 
no_person = True 

prev_error = None 
prev_time = None 


Kp = -0.01
Kd = -0.001 #-0.05  # You can tune this

Kp *= 1 
Kd *= 1
FORWARD_SPEED = 5
PERSON_SWITCH = False

print(sport_client.SwitchGait(2)) # fast trot 
print(sport_client.SpeedLevel(1)) # fast mode 

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


    back, front = camera.receive_image() # .copy()
    if front is None:
        continue
    info = line_detector.detect_line(front) # follow line 
    cv2.putText(info["frame"], "Control Status: " + ("active" if active_control else "disabled"), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(info["frame"], "Tracking status: " + info["message"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # detect person 
    rvecs, tvecs, image, ids = estimate_aruco_pose(back.copy(), camera_matrix, dist_coeffs) # TODO: THIS MIGHT NOT BE CORRECT ANYMORE 
    tag_location = None
    if tvecs is not None and ids[0] == 0: # second 
        tag_location = tvecs[0] # currently only tracking one tag 
        tag_location = tag_location[:, 0] # removethe exgtradimension 
        distance = tag_location[2]
        cv2.putText(info["frame"], "PERSON DISTANCE " + str(round(distance, 1)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2, cv2.LINE_AA)
        no_person = False
    else:
        no_person = True 
        cv2.putText(info["frame"], "PERSON TAG NOT DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 170), 2, cv2.LINE_AA)

    if not info["success"] or (no_person and PERSON_SWITCH):
        # show the error message on the camera 

        cv2.imshow("Visual", info["frame"])
        cv2.imshow("Back", back)

        cv2.waitKey(1) # this is to allow the frame to be shown 
        continue 

    # cv2.putText(info["frame"], str(round(info["angle"], 2)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

    best_line = info["best_line"]

    # print(info["angle"], info["x_at_target"])
    # print(best_line)
    # color_output.append_data(cv2.cvtColor(front,cv2.COLOR_BGR2RGB))
    color_output.append_data(cv2.cvtColor(back,cv2.COLOR_BGR2RGB))


    # angle_error= -0.02 * (info["angle"] - 90)

    # Persistent storage (you'll need to initialize this in your control loop)
    current_time = time.time() 
    if prev_error is None:
        prev_error = info["x_at_target"]
        prev_time = time.time()

    error = info["x_at_target"]
    P = Kp * error

    # Calculate time delta
    dt = current_time - prev_time 
    if dt == 0:
        D = 0
    else: 
        # Calculate errors
        d_error = (error - prev_error) / dt
        # Compute P and D terms
        D = Kd * d_error

    control_output = P + D

    prev_error = error
    prev_time = current_time


    # sport_client.Move(scaled_distance_error, scaled_position_error, yaw_error)
    # scaled_position_error = -0.01 * info["x_at_target"]
        # scaled_position_error = -0.005 * info["x_at_target"]
    print(control_output)
    # control_output = np.clip(control_output, -1.5, 1.5)

    cv2.putText(info["frame"], f"P: {round(P, 2)}, D: {round(D, 2)}, S: {FORWARD_SPEED}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2, cv2.LINE_AA)

    if active_control: 
        # sport_client.Move(4, 0, np.clip(scaled_position_error, -1.5, 1.5)) #  scaled_position_error)
        sport_client.Move(FORWARD_SPEED, 0, control_output) #  scaled_position_error)

    # forwards, sideways, rotation

    cv2.imshow("Visual", info["frame"])
    cv2.imshow("Back", back)


    time_elapsed = time.time() - start 
    # print(time_elapsed)
    time.sleep(max((1/frame_rate) - time_elapsed, 0))
    start = time.time() 

color_output.close()