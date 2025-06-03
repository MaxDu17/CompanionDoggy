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

from dog_line import LineDetector 

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


# Create a window for the trackbars
cv2.namedWindow("Trackbars")

def nothing():
    pass



while True: # MAIN EXECUTION LOOP 
    # safety  
    if remoteControl.getEstopState() == 1: 
        sport_client.Damp() 
        last_tag = None 


    front = camera.receive_image(crop = "back").copy()
    if front is None:
        continue
    info = line_detector.detect_line(front) 
    if not info["success"]:
        # show the error message on the camera 
        cv2.putText(info["frame"], info["message"], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow("Visual", info["frame"])
        cv2.waitKey(1) # this is to allow the frame to be shown 
        continue 

    best_line = info["best_line"]
    cv2.imshow("Visual", info["frame"])
    print(best_line)
    color_output.append_data(cv2.cvtColor(front,cv2.COLOR_BGR2RGB))


    # sport_client.Move(scaled_distance_error, scaled_position_error, yaw_error)
    scaled_position_error = 0 
    sport_client.Move(0, 0, scaled_position_error)
    # forwards, sideways, rotation



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