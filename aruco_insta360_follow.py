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
# from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from datetime import datetime
now = datetime.now() # current date and time
date_time = now.strftime("%m_%d_%Y__%H_%M_%S")

from collections import deque

from safety_stack import RemoteHandler 

from scipy.spatial.transform import Rotation

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

HEADLESS_MODE = True
DEVELOP_MODE = True
last_tag = None 
last_error = 0 
integral_error = 0
vanish_counter = 0 



camera = Insta360SharedMem() # ('127.0.0.1', 8080)
cam = Insta360Calibrated(
    camera = camera, camera_resolution=(720, 720), image_save_path='images',camera_calibration_save_path='./camera_calibration'
)
cam.load_calibration("Insta360Camera/camera_calibration/fisheye_calibration.json")
cam.start_streaming()

while True: # MAIN EXECUTION LOOP 
    # safety  
    if remoteControl.getEstopState() == 1: 
        sport_client.Damp() 
        last_tag = None 

    read = cam.get_camera_frame()
    if read is None:
        continue 
    front, back = read.front_rgb, read.back_rgb 
    # Define camera intrinsic parameters (example values, replace with actual calibration data)
    camera_matrix = cam.K
    dist_coeffs = cam.D
    rvecs, tvecs, image = estimate_aruco_pose(front.copy(), camera_matrix, dist_coeffs)
    if rvecs is not None:
        for i in range(len(rvecs)):
            # print(f"Marker {i}: Rotation Vector: {rvecs[i].flatten()} Translation Vector: {tvecs[i].flatten()}")
            print(f"Marker {i}: Translation Vector: {tvecs[i].flatten()}")

    tag_location = tvecs[0] # currently only tracking one tag 
        
    if last_tag is not None: 
        # TODO: what happens to PD control when the tag isn't detected for a bit? 
        position_error = 0.01 * tag_location[0]
        velocity_error = position_error - last_error 
        integral_error += position_error 
        # turn position into velocity 
        pd_error = position_error # + 1 * velocity_error # + 0.05 * integral_error 
        scaled_position_error = -np.clip(pd_error, -1, 1)
        # print(pd_error)
        # print(velocity_error)
        # print(integral_error)
       
        distance_error = 0.05 * (tag_location[2] - 500)
        scaled_distance_error = np.clip(distance_error, -1.5, 2)
        last_error = position_error 
        # print(distance_error)
        
        # rotation = Rotation.from_matrix(last_tag.pose_R).as_euler('xyz') # , degrees = True)
        # yaw_error = -rotation[1] 
        # scaled_yaw_error = np.clip(yaw_error, -0.5, 0.5)
        # print(yaw_error)

        
        # sport_client.Move(scaled_distance_error, scaled_position_error, yaw_error)
        sport_client.Move(scaled_distance_error, 0, scaled_position_error)
        # forwards, sideways, rotation 
    
    color_output.append_data(cv2.cvtColor(color_img,cv2.COLOR_BGR2RGB))
    
    if not HEADLESS_MODE: 
        cv2.imshow("Output", color_img)
        if cv2.waitKey(20) == 27:
            break

color_output.close()