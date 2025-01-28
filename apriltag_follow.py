from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2
import numpy as np
import sys
from dt_apriltags import Detector

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

from collections import deque

from safety_stack import RemoteHandler 

from scipy.spatial.transform import Rotation

at_detector = Detector(families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

if len(sys.argv)>1:
    ChannelFactoryInitialize(0, sys.argv[1])
else:
    ChannelFactoryInitialize(0)

client = VideoClient()  # Create a video client
client.SetTimeout(3.0)
client.Init()

# Open the default camera
cam = cv2.VideoCapture(0)
# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width)
print(frame_height)

sport_client = SportClient()  
sport_client.SetTimeout(10.0)
sport_client.Init()

remoteControl = RemoteHandler()
remoteControl.Init()

# position_queue = deque(maxlen = 3)

HEADLESS_MODE = False
DEVELOP_MODE = True
last_tag = None 
last_error = 0 
vanish_counter = 0 

def get_onboard_camera_image():
    code, data = client.GetImageSample()
    if code != 0: 
        print("Error with camera read!")
        return -1, False 

    # Convert to numpy image
    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR) #cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return color_img, True 

while True: # MAIN EXECUTION LOOP 
    # safety  
    if remoteControl.getEstopState() == 1: 
        sport_client.Damp() 
        last_tag = None 

    # color_img, status = get_onboard_camera_image()
    status, color_img = cam.read()
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)


    cameraMatrix = np.load("calibration_matrix.npy")
    camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

    # if visualization:
    #     cv2.imshow('Original image',img)

    tags = at_detector.detect(img, True, camera_params, 0.1)

    # tags = at_detector.detect(img) # , True, camera_params, parameters['sample_test']['tag_size'])


    if not HEADLESS_MODE:
        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 10)
            cv2.circle(color_img, (int(tag.center[0]), int(tag.center[1])), 20, (255, 0, 0), -1)
            cv2.putText(color_img, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255))
            print(tag.pose_t)
            # print(tag.pose_R)
            # rot = Rotation.from_matrix(tag.pose_R).as_euler("YZX")
            # print(rot)
                
    
    if len(tags) > 0: 
        last_tag = tags[-1]  # this ensures that we always have something to track 
        vanish_counter = 0 
     
    if len(tags) == 0:
        # print("################ Tag not detected! ###########")
        vanish_counter += 1 
        if vanish_counter > 5:
            last_tag = None # reset if we haven't seen the tag in a moment 

        
    if last_tag is not None: 
        tag_location = last_tag.pose_t 
    





        # # TODO: approximate orientation and size of the tag using camera parameters, using that approximate distance 
        # # approximating orientation allows us to match the rotation of the tag 
        # rough_scale = np.linalg.norm(last_tag.corners[0] - last_tag.corners[2]) 
        # HARDCODED_DISTANCE = 110 
        # distance_error = rough_scale - HARDCODED_DISTANCE 
        # scaled_distance_error = np.clip(distance_error / 100, -0.5, 0.5)
        
        # p_distance_signal = -scaled_distance_error 
        # # print(scaled_distance_error)
        # # print(rough_scale)
        
        # # track the last detected tag 
        # center = last_tag.center 
        # # center is (x, y) and image is (y, x)
        # position_error = (center[0] - color_img.shape[1] / 2) # -1 accounting for mirroring 
        # scaled_position_error = np.clip(position_error / (color_img.shape[1]), -0.5, 0.5) # conservative 
        # error_delta = scaled_position_error - last_error 
        # last_error = scaled_position_error 
        
        # pd_signal = -scaled_position_error + 0.5 * error_delta # the D controller 
        # # print(-scaled_position_error, pd_signal) 
        
        # print(step_value)
        if not DEVELOP_MODE: 
            sport_client.Move(p_distance_signal, pd_signal,0)
        # sport_client.Move(pd_distance_signal, 0,0)
        
        
    if not HEADLESS_MODE: 
        cv2.imshow("Output", color_img)
        if cv2.waitKey(20) == 27:
            break

