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


# if __name__ == "__main__":

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

code, data = client.GetImageSample()

sport_client = SportClient()  
  
sport_client.SetTimeout(10.0)
sport_client.Init()


# Request normal when code==0
while code == 0:
    # Get Image data from Go2 robot
    code, data = client.GetImageSample()

    # Convert to numpy image
    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
    # image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    img = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
        
        
    tags = at_detector.detect(img) # , True, camera_params, parameters['sample_test']['tag_size'])

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 10)
        cv2.circle(color_img, (int(tag.center[0]), int(tag.center[1])), 20, (255, 0, 0), -1)

        # print(tag.center[0] - color_img.shape[1] / 2)

        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))
    
    if len(tags) > 0: 
        # track the last detected tag 
        center = tags[-1].center 
        offset = -1 * (center[0] - color_img.shape[1] / 2) # -1 accounting for mirroring 
        # if abs(offset) > 100:
        step_value = np.clip(offset / 1000, -0.5, 0.5) # conservative 
        print(step_value)
        sport_client.Move(0,step_value,0)


    # cv2.imshow("Output", color_img)


    if cv2.waitKey(20) == 27:
        break

