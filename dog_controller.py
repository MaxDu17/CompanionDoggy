from global_state import GlobalState
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
from datetime import datetime
from collections import deque
from safety_stack import RemoteHandler 
from scipy.spatial.transform import Rotation
import json 

if len(sys.argv)>1:
    ChannelFactoryInitialize(0, sys.argv[1])
else:
    ChannelFactoryInitialize(0)

from dog_line import LineDetector, create_color_sliders  

class DogController:
    def __init__(self, global_state: GlobalState):
        self.global_state = global_state
        from unitree_sdk2py.go2.video.video_client import VideoClient

        now = datetime.now() # current date and time
        date_time = now.strftime("%m_%d_%Y__%H_%M_%S")

        self.color_output = imageio.get_writer(f"videos/{date_time}.mp4", fps = 10)

        self.speed_list = list() # this keeps track of 

        self.sport_client = SportClient()  
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        self.remoteControl = RemoteHandler()
        self.remoteControl.Init()

        self.color_output = imageio.get_writer(f"videos/{date_time}.mp4", fps = 10)
        self.frame_rate = 10 

        self.camera = Insta360SharedMem() # ('127.0.0.1', 8080)

        with open("Insta360Camera/camera_calibration/fisheye_calibration.json", "r") as f:
            calibration = json.load(f)
        self.camera_matrix = np.array(calibration["K"])
        self.dist_coeffs = np.array(calibration["D"])

        self.line_detector = LineDetector(
            width = 720, height = 720,
            K = self.camera_matrix,
            D = self.dist_coeffs
        )

        create_color_sliders("white") #line_detector.preload_colors)

    def run_warmup(self, duration = 30): 
        self.global_state.lock_set("person_distance", 100)
        last_tag = None 
        last_error = 0 


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
        PERSON_SWITCH = False

        print(self.sport_client.SwitchGait(2)) # fast trot 
        print(self.sport_client.SpeedLevel(1)) # fast mode 

        start_time = time.time()
        start = time.time()
        inactive_time = 0 
        start_inactive = time.time()

        while True: # MAIN EXECUTION LOOP 
            # safety stuff 
            if self.remoteControl.getEstopState() == 1: 
                self.sport_client.Damp() 
                last_tag = None 
            if self.remoteControl.getDisableState() == 1 and not ispressing:
                active_control = not active_control 
                ispressing = True 
                if not active_control: # this logic 
                    start_inactive = time.time()
                else: 
                    inactive_time += (time.time() - start_inactive)
            if self.remoteControl.getDisableState() == 0:
                ispressing = False 
            
            while True: #this needs to go up top to prevent program from freezinggi
                key = cv2.waitKey(1)
                if key == -1:  # no keycode reported
                    break
                if key == ord('q'):
                    quit()

            back, front = self.camera.receive_image() # .copy()
            if front is None:
                continue
            info = self.line_detector.detect_line(front) # follow line 
            cv2.putText(info["frame"], "Control Status: " + ("active" if active_control else "disabled"), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(info["frame"], "Tracking status: " + info["message"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # detect person 
            rvecs, tvecs, image, ids = estimate_aruco_pose(back.copy(), self.camera_matrix, self.dist_coeffs) # TODO: THIS MIGHT NOT BE CORRECT ANYMORE 
            tag_location = None
            scaled_distance_error = 0 # don't move forward unless you see the person 
            if tvecs is not None and ids[0] == 0: # second 
                tag_location = tvecs[0] # currently only tracking one tag 
                tag_location = tag_location[:, 0] # removethe exgtradimension 
                distance = tag_location[2]
                distance_error = -0.01 * (tag_location[2] - 750) # negate because we're backwards. Adjust the value for good following distance 

                scaled_distance_error = np.clip(distance_error, 0, 3) # only allow forward motion 
                cv2.putText(info["frame"], "SPEED: " + str(round(scaled_distance_error, 1)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2, cv2.LINE_AA)
                cv2.putText(info["frame"], "TIME_ELAPSED: " + str(round(time.time() - start_time - inactive_time, 1)), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2, cv2.LINE_AA)

                print(scaled_distance_error)

                self.speed_list.append(scaled_distance_error)

            if not info["success"]:
                # show the error message on the camera 

                cv2.imshow("Visual", info["frame"])
                cv2.imshow("Back", back)

                cv2.waitKey(1) # this is to allow the frame to be shown 
                continue 


            best_line = info["best_line"]
            self.color_output.append_data(cv2.cvtColor(back,cv2.COLOR_BGR2RGB))

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

            cv2.putText(info["frame"], f"P: {round(P, 2)}, D: {round(D, 2)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2, cv2.LINE_AA)

            if active_control: 
                self.sport_client.Move(scaled_distance_error, 0, control_output) #  scaled_position_error)

            # forwards, sideways, rotation

            cv2.imshow("Visual", info["frame"])
            cv2.imshow("Back", back)

            time_elapsed = time.time() - start 
            # print(time_elapsed)
            time.sleep(max((1/self.frame_rate) - time_elapsed, 0))
            start = time.time() 

            if duration is not None and active_control and time.time() - start_time - inactive_time > duration:
                return np.mean(self.speed_list)

    def run_interval(self, speed: int, duration: int = 30):
        base_speed = speed 
        self.run_fixed_speed(duration, mode = "run") # start faster 
        self.run_fixed_speed(duration, mode = "walk") # slower 
        self.run_fixed_speed(duration, mode = "run") # faster 
        self.run_fixed_speed(duration, mode = "walk") # slower 

    def run_fixed_speed(self, speed: int, duration: int = None, mode = "run"):
        last_tag = None 
        last_error = 0 

        # speed = self.global_state.lock_get("speed") # minutes per mile # TODO: ENABLE SPEED SETTING ON ui

        ispressing = False 
        active_control = False # needs an orange button press to start and stop 
        do_stop = False 
        no_person = True 

        prev_error = None 
        prev_time = None 


        Kp = -0.01
        Kd = -0.001 #-0.05  # You can tune this
        FORWARD_SPEED = speed #5 / (1 / 14) * (1 / speed) # JENN: is this right?
        PERSON_SWITCH = False
        
        if mode == "run":
            print(self.sport_client.SwitchGait(2)) # fast trot 
            print(self.sport_client.SpeedLevel(1)) # fast mode 
        elif mode == "walk":
            print(self.sport_client.SwitchGait(1)) # regular trot 
            print(self.sport_client.SpeedLevel(0)) # regular mode  
        else:
            raise Exception("Invalid mode")

        start_time = time.time()
        start = time.time()
        inactive_time = 0 
        start_inactive = time.time()

        while True: # MAIN EXECUTION LOOP 
            # safety  
            if self.remoteControl.getEstopState() == 1: 
                self.sport_client.Damp() 
                last_tag = None 
            if self.remoteControl.getDisableState() == 1 and not ispressing:
                active_control = not active_control 
                ispressing = True 
                if not active_control: # this logic 
                    start_inactive = time.time()
                else: 
                    inactive_time += (time.time() - start_inactive)
            if self.remoteControl.getDisableState() == 0:
                ispressing = False 
            
            while True: #this needs to go up top to prevent program from freezinggi
                key = cv2.waitKey(1)
                if key == -1:  # no keycode reported
                    break
                if key == ord('q'):
                    do_stop = True
            if do_stop:
                break 


            back, front = self.camera.receive_image() # .copy()
            if front is None:
                continue
            info = self.line_detector.detect_line(front) # follow line 
            cv2.putText(info["frame"], "Control Status: " + ("active" if active_control else "disabled"), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(info["frame"], "Tracking status: " + info["message"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # detect person 
            rvecs, tvecs, image, ids = estimate_aruco_pose(back.copy(), self.camera_matrix, self.dist_coeffs) # TODO: THIS MIGHT NOT BE CORRECT ANYMORE 
            tag_location = None
            if tvecs is not None and ids[0] == 0: # second 
                tag_location = tvecs[0] # currently only tracking one tag 
                tag_location = tag_location[:, 0] # removethe exgtradimension 
                distance = tag_location[2]
                cv2.putText(info["frame"], "PERSON DISTANCE " + str(round(distance, 1)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2, cv2.LINE_AA)
                no_person = False
                self.global_state.lock_set("person_distance", distance)
                print(distance)
            else:
                no_person = True 
                self.global_state.lock_set("person_distance", np.inf)
                cv2.putText(info["frame"], "PERSON TAG NOT DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 170), 2, cv2.LINE_AA)

            if not info["success"] or (no_person and PERSON_SWITCH):
                # show the error message on the camera 

                cv2.imshow("Visual", info["frame"])
                cv2.imshow("Back", back)

                cv2.waitKey(1) # this is to allow the frame to be shown 
                continue 


            best_line = info["best_line"]
            self.color_output.append_data(cv2.cvtColor(back,cv2.COLOR_BGR2RGB))

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
            # print(control_output)

            cv2.putText(info["frame"], f"P: {round(P, 2)}, D: {round(D, 2)}, S: {FORWARD_SPEED}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2, cv2.LINE_AA)

            if active_control: 
                # sport_client.Move(4, 0, np.clip(scaled_position_error, -1.5, 1.5)) #  scaled_position_error)
                self.sport_client.Move(FORWARD_SPEED, 0, control_output) #  scaled_position_error)

            # forwards, sideways, rotation

            cv2.imshow("Visual", info["frame"])
            cv2.imshow("Back", back)

            time_elapsed = time.time() - start 
            # print(time_elapsed)
            time.sleep(max((1/self.frame_rate) - time_elapsed, 0))
            start = time.time() 

            if duration is not None and active_control and time.time() - start_time - inactive_time > duration:
                return np.mean(self.speed_list)
