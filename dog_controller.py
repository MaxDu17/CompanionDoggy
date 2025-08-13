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

import socket
import sys

SOCKET_PATH = "/tmp/audio_socket"  # Inside container mount point


def play_audio(file_path):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)
    client.sendall(file_path.encode())
    client.close()

    print(f"[DOCKER] Sent play request for: {file_path}")


class DogController:
    def __init__(self, global_state: GlobalState, timestamps):
        self.global_state = global_state
        from unitree_sdk2py.go2.video.video_client import VideoClient

        now = datetime.now() # current date and time
        date_time = now.strftime("%m_%d_%Y__%H_%M_%S")

        self.color_output = imageio.get_writer(f"videos/{date_time}.mp4", fps = 10)

        self.speed_list = list() # this keeps track of .
        self.distances_list = list() 

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
        self.timestamps = timestamps 


    def run_warmup(self, duration = 30): 
        last_tag = None 
        last_error = 0 

        current_speed = 2
        self.global_state.lock_set("speed", current_speed) # start aligning the global state 
        CHECK_WINDOW = 15 # around 1.5 seconds 

        ispressing = False 
        active_control = False # needs an orange button press to start and stop 
        do_stop = False 
        no_person = True 

        prev_error = None 
        prev_time = None 


        Kp = -0.01
        Kd = -0.001 #-0.05  # You can tune this

        PERSON_SWITCH = False

        # print(self.sport_client.SwitchGait(2)) # fast trot 
        # print(self.sport_client.SpeedLevel(1)) # fast mode 

        start_time = time.time()
        start = time.time()
        inactive_time = 0 
        start_inactive = time.time()

        played_sound = False 

        while True: # MAIN EXECUTION LOOP 
            # safety stuff 
            current_speed = self.global_state.lock_get("speed")
            if self.remoteControl.getEstopState() == 1: 
                self.sport_client.Damp() 
                last_tag = None 
            if self.remoteControl.getDisableState() == 1 and not ispressing:
                active_control = not active_control 
                ispressing = True 
                if not active_control: # this logic 
                    start_inactive = time.time()
                    self.global_state.lock_set("mode", "Warmup_Idle")
                    self.timestamps.append(("Warmup: Idle", time.time()))

                else: 
                    inactive_time += (time.time() - start_inactive)
                    self.distances_list.clear() #
                    self.global_state.lock_set("mode", "Warmup")
                    self.timestamps.append(("Warmup: Active", time.time()))


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
                self.global_state.lock_set("person_distance", distance)

                # distance_error = -0.01 * (tag_location[2] - 750) # negate because we're backwards. Adjust the value for good following distance 

                # scaled_distance_error = np.clip(distance_error, 0, 3) # only allow forward motion 
                cv2.putText(info["frame"], "SPEED: " + str(round(scaled_distance_error, 1)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2, cv2.LINE_AA)
                cv2.putText(info["frame"], "TIME_ELAPSED: " + str(round(time.time() - start_time - inactive_time, 1)), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2, cv2.LINE_AA)

                # print(scaled_distance_error)

                # self.speed_list.append(scaled_distance_error)
                self.distances_list.append(distance)
            else:
                self.global_state.lock_set("person_distance", None)


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
                # this is responsible for the speed control logic 
                if len(self.distances_list) > CHECK_WINDOW:
                    average_dists = self.distances_list[-1] - self.distances_list[0]
                    print(current_speed, average_dists)

                    if average_dists < -100:
                        print("BUMPING UP SPEED")
                        current_speed += 0.5
                        current_speed = min(4, current_speed)
                    elif average_dists > 200: # asymmetrical speeding up and slowing down 
                        print("BUMPING DOWN SPEED")
                        current_speed -= 0.5
                        current_speed = max(1, current_speed)
                    self.global_state.lock_set("speed", current_speed)
                    self.distances_list.clear()
                self.speed_list.append(current_speed)
                self.sport_client.Move(current_speed, 0, control_output) #  scaled_position_error)

            # forwards, sideways, rotation

            cv2.imshow("Visual", info["frame"])
            cv2.imshow("Back", back)

            time_elapsed = time.time() - start 
            # print(time_elapsed)
            time.sleep(max((1/self.frame_rate) - time_elapsed, 0))
            start = time.time() 

            if active_control and time.time() - start_time - inactive_time > duration - 5 and not played_sound:
                played_sound = True 
                play_audio("/home/max/CompanionDoggy/assets/WarningTone5s.mp3")


            if duration is not None and active_control and time.time() - start_time - inactive_time > duration:
                self.timestamps.append(("Warmup: FINISHED", time.time()))

                return np.mean(self.speed_list)

    def run_interval(self, speed: int, duration: int = 30):
        base_speed = speed 
        if base_speed < 1.5:
            # too slow for slower mode, need to bump up the starting speed 
            slow_speed = base_speed 
            fast_speed = base_speed + 1 
        if base_speed >= 2.5:
            slow_speed = base_speed - 1 
            fast_speed = base_speed 
        else:
            slow_speed = base_speed - 0.5 
            fast_speed = base_speed + 0.5 

        self.timestamps.append(("Entering Fast Interval: Idle", time.time()))
        print("START FAST")
        self.run_fixed_speed(duration = duration, speed = fast_speed, mode = "run", default_control = False) # start faster 
        self.timestamps.append(("Entering Slow Interval: Active" , time.time()))
        print("START SLOW")
        self.run_fixed_speed(duration = duration, speed = slow_speed, mode = "run", default_control = True) # slower 
        self.timestamps.append(("Entering Fast Interval: Active", time.time()))
        print("START FAST")
        self.run_fixed_speed(duration = duration, speed = fast_speed, mode = "run", default_control = True) # faster 
        print("START SLOW")
        self.timestamps.append(("Entering Slow Interval: Active", time.time()))
        self.run_fixed_speed(duration = duration, speed = slow_speed, mode = "run", default_control = True) # slower 
        print("DONE")
        return slow_speed, fast_speed 

    def run_fixed_speed(self, speed: int, duration: int = None, mode = "run", default_control = False):
        last_tag = None 
        last_error = 0 
        speed = self.global_state.lock_set("speed", speed)


        ispressing = False 
        active_control = default_control #  False # needs an orange button press to start and stop 
        do_stop = False 
        no_person = True 

        prev_error = None 
        prev_time = None 


        Kp = -0.01
        Kd = -0.001 #-0.05  # You can tune this
        PERSON_SWITCH = True
        
        # if mode == "run":
        #     print(self.sport_client.SwitchGait(2)) # fast trot 
        #     print(self.sport_client.SpeedLevel(1)) # fast mode 
        # elif mode == "walk":
        #     print(self.sport_client.SwitchGait(1)) # regular trot 
        #     print(self.sport_client.SpeedLevel(0)) # regular mode  
        # else:
        #     raise Exception("Invalid mode")

        start_time = time.time()
        start = time.time()
        inactive_time = 0 
        start_inactive = time.time()

        played_sound = False 

        while True: # MAIN EXECUTION LOOP 
            # safety  
            speed = self.global_state.lock_get("speed")
            if self.remoteControl.getEstopState() == 1: 
                self.sport_client.Damp() 
                last_tag = None 
            if self.remoteControl.getDisableState() == 1 and not ispressing:
                active_control = not active_control 
                ispressing = True 

                if not active_control: # this logic 
                    start_inactive = time.time()
                    self.timestamps.append(("Constant speed: Idle", time.time()))

                    self.global_state.lock_set("mode", "Run_Idle")

                else: 
                    inactive_time += (time.time() - start_inactive)
                    self.global_state.lock_set("mode", "Run")
                    self.timestamps.append(("Constant speed: Active", time.time()))


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
                self.global_state.lock_set("person_distance", None)
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

            cv2.putText(info["frame"], f"P: {round(P, 2)}, D: {round(D, 2)}, S: {speed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2, cv2.LINE_AA)

            if active_control: 
                # sport_client.Move(4, 0, np.clip(scaled_position_error, -1.5, 1.5)) #  scaled_position_error)
                self.sport_client.Move(speed, 0, control_output) #  scaled_position_error)

            # forwards, sideways, rotation

            cv2.imshow("Visual", info["frame"])
            cv2.imshow("Back", back)

            time_elapsed = time.time() - start 
            # print(time_elapsed)
            time.sleep(max((1/self.frame_rate) - time_elapsed, 0))
            start = time.time() 

            if active_control and time.time() - start_time - inactive_time > duration - 5 and not played_sound:
                played_sound = True 
                play_audio("/home/max/CompanionDoggy/assets/WarningTone5s.mp3")


            if duration is not None and active_control and time.time() - start_time - inactive_time > duration:
                self.timestamps.append(("Constant Speed: DONE", time.time()))

                return np.mean(self.speed_list)
