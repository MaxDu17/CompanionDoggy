# Importing Libraries
import cv2
import mediapipe as mp

from Insta360Camera.CalibratedInsta360 import Insta360Calibrated 
from Insta360Camera.Insta360_x4_client import Insta360SharedMem 


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

current_name = ""
# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # print(result.hand_landmarks)
    global current_name
    if len(result.hand_landmarks) > 0:
        landmark_x = [x.x for x in result.hand_landmarks[0]]
        x_range = max(landmark_x) - min(landmark_x)
        # print(x_range)
    gestures = result.gestures
    if len(gestures) > 0:
        print(gestures[0][0].category_name)
        current_name = gestures[0][0].category_name
    else:
        current_name = ""
    # print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


recognizer = vision.GestureRecognizer.create_from_options(options)
#
import time



camera = Insta360SharedMem() # ('127.0.0.1', 8080)
cam = Insta360Calibrated(
    camera = camera, camera_resolution=(720, 720), image_save_path='images',camera_calibration_save_path='./camera_calibration'
)
cam.load_calibration("Insta360Camera/camera_calibration/fisheye_calibration.json")
cam.start_streaming()

while True:
    # Read video frame by frame
    read = cam.get_camera_frame()
    if read is None:
        continue 
    front, back = read.front_rgb, read.back_rgb 
    img = back.copy() 
    # img = cam.undistort_frame(back) 
    # success, img = cap.read()

    # Flip the image(frame)
    # img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    current_timestamp_ms = int(time.time() * 1000)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

    recognizer.recognize_async(mp_image, current_timestamp_ms) #, time.time() - start)
    cv2.putText(img, current_name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

    # image, text, org, font, fontScale, color, thickness, lineType
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    time.sleep(0.25)