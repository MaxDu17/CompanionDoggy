from Insta360Camera.CalibratedInsta360 import Insta360Calibrated 
from Insta360Camera.Insta360_x4_client import Insta360SharedMem 
import cv2 


# from dt_apriltags import Detector
import numpy as np
import os
import cv2 

# at_detector = Detector(families='tagStandard41h12',
#                        nthreads=1,
#                        quad_decimate=1.0,
#                        quad_sigma=0.0,
#                        refine_edges=1,
#                        decode_sharpening=0.25,
#                        debug=0)


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        print(mtx, distortion)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def estimate_aruco_pose(image, camera_matrix, dist_coeffs, marker_length=160, aruco_dict_type=cv2.aruco.DICT_6X6_250):
    """
    Estimates the position and orientation of an ArUco tag.
    
    Parameters:
    - image_path: Path to the input image containing the ArUco tag.
    - camera_matrix: Intrinsic matrix of the camera (3x3 numpy array).
    - dist_coeffs: Distortion coefficients of the camera (1x5 or 1x8 numpy array).
    - marker_length: The length of the ArUco marker's side in meters.
    - aruco_dict_type: Type of ArUco dictionary to use.
    
    Returns:
    - rvecs: Rotation vectors of detected markers.
    - tvecs: Translation vectors of detected markers.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    
    if ids is None:
        print("No ArUco markers detected.")
        return None, None, image
    
    # Estimate pose of each detected marker
    rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
    
    # Draw detected markers and axes
    for i in range(len(ids)):
        cv2.aruco.drawDetectedMarkers(image, corners)
        # cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length * 0.5)
    
    # # Display the image with markers and pose
    # cv2.imshow("Detected ArUco Markers", image)
    
    return rvecs, tvecs, image


camera = Insta360SharedMem() # ('127.0.0.1', 8080)
cam = Insta360Calibrated(
    camera = camera, camera_resolution=(720, 720), image_save_path='images',camera_calibration_save_path='./camera_calibration'
)
# cam.calibrate_camera(chessboard_size=(8, 6), square_size=2.3, num_images=30)
cam.load_calibration("Insta360Camera/camera_calibration/fisheye_calibration.json")
cam.start_streaming()


while True:
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
            print(f"Marker {i}: Rotation Vector: {rvecs[i].flatten()} Translation Vector: {tvecs[i].flatten()}")

    # img = front 
    # bwimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # tags = at_detector.detect(bwimg) # , True, camera_params, parameters['sample_test']['tag_size'])
    # print(tags)

    # for tag in tags:
    #     for idx in range(len(tag.corners)):
    #         cv2.line(img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

    #     cv2.putText(img, str(tag.tag_id),
    #                 org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
    #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=0.8,
    #                 color=(0, 0, 255))

    cv2.imshow("Front", image)

    # cv2.imshow("Back", back)
    cv2.waitKey(1)


# cam.start_streaming()
# cam.live_streaming()
# cam.live_streaming(undistort=True)  # Start with distortion by default
cam.live_streaming_comparison()
# cam.stop_streaming()
cv2.destroyAllWindows()
