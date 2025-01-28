import numpy as np
import cv2 as cv
import glob
import time

NEED_RECAPTURE = False
if NEED_RECAPTURE:

    # Open the default camera
    cam = cv.VideoCapture(0)
    # Get the default frame width and height
    frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    imcount = 0
    img_list = list()
    do_stop = False
    to_capture = False
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    while True:
        ret = False
        _, frame = cam.read()
        # Display the captured frame


        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        if to_capture: # when we are ready
            ret, corners = cv.findChessboardCorners(gray, (8, 6), None)
        # If found, add object points, image points (after refining them)
        if to_capture and ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.imwrite(f"calibration/{imcount}.jpg", frame)
            imcount += 1
            cv.drawChessboardCorners(frame, (8, 6), corners2, ret)
            to_capture = False

        cv.imshow('img', frame)
        while True:
            key = cv.waitKey(1)
            if key == -1:  # no keycode reported
                break
            if key == ord('q'):
                do_stop = True
            if key == ord('c'):
                to_capture = True

        if do_stop:
            break

        if ret:
            time.sleep(2)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

else:
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('calibration/*.jpg')
    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8,6), None)
        # If found, add object points, image points (after refining them)
        print(ret)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (8,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.save("calibration_matrix.npy", mtx)
    print(mtx)