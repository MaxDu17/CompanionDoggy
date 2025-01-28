from dt_apriltags import Detector
import numpy
import os
import cv2 

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

img = cv2.imread("test_image_rotation_0.png", cv2.IMREAD_GRAYSCALE)

tags = at_detector.detect(img) # , True, camera_params, parameters['sample_test']['tag_size'])
print(tags)

color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

for tag in tags:
    for idx in range(len(tag.corners)):
        cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

    cv2.putText(color_img, str(tag.tag_id),
                org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255))

cv2.imwrite("test.png", color_img)

# if visualization:
#     cv2.imshow('Detected tags', color_img)

#     k = cv2.waitKey(0)
#     if k == 27:         # wait for ESC key to exit
#         cv2.destroyAllWindows()