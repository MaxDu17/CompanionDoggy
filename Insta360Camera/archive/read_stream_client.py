import socket
import numpy as np
import cv2


from dt_apriltags import Detector
import numpy
import os
import cv2 

at_detector = Detector(families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=2.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)



def set_up_receiver(server_ip, server_port):
    # Create a socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((server_ip, server_port))
    print("Connected to server")
    return client_socket 

def receive_image(server_socket):
    buffer_size = int.from_bytes(server_socket.recv(4), byteorder='little')

    # Receive the actual image data
    buffer = b""
    while len(buffer) < buffer_size:
        buffer += server_socket.recv(buffer_size - len(buffer))

    # Convert the byte buffer to a numpy array and decode it as an image
    np_arr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img 


if __name__ == "__main__":
    server_socket =  set_up_receiver('127.0.0.1', 8080)
    while True:
        img = receive_image(server_socket)  # Listen on localhost and port 8080
        if img is None:
            continue 
        bwimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(bwimg) # , True, camera_params, parameters['sample_test']['tag_size'])
        print(tags)

        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

            cv2.putText(img, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255))

        # cv2.imwrite("test.png", color_img)
        cv2.imshow("Annotated Image", img)
        # cv2.imshow("bw Image", bwimg)
        cv2.waitKey(1)       