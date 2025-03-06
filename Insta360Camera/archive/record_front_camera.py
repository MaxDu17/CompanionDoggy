import socket
import numpy as np
import cv2


import imageio 
import numpy
import os
import cv2 
import time 




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
    writer = imageio.get_writer('recorded_calibration.mp4', fps=20)
    beg = time.time() 

    while time.time() - beg < 90: # 90 seconds 
        img = receive_image(server_socket)  # Listen on localhost and port 8080
        if img is None:
            continue 
     
        # cv2.imwrite("test.png", color_img)
        cropped_img = img[:, 0: img.shape[1] // 2]

        writer.append_data(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB) )
        cv2.imshow("Annotated Image", cropped_img)
        # cv2.imshow("bw Image", bwimg)
        cv2.waitKey(1)       