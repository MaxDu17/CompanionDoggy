import socket
import numpy as np
import cv2
import threading 


import numpy
import os
import cv2 
import time 

class Insta360SharedMem:
    

# '127.0.0.1', 8080
class Insta360Socket:
    def __init__(self, ip, port):
        self.server_socket =  self.set_up_receiver(ip, port)
        self.most_recent_frame = None 
        thread = threading.Thread(target=self.monitor_stream, daemon=True)
        thread.start()

    def receive_image(self, crop = None):
        if self.most_recent_frame is not None: 
            if crop == "front":
                img = self.most_recent_frame[:, 0: self.most_recent_frame.shape[1] // 2]
            elif crop == "back":
                img = self.most_recent_frame[:, self.most_recent_frame.shape[1] // 2:]
            else:
                img = self.most_recent_frame 
            
            return img 
        return self.most_recent_frame 
    

    def monitor_stream(self):
        while True:
            self.most_recent_frame = self._receive_image() # this will hang until we get something 
            # time.sleep(0.01) 

    def set_up_receiver(self, server_ip, server_port):
        # Create a socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((server_ip, server_port))
        print("Connected to server")
        return client_socket 

    def _receive_image(self):
        buffer_size = int.from_bytes(self.server_socket.recv(4), byteorder='little')

        # Receive the actual image data
        buffer = b""
        while len(buffer) < buffer_size:
            buffer += self.server_socket.recv(buffer_size - len(buffer))

        # Convert the byte buffer to a numpy array and decode it as an image
        np_arr = np.frombuffer(buffer, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
        return img 


if __name__ == "__main__":
    camera = Insta360('127.0.0.1', 8080)
    import time 

    while True:
        img = camera.receive_image(crop = "front")
        # print(img.shape)
        if img is not None:
        # cv2.imwrite("test.png", color_img)
            cv2.imshow("Annotated Image", img)

        # cv2.imshow("bw Image", bwimg)
        cv2.waitKey(1)       