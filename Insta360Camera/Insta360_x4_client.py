import socket
import numpy as np
import cv2
import threading 


import numpy
import os
import cv2 
import time 


import mmap
import posix_ipc
import time
import os
import cv2
import numpy as np 


class Insta360SharedMem:
    def __init__(self):
        # Constants that match the C++ code
        SHARED_MEM_NAME = "shared_image"
        SEMAPHORE_NAME = "image_semaphore"
        self.IMAGE_SIZE = 1440 * 720 * 3 #1024 * 1024  # 1 MB
        # self.IMAGE_SIZE = 720 * 360 * 3 #1024 * 1024  # 1 MB


        # Open the shared memory using /dev/shm/
        # /dev/shm is where POSIX shared memory is stored
        shm_path = f"/dev/shm/{SHARED_MEM_NAME}"

        os.system("sudo chmod 666 " + shm_path)
        os.system("sudo chmod 666 /dev/shm/sem." + SEMAPHORE_NAME)

        # Open the shared memory file
        self.fd = os.open(shm_path, os.O_RDWR)
        self.shared_memory = mmap.mmap(self.fd, self.IMAGE_SIZE, access=mmap.ACCESS_READ)

        # Open the semaphore
        self.semaphore = posix_ipc.Semaphore("/" + SEMAPHORE_NAME)
    
    def receive_image(self, crop = None):
        # self.semaphore.acquire()
        data = self.shared_memory[:self.IMAGE_SIZE]
        array = np.frombuffer(data, dtype=np.uint8).reshape(720, 1440, 3)
        # array = np.frombuffer(data, dtype=np.uint8).reshape(360, 720, 3)

        if crop == "front":
            return array[:, 0: array.shape[1] // 2]
        elif crop == "back":
            return array[:, array.shape[1] // 2:]
        else:
            return array[:, 0: array.shape[1] // 2],  array[:, array.shape[1] // 2:]
    
    def clean_up(self):
        self.shared_memory.close()
        os.close(self.fd)

        


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
    camera = Insta360SharedMem() # ('127.0.0.1', 8080)
    import time 

    while True:
        img = camera.receive_image(crop = "front")
        # print(img.shape)
        if img is not None:
        # cv2.imwrite("test.png", color_img)
            cv2.imshow("Annotated Image", img)

        # cv2.imshow("bw Image", bwimg)
        cv2.waitKey(1)       