import mmap
import posix_ipc
import time
import os
import cv2
import numpy as np 

# Constants that match the C++ code
SHARED_MEM_NAME = "/shared_image"
SEMAPHORE_NAME = "/image_semaphore"
IMAGE_SIZE = 1024 * 1024  # 1 MB

# Open the shared memory using /dev/shm/
# /dev/shm is where POSIX shared memory is stored
shm_path = f"/dev/shm{SHARED_MEM_NAME}"

# Open the shared memory file
fd = os.open(shm_path, os.O_RDWR)
shared_memory = mmap.mmap(fd, IMAGE_SIZE, access=mmap.ACCESS_READ)

# Open the semaphore
semaphore = posix_ipc.Semaphore(SEMAPHORE_NAME)

# Simulate reading data from the shared memory
for i in range(100):  # Adjust this to match the number of images
    # Wait for the semaphore to be posted (indicating image is ready)
    semaphore.acquire()

    # Read the shared memory data
    data = shared_memory[:IMAGE_SIZE]
    array = np.frombuffer(data, dtype=np.uint8).reshape((1024, 1024))
    # print(array)
    # import ipdb 
    # ipdb.set_trace()
    cv2.imshow("test", array)
    cv2.waitKey(1) 
  
    print(f"Received image {i + 1} of size {len(data)} bytes")

    # Simulate processing (sleep)

# Clean up
shared_memory.close()
os.close(fd)
