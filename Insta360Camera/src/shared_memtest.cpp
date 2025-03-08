#include <iostream>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <vector>

#define SHARED_MEM_NAME "/shared_image"
#define SEMAPHORE_NAME "/image_semaphore"
#define IMAGE_SIZE 1024 * 1024 // 1 MB image
#define NUM_IMAGES 100

struct SharedMemory {
    char data[IMAGE_SIZE];
};

int main() {
    // Open shared memory
    int shm_fd = shm_open(SHARED_MEM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Error creating shared memory." << std::endl;
        return -1;
    }

    // Set the size of the shared memory
    if (ftruncate(shm_fd, sizeof(SharedMemory)) == -1) {
        std::cerr << "Error setting size of shared memory." << std::endl;
        return -1;
    }

    // Map shared memory
    SharedMemory* shm_ptr = (SharedMemory*) mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Error mapping shared memory." << std::endl;
        return -1;
    }

    // Open semaphore
    sem_t* sem = sem_open(SEMAPHORE_NAME, O_CREAT, 0666, 0);  // Initial value 0, i.e., not ready
    if (sem == SEM_FAILED) {
        std::cerr << "Error creating semaphore." << std::endl;
        return -1;
    }

    // Simulate sending images
    for (int i = 0; i < NUM_IMAGES; ++i) {
        // Simulate an image (for example, just fill with a pattern)
        memset(shm_ptr->data, ((i * 20) % 256), IMAGE_SIZE);

        // Signal that the image is ready
        sem_post(sem);

        std::cout << "Sent image " << i + 1 << std::endl;

        // Sleep to simulate processing time
        sleep(1);
    }

    // Cleanup
    sem_close(sem);
    munmap(shm_ptr, sizeof(SharedMemory));
    close(shm_fd);

    return 0;
}