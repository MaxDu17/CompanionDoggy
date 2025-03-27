#include <iostream>
#include <thread>
#include "camera/camera.h"
#include "camera/photography_settings.h"
#include "camera/device_discovery.h"
#include <regex>

#include <opencv2/opencv.hpp>
// #include "opencv4"

#include <netinet/in.h>
#include <thread>
#include <chrono>

#include <fstream>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

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
#define IMAGE_SIZE 1440 * 720 * 3// 1 MB image
// #define IMAGE_SIZE 720 * 360 * 3// 1 MB image


struct SharedMemory {
    char data[IMAGE_SIZE];
};


class StreamProcessor : public ins_camera::StreamDelegate {
public:
    StreamProcessor() {
        //setting up the processing pipeline for the images 

        // Find the decoder for the h264
        // AVCodec* codec = avcodec_find_decoder_by_name("h264_cuvid");
        // if (!codec) {
        //     std::cerr << "NVIDIA decoder not available!" << std::endl;
        //     exit(1); 
        // }

        codec = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!codec) {
            std::cerr << "Codec not found\n";
            exit(1);
        }

        codecCtx = avcodec_alloc_context3(codec);
        codecCtx->flags2 |= AV_CODEC_FLAG2_FAST;
        codecCtx->thread_count = 4;  // Set the number of threads, e.g., 4
        codecCtx->thread_type =FF_THREAD_FRAME; // FF_THREAD_FRAME; // Threading per frame (can also be FF_THREAD_SLICE for slice-level threading)

        AVDictionary* opts = NULL;
        av_dict_set(&opts, "preset", "fast", 0);
        av_dict_set(&opts, "tune", "zerolatency", 0);


        if (!codecCtx) {
            std::cerr << "Could not allocate video codec context\n";
            exit(1);
        }

        // Open codec
        if (avcodec_open2(codecCtx, codec, &opts) < 0) {
        // if (avcodec_open2(codecCtx, codec, nullptr) < 0) {

            std::cerr << "Could not open codec\n";
            exit(1);
        }

        avFrame = av_frame_alloc();
        pkt = av_packet_alloc();

        shm_unlink(SHARED_MEM_NAME); //necessary to work under sudo 

         // Open shared memory
        int shm_fd = shm_open(SHARED_MEM_NAME, O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            std::cerr << "Error creating shared memory." << std::endl;
            exit(1);
        }

        // Set the size of the shared memory
        if (ftruncate(shm_fd, sizeof(SharedMemory)) == -1) {
            std::cerr << "Error setting size of shared memory." << std::endl;
            exit(1);
        }

        // Map shared memory
        shm_ptr = (SharedMemory*) mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shm_ptr == MAP_FAILED) {
            std::cerr << "Error mapping shared memory." << std::endl;
            exit(1);
        }

        // Open semaphore
        sem = sem_open(SEMAPHORE_NAME, O_CREAT, 0666, 0);  // Initial value 0, i.e., not ready
        if (sem == SEM_FAILED) {
            std::cerr << "Error creating semaphore." << std::endl;
            exit(1);
        }

    }
    ~StreamProcessor() {
        av_frame_free(&avFrame);
        av_packet_free(&pkt);
        avcodec_free_context(&codecCtx);
    }

    void setClient(int new_socket){
        client_socket = new_socket; 
    }

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {
        // std::cout << "on audio data:" << std::endl;
    }
    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index = 0) override {
        // Feed data into packet

        // if(time_taken < 10){
        //     return; 
        // }
        // std::cout << data << std::endl; 
        // std::cout << time_taken << std::endl; 
        // std::cout << "begin" << std::endl; 
        start = std::chrono::system_clock::now(); 

        if (stream_index == 0) {
            pkt->data = const_cast<uint8_t*>(data);
            pkt->size = size;

            // // Send the packet to the decoder
            if (avcodec_send_packet(codecCtx, pkt) == 0) {
                // Receive frame from decoder
                if (avcodec_receive_frame(codecCtx, avFrame) == 0) { //used to be "while"
                    int width = avFrame->width;
                    int height = avFrame->height;
                    int chromaHeight = height / 2;
                    int chromaWidth = width / 2;

                    int y_stride = avFrame->linesize[0];
                    int u_stride = avFrame->linesize[1];
                    int v_stride = avFrame->linesize[2];

                    cv::Mat yuv(height * 3 / 2, width, CV_8UC1, cv::Scalar(0));

                    // Copy Y plane
                    for (int i = 0; i < height; i++) {
                        memcpy(yuv.data + i * width, avFrame->data[0] + i * y_stride, width);
                    }

                    // Copy U plane
                    for (int i = 0; i < chromaHeight; i++) {
                        memcpy(yuv.data + width * height + i * chromaWidth, avFrame->data[1] + i * u_stride, chromaWidth);
                    }

                    // Copy V plane
                    for (int i = 0; i < chromaHeight; i++) {
                        memcpy(yuv.data + width * height + chromaWidth * chromaHeight + i * chromaWidth, avFrame->data[2] + i * v_stride, chromaWidth);
                    }


                         // Convert the YUV420P frame to BGR
                    end = std::chrono::system_clock::now(); 
                    double time_taken = std::chrono::duration<double, std::milli>(end-start).count(); 
                    std::cout << time_taken << std::endl; 
                    cv::Mat bgr;
                    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
                    // sendMatrix(bgr); 
                    cv::Mat smaller; 
                    cv::resize(bgr, smaller, cv::Size(width / 2, height / 2), cv::INTER_LINEAR);
                    
                    // sem_wait(sem); 
                    // std::cout << avFrame->width << " " << avFrame->height << std::endl; 
                    memcpy(shm_ptr->data, smaller.data, IMAGE_SIZE);
                    // sem_post(sem); //ready! 
                }
            }
        }
        start = std::chrono::system_clock::now(); 
    }
    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
    }
    void OnExposureData(const ins_camera::ExposureData& data) override {
        //fprintf(file2_, "timestamp:%lld shutter_speed_s:%f\n", data.timestamp, data.exposure_time);
    }

private:
    // FILE* file1_;
    // FILE* file2_;
    int64_t last_timestamp = 0;
    int client_socket = -1; 
    SharedMemory* shm_ptr; 
    int frame_count = 0; 
    std::chrono::high_resolution_clock::time_point start = std::chrono::system_clock::now(); 
    std::chrono::high_resolution_clock::time_point end = std::chrono::system_clock::now(); 
    sem_t* sem; 
    // int server_fd; 

    // int client_socket; 

    AVCodec* codec;
    AVCodecContext* codecCtx;
    AVFrame* avFrame;
    AVPacket* pkt;
    struct SwsContext* img_convert_ctx;
};



std::shared_ptr<ins_camera::Camera> cam; //global variable! 

void onExit(){
   if (cam->StopLiveStreaming()) {
            std::cout << "Successfully closed stream!" << std::endl;
        }
        else {
            std::cerr << "failed to stop live." << std::endl;
        }
    cam->Close();
}

#include <iostream>
#include <csignal>
#include <cstdlib>

// Function to handle SIGINT (Ctrl+C)
void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received.\n";
    std::cout << "Cleaning up before exit...\n";
    if(cam == nullptr){
        std::exit(signum); // Calls `atexit` functions before exiting
    }
    if (cam->StopLiveStreaming()) {
        std::cout << "Successfully closed stream!" << std::endl;
    }
    else {
        std::cerr << "failed to stop live." << std::endl;
    }
    cam->Close();

    std::exit(signum); // Calls `atexit` functions before exiting
}


int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler); 
    // std::atexit(onExit); 
    std::cout << "begin open camera" << std::endl;
    ins_camera::DeviceDiscovery discovery;
    auto list = discovery.GetAvailableDevices();
    for (int i = 0; i < list.size(); ++i) {
        auto desc = list[i];
        std::cout << "serial:" << desc.serial_number << "\t"
            << "camera type:" << int(desc.camera_type) << "\t"
            << "lens type:" << int(desc.lens_type) << std::endl;
    }

    if (list.size() <= 0) {
        std::cerr << "no device found." << std::endl;
        exit(EXIT_FAILURE); 
    }

    cam = std::make_shared<ins_camera::Camera>(list[0].info);


    //ins_camera::Camera cam(list[0].info);
    if (!cam->Open()) {
        std::cerr << "failed to open camera" << std::endl;
        exit(EXIT_FAILURE); 
    }

    auto exposure = std::make_shared<ins_camera::ExposureSettings>();
    exposure->SetExposureMode(ins_camera::PhotographyOptions_ExposureMode::PhotographyOptions_ExposureOptions_Program_MANUAL);//set to manual exposure mode
    exposure->SetIso(500); // set iso to 400
    exposure->SetShutterSpeed(1.0/300.0); // set shutter to 1/120 second.
    // auto success = cam.SetExposureSettings(ins_camera::CameraFunctionMode::FUNCTION_MODE_NORMAL_VIDEO, exposure);
    // auto ret = cam->SetExposureSettings(ins_camera::CameraFunctionMode::FUNCTION_MODE_NORMAL_IMAGE, exposure);

    auto success = cam->SetExposureSettings(ins_camera::CameraFunctionMode::FUNCTION_MODE_LIVE_STREAM, exposure);

    std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<StreamProcessor>();
    cam->SetStreamDelegate(delegate);
    discovery.FreeDeviceDescriptors(list);

    std::cout << "Succeed to open camera..." << std::endl;

    auto camera_type = cam->GetCameraType();

    // auto start = time(NULL);
    // cam->SyncLocalTimeToCamera(start);

    ins_camera::LiveStreamParam param;
    // param.video_resolution = ins_camera::VideoResolution::RES_1920_1920P24;
    // param.lrv_video_resulution = ins_camera::VideoResolution::RES_1920_1920P24;
    // param.video_resolution = ins_camera::VideoResolution::RES_480_240P30 ;
    // param.lrv_video_resulution = ins_camera::VideoResolution::RES_480_240P30;
    // param.video_resolution = ins_camera::VideoResolution::RES_720_360P30;
    // param.lrv_video_resulution = ins_camera::VideoResolution::RES_720_360P30;
    
    param.video_bitrate = 1024 * 1024 / 6;
    param.enable_audio = false;
    param.using_lrv = false;
    std::cout << "trying to start stream" << std::endl; 
    if (cam->StartLiveStreaming(param)) {
        std::cout << "successfully started live stream" << std::endl;
    }
    while(true); //hang until done 
}