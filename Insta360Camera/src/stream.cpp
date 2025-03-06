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



class StreamProcessor : public ins_camera::StreamDelegate {
public:
    StreamProcessor() {
        //setting up the processing pipeline for the images 

        // Find the decoder for the h264
        codec = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!codec) {
            std::cerr << "Codec not found\n";
            exit(1);
        }

        codecCtx = avcodec_alloc_context3(codec);
        codecCtx->flags2 |= AV_CODEC_FLAG2_FAST;
        if (!codecCtx) {
            std::cerr << "Could not allocate video codec context\n";
            exit(1);
        }

        // Open codec
        if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
            std::cerr << "Could not open codec\n";
            exit(1);
        }

        avFrame = av_frame_alloc();
        pkt = av_packet_alloc();

    }
    ~StreamProcessor() {
        av_frame_free(&avFrame);
        av_packet_free(&pkt);
        avcodec_free_context(&codecCtx);
    }

    void setClient(int new_socket){
        client_socket = new_socket; 
    }


    void sendMatrix(const cv::Mat& mat) {
        if(client_socket == -1){ //if there isn't a connection, then don't send anything 
            return;
        }
    // Convert cv::Mat to a byte buffer (serialization)
        std::vector<uchar> buffer;
        cv::imencode(".jpg", mat, buffer);  // You can use any encoding like .jpg, .png, etc.

        // Send the size of the buffer first
        int bufferSize = buffer.size();
        // std::cout << bufferSize << " " << sizeof(bufferSize) << std::endl; 
        ssize_t result = send(client_socket, &bufferSize, sizeof(bufferSize), MSG_NOSIGNAL); 
        if(result < 0){
            client_socket = -1; 
            return;
        }
        // Send the actual data (the encoded image)
        result = send(client_socket, buffer.data(), bufferSize, MSG_NOSIGNAL); 
        if(result < 0){
            client_socket = -1; 
            return;
        }
    }

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {
        // std::cout << "on audio data:" << std::endl;
    }
    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index = 0) override {
        // Feed data into packet
        if (stream_index == 0) {
            pkt->data = const_cast<uint8_t*>(data);
            pkt->size = size;

            // // Send the packet to the decoder
            if (avcodec_send_packet(codecCtx, pkt) == 0) {
                // Receive frame from decoder
                while (avcodec_receive_frame(codecCtx, avFrame) == 0) {
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
                    cv::Mat bgr;
                    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
                    // sendMatrix(bgr); 
                    cv::Mat smaller; 
                    cv::resize(bgr, smaller, cv::Size(width / 2, height / 2), cv::INTER_LINEAR);
                    sendMatrix(smaller); 
                }
            }
        }
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
    param.video_resolution = ins_camera::VideoResolution::RES_720_360P30;
    param.lrv_video_resulution = ins_camera::VideoResolution::RES_720_360P30;
    param.video_bitrate = 1024 * 1024 / 2;
    param.enable_audio = false;
    param.using_lrv = false;
    std::cout << "trying to start stream" << std::endl; 
    if (cam->StartLiveStreaming(param)) {
        std::cout << "successfully started live stream" << std::endl;
    }

    //SET UP THE SERVER!! 
    int server_fd; 
    std::string SERVER_IP = "127.0.0.1";  // Replace with your receiver's IP address
    int SERVER_PORT = 8080; 

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Socket creation failed" << std::endl;
        exit(EXIT_FAILURE); 
    }
    
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(SERVER_PORT);

    // Bind socket to port
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

      // Listen for incoming connections
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }
    std::cout << "Listening on port " << SERVER_PORT << "..." << std::endl;

   
    int new_socket; 
    while(true){
        std::cout << "Waiting for another connection" << std::endl; 
    // Accept a client connection
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            perror("Accept failed");
            exit(EXIT_FAILURE);
        }
        StreamProcessor* processPtr = dynamic_cast<StreamProcessor*>(delegate.get()); //allows us to access the original function 
        processPtr->setClient(new_socket); 
    }
    return 0;
}