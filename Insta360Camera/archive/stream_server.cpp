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



class TestStreamDelegate : public ins_camera::StreamDelegate {
public:
    TestStreamDelegate() {
        std::string SERVER_IP = "127.0.0.1";  // Replace with your receiver's IP address
        int SERVER_PORT = 8080;

        // file1_ = fopen("./01.h264", "wb");
        // file2_ = fopen("./02.h264", "wb");

        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "Socket creation failed" << std::endl;
            return;
        }



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
        // av_init_packet(&pkt); 
        
        //decoding stuff 
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(SERVER_PORT);

        // Convert IP address from string to binary form
        if (inet_pton(AF_INET, SERVER_IP.c_str(), &serverAddr.sin_addr) <= 0) {
            std::cerr << "Invalid address or address not supported" << std::endl;
            close(sock);
            return;
        }

        // Connect to the Python server
        if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "Connection failed" << std::endl;
            close(sock);
            return;
        }

    }
    ~TestStreamDelegate() {
        // fclose(file1_);
        // fclose(file2_);
        close(sock); 
        av_frame_free(&avFrame);
        av_packet_free(&pkt);
        avcodec_free_context(&codecCtx);
    }


    void sendMatrix(const cv::Mat& mat) {
    // Convert cv::Mat to a byte buffer (serialization)
        std::vector<uchar> buffer;
        cv::imencode(".jpg", mat, buffer);  // You can use any encoding like .jpg, .png, etc.

        // Send the size of the buffer first
        int bufferSize = buffer.size();
        std::cout << bufferSize << " " << sizeof(bufferSize) << std::endl; 
        send(sock, &bufferSize, sizeof(bufferSize), 0);

        // Send the actual data (the encoded image)
        send(sock, buffer.data(), bufferSize, 0);
    }

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {
        std::cout << "on audio data:" << std::endl;
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

                    cv::Mat smaller; 
                    cv::resize(bgr, smaller, cv::Size(width / 4, height / 4), cv::INTER_LINEAR);
                    // cv::cvtColor(yuv, bgr, cv::COLOR_YUV420p2RGB);

                    
                    // cv::imwrite("test.png", bgr); 
                    
                    // cv::imshow("testing", yuv); 
                    sendMatrix(smaller); 
                }
            }
        }
        // std::cout << "on video frame:" << size << ";" << timestamp << std::endl;
        // if (stream_index == 0) {
        //     sendImage(data, size); 
        //     // fwrite(data, sizeof(uint8_t), size, file1_);
        // }
        // if (stream_index == 1) {
        //     fwrite(data, sizeof(uint8_t), size, file2_);
        // }
    }
    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        //for (auto& gyro : data) {
        //	if (gyro.timestamp - last_timestamp > 2) {
        //		fprintf(file1_, "timestamp:%lld package_size = %d  offtimestamp = %lld gyro:[%f %f %f] accel:[%f %f %f]\n", gyro.timestamp, data.size(), gyro.timestamp - last_timestamp, gyro.gx, gyro.gy, gyro.gz, gyro.ax, gyro.ay, gyro.az);
        //	}
        //	last_timestamp = gyro.timestamp;
        //}
    }
    void OnExposureData(const ins_camera::ExposureData& data) override {
        //fprintf(file2_, "timestamp:%lld shutter_speed_s:%f\n", data.timestamp, data.exposure_time);
    }

private:
    FILE* file1_;
    FILE* file2_;
    int64_t last_timestamp = 0;
    int sock; 

    AVCodec* codec;
    AVCodecContext* codecCtx;
    AVFrame* avFrame;
    AVPacket* pkt;
    struct SwsContext* img_convert_ctx;
};

std::shared_ptr<ins_camera::Camera> cam; 

void onExit(){
   if (cam->StopLiveStreaming()) {
            std::cout << "Successfully closed stream!" << std::endl;
        }
        else {
            std::cerr << "failed to stop live." << std::endl;
        }
    cam->Close();
}


int main(int argc, char* argv[]) {
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
        return -1;
    }

    cam = std::make_shared<ins_camera::Camera>(list[0].info);
    //ins_camera::Camera cam(list[0].info);
    if (!cam->Open()) {
        std::cerr << "failed to open camera" << std::endl;
        return -1;
    }

    //std::cout << "http base url:" << cam->GetHttpBaseUrl() << std::endl;

    std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<TestStreamDelegate>();
    cam->SetStreamDelegate(delegate);

    discovery.FreeDeviceDescriptors(list);

    std::cout << "Succeed to open camera..." << std::endl;

    auto camera_type = cam->GetCameraType();

    auto start = time(NULL);
    cam->SyncLocalTimeToCamera(start);

    ins_camera::LiveStreamParam param;
    param.video_resolution = ins_camera::VideoResolution::RES_720_360P30;
    param.lrv_video_resulution = ins_camera::VideoResolution::RES_720_360P30;
    param.video_bitrate = 1024 * 1024 / 2;
    param.enable_audio = false;
    param.using_lrv = false;
    if (cam->StartLiveStreaming(param)) {
        std::cout << "successfully started live stream" << std::endl;
    }
    while(true); //hang until we quit 

    return 0;
}