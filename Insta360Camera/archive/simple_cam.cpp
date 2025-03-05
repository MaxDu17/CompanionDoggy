#include <iostream>
#include <thread>

#include "camera/camera.h"
#include "camera/photography_settings.h"
#include "camera/device_discovery.h"
#include "camera/ins_types.h"

#include "stream/stream_delegate.h"
#include "stream/stream_types.h"

#include "regex"

#include <thread>
#include <chrono>
int main(){
    ins_camera::DeviceDiscovery discovery;
    auto list = discovery.GetAvailableDevices();
    if(list.size() == 0){
        std::cout << "Your device is either not connected or you're not using sudo!" << std::endl; 
    }
    for(int i = 0;i < list.size(); ++i) {
        std::cout << "device:" << list[i].serial_number << std::endl;
    }
}


