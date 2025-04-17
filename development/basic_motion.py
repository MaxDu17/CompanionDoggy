import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
import math
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, sys.argv[1])

    sport_client = SportClient()  
    state_monitor = RobotStateClient() 
    # state_monitor.Init()
    # print(state_monitor.ServiceList())
    
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    # sport_client.StandUp()
    # time.sleep(3)
    # while True:
    #     print(sport_client.GetState())
    input("press enter to walk")
    
    sport_client.Move(0.3,0,0)
    # sport_client.CrossStep()
    print("done!")