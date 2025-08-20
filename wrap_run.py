import threading
from ui import GUI
from global_state import GlobalState
from dog_controller import DogController
import argparse
from threading import Thread
import signal
import time 
import os 
LOGS_DIR = "logs/"
import json 

def dump(timestamps, name):
    with open(os.path.join(LOGS_DIR, name + "_"     + str(round(time.time(), 0)) + ".txt"), "w") as f:
        for step in timestamps: 
            f.write(f"{step[0]},{step[1]}\n")


def main():
    name = input("Who is on the dog? \n")
    global_state = GlobalState()
    timestamps = list() 
    timestamps.append(("Start Program", time.time()))

    # Start the GUI in a separate thread
    gui = GUI(global_state)
    gui_thread = Thread(target=gui.demo.queue().launch, daemon=True)
    gui_thread.start()

    # Start dog controller in a separate thread

    dog_controller = DogController(global_state, timestamps)
    global_state.lock_set("mode", "Warmup_Idle")

    average_speed = dog_controller.run_warmup(duration = 30)
    global_state.lock_set("mode", "Run_Idle")
    global_state.lock_set("speed", average_speed)
    dump(timestamps, name)
    print(average_speed)
    base_speed = average_speed
    if base_speed < 1.5:
        # too slow for slower mode, need to bump up the starting speed 
        slow_speed = base_speed 
        fast_speed = base_speed + 1 
    if base_speed >= 2.5:
        slow_speed = base_speed - 1 
        fast_speed = base_speed 
    else:
        slow_speed = base_speed - 0.5 
        fast_speed = base_speed + 0.5 
    print(base_speed, slow_speed, fast_speed)
    print("----------")



    slow_speed, fast_speed = dog_controller.run_interval(speed = average_speed, duration = 30, logger = dump, name = name)
    timestamps.append(("FAST SPEED", fast_speed))
    timestamps.append(("SLOW SPEED", slow_speed))
    print(f"Fast speed: {fast_speed}. Slow speed: {slow_speed}")
    dump(timestamps, name)
    # with open(os.path.join(LOGS_DIR, name + "_"     + str(round(time.time(), 0)) + ".txt"), "w") as f:
    #     for step in timestamps: 
    #         f.write(f"{step[0]},{step[1]}\n")


if __name__ == "__main__":
    main() # args.mode, args.speed)